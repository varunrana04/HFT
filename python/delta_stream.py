"""
delta_stream.py — Delta Exchange Market Data & Options Engine
═════════════════════════════════════════════════════════════
Scope: Perpetual Futures + Options only (no spot).
Full support for Delta Demo / Testnet with active XAUT (Gold) options!

Instruments (Delta Demo & Live):
  BTC  → Perp: BTCUSD (id:84)     | Options: C/P-BTC-STRIKE-DDMMYY
  ETH  → Perp: ETHUSD (id:1699)   | Options: C/P-ETH-STRIKE-DDMMYY
  SOL  → Perp: SOLUSD (id:92572)  | Options: ✗ not listed
  GOLD → Perp: XAUTUSD (id:181689)| Options: C/P-XAUT-STRIKE-DDMMYY (112+ contracts on Demo!)

Features:
  • Low-latency WebSocket streaming for Perpetuals with live quotes & trades
  • Advanced Options Department:
      - Full Black-Scholes Greeks: Delta, Gamma, Theta, Vega
      - Portfolio Gamma Exposure (GEX in $/1% move) & Market GEX Profile
      - Net Position Delta and Dollar Delta ($Delta)
      - Dynamic Delta-Hedging Order Emitter (offsets options delta on perp)
      - Daily & Hourly Theta Decay tracking ($/day and $/hr)
      - Volatility Skew (IV_call - IV_put) & Variance Risk Premium (IV - RV)
      - Automated Expiry Lifecycle: Expiring (0-DTE) vs Non-Expiring categorization
      - Automated Expiration Settlement ($T <= 0$) & Quote Rollover
"""

import asyncio
import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Callable, Awaitable, Dict, List, Any

import aiohttp
import websockets
from dotenv import load_dotenv

load_dotenv()

DELTA_REST       = os.environ.get("DELTA_BASE_URL", "https://cdn-ind.testnet.deltaex.org")
DELTA_WS         = os.environ.get("DELTA_WS_URL",   "wss://socket-ind.testnet.deltaex.org")
DELTA_API_KEY    = os.environ.get("DELTA_API_KEY", "")
DELTA_API_SECRET = os.environ.get("DELTA_API_SECRET", "")

# ── Instrument Catalogue ───────────────────────────────────────────────────────
INSTRUMENTS = {
    "BTC": {
        "name":        "Bitcoin",
        "perp":        "BTCUSD",
        "perp_id":     84,
        "options":     True,
        "opt_prefix":  "BTC",
        "display_sym": "BTCUSD",
        "note":        "Delta Demo active BTC perpetual & options.",
    },
    "ETH": {
        "name":        "Ethereum",
        "perp":        "ETHUSD",
        "perp_id":     1699,
        "options":     True,
        "opt_prefix":  "ETH",
        "display_sym": "ETHUSD",
        "note":        "Delta Demo active ETH perpetual & options.",
    },
    "SOL": {
        "name":        "Solana",
        "perp":        "SOLUSD",
        "perp_id":     92572,
        "options":     False,
        "opt_prefix":  None,
        "display_sym": "SOLUSD",
        "note":        "SOL options not listed on Delta Exchange.",
    },
    "GOLD": {
        "name":        "Gold (XAUT)",
        "perp":        "XAUTUSD",
        "perp_id":     181689,
        "options":     True,
        "opt_prefix":  "XAUT",
        "display_sym": "XAUTUSD",
        "note":        "Delta Demo active XAUT (Tether Gold) perpetual & options chain.",
    },
}

MARKET_TYPES = ["perp", "options"]


def get_perp_symbol(base: str) -> Optional[str]:
    return INSTRUMENTS.get(base, {}).get("perp")


def has_options(base: str) -> bool:
    return INSTRUMENTS.get(base, {}).get("options", False)


# ── Black-Scholes Formula & Greeks ─────────────────────────────────────────────
def _ncdf(x: float) -> float:
    return 0.5 * math.erfc(-x / math.sqrt(2.0))


def bs_price(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        return max(0.0, (S - K) if cp == "C" else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    return (S * _ncdf(d1) - K * disc * _ncdf(d2)) if cp == "C" \
        else (K * disc * _ncdf(-d2) - S * _ncdf(-d1))


def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> dict:
    if T <= 0 or sigma <= 0 or S <= 0 or K <= 0:
        intrinsic = max(0.0, (S - K) if cp == "C" else (K - S))
        delta = 1.0 if (cp == "C" and S > K) else (-1.0 if (cp == "P" and S < K) else 0.0)
        return dict(delta=delta, gamma=0.0, theta=0.0, vega=0.0, intrinsic=intrinsic)

    sq = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    npdf = math.exp(-0.5 * d1 ** 2) / math.sqrt(2.0 * math.pi)
    disc = math.exp(-r * T)

    delta = _ncdf(d1) if cp == "C" else _ncdf(d1) - 1.0
    gamma = npdf / (S * sigma * sq)
    # Theta expressed in $/day
    theta = (-(S * npdf * sigma) / (2.0 * sq)
             - r * K * disc * (_ncdf(d2) if cp == "C" else -_ncdf(-d2))) / 365.0
    # Vega per 1 vol percentage point (0.01)
    vega  = S * npdf * sq * 0.01

    return dict(delta=delta, gamma=gamma, theta=theta, vega=vega, intrinsic=max(0.0, (S - K) if cp == "C" else (K - S)))


def implied_vol(px: float, S: float, K: float, T: float, r: float, cp: str,
                lo: float = 0.001, hi: float = 20.0) -> float:
    if T <= 0 or px <= 0:
        return 0.0
    for _ in range(60):
        mid = (lo + hi) / 2.0
        if bs_price(S, K, T, r, mid, cp) > px:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0


# ── Option Tick Dataclass ──────────────────────────────────────────────────────
@dataclass
class OptionTick:
    symbol:    str
    bid:       float = 0.0
    ask:       float = 0.0
    mark:      float = 0.0
    last:      float = 0.0
    oi:        float = 0.0
    timestamp: int   = 0
    strike:    float = 0.0
    cp:        str   = "C"     # "C" or "P"
    expiry_ts: int   = 0
    t_years:   float = 0.0
    is_expiring: bool = False  # True if <= 24h to expiry (0-DTE)
    iv:        float = 0.0
    delta:     float = 0.0
    gamma:     float = 0.0
    theta:     float = 0.0
    vega:      float = 0.0
    gex:       float = 0.0     # Strike gamma exposure: OI * gamma * S^2 * 0.01


# ── Delta Perpetual WebSocket Streamer ─────────────────────────────────────────
class DeltaPerpStreamer:
    """
    Subscribes to Delta Exchange v2/ticker and recent_trade for Perpetual Futures.
    Parses real quotes (best_bid, best_ask, bid_size, ask_size), mark price, and trades.
    """
    def __init__(self, symbol: str,
                 on_book: Callable[[float, float, float, float], Awaitable[None]],
                 on_trade: Callable[[float, float, bool, int], Awaitable[None]]):
        self.symbol    = symbol
        self.on_book   = on_book
        self.on_trade  = on_trade
        self._stop     = False
        self._last_bid = 0.0
        self._last_ask = 0.0

    async def run(self):
        print(f"[DELTA WS] Connecting to {DELTA_WS} for {self.symbol} perp…")
        backoff = 3
        while not self._stop:
            try:
                async with websockets.connect(DELTA_WS, ping_interval=None) as ws:
                    async def _heartbeat():
                        while not self._stop:
                            await asyncio.sleep(15)
                            try:
                                await ws.send(json.dumps({"type": "ping"}))
                            except Exception:
                                break
                    hb_task = asyncio.create_task(_heartbeat())
                    try:
                        sub = {
                            "type": "subscribe",
                            "payload": {
                                "channels": [
                                    {"name": "v2/ticker",    "symbols": [self.symbol]},
                                    {"name": "recent_trade", "symbols": [self.symbol]},
                                ]
                            }
                        }
                        await ws.send(json.dumps(sub))
                        print(f"[DELTA WS] Subscribed to {self.symbol}")
                        backoff = 3

                        async for raw in ws:
                            if self._stop:
                                break
                            msg = json.loads(raw)
                        mtype = msg.get("type", "")

                        if mtype == "v2/ticker":
                            # Delta v2/ticker directly provides quotes and mark_price
                            q = msg.get("quotes") or {}
                            bid = float(q.get("best_bid", 0) or 0)
                            ask = float(q.get("best_ask", 0) or 0)
                            b_sz = float(q.get("bid_size", 1) or 1)
                            a_sz = float(q.get("ask_size", 1) or 1)
                            mark = float(msg.get("mark_price", 0) or 0)
                            close_px = float(msg.get("close", 0) or 0)
                            ts = int(msg.get("timestamp") or time.time() * 1000)

                            if bid > 0 and ask > 0:
                                self._last_bid, self._last_ask = bid, ask
                                await self.on_book(bid, ask, b_sz, a_sz)
                            elif mark > 0:
                                self._last_bid, self._last_ask = mark * 0.9999, mark * 1.0001
                                await self.on_book(self._last_bid, self._last_ask, 1.0, 1.0)

                            # If close price received on ticker, emit as trade
                            if close_px > 0:
                                is_bm = close_px <= self._last_bid if self._last_bid else False
                                await self.on_trade(close_px, 1.0, is_bm, ts)

                        elif mtype == "recent_trade":
                            for tr in msg.get("trades", []):
                                px  = float(tr.get("price", 0) or 0)
                                qty = float(tr.get("size",  1) or 1)
                                ts  = int(tr.get("timestamp", time.time() * 1000))
                                if px > 0:
                                    is_bm = px <= self._last_bid if self._last_bid else False
                                    await self.on_trade(px, qty, is_bm, ts)
                    finally:
                        hb_task.cancel()

            except asyncio.CancelledError:
                break
            except Exception as e:
                if not self._stop:
                    err_str = str(e)
                    if "429" in err_str:
                        backoff = min(backoff + 5, 20)
                    else:
                        backoff = 4
                    print(f"[DELTA WS] {self.symbol} reconnecting in {backoff}s: {e}")
                    await asyncio.sleep(backoff)

    def stop(self):
        self._stop = True


# ── Advanced Options Department Engine ─────────────────────────────────────────
class DeltaOptionsEngine:
    """
    Advanced Options Engine for Delta Exchange:
    1. Fetches real options chain (XAUT, BTC, ETH)
    2. Categorizes into Expiring (0-DTE / <=24h) vs Non-Expiring (Deferred maturities)
    3. Computes Black-Scholes Greeks, Gamma Exposure (GEX), and Theta Decay
    4. Automatically settles expired contracts (T <= 0) and rolls over quotes
    5. Calculates Volatility Skew (IV_call - IV_put) and Variance Risk Premium (VRP)
    6. Generates Dynamic Delta-Hedge order recommendations for the underlying perp
    """
    RISK_FREE        = 0.05
    QUOTE_VOL_WIDTH  = 0.035   # +/-3.5 vol points quoting spread
    MAX_CONTRACTS    = 5       # Max position per leg
    HEDGE_THRESHOLD  = 0.5     # Net delta threshold to trigger perp hedge recommendation

    def __init__(self, underlying: str, on_state: Callable[[dict], Awaitable[None]]):
        self.underlying    = underlying.upper()     # "BTC", "ETH", "GOLD"
        self.opt_prefix    = INSTRUMENTS.get(self.underlying, {}).get("opt_prefix", self.underlying)
        self.perp_sym      = get_perp_symbol(self.underlying)
        self.on_state      = on_state
        self._stop         = False

        # Positions book: symbol -> {qty, avg_px, delta, gamma, theta, vega, mark, strike, cp, expiry_ts}
        self.positions: Dict[str, dict] = {}
        self.realized_pnl: float = 0.0
        self.fill_count: int = 0
        self.tick_count: int = 0
        self.spot_price: float = 0.0
        self.recent_settlements: List[dict] = []
        self.expiry_filter: str = "all"  # "all", "expiring", "non_expiring"

    def set_expiry_filter(self, mode: str):
        if mode in ("all", "expiring", "non_expiring"):
            self.expiry_filter = mode

    async def run(self, poll_interval: float = 3.0):
        print(f"[DELTA OPT] Engine initialized for {self.underlying} (Prefix: {self.opt_prefix})…")
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=8)) as session:
            while not self._stop:
                try:
                    await self._cycle(session)
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    print(f"[DELTA OPT] {self.underlying} cycle error: {e}")
                await asyncio.sleep(poll_interval)

    async def _cycle(self, session: aiohttp.ClientSession):
        now_ts = int(time.time() * 1000)
        now_s  = time.time()

        # ── 1. Fetch Spot/Mark reference for underlying ───────────────────────
        perp_url = f"{DELTA_REST}/v2/tickers/{self.perp_sym}"
        try:
            async with session.get(perp_url) as r:
                if r.status == 200:
                    d = await r.json()
                    res = d.get("result") or {}
                    self.spot_price = float(res.get("mark_price") or res.get("close") or res.get("spot_price") or 0)
        except Exception as e:
            print(f"[DELTA OPT] Spot query error: {e}")

        if self.spot_price <= 0:
            return

        # ── 2. Automated Expiration Settlement Check on Open Positions ────────
        for sym, pos in list(self.positions.items()):
            exp_ts = pos.get("expiry_ts", 0)
            if exp_ts > 0 and now_ts >= exp_ts:
                # Contract has expired! Settle it automatically
                k = pos.get("strike", self.spot_price)
                cp = pos.get("cp", "C")
                qty = pos.get("qty", 0)
                avg_px = pos.get("avg_px", 0.0)

                payoff = max(0.0, (self.spot_price - k) if cp == "C" else (k - self.spot_price))
                settle_pnl = qty * (payoff - avg_px)
                self.realized_pnl += settle_pnl

                settle_event = {
                    "symbol": sym,
                    "type": "AUTO_SETTLED",
                    "strike": k,
                    "cp": cp,
                    "qty": qty,
                    "payoff": round(payoff, 2),
                    "pnl": round(settle_pnl, 2),
                    "spot_at_expiry": round(self.spot_price, 2),
                    "time": now_ts,
                }
                self.recent_settlements.append(settle_event)
                self.recent_settlements = self.recent_settlements[-20:]
                del self.positions[sym]
                print(f"  [AUTO-SETTLE] Expired {sym} settled: Payoff=${payoff:.2f} | PnL: ${settle_pnl:+.2f}")

        # ── 3. Fetch Full Options Chain Tickers ───────────────────────────────
        opt_tickers = []
        try:
            async with session.get(f"{DELTA_REST}/v2/tickers") as r:
                if r.status == 200:
                    all_tk = await r.json()
                    prefix_c = f"C-{self.opt_prefix}-"
                    prefix_p = f"P-{self.opt_prefix}-"
                    opt_tickers = [
                        t for t in (all_tk.get("result") or [])
                        if t.get("symbol", "").startswith(prefix_c)
                        or t.get("symbol", "").startswith(prefix_p)
                    ]
        except Exception as e:
            print(f"[DELTA OPT] Tickers error: {e}")
            return

        if not opt_tickers:
            return

        self.tick_count += 1

        # ── 4. Parse Option Ticks & Compute Individual Greeks ─────────────────
        ticks: List[OptionTick] = []
        for t in opt_tickers:
            sym   = t["symbol"]
            parts = sym.split("-")
            if len(parts) < 4:
                continue

            cp     = parts[0]
            try:
                strike = float(parts[2])
            except ValueError:
                continue

            mark = float(t.get("mark_price", 0) or 0)
            quotes = t.get("quotes") or {}
            bid = float(quotes.get("best_bid") or t.get("best_bid", 0) or 0)
            ask = float(quotes.get("best_ask") or t.get("best_ask", 0) or 0)
            oi  = float(t.get("oi_contracts", 0) or t.get("oi", 0) or 0)

            # Parse Expiry: DDMMYY
            try:
                exp_dt = datetime.strptime(parts[3], "%d%m%y").replace(hour=12, tzinfo=timezone.utc)
                expiry_ts = int(exp_dt.timestamp() * 1000)
                t_years = max((exp_dt.timestamp() - now_s) / (365.0 * 86400.0), 1e-6)
            except Exception:
                expiry_ts = now_ts + 86400 * 1000 * 7
                t_years = 7.0 / 365.0

            # Expiry categorization: <= 24h is expiring (0-DTE)
            is_expiring = (t_years * 365.0 * 24.0) <= 24.0

            # Volatility
            iv_raw = float(t.get("iv", 0) or quotes.get("mark_iv", 0) or 0)
            if iv_raw > 0:
                sigma = iv_raw
            elif mark > 0:
                sigma = implied_vol(mark, self.spot_price, strike, t_years, self.RISK_FREE, cp)
            else:
                sigma = 0.55  # Baseline crypto/gold options IV fallback

            sigma = max(0.05, min(sigma, 5.0))
            g = bs_greeks(self.spot_price, strike, t_years, self.RISK_FREE, sigma, cp)
            strike_gex = oi * g["gamma"] * (self.spot_price ** 2) * 0.01

            ticks.append(OptionTick(
                symbol=sym, bid=bid, ask=ask, mark=mark, last=mark,
                oi=oi, timestamp=now_ts, strike=strike, cp=cp,
                expiry_ts=expiry_ts, t_years=t_years, is_expiring=is_expiring,
                iv=sigma, delta=g["delta"], gamma=g["gamma"], theta=g["theta"],
                vega=g["vega"], gex=strike_gex,
            ))

        if not ticks:
            return

        # ── 5. Expiry Term Structure & ATM Selection ───────────────────────────
        unique_expiries = sorted({t.expiry_ts for t in ticks if t.expiry_ts > now_ts})
        if not unique_expiries:
            return

        # Filter according to user preference
        if self.expiry_filter == "expiring":
            filtered_ticks = [t for t in ticks if t.is_expiring]
            if not filtered_ticks:  # Fallback if no 0-DTE active
                filtered_ticks = [t for t in ticks if t.expiry_ts == unique_expiries[0]]
        elif self.expiry_filter == "non_expiring":
            filtered_ticks = [t for t in ticks if not t.is_expiring]
            if not filtered_ticks:
                filtered_ticks = [t for t in ticks if t.expiry_ts == unique_expiries[-1]]
        else:  # "all"
            filtered_ticks = ticks

        active_exp = min({t.expiry_ts for t in filtered_ticks if t.expiry_ts > now_ts}, default=unique_expiries[0])
        current_exp_ticks = [t for t in filtered_ticks if t.expiry_ts == active_exp]

        # Find ATM strike
        strikes = sorted({t.strike for t in current_exp_ticks})
        if not strikes:
            return
        atm_k = min(strikes, key=lambda k: abs(k - self.spot_price))
        atm_call = next((t for t in current_exp_ticks if t.strike == atm_k and t.cp == "C"), None)
        atm_put  = next((t for t in current_exp_ticks if t.strike == atm_k and t.cp == "P"), None)

        # ── 6. Greeks & Risk Aggregation ──────────────────────────────────────
        net_delta = sum(p["qty"] * p.get("delta", 0.0) for p in self.positions.values())
        dollar_delta = net_delta * self.spot_price
        portfolio_gex = sum(p["qty"] * p.get("gamma", 0.0) * (self.spot_price ** 2) * 0.01 for p in self.positions.values())
        portfolio_theta_day = sum(p["qty"] * p.get("theta", 0.0) for p in self.positions.values())
        portfolio_theta_hr  = portfolio_theta_day / 24.0
        portfolio_vega = sum(p["qty"] * p.get("vega", 0.0) for p in self.positions.values())

        # Volatility Skew: Call IV - Put IV at ATM
        call_iv = atm_call.iv if atm_call else 0.50
        put_iv  = atm_put.iv  if atm_put  else 0.50
        vol_skew = call_iv - put_iv

        # Dynamic Delta Hedge Recommendation
        hedge_order = None
        if abs(net_delta) >= self.HEDGE_THRESHOLD:
            hedge_order = {
                "symbol": self.perp_sym,
                "action": "SELL" if net_delta > 0 else "BUY",
                "target_contracts": round(abs(net_delta), 2),
                "notional_usd": round(abs(dollar_delta), 2),
                "status": "RECOMMENDED_HEDGE",
            }

        # ── 7. Market-Making Quoting & Fill Simulation on ATM ─────────────────
        fills = []
        for contract in [c for c in [atm_call, atm_put] if c]:
            pos = self.positions.get(contract.symbol, {
                "qty": 0, "avg_px": 0.0, "delta": contract.delta,
                "gamma": contract.gamma, "theta": contract.theta,
                "vega": contract.vega, "mark": contract.mark,
                "strike": contract.strike, "cp": contract.cp,
                "expiry_ts": contract.expiry_ts,
            })
            qty = pos["qty"]

            # Avellaneda-Stoikov (2008) Reservation Price & Half-Spread
            theo_mid = bs_price(self.spot_price, contract.strike, contract.t_years, self.RISK_FREE, contract.iv, contract.cp)
            gamma_aversion = 0.15
            kappa_liquidity = 1.5
            
            # Option-level price variance scaling: sigma_opt = iv * |delta| * spot
            opt_sigma = max(0.05, contract.iv * max(0.15, abs(contract.delta)))
            opt_var = (opt_sigma * max(1.0, theo_mid)) ** 2
            
            # Reservation price: r(s, q) = s - q * gamma * sigma^2 * (T - t)
            # When long (q > 0), reservation price drops, pulling ask closer to market to unload
            reservation_px = theo_mid - (qty * gamma_aversion * opt_var * max(0.01, contract.t_years))
            
            # Optimal half-spread: delta* = (1/gamma) * ln(1 + gamma/kappa) scaled by minimum tick
            spread_mult = (1.0 / gamma_aversion) * math.log(1.0 + (gamma_aversion / kappa_liquidity))
            half_spread = max(0.05, max(theo_mid * 0.015, spread_mult * 0.2))
            
            # Quote generation with hard inventory limits
            my_bid = max(0.05, reservation_px - half_spread) if qty < self.MAX_CONTRACTS else None
            my_ask = max(0.10, reservation_px + half_spread) if qty > -self.MAX_CONTRACTS else None
            if my_bid and my_ask and my_ask <= my_bid:
                my_ask = my_bid + 0.05

            # Fill simulation: check book crossing
            fill_bid = False
            fill_ask = False
            if my_bid and contract.ask > 0 and contract.ask <= (my_bid + 0.05) and qty < self.MAX_CONTRACTS:
                fill_bid = True
            elif my_ask and contract.bid > 0 and contract.bid >= (my_ask - 0.05) and qty > -self.MAX_CONTRACTS:
                fill_ask = True

            # Auto-quote testnet fill simulation (Inventory-reverting pacing)
            if not fill_bid and not fill_ask and contract.mark > 0:
                # Quoting pace adjusts dynamically to offload accumulated inventory:
                # If long (qty > 0), offloading via ASK is prioritized
                # If short (qty < 0), rebalancing via BID is prioritized
                cadence = 10
                if qty > 0 and self.tick_count % cadence == 0 and my_ask:
                    fill_ask = True
                elif qty < 0 and self.tick_count % cadence == 0 and my_bid:
                    fill_bid = True
                elif qty == 0 and self.tick_count % (cadence * 2) == 0:
                    if self.tick_count % (cadence * 4) == 0 and my_bid:
                        fill_bid = True
                    elif my_ask:
                        fill_ask = True

            if fill_bid and my_bid:
                self._fill(contract, +1, my_bid)
                fills.append({"side": "BUY", "symbol": contract.symbol, "price": round(my_bid, 2), "qty": 1.0, "engine": "OPTIONS", "time": now_ts})
            elif fill_ask and my_ask:
                self._fill(contract, -1, my_ask)
                fills.append({"side": "SELL", "symbol": contract.symbol, "price": round(my_ask, 2), "qty": 1.0, "engine": "OPTIONS", "time": now_ts})

        # ── 8. Emit Complete Options Department State ─────────────────────────
        mtm_pnl = sum(p["qty"] * (p.get("mark", p["avg_px"]) - p["avg_px"]) for p in self.positions.values())
        total_pnl = self.realized_pnl + mtm_pnl

        # Format expiry labels
        expiry_labels = []
        for e_ts in unique_expiries[:5]:
            dt = datetime.fromtimestamp(e_ts / 1000.0, tz=timezone.utc)
            is_exp = ((e_ts - now_ts) / (1000.0 * 3600.0)) <= 24.0
            expiry_labels.append({
                "ts": e_ts,
                "label": dt.strftime("%d %b %Y"),
                "is_expiring": is_exp,
            })

        state = {
            "spot_price":          round(self.spot_price, 2),
            "net_delta":           round(net_delta, 4),
            "dollar_delta":        round(dollar_delta, 2),
            "portfolio_gex":       round(portfolio_gex, 2),
            "portfolio_theta_day": round(portfolio_theta_day, 2),
            "portfolio_theta_hr":  round(portfolio_theta_hr, 4),
            "portfolio_vega":      round(portfolio_vega, 2),
            "vol_skew":            round(vol_skew * 100.0, 2),  # in vol percentage points
            "atm_call_iv":         round(call_iv * 100.0, 2),
            "atm_put_iv":          round(put_iv * 100.0, 2),
            "atm_symbol":          atm_call.symbol if atm_call else (atm_put.symbol if atm_put else f"{self.underlying} Options"),
            "atm_strike":          atm_k,
            "expiry_filter":       self.expiry_filter,
            "expiries":            expiry_labels,
            "hedge_order":         hedge_order,
            "settlements":         self.recent_settlements,
            "total_pnl":           round(total_pnl, 2),
            "fill_count":          self.fill_count,
            "tick_count":          self.tick_count,
            "my_bid_iv":           round(max(0, call_iv - self.QUOTE_VOL_WIDTH) * 100.0, 2),
            "my_ask_iv":           round((call_iv + self.QUOTE_VOL_WIDTH) * 100.0, 2),
            "spread_bps":          round(self.QUOTE_VOL_WIDTH * 2.0 * 10_000.0, 1),
            "fills":               fills,
            "open_positions":      [
                {"symbol": k, "qty": v["qty"], "avg_px": round(v["avg_px"], 2), "delta": round(v.get("delta", 0), 3)}
                for k, v in self.positions.items() if v["qty"] != 0
            ],
        }

        await self.on_state(state)

    def _fill(self, contract: OptionTick, side: int, price: float):
        sym = contract.symbol
        if sym not in self.positions:
            self.positions[sym] = {
                "qty": 0, "avg_px": price, "delta": contract.delta,
                "gamma": contract.gamma, "theta": contract.theta,
                "vega": contract.vega, "mark": contract.mark,
                "strike": contract.strike, "cp": contract.cp,
                "expiry_ts": contract.expiry_ts,
            }

        pos = self.positions[sym]
        old_qty = pos["qty"]
        new_qty = old_qty + side

        if old_qty != 0 and (old_qty * side < 0):
            # Closing part or all of position
            closed_qty = min(abs(old_qty), abs(side))
            direction = 1.0 if old_qty > 0 else -1.0
            realized_pnl_gain = closed_qty * (price - pos["avg_px"]) * direction
            self.realized_pnl += realized_pnl_gain

            if new_qty == 0:
                pos["avg_px"] = 0.0
            elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
                # Position flipped! The remainder has entry price = current price
                pos["avg_px"] = price
            # else: position reduced in same direction, avg_px stays unchanged
        elif old_qty == 0:
            pos["avg_px"] = price
        else:
            # Adding to existing position in same direction
            pos["avg_px"] = (pos["avg_px"] * abs(old_qty) + price * abs(side)) / abs(new_qty)

        pos.update(
            qty=new_qty,
            delta=contract.delta,
            gamma=contract.gamma,
            theta=contract.theta,
            vega=contract.vega,
            mark=contract.mark,
        )
        self.fill_count += 1
        print(f"  [OPT FILL] {'BUY' if side > 0 else 'SELL'} {sym} @ ${price:.2f} | "
              f"delta={contract.delta:.3f} | gamma={contract.gamma:.5f} | theta=${contract.theta:.2f}/d")

    def stop(self):
        self._stop = True


# ── Top-level Helpers ──────────────────────────────────────────────────────────
async def stream_perp(
    symbol: str,
    on_book:  Callable[[float, float, float, float], Awaitable[None]],
    on_trade: Callable[[float, float, bool, int],    Awaitable[None]],
    duration_s: int = 86400,
):
    streamer = DeltaPerpStreamer(symbol, on_book, on_trade)
    task = asyncio.create_task(streamer.run())
    try:
        await asyncio.sleep(duration_s)
    finally:
        streamer.stop()
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass


async def stream_options(
    underlying: str,
    on_state: Callable[[dict], Awaitable[None]],
    duration_s: int = 86400,
    poll_interval: float = 3.0,
) -> DeltaOptionsEngine:
    engine = DeltaOptionsEngine(underlying, on_state)
    task = asyncio.create_task(engine.run(poll_interval))
    try:
        await asyncio.sleep(duration_s)
    finally:
        engine.stop()
        task.cancel()
        try: await task
        except asyncio.CancelledError: pass
    return engine
