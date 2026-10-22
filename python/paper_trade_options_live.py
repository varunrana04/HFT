"""
paper_trade_options_live.py
─────────────────────────────────────────────────────────────────────────────
Paper-trading options market-maker on Binance European Options (eAPI).

Architecture
────────────
• Streams the Binance eAPI /eapi/v1/ticker REST poll (WebSocket stream for
  options trades uses a different auth flow — REST polling at 1s is simpler
  and avoids the extra exchange-key requirement).
• Computes Black-Scholes Greeks locally as a fallback when Binance doesn't
  return them (happens for deep OTM/ITM strikes).
• Quotes a two-sided market (bid/ask) on the ATM straddle of the nearest
  expiry using a simple skew model.
• Tracks realized PnL with MTM from the options mark price.
• Designed to be swapped in where paper_trade_rl_live.RLLimitSimulator is
  used (same on_book_tick / on_trade_tick interface where possible).

NO REAL ORDERS ARE PLACED — paper simulation only.
"""

import asyncio
import math
import time
import json
import datetime
from dataclasses import dataclass, field
from typing import Optional
import aiohttp

# ── Black-Scholes helper ──────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Accurate normal CDF using math.erfc."""
    return 0.5 * math.erfc(-x / math.sqrt(2))

def bs_price(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> float:
    """European Black-Scholes price. cp = 'C' or 'P'."""
    if T <= 0 or sigma <= 0:
        return max(0.0, (S - K) if cp == 'C' else (K - S))
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
    d2 = d1 - sigma * math.sqrt(T)
    disc = math.exp(-r * T)
    if cp == 'C':
        return S * _norm_cdf(d1) - K * disc * _norm_cdf(d2)
    else:
        return K * disc * _norm_cdf(-d2) - S * _norm_cdf(-d1)

def bs_greeks(S: float, K: float, T: float, r: float, sigma: float, cp: str) -> dict:
    """Return delta, gamma, theta (per day), vega (per 1% vol)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0}
    sq_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sq_T)
    d2 = d1 - sigma * sq_T
    nd1 = math.exp(-0.5 * d1**2) / math.sqrt(2 * math.pi)   # pdf
    disc = math.exp(-r * T)
    delta = _norm_cdf(d1) if cp == 'C' else _norm_cdf(d1) - 1.0
    gamma = nd1 / (S * sigma * sq_T)
    # theta per calendar day
    theta = (
        -S * nd1 * sigma / (2 * sq_T)
        - r * K * disc * (_norm_cdf(d2) if cp == 'C' else -_norm_cdf(-d2))
    ) / 365.0
    vega = S * nd1 * sq_T * 0.01     # per 1% change in vol
    return {"delta": delta, "gamma": gamma, "theta": theta, "vega": vega}

def implied_vol(market_price: float, S: float, K: float, T: float, r: float, cp: str,
                lo=0.001, hi=20.0, tol=1e-6, max_iter=100) -> float:
    """Bisection IV solver."""
    for _ in range(max_iter):
        mid = (lo + hi) / 2.0
        val = bs_price(S, K, T, r, mid, cp) - market_price
        if abs(val) < tol:
            return mid
        if val > 0:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2.0

# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class OptionContract:
    symbol:     str       # e.g. BTC-241227-90000-C
    underlying: str       # BTCUSDT
    strike:     float
    expiry_ms:  int       # Binance epoch ms
    cp:         str       # C or P
    # Live market data
    bid:        float = 0.0
    ask:        float = 0.0
    mark:       float = 0.0
    last:       float = 0.0
    iv:         float = 0.0
    delta:      float = 0.0
    gamma:      float = 0.0
    theta:      float = 0.0
    vega:       float = 0.0

    @property
    def tte_years(self) -> float:
        """Time-to-expiry in years from now."""
        remaining_ms = self.expiry_ms - int(time.time() * 1000)
        return max(remaining_ms / (1000 * 86400 * 365), 1e-6)

    @property
    def expiry_date(self) -> str:
        return datetime.datetime.utcfromtimestamp(self.expiry_ms / 1000).strftime('%Y-%m-%d')


@dataclass
class OptionsPosition:
    contract:  OptionContract
    qty:       int = 0      # + long, - short
    avg_price: float = 0.0

    @property
    def mtm_pnl(self) -> float:
        return self.qty * (self.contract.mark - self.avg_price)

    @property
    def delta_exposure(self) -> float:
        return self.qty * self.contract.delta


# ── Options Paper Simulator ───────────────────────────────────────────────────

class OptionsMarketMaker:
    """
    Paper market-maker for Binance European Options.
    Quotes ATM straddle on nearest expiry; fills when bid/ask cross market.
    """
    RISK_FREE_RATE = 0.05     # 5% annualised
    QUOTE_WIDTH_IV = 0.02     # quote at IV ± 2% (vol points)
    MAX_DELTA_EXPOSURE = 10.0 # flatten if |net delta| > 10
    MAX_CONTRACTS    = 5      # max contracts per leg

    def __init__(self):
        self.positions: dict[str, OptionsPosition] = {}
        self.realized_pnl: float = 0.0
        self.fill_count:   int   = 0
        self.tick_count:   int   = 0
        self.chain:        dict[str, OptionContract] = {}  # symbol → contract

        # State for the dashboard
        self.spot_price:   float = 0.0
        self.atm_call:     Optional[OptionContract] = None
        self.atm_put:      Optional[OptionContract] = None
        self.my_bid_iv:    float = 0.0
        self.my_ask_iv:    float = 0.0
        self.net_delta:    float = 0.0

    def update_contract(self, c: OptionContract, spot: float):
        """Update contract Greeks from Binance ticker data."""
        self.chain[c.symbol] = c
        # Fill in BS Greeks if Binance didn't return them
        if c.delta == 0.0 and c.iv == 0.0 and c.mark > 0 and spot > 0:
            T = c.tte_years
            c.iv = implied_vol(c.mark, spot, c.strike, T, self.RISK_FREE_RATE, c.cp)
        if c.iv > 0 and spot > 0:
            T = c.tte_years
            g = bs_greeks(spot, c.strike, T, self.RISK_FREE_RATE, c.iv, c.cp)
            c.delta = g["delta"]
            c.gamma = g["gamma"]
            c.theta = g["theta"]
            c.vega  = g["vega"]

    def select_atm(self, spot: float) -> tuple[Optional[OptionContract], Optional[OptionContract]]:
        """Find ATM call + put on nearest expiry."""
        # Group by expiry
        expiries = sorted(set(c.expiry_ms for c in self.chain.values()))
        if not expiries:
            return None, None
        nearest = expiries[0]
        contracts = [c for c in self.chain.values() if c.expiry_ms == nearest]
        # Pick strike closest to spot
        strikes = sorted(set(c.strike for c in contracts))
        if not strikes:
            return None, None
        atm_strike = min(strikes, key=lambda k: abs(k - spot))
        call = next((c for c in contracts if c.strike == atm_strike and c.cp == 'C'), None)
        put  = next((c for c in contracts if c.strike == atm_strike and c.cp == 'P'), None)
        return call, put

    def quote_and_fill(self, spot: float, ts_ms: int):
        """Main market-making logic — called every poll cycle."""
        self.spot_price = spot
        self.tick_count += 1

        call, put = self.select_atm(spot)
        self.atm_call = call
        self.atm_put  = put

        # Compute net delta of book
        self.net_delta = sum(p.delta_exposure for p in self.positions.values())

        fills = []
        for contract in [call, put]:
            if not contract or contract.mark <= 0 or contract.iv <= 0:
                continue

            pos = self.positions.get(contract.symbol)
            qty = pos.qty if pos else 0
            leg_max = self.MAX_CONTRACTS

            # Delta-adjusted skew: if long too much delta → bias ask tighter
            skew = 0.0
            if self.net_delta > self.MAX_DELTA_EXPOSURE * 0.5:
                skew = self.QUOTE_WIDTH_IV * 0.5   # sell more → tighten ask
            elif self.net_delta < -self.MAX_DELTA_EXPOSURE * 0.5:
                skew = -self.QUOTE_WIDTH_IV * 0.5  # buy more → tighten bid

            T = contract.tte_years
            S, K, r = spot, contract.strike, self.RISK_FREE_RATE

            bid_iv = max(0.001, contract.iv - self.QUOTE_WIDTH_IV + skew)
            ask_iv = contract.iv + self.QUOTE_WIDTH_IV + skew
            self.my_bid_iv = bid_iv
            self.my_ask_iv = ask_iv

            my_bid_px = bs_price(S, K, T, r, bid_iv, contract.cp)
            my_ask_px = bs_price(S, K, T, r, ask_iv, contract.cp)

            # Simulate fill: market crosses our bid → we get long
            if contract.ask > 0 and contract.ask <= my_bid_px and qty < leg_max:
                self._fill(contract, +1, my_bid_px, ts_ms)
                fills.append({"side": "BUY", "symbol": contract.symbol, "price": my_bid_px, "iv": bid_iv})

            # Market crosses our ask → we get short
            elif contract.bid > 0 and contract.bid >= my_ask_px and qty > -leg_max:
                self._fill(contract, -1, my_ask_px, ts_ms)
                fills.append({"side": "SELL", "symbol": contract.symbol, "price": my_ask_px, "iv": ask_iv})

        return fills

    def _fill(self, contract: OptionContract, side: int, price: float, ts_ms: int):
        sym = contract.symbol
        if sym not in self.positions:
            self.positions[sym] = OptionsPosition(contract=contract, qty=0, avg_price=price)
        pos = self.positions[sym]
        if pos.qty == 0:
            pos.avg_price = price
        else:
            old_qty = pos.qty
            new_qty = old_qty + side
            if old_qty != 0 and (old_qty * side < 0):
                closed_qty = min(abs(old_qty), abs(side))
                direction = 1.0 if old_qty > 0 else -1.0
                self.realized_pnl += closed_qty * (price - pos.avg_price) * direction
                if new_qty == 0:
                    pos.avg_price = 0.0
                elif (old_qty > 0 and new_qty < 0) or (old_qty < 0 and new_qty > 0):
                    pos.avg_price = price
            elif old_qty == 0:
                pos.avg_price = price
            else:
                pos.avg_price = (pos.avg_price * abs(old_qty) + price * abs(side)) / abs(new_qty)
            pos.qty = new_qty
        self.fill_count += 1
        print(f"  [OPT FILL] {'BUY' if side > 0 else 'SELL'} {contract.symbol} @ ${price:.2f} | IV={contract.iv:.1%} | delta={contract.delta:.3f}")

    @property
    def total_pnl(self) -> float:
        mtm = sum(p.mtm_pnl for p in self.positions.values())
        return self.realized_pnl + mtm

    def state_dict(self) -> dict:
        positions = [
            {
                "symbol": p.contract.symbol,
                "qty": p.qty,
                "avg_price": round(p.avg_price, 4),
                "mark": round(p.contract.mark, 4),
                "delta": round(p.contract.delta, 4),
                "mtm_pnl": round(p.mtm_pnl, 4),
            }
            for p in self.positions.values() if p.qty != 0
        ]
        return {
            "spot_price":   round(self.spot_price, 2),
            "net_delta":    round(self.net_delta, 4),
            "my_bid_iv":    round(self.my_bid_iv, 4),
            "my_ask_iv":    round(self.my_ask_iv, 4),
            "realized_pnl": round(self.realized_pnl, 4),
            "total_pnl":    round(self.total_pnl, 4),
            "fill_count":   self.fill_count,
            "tick_count":   self.tick_count,
            "positions":    positions,
            "atm_symbol":   self.atm_call.symbol if self.atm_call else None,
            "atm_strike":   self.atm_call.strike if self.atm_call else None,
            "atm_expiry":   self.atm_call.expiry_date if self.atm_call else None,
            "atm_call_iv":  round(self.atm_call.iv, 4) if self.atm_call and self.atm_call.iv else None,
            "atm_put_iv":   round(self.atm_put.iv, 4)  if self.atm_put  and self.atm_put.iv  else None,
        }


# ── Async Streaming Functions ─────────────────────────────────────────────────

async def fetch_options_chain(session: aiohttp.ClientSession, underlying: str) -> list[dict]:
    """Fetch the full options chain for BTC or ETH from Binance eAPI."""
    url = "https://eapi.binance.com/eapi/v1/exchangeInfo"
    async with session.get(url) as r:
        data = await r.json()
    return [s for s in data.get("optionSymbols", []) if s["underlying"] == underlying]

async def fetch_tickers(session: aiohttp.ClientSession, symbols: list[str]) -> list[dict]:
    """Fetch live ticker data (bid/ask/mark/IV/Greeks) for given option symbols."""
    url = "https://eapi.binance.com/eapi/v1/ticker"
    async with session.get(url) as r:
        all_tickers = await r.json()
    sym_set = set(symbols)
    return [t for t in all_tickers if t.get("symbol") in sym_set]

async def fetch_spot_price(session: aiohttp.ClientSession, symbol: str) -> float:
    """Fetch current spot price from Binance REST."""
    url = f"https://api.binance.com/api/v3/ticker/price?symbol={symbol}"
    async with session.get(url) as r:
        data = await r.json()
    return float(data.get("price", 0))


async def stream_options(underlying: str, mm: OptionsMarketMaker,
                         duration_s: int = 3600, poll_interval: float = 2.0,
                         state_cb=None):
    """
    Main options streaming loop.
    underlying = "BTCUSDT" or "ETHUSDT"
    state_cb   = async callback(dict) called every tick — used by api_server to push to WS
    Streams REST ticker every poll_interval seconds (Binance options don't expose
    a public unauthenticated WS trade stream — their WS requires API key).
    """
    print(f"[OPTIONS] Starting {underlying} options stream for {duration_s}s…")
    spot_sym = underlying   # BTCUSDT

    async with aiohttp.ClientSession() as session:
        # 1. Build chain (nearest 2 expiries, strikes within 20% of spot)
        print("[OPTIONS] Fetching chain…")
        all_contracts = await fetch_options_chain(session, underlying)
        spot = await fetch_spot_price(session, spot_sym)

        # Keep nearest 2 expiries
        expiries = sorted(set(c["expiryDate"] for c in all_contracts))[:2]
        # Filter strikes within ±20% of spot
        relevant = [
            c for c in all_contracts
            if c["expiryDate"] in expiries
            and spot * 0.80 <= float(c["strikePrice"]) <= spot * 1.20
        ]
        symbols = [c["symbol"] for c in relevant]
        print(f"[OPTIONS] Chain: {len(symbols)} contracts, expiries: {expiries}")

        # Seed the chain
        for c in relevant:
            cp = "C" if c["symbol"].endswith("-C") else "P"
            mm.chain[c["symbol"]] = OptionContract(
                symbol=c["symbol"], underlying=underlying,
                strike=float(c["strikePrice"]), expiry_ms=int(c["expiryDate"]),
                cp=cp,
            )

        start = time.time()
        while time.time() - start < duration_s:
            try:
                # 2. Refresh spot price
                spot = await fetch_spot_price(session, spot_sym)

                # 3. Fetch tickers for all relevant symbols
                tickers = await fetch_tickers(session, symbols)
                for tk in tickers:
                    sym = tk["symbol"]
                    if sym not in mm.chain:
                        continue
                    c = mm.chain[sym]
                    c.bid  = float(tk.get("bidPrice",  0) or 0)
                    c.ask  = float(tk.get("askPrice",  0) or 0)
                    c.mark = float(tk.get("markPrice", 0) or 0)
                    c.last = float(tk.get("lastPrice", 0) or 0)
                    c.iv   = float(tk.get("impliedVolatility", 0) or 0)
                    c.delta= float(tk.get("delta", 0) or 0)
                    c.gamma= float(tk.get("gamma", 0) or 0)
                    c.theta= float(tk.get("theta", 0) or 0)
                    c.vega = float(tk.get("vega",  0) or 0)
                    mm.update_contract(c, spot)

                # 4. Quote & fill
                ts_ms = int(time.time() * 1000)
                fills = mm.quote_and_fill(spot, ts_ms)

                if state_cb:
                    await state_cb(mm.state_dict())

                await asyncio.sleep(poll_interval)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[OPTIONS] Error: {e}")
                await asyncio.sleep(5)

    print(f"[OPTIONS] Done. PnL=${mm.total_pnl:.2f}, Fills={mm.fill_count}")


# ── Standalone entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--underlying", default="BTCUSDT")
    ap.add_argument("--duration", type=int, default=3600)
    ap.add_argument("--interval", type=float, default=2.0)
    args = ap.parse_args()

    mm = OptionsMarketMaker()

    async def _print_state(s: dict):
        print(f"  [OPT] Spot=${s['spot_price']} | NetDelta={s['net_delta']:.3f} | PnL=${s['total_pnl']:.2f} | Fills={s['fill_count']}")

    asyncio.run(stream_options(args.underlying, mm, args.duration, args.interval, _print_state))
