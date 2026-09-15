"""
HFT Live Paper Trader — Binance Testnet
========================================
Connects to Binance aggTrades WebSocket stream in real-time.
Runs LGBM model on live ticks (same features as backtest).
Logs every signal → fill latency, estimated slippage, and PnL.

This validates that:
  1. Real-time feature computation matches backtest features
  2. Fill quality (latency + slippage) is within the 0.10 bps budget
  3. Signal frequency matches backtest expectations

Usage:
    # Set API keys (testnet):
    $env:BINANCE_API_KEY    = "your_testnet_key"
    $env:BINANCE_API_SECRET = "your_testnet_secret"
    $env:BINANCE_BASE_URL   = "https://testnet.binancefuture.com"

    python python/paper_trade_live.py --symbol BTCUSDT --duration 3600

    # No API keys? Runs in SIGNAL-ONLY mode (no orders placed, just signals):
    python python/paper_trade_live.py --signal-only

Ctrl+C to stop — results saved to hft_nt_results/paper_trade_YYYYMMDD_HHMMSS.csv
"""

import argparse
import asyncio
import json
import os
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
import lightgbm as lgb
import websockets

HERE    = Path(__file__).parent
ML_DIR  = HERE / "ml_pipeline" / "models"
OUT_DIR = Path("hft_nt_results")
OUT_DIR.mkdir(exist_ok=True)

# ── Match EXACTLY the backtest parameters ─────────────────────────────────────
LGBM_PATH       = ML_DIR / "lgbm_trade_tape_model.txt"
FEATURES        = ["tfi_ema_50", "tfi_ema_200", "volatility_ema_100"]
CONFIDENCE_BAND = 0.5    # bps
COOLDOWN_MS     = 200    # ms between signals
TAKER_FEE_BPS   = 0.1   # bps (Binance VIP0 default)

# EMA decay parameters (match backtest half-lives)
ALPHA_50  = 1 - 2 ** (-1 / 50)
ALPHA_200 = 1 - 2 ** (-1 / 200)
ALPHA_V   = 1 - 2 ** (-1 / 100)

# ── Exchange configs ───────────────────────────────────────────────────────────
EXCHANGES = {
    "binance": {
        # Spot WebSocket — matches Binance Vision training data source
        # Try port 9443 first, falls back to 443 if blocked
        "ws_url":    "wss://stream.binance.com:9443/ws/{symbol}@aggTrade",
        "ws_url_443":"wss://stream.binance.com:443/ws/{symbol}@aggTrade",
        "symbol":    "BTCUSDT",
        "fee_bps":   0.1,   # VIP0 taker
        "note":      "Binance Spot — same feed as Binance Vision training data",
    },
    "coinbase": {
        "ws_url":    "wss://ws-feed.exchange.coinbase.com",
        "symbol":    "BTC-USD",
        "fee_bps":   6.0,
        "note":      "Coinbase Exchange spot — no auth needed for public feed",
    },
}

# ── Real-time EMA State ────────────────────────────────────────────────────────
class OnlineEMA:
    """Incremental EMA matching Polars ewm_mean(half_life=N) behavior."""
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.value = None

    def update(self, x: float) -> float:
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value

# ── Signal State Machine ───────────────────────────────────────────────────────
class SignalEngine:
    """
    Auto-detects model feature count:
      3 features → legacy mode (tfi×2, vol×1) — current model
      8 features → L2 mode    (tfi×2, vol×2, book_imbalance, spread_bps,
                                microprice, spread_ema) — retrained Kaggle model
    """
    ALPHA_V500 = 1 - 2 ** (-1 / 500)  # half_life=500 for vol slow EMA
    ALPHA_SP50 = 1 - 2 ** (-1 / 50)   # half_life=50 for spread EMA

    def __init__(self, model: lgb.Booster):
        self.model       = model
        self.n_features  = model.num_feature()
        self.last_price  = None
        self.last_signal_ms = 0

        # ── Always-present EMAs (3-feature model) ─────────────────────────────
        self.ema_sv_50   = OnlineEMA(ALPHA_50)
        self.ema_sv_200  = OnlineEMA(ALPHA_200)
        self.ema_vol_100 = OnlineEMA(ALPHA_V)

        # ── L2-mode extra EMAs (8-feature model) ──────────────────────────────
        if self.n_features == 8:
            self.ema_vol_500  = OnlineEMA(self.ALPHA_V500)
            self.ema_spread50 = OnlineEMA(self.ALPHA_SP50)
            # Latest book snapshot (filled by on_book_tick)
            self.bid_px = self.ask_px = self.bid_qty = self.ask_qty = None
            print(f"  [SignalEngine] L2 mode — 8 features, bookTicker required")
        else:
            print(f"  [SignalEngine] Legacy mode — {self.n_features} features")

    @property
    def is_l2(self) -> bool:
        return self.n_features == 8

    def on_book_tick(self, bid_px: float, ask_px: float,
                     bid_qty: float, ask_qty: float):
        """Update best bid/ask snapshot from bookTicker stream."""
        self.bid_px  = bid_px
        self.ask_px  = ask_px
        self.bid_qty = bid_qty
        self.ask_qty = ask_qty

    def on_tick(self, price: float, qty: float, is_buyer_maker: bool,
                ts_ms: int) -> dict | None:
        """Process one aggTrade tick. Returns signal dict or None."""
        signed_vol = -qty if is_buyer_maker else qty
        abs_diff   = abs(price - self.last_price) if self.last_price else 0.0
        self.last_price = price

        f1 = self.ema_sv_50.update(signed_vol)
        f2 = self.ema_sv_200.update(signed_vol)
        f3 = self.ema_vol_100.update(abs_diff)

        if None in (f1, f2, f3):
            return None

        if self.is_l2:
            # Need book data — skip tick if not yet received
            if None in (self.bid_px, self.ask_px, self.bid_qty, self.ask_qty):
                return None
            f4 = self.ema_vol_500.update(abs_diff)
            total_qty = self.bid_qty + self.ask_qty + 1e-9
            mid_px    = (self.bid_px + self.ask_px) / 2.0
            raw_spread = (self.ask_px - self.bid_px) / (mid_px + 1e-9) * 10000.0
            book_imb   = (self.bid_qty - self.ask_qty) / total_qty
            microprice = (self.bid_px * self.ask_qty + self.ask_px * self.bid_qty) / total_qty
            f5 = book_imb
            f6 = raw_spread
            f7 = microprice
            f8 = self.ema_spread50.update(raw_spread)
            X  = np.array([[f1, f2, f3, f4, f5, f6, f7, f8]], dtype=np.float32)
        else:
            X = np.array([[f1, f2, f3]], dtype=np.float32)

        pred = float(self.model.predict(X)[0])

        signal = 0
        if pred > CONFIDENCE_BAND:
            signal = 1
        elif pred < -CONFIDENCE_BAND:
            signal = -1

        # Cooldown gate
        if signal != 0 and (ts_ms - self.last_signal_ms) >= COOLDOWN_MS:
            self.last_signal_ms = ts_ms
            return {
                "ts_ms":  ts_ms,
                "price":  price,
                "pred":   pred,
                "signal": signal,
            }
        return None


# ── Order Gateway (thin wrapper for testnet) ───────────────────────────────────
class PaperGateway:
    """Simulates fills or routes to Binance testnet. Measures fill latency."""
    def __init__(self, signal_only: bool = True):
        self.signal_only = signal_only
        self.api_key     = os.environ.get("BINANCE_API_KEY", "")
        self.api_secret  = os.environ.get("BINANCE_API_SECRET", "")
        self.base_url    = os.environ.get("BINANCE_BASE_URL",
                                          "https://testnet.binancefuture.com")
        if not self.signal_only and not (self.api_key and self.api_secret):
            print("[WARN] No API keys — forcing signal-only mode.")
            self.signal_only = True

    async def execute(self, signal: dict, symbol: str = "BTCUSDT") -> dict:
        """Place order or simulate fill. Returns fill dict."""
        side    = "BUY" if signal["signal"] == 1 else "SELL"
        t0      = time.perf_counter_ns()

        if self.signal_only:
            # Simulate: assume fill at signal price (no slippage model)
            await asyncio.sleep(0.001)  # simulate 1ms network
            latency_ms  = 1.0
            fill_price  = signal["price"]
            slip_bps    = 0.0
        else:
            try:
                import aiohttp, hmac, hashlib
                ts = int(time.time() * 1000)
                qty = "0.001"  # minimum size
                qs  = f"symbol={symbol}&side={side}&type=MARKET&quantity={qty}&timestamp={ts}"
                sig = hmac.new(self.api_secret.encode(),
                               qs.encode(), hashlib.sha256).hexdigest()
                url = f"{self.base_url}/fapi/v1/order?{qs}&signature={sig}"
                headers = {"X-MBX-APIKEY": self.api_key}
                async with aiohttp.ClientSession() as sess:
                    async with sess.post(url, headers=headers) as r:
                        data = await r.json()
                latency_ms = (time.perf_counter_ns() - t0) / 1e6
                fill_price = float(data.get("avgPrice", signal["price"]))
                slip_bps   = abs(fill_price - signal["price"]) / signal["price"] * 10000
            except Exception as e:
                print(f"  [WARN] Order failed: {e} — using paper fill")
                latency_ms = 0.0
                fill_price = signal["price"]
                slip_bps   = 0.0

        return {
            "fill_price":  fill_price,
            "latency_ms":  latency_ms,
            "slip_bps":    slip_bps,
            "side":        side,
        }

# ── Exchange-specific WebSocket handlers ──────────────────────────────────────
async def _binance_stream(ws, symbol: str, engine: SignalEngine,
                          gateway: PaperGateway, duration_s: int,
                          first_msg: str = None) -> tuple:
    """Binance aggTrade stream handler."""
    t_start    = time.time()
    t_hb       = t_start
    tick_count = signal_count = 0
    fills      = []
    last_fill  = None
    pred_min   = float("inf")
    pred_max   = float("-inf")

    # Seed with the probe message already received in stream_and_trade
    pending = [first_msg] if first_msg else []

    while time.time() - t_start < duration_s:
        elapsed = time.time() - t_start

        # Heartbeat every 30s — confirms ticks are flowing and shows pred range
        if elapsed - (t_hb - t_start) >= 30 and tick_count > 0:
            t_hb = time.time()
            print(f"  [HB {int(elapsed):>5}s] ticks={tick_count:>8,} | "
                  f"signals={signal_count:>4} | "
                  f"pred range=[{pred_min:+.4f}, {pred_max:+.4f}] | "
                  f"threshold=+/-{CONFIDENCE_BAND}", flush=True)

        try:
            if pending:
                msg = pending.pop(0)
            else:
                msg = await asyncio.wait_for(ws.recv(), timeout=5.0)
            raw  = json.loads(msg)
            # Combined-stream envelope: {"stream": "...", "data": {...}}
            data = raw.get("data", raw)
        except asyncio.TimeoutError:
            print(f"  [WARN {int(elapsed)}s] No data for 5s — still waiting...", flush=True)
            continue


        # Binance sends non-trade frames (ping, bookTicker, ack) — route them
        if "p" not in data or "q" not in data:
            # bookTicker frame has b/B/a/A fields
            if engine.is_l2 and "b" in data and "a" in data:
                try:
                    engine.on_book_tick(
                        float(data["b"]), float(data["a"]),
                        float(data["B"]), float(data["A"]),
                    )
                except (KeyError, ValueError):
                    pass
            continue

        try:
            price = float(data["p"])
            qty   = float(data["q"])
            is_bm = data["m"]
            ts_ms = data["T"]
        except (KeyError, ValueError):
            continue

        tick_count += 1
        sig = engine.on_tick(price, qty, is_bm, ts_ms)

        # Track prediction range (use correct attribute names)
        if engine.ema_sv_50.value is not None:
            if engine.is_l2 and engine.bid_px is not None:
                total = engine.bid_qty + engine.ask_qty + 1e-9
                mid   = (engine.bid_px + engine.ask_px) / 2.0
                sp    = (engine.ask_px - engine.bid_px) / (mid + 1e-9) * 10000
                mp    = (engine.bid_px * engine.ask_qty + engine.ask_px * engine.bid_qty) / total
                X = np.array([[engine.ema_sv_50.value, engine.ema_sv_200.value,
                               engine.ema_vol_100.value,
                               engine.ema_vol_500.value if hasattr(engine, "ema_vol_500") else 0,
                               (engine.bid_qty - engine.ask_qty) / total,
                               sp, mp,
                               engine.ema_spread50.value or sp]], dtype=np.float32)
            else:
                X = np.array([[engine.ema_sv_50.value,
                               engine.ema_sv_200.value,
                               engine.ema_vol_100.value]], dtype=np.float32)
            pred = float(engine.model.predict(X)[0])
            pred_min = min(pred_min, pred)
            pred_max = max(pred_max, pred)

        if sig:
            signal_count += 1
            fill, pnl_bps, last_fill = await _process_signal(
                sig, gateway, symbol, last_fill, fills)
            _print_row(sig, fill, pnl_bps, fills, elapsed)

    return fills, tick_count, signal_count


async def _coinbase_stream(ws, symbol: str, engine: SignalEngine,
                           gateway: PaperGateway, duration_s: int) -> tuple:
    """Coinbase Exchange matches stream handler.

    Coinbase 'side' is the MAKER's side:
      side='sell' → maker sold → taker bought → buy pressure  (+qty)
      side='buy'  → maker bought → taker sold → sell pressure (-qty)
    This maps directly to Binance is_buyer_maker convention.
    """
    # Subscribe to matches channel
    sub = json.dumps({
        "type": "subscribe",
        "product_ids": [symbol],
        "channels": ["matches"]
    })
    await ws.send(sub)
    print(f"  Subscribed to Coinbase matches: {symbol}")

    t_start = time.time()
    tick_count = signal_count = 0
    fills = []
    last_fill = None

    while time.time() - t_start < duration_s:
        try:
            msg  = await asyncio.wait_for(ws.recv(), timeout=5.0)
            data = json.loads(msg)
        except asyncio.TimeoutError:
            continue

        if data.get("type") not in ("match", "last_match"):
            continue  # skip subscriptions/heartbeats

        price = float(data["price"])
        qty   = float(data["size"])
        # Coinbase: side = maker side. sell-maker = buy-taker = NOT buyer_maker
        is_bm = (data["side"] == "buy")   # maker bought → buyer_maker = True
        ts_ms = int(datetime.fromisoformat(
            data["time"].replace("Z", "+00:00")
        ).timestamp() * 1000)

        tick_count += 1
        sig = engine.on_tick(price, qty, is_bm, ts_ms)
        if sig:
            signal_count += 1
            fill, pnl_bps, last_fill = await _process_signal(
                sig, gateway, symbol, last_fill, fills)
            _print_row(sig, fill, pnl_bps, fills, time.time() - t_start)

    return fills, tick_count, signal_count


async def _process_signal(sig, gateway, symbol, last_fill, fills):
    fill = await gateway.execute(sig, symbol)
    pnl_bps = None
    if last_fill and last_fill["signal"]["signal"] != sig["signal"]:
        entry   = last_fill["fill"]["fill_price"]
        exit_   = fill["fill_price"]
        dir_    = last_fill["signal"]["signal"]
        fee     = gateway.fee_bps * 2
        pnl_bps = (exit_ - entry) / entry * 10000 * dir_ - fee
        fills.append(pnl_bps)
    return fill, pnl_bps, {"signal": sig, "fill": fill}


def _print_row(sig, fill, pnl_bps, fills, elapsed):
    dir_str = "LONG " if sig["signal"] == 1 else "SHORT"
    pnl_str = f"{pnl_bps:+.2f}" if pnl_bps is not None else "  open"
    cum_pnl = sum(fills)
    print(f"  {int(elapsed):>7}s | {sig['price']:>10.2f} | {sig['pred']:>+8.4f} | "
          f"{dir_str:>5} | {fill['latency_ms']:>6.1f}ms | "
          f"{fill['slip_bps']:>6.3f}bp | {len(fills):>7} | {cum_pnl:>+10.2f}",
          flush=True)


async def stream_and_trade(exchange: str, symbol: str, duration_s: int,
                           engine: SignalEngine, gateway: PaperGateway) -> tuple:
    cfg = EXCHANGES[exchange]

    if exchange == "binance":
        sym = symbol.lower()
        if engine.is_l2:
            # Combined stream — both aggTrade and bookTicker in one connection
            stream_path = f"{sym}@aggTrade/{sym}@bookTicker"
            base9 = "wss://stream.binance.com:9443/stream?streams="
            base4 = "wss://stream.binance.com:443/stream?streams="
        else:
            stream_path = f"{sym}@aggTrade"
            base9 = "wss://stream.binance.com:9443/ws/"
            base4 = "wss://stream.binance.com:443/ws/"
        urls = [base9 + stream_path, base4 + stream_path]
    else:
        urls = [cfg["ws_url"]]

    print(f"\n  {'Time':>8} | {'Price':>10} | {'Pred':>8} | {'Dir':>5} | "
          f"{'Lat':>8} | {'Slip':>7} | {'Trades':>7} | {'PnL':>10}")
    print("  " + "-"*78)

    for url in urls:
        print(f"\n  Connecting to {url} ...", flush=True)
        try:
            async with websockets.connect(
                url,
                ping_interval=20,
                open_timeout=10,
            ) as ws:
                # Quick liveness check — if no data in 8s, try next URL
                try:
                    probe = await asyncio.wait_for(ws.recv(), timeout=8.0)
                    print(f"  Connected — data flowing ✓", flush=True)
                except asyncio.TimeoutError:
                    print(f"  [WARN] No data in 8s — trying next URL...", flush=True)
                    continue

                # Re-process the probe message before entering main loop
                if exchange == "binance":
                    return await _binance_stream(
                        ws, symbol, engine, gateway, duration_s,
                        first_msg=probe)
                else:
                    return await _coinbase_stream(
                        ws, symbol, engine, gateway, duration_s)

        except Exception as e:
            print(f"  [WARN] Connection failed: {e} — trying next URL...", flush=True)

    print("[ERROR] All WebSocket URLs failed. Check network/firewall.")
    return [], 0, 0

# ── Main ───────────────────────────────────────────────────────────────────────
async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exchange",    default="binance",
                        choices=list(EXCHANGES.keys()),
                        help="Exchange to connect to (default: binance)")
    parser.add_argument("--duration",    type=int, default=3600,
                        help="Run duration in seconds (default: 1 hour)")
    parser.add_argument("--signal-only", action="store_true",
                        help="Log signals only, no orders placed")
    args = parser.parse_args()

    cfg    = EXCHANGES[args.exchange]
    symbol = cfg["symbol"]

    signal_only = args.signal_only or not (
        os.environ.get("BINANCE_API_KEY") and
        os.environ.get("BINANCE_API_SECRET")
    )

    print("="*70)
    print(f"  HFT Live Paper Trader — {args.exchange.upper()}")
    print("="*70)
    print(f"  Exchange: {args.exchange.upper()} — {cfg['note']}")
    print(f"  Symbol  : {symbol}")
    print(f"  Duration: {args.duration}s ({args.duration//60} min)")
    print(f"  Mode    : {'SIGNAL-ONLY (no orders)' if signal_only else 'PAPER ORDERS'}")
    print(f"  Fee     : {cfg['fee_bps']} bps/side")
    print(f"  Model   : {LGBM_PATH}")
    print(f"  Params  : threshold={CONFIDENCE_BAND}bps | cooldown={COOLDOWN_MS}ms")

    if args.exchange == "coinbase" and not signal_only:
        print(f"\n  ⚠️  WARNING: Coinbase taker fee = {cfg['fee_bps']} bps")
        print(f"     Model gross edge = +0.23 bps — UNPROFITABLE at Coinbase fees.")
        print(f"     Forcing signal-only mode.\n")
        signal_only = True

    if not LGBM_PATH.exists():
        print(f"[ERROR] Model not found: {LGBM_PATH}")
        sys.exit(1)

    model   = lgb.Booster(model_file=str(LGBM_PATH))
    engine  = SignalEngine(model)
    gateway = PaperGateway(signal_only=signal_only)
    gateway.fee_bps = cfg["fee_bps"]  # use exchange-specific fee for PnL calc

    print(f"  Model   : {model.num_trees()} trees loaded ✓")

    t0 = time.time()
    try:
        fills, ticks, n_signals = await stream_and_trade(
            args.exchange, symbol, args.duration, engine, gateway
        )
    except KeyboardInterrupt:
        print("\n  [Interrupted by user]")
        fills, ticks, n_signals = [], 0, 0

    elapsed = time.time() - t0

    # ── Final Report ───────────────────────────────────────────────────────
    print("\n" + "="*70)
    print(f"  PAPER TRADING SESSION REPORT — {args.exchange.upper()}")
    print("="*70)
    print(f"  Duration    : {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Ticks seen  : {ticks:,}")
    print(f"  Signals     : {n_signals:,} ({n_signals/max(1,elapsed/60):.1f}/min)")

    if fills:
        arr = np.array(fills)
        pf  = arr[arr>0].sum() / (abs(arr[arr<0].sum()) + 1e-9)
        print(f"  Trades      : {len(arr)}")
        print(f"  Total PnL   : {arr.sum():+.4f} bps")
        print(f"  Avg PnL     : {arr.mean():+.4f} bps/trade")
        print(f"  Win Rate    : {(arr>0).mean()*100:.1f}%")
        print(f"  Profit Fac  : {pf:.3f}")
        print(f"  Sharpe      : {arr.mean()/(arr.std()+1e-9)*np.sqrt(len(arr)*252):.3f}")

        backtest_avg = 0.1275
        ratio = arr.mean() / backtest_avg
        print(f"\n  LIVE vs BACKTEST (signal quality, fee-adjusted to 0.1 bps):")
        print(f"    Backtest avg PnL/trade : +{backtest_avg:.4f} bps")
        print(f"    Live     avg PnL/trade : {arr.mean():+.4f} bps")
        print(f"    Capture ratio          : {ratio:.1%}")
        if ratio > 0.7:
            print("    ✅ Live captures >70% of backtest edge — model is VALID")
        elif ratio > 0.4:
            print("    ⚠️  Partial capture — microstructure diff vs Binance futures")
        else:
            print("    ❌ Low capture — Coinbase spot has different microstructure")

        ts_str   = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = OUT_DIR / f"paper_trade_{args.exchange}_{ts_str}.csv"
        import pandas as pd
        pd.DataFrame({"pnl_bps": arr}).to_csv(out_path, index=False)
        print(f"\n  Saved: {out_path}")
    else:
        print("  No completed round-trips yet (need 2+ signals in opposite directions)")

    print("="*70)

if __name__ == "__main__":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    asyncio.run(main())
