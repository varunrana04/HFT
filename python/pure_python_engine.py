import time
import math
import struct
from enum import Enum

INVALID_PRICE = -1

class Side(Enum):
    BID = 0
    ASK = 1

class EngineMode(Enum):
    LIVE = 0
    BACKTEST = 1

class StrategyConfig:
    def __init__(self):
        self.alpha_entry_threshold   = 3.5
        self.max_position_pct        = 0.1
        self.min_warmup_ticks        = 100
        self.initial_capital         = 10_000_000.0
        self.alpha_short_multiplier  = 1.2
        self.spread_alpha_multiplier = 0.05
        self.min_take_profit_bps     = 5.0
        self.maker_fee_pct           = -0.00005
        self.order_size_btc          = 1.0
        self.max_position_btc        = 5.0
        self.daily_loss_limit_usd    = 50_000.0    # 0.5% of 10M — recalibrated on boot
        self.vpin_halt_threshold     = 0.70
        self.execution_cooldown_ns   = 1_000_000_000 # 1 second cooldown

class BookSnapshot:
    def __init__(self):
        self.timestamp_ns   = 0
        self.best_bid_price = INVALID_PRICE
        self.best_bid_qty   = 0
        self.best_ask_price = INVALID_PRICE
        self.best_ask_qty   = 0
        self.bid_count      = 0
        self.ask_count      = 0
        
    def is_valid(self) -> bool:
        return self.best_bid_price > 0 and self.best_ask_price > 0

class Trade:
    def __init__(self):
        self.timestamp_ns = 0
        self.price        = 0
        self.qty          = 0
        self.side         = Side.BID

class FeatureVector:
    def __init__(self):
        self.timestamp_ns      = 0
        self.vpin              = 0.5
        self.microprice        = 0.0
        self.spread_bps        = 2.0
        self.realized_vol      = 0.01
        self.ofi               = 0.0
        self.obi               = 0.0
        self.combined_alpha    = 0.0
        self.regime            = 0
        self.cvd               = 0.0
        self.hawkes_intensity  = 0.0

class PerformanceMetrics:
    def __init__(self):
        self.total_trades  = 0
        self.realized_pnl  = 0.0
        self.total_pnl     = 0.0
        self.win_rate      = 0.0
        self.sharpe_ratio  = 0.0
        self.max_drawdown  = 0.0

class TradeRecord:
    def __init__(self):
        self.timestamp_ns = 0
        self.side         = Side.BID
        self.entry_price  = 0
        self.exit_price   = 0
        self.quantity     = 0.0
        self.pnl          = 0.0
        self.slippage     = 0.0


WEIGHT_NAMES = ["w_obi","w_vpin","w_vol","w_spread","w_ofi","w_microprice","w_bias"]

def _load_signal_weights(path):
    try:
        with open(path, "rb") as f:
            raw = f.read(7 * 8)
        values = struct.unpack("<7d", raw)
        weights = dict(zip(WEIGHT_NAMES, values))
        print(f"[MOCK ENGINE] Loaded signal weights: " + ", ".join(f"{k}={v:.4f}" for k,v in weights.items()))
        return weights
    except Exception as e:
        print(f"[MOCK ENGINE] Cannot read {path}: {e} -- using hand-tuned fallback.")
        return {"w_obi":0.224,"w_vpin":-0.242,"w_vol":0.101,"w_spread":-0.238,"w_ofi":0.006,"w_microprice":0.189,"w_bias":0.0}


class StrategyEngine:
    def __init__(self, config: StrategyConfig):
        self.config          = config
        self.mode            = EngineMode.BACKTEST
        self.pos             = 0.0
        self._entry_price    = 0
        self._cash           = config.initial_capital
        self._equity         = config.initial_capital
        self._peak_equity    = config.initial_capital
        self._metrics        = PerformanceMetrics()
        self._last_features  = FeatureVector()
        self._records        = []
        self._records        = []
        self._ticks          = 0
        self._stat_arb       = False
        self._halted         = False
        self._weights        = {}
        self._prev_micro     = 0.0
        self._micro_ret      = 0.0
        self._vpin_bvol      = 0.0
        self._vpin_buyvol    = 0.0
        self._vpin_window    = []
        self._VPIN_BUCKET    = 50.0
        self._last_trade_ns   = 0
        # Session-start equity (set after loading DB state, reset daily)
        # Kill switch measures loss from HERE, not from construction-time initial_capital
        self._session_start_equity = config.initial_capital
        # CVD (Cumulative Volume Delta)
        self._cvd_buffer     = []
        self._CVD_WINDOW     = 200
        # Hawkes intensity (self-exciting point process)
        self._hawkes_intensity = 0.0
        self._HAWKES_DECAY   = 0.5   # decay rate λ (per second)
        self._HAWKES_MU      = 0.02  # base intensity per trade
        self._last_hawkes_ns = 0
        # OFI (Order Flow Imbalance) — delta of top-of-book qty between snapshots
        self._prev_bid_qty   = 0
        self._prev_ask_qty   = 0
        
    def pending_order(self):
        class DummyOrder:
            def __init__(self):
                self.active = False
                self.timestamp_ns = 0
                self.side = Side.BID
                self.price = 0
                self.qty = 0
        return DummyOrder()

    def load_model(self, path):
        self._weights = _load_signal_weights(path)
        return True
        
    def set_weights(self, weights_list):
        if len(weights_list) >= 6:
            self._weights = {
                "w_obi": weights_list[0],
                "w_vpin": weights_list[1],
                "w_vol": weights_list[2],
                "w_spread": weights_list[3],
                "w_ofi": weights_list[4],
                "w_microprice": weights_list[5],
                "w_bias": weights_list[6] if len(weights_list) > 6 else 0.0
            }

    def has_model(self):
        return bool(self._weights)

    def set_stat_arb_valid(self, valid):
        self._stat_arb = valid

    def trade_journal(self):
        return list(self._records)

    def equity(self):
        return self._equity

    def set_mode(self, mode):
        self.mode = mode

    def set_position(self, pos: int):
        self.pos = pos / 1e8

    def set_avg_entry_price(self, price: float):
        self._entry_price = int(price * 1e8)
        
    def set_realized_pnl(self, pnl: float):
        self._metrics.realized_pnl = pnl
        self._cash = self.config.initial_capital + pnl
        self._equity = self._cash
        # Anchor the session baseline AFTER restoring historical PnL
        self._session_start_equity = self._equity
        self._peak_equity          = self._equity

    def realized_pnl(self) -> float:
        """Return realized PnL — mirrors the C++ engine interface used by log_flusher_loop."""
        return self._metrics.realized_pnl


    def new_trading_day(self):
        """
        Called at UTC Midnight. Rebases the initial capital to the current equity
        so that the daily loss limit tracks a new intraday watermark, and resets daily metrics.
        """
        self.config.initial_capital    = self._equity
        self._session_start_equity     = self._equity   # new day baseline
        self._metrics.realized_pnl     = 0.0
        self._metrics.total_pnl        = 0.0
        self._metrics.total_trades     = 0
        self._peak_equity              = self._equity
        print(f"[ENGINE] New Trading Day Rollover. Capital rebased to ${self._equity:.2f}")

    def update_kill_switch_state(self, timestamp_ms: int):
        loss = self._session_start_equity - self._equity
        if loss >= self.config.daily_loss_limit_usd:
            self._halted = True
            print(f"[RISK] KILL SWITCH TRIPPED AT BOOT: Loss ${loss:.2f} >= ${self.config.daily_loss_limit_usd}. HALTED.")
            
        if abs(self.pos) > self.config.max_position_btc:
            self._halted = True
            print(f"[RISK] KILL SWITCH TRIPPED AT BOOT: Pos {self.pos} > Max {self.config.max_position_btc}. HALTED.")

    def is_trading_halted(self, timestamp_ms: int) -> bool:
        """
        Returns True if the engine's internal kill switch has been tripped.
        Loss is measured from _session_start_equity (set at boot / daily rollover),
        NOT from config.initial_capital which is rebased each midnight.
        """
        loss = self._session_start_equity - self._equity
        if loss >= self.config.daily_loss_limit_usd and not self._halted:
            self._halted = True
            print(f"[RISK] KILL SWITCH TRIPPED: Intraday loss ${loss:.2f} >= limit ${self.config.daily_loss_limit_usd}")

        if abs(self.pos) > self.config.max_position_btc and not self._halted:
            self._halted = True
            print(f"[RISK] KILL SWITCH TRIPPED: Position {self.pos:.6f} BTC exceeds max {self.config.max_position_btc} BTC")

        return self._halted

    def on_book_update(self, book):
        if book.best_bid_price <= 0:
            return
        mid = (book.best_bid_price + book.best_ask_price) / 2.0
        if self.pos != 0.0:
            self._equity = self._cash + self.pos * (mid - self._entry_price) / 1e8
        bid_q = book.best_bid_qty
        ask_q = book.best_ask_qty
        total_q = bid_q + ask_q
        if total_q > 0:
            self._last_features.obi = (bid_q - ask_q) / total_q
            # OFI: (Δbid_qty - Δask_qty) / total_qty
            # Positive = more aggressive buying; Negative = more aggressive selling
            d_bid = bid_q - self._prev_bid_qty
            d_ask = ask_q - self._prev_ask_qty
            self._last_features.ofi = (d_bid - d_ask) / (total_q + 1e-8)
            self._prev_bid_qty = bid_q
            self._prev_ask_qty = ask_q
            mp = (book.best_bid_price * ask_q + book.best_ask_price * bid_q) / total_q
            mp_btc = mp / 1e8
            if self._prev_micro > 0:
                self._micro_ret = (mp_btc - self._prev_micro) / self._prev_micro
            self._prev_micro = mp_btc
            self._last_features.microprice = mp_btc
        self._last_features.spread_bps = (
            (book.best_ask_price - book.best_bid_price) / book.best_bid_price * 10000.0
        )

    def on_trade(self, trade, book):
        self._ticks += 1

        # ── CVD + Hawkes (track ALL market trades, including warmup) ──────────
        vol_btc = abs(getattr(trade, 'qty', getattr(trade, 'quantity', 0))) / 1e8
        # ASK side = aggressive buyer = positive CVD; BID side = aggressive seller = negative
        cvd_sign = +1.0 if trade.side == Side.ASK else -1.0
        self._cvd_buffer.append(cvd_sign * vol_btc)
        if len(self._cvd_buffer) > self._CVD_WINDOW:
            self._cvd_buffer.pop(0)
        self._last_features.cvd = sum(self._cvd_buffer)

        now_ns = int(time.time() * 1e9)
        if self._last_hawkes_ns > 0:
            dt_s = max((now_ns - self._last_hawkes_ns) / 1e9, 0.0)
            self._hawkes_intensity = (
                self._hawkes_intensity * math.exp(-self._HAWKES_DECAY * dt_s)
                + self._HAWKES_MU
            )
        else:
            self._hawkes_intensity = self._HAWKES_MU
        self._last_hawkes_ns = now_ns
        self._last_features.hawkes_intensity = min(self._hawkes_intensity, 10.0)
        # ─────────────────────────────────────────────────────────────────────

        if self._ticks < self.config.min_warmup_ticks:
            return
        if self._halted:
            return

        # Daily loss limit (measured from session start, not construction-time capital)
        loss = self._session_start_equity - self._equity
        if loss >= self.config.daily_loss_limit_usd:
            self._halted = True
            print(f"[RISK] DAILY LOSS LIMIT: ${loss:.2f} >= ${self.config.daily_loss_limit_usd}. HALTED.")
            return

        # Cooldown check
        if time.time() * 1e9 - self._last_trade_ns < getattr(self.config, 'execution_cooldown_ns', 1_000_000_000):
            return

        # VPIN accumulator
        trade_vol_btc = abs(trade.qty) / 1e8
        self._vpin_bvol += trade_vol_btc
        if trade.side == Side.BID:
            self._vpin_buyvol += trade_vol_btc
        if self._vpin_bvol >= self._VPIN_BUCKET:
            sell_vol = self._vpin_bvol - self._vpin_buyvol
            imb = abs(self._vpin_buyvol - sell_vol) / self._vpin_bvol
            self._vpin_window.append(imb)
            if len(self._vpin_window) > 50:
                self._vpin_window.pop(0)
            self._vpin_bvol = 0.0
            self._vpin_buyvol = 0.0

        vpin = float(sum(self._vpin_window) / len(self._vpin_window)) if self._vpin_window else 0.5
        self._last_features.vpin = vpin

        # VPIN kill gate
        if vpin > self.config.vpin_halt_threshold:
            return

        # Compute alpha from signal weights
        w = self._weights
        if w:
            raw_alpha = (
                w.get("w_obi",        0.224) * self._last_features.obi +
                w.get("w_vpin",      -0.242) * (vpin - 0.5) +
                w.get("w_vol",        0.101) * self._last_features.realized_vol +
                w.get("w_spread",    -0.238) * (self._last_features.spread_bps - 2.0) +
                w.get("w_ofi",        0.006) * self._last_features.ofi +
                w.get("w_microprice", 0.189) * self._micro_ret * 1e4 +
                w.get("w_bias",       0.0)
            )
            self._last_features.combined_alpha = max(-1.0, min(1.0, raw_alpha))

        alpha      = self._last_features.combined_alpha
        spread_bps = self._last_features.spread_bps
        regime     = int(self._last_features.regime)

        # ── Regime-aware threshold adjustment ────────────────────
        #  State 0: Low-vol trend  → allow both directions, standard thresholds
        #  State 1: High-vol chaos → widen thresholds 50% on both sides (protect capital)
        #  State 2: Mean-reversion → tighten short (fade rallies), loosen long (buy dips)
        #  State 3: Crisis         → blocked upstream in bridge loop
        regime_long_mult  = 1.0
        regime_short_mult = 1.0
        if regime == 1:    # High volatility: stand down
            regime_long_mult  = 1.5
            regime_short_mult = 1.5
        elif regime == 2:  # Mean-reversion: be more aggressive on counter-trend
            regime_long_mult  = 0.8   # easier to go long (buy dips)
            regime_short_mult = 0.8   # easier to go short (sell rips)

        base_thr     = self.config.alpha_entry_threshold
        spread_addon = spread_bps * self.config.spread_alpha_multiplier
        long_thr     = base_thr * regime_long_mult  + spread_addon
        short_thr    = base_thr * self.config.alpha_short_multiplier * regime_short_mult + spread_addon

        order_qty = self.config.order_size_btc
        max_pos   = self.config.max_position_btc
        if max_pos > 0 and abs(self.pos) > 0:
            size_scale = max(0.1, 1.0 - abs(self.pos) / max_pos)
            order_qty *= size_scale

        if alpha > long_thr and self.pos <= 0 and abs(self.pos) < max_pos and book.best_ask_price > 0:
            self._execute(book.best_ask_price, order_qty, Side.BID)
        elif alpha < -short_thr and self.pos >= 0 and abs(self.pos) < max_pos and book.best_bid_price > 0:
            self._execute(book.best_bid_price, order_qty, Side.ASK)

    def position(self):
        return self.pos

    def last_features(self):
        return self._last_features

    def metrics(self):
        return self._metrics

    def extract_execution_history(self):
        return list(self._records)

    def flatten(self, book):
        if self.pos == 0.0:
            return
        if self.pos > 0 and book.best_bid_price > 0:
            self._execute(book.best_bid_price, abs(self.pos), Side.ASK)
            print(f"[RISK] EOD Flatten: SOLD {abs(self.pos):.6f} BTC at ${book.best_bid_price/1e8:.2f}")
        elif self.pos < 0 and book.best_ask_price > 0:
            self._execute(book.best_ask_price, abs(self.pos), Side.BID)
            print(f"[RISK] EOD Flatten: BOUGHT {abs(self.pos):.6f} BTC at ${book.best_ask_price/1e8:.2f}")

    def simulate_fill(self, side, price_int: int, qty_int: int, is_maker: bool = False):
        """
        Called by user_data_loop when a real Binance fill arrives via the
        User Data Stream. Routes the actual execution data into the engine
        PnL accounting, bypassing the engine's internal signal logic.

        Args:
            side      : Side.BID (buy fill) or Side.ASK (sell fill)
            price_int : execution price in fixed-point (price * 1e8)
            qty_int   : executed quantity in fixed-point (qty * 1e8)
            is_maker  : True for maker fills (lower fee); False for taker
        """
        qty = qty_int / 1e8
        fee_pct = self.config.maker_fee_pct if is_maker else 0.00015
        fee = price_int / 1e8 * qty * abs(fee_pct)

        if side == Side.BID:
            # Buy fill
            self._cash       -= fee
            self._entry_price = price_int
            self.pos         += qty
        else:
            # Sell fill — realize PnL if closing a long
            if self.pos > 0:
                pnl = self.pos * (price_int - self._entry_price) / 1e8 - fee
                self._cash              += pnl
                self._metrics.realized_pnl += pnl
                self._metrics.total_pnl    += pnl
                self.pos                -= qty
                if self.pos <= 0:
                    self._entry_price = 0
            else:
                self._cash -= fee
                self.pos   -= qty
                self._entry_price = price_int

        self._equity = self._cash
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity
        print(f"[ENGINE] simulate_fill: {'BUY' if side==Side.BID else 'SELL'} "
              f"{qty:.6f} @ ${price_int/1e8:.2f} | equity=${self._equity:.2f}")


    def _execute(self, price_int, qty, side):
        qty_signed = qty if side == Side.BID else -qty
        taker_fee  = price_int / 1e8 * abs(qty) * 0.00015
        is_closing = (self.pos > 0 and qty_signed < 0) or (self.pos < 0 and qty_signed > 0)

        r = TradeRecord()
        r.timestamp_ns = int(time.time() * 1e9)
        self._last_trade_ns = r.timestamp_ns
        r.side         = side
        r.quantity     = qty_signed

        if is_closing:
            pnl              = self.pos * (price_int - self._entry_price) / 1e8 - taker_fee
            self._cash      += pnl
            self._equity     = self._cash
            self._metrics.realized_pnl += pnl
            self._metrics.total_pnl    += pnl
            r.entry_price    = 0
            r.exit_price     = price_int
            self.pos        += qty_signed
            self._entry_price = 0
        else:
            self._cash       -= taker_fee
            self._entry_price = price_int
            self.pos         += qty_signed
            r.entry_price     = price_int
            r.exit_price      = 0

        if self._equity > self._peak_equity:
            self._peak_equity = self._equity
        dd = self._peak_equity - self._equity
        if dd > self._metrics.max_drawdown:
            self._metrics.max_drawdown = dd

        self._metrics.total_trades += 1
        self._records.append(r)
