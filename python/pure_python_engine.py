import time
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
        self.daily_loss_limit_usd    = 500.0
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

class Trade:
    def __init__(self):
        self.timestamp_ns = 0
        self.price        = 0
        self.qty          = 0
        self.side         = Side.BID

class FeatureVector:
    def __init__(self):
        self.timestamp_ns   = 0
        self.vpin           = 0.5
        self.microprice     = 0.0
        self.spread_bps     = 2.0
        self.realized_vol   = 0.01
        self.ofi            = 0.0
        self.obi            = 0.0
        self.combined_alpha = 0.0
        self.regime         = 0

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
    def __init__(self, config):
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
        self._last_trade_ns  = 0

    def load_model(self, path):
        self._weights = _load_signal_weights(path)
        return True

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
        if self._ticks < self.config.min_warmup_ticks:
            return
        if self._halted:
            return

        # Daily loss limit
        loss = self.config.initial_capital - self._equity
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
