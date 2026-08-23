"""
bullish_engine.py — Directional Long-Side Specialist Engine
============================================================
Optimized exclusively for BUY-side maker limit orders.

Architecture rationale:
  The combined engine (pure_python_engine.py) balances long and short
  signals and can cancel out directional edge by hedging itself.
  The Bullish Engine removes all short-side logic and instead:

  1. Uses signals with known BULLISH structural predictors:
       - OBI  (positive order book imbalance → more bids → bullish)
       - OFI  (buy-side flow domination)
       - Microprice pulling above mid (aggressive buyers lifting offers)

  2. Applies regime-specific boosts:
       - Regime 0 (low-vol trend): full size, tight threshold
       - Regime 1 (high-vol):      reduce size 50%, widen threshold
       - Regime 2 (mean-revert):   aggressive entry on dips (buy the fear)
       - Regime 3 (crisis):        hard halt

  3. Inventory management via Avellaneda-Stoikov inspired skew:
       As long inventory grows, entry threshold rises to prevent
       runaway directional exposure. Max position: 3× order size.

  4. Exit: time-based OR alpha-decay. No short entry allowed.

  5. VPIN kill gate set LOWER than the combined engine (0.60 vs 0.70):
       The long side is more sensitive to toxic buy-side flow — a
       rising VPIN on the buy side means informed sellers are offloading,
       not buyers accumulating. We step aside earlier.

Signal weights (long-only optimized):
  w_obi        = +0.35  (strongest bullish predictor on bid-side)
  w_ofi        = +0.28  (buy flow imbalance)
  w_microprice = +0.22  (micro pulling above mid = pressure)
  w_stat_arb   = +0.10  (mean-reversion: buy below mean)
  w_vol        = +0.05  (rising vol → momentum continuation)
  w_vpin       = -0.00  (excluded: always-positive, no directional info)
  w_spread     =  0.00  (excluded: wide spread is equally bad long/short)
  bias         = +0.002 (slight long bias reflecting crypto's upward drift)
"""

import time
from enum import Enum


# ── Shared data structures (mirror pure_python_engine) ──────────────
class Side(Enum):
    BID = 0
    ASK = 1


class EngineMode(Enum):
    LIVE     = 0
    BACKTEST = 1


class BullishConfig:
    """Configuration specific to the long-side specialist engine."""
    def __init__(self):
        self.initial_capital       = 10_000_000.0
        self.order_size_btc        = 1.0          # 1 BTC min fill
        self.max_position_btc      = 3.0          # Max 3× = 3 BTC long
        self.alpha_entry_threshold = 0.04
        self.spread_alpha_mult     = 0.14         # was 0.04 — red-zone fix (3D surface)
        self.min_take_profit_bps   = 4.0
        self.maker_fee_pct         = -0.00005
        self.taker_fee_pct         =  0.00015
        self.vpin_halt_threshold   = 0.55         # was 0.60 — long side most exposed to sellers
        self.daily_loss_limit_usd  = 15_000.0
        self.min_warmup_ticks      = 1000
        self.execution_cooldown_ns = 750_000_000
        self.max_spread_bps_cutoff = 3.0          # hard stop: no longs above 3.0 bps spread

        # Inventory skew: raise threshold by this per BTC already held
        # Forces larger alpha required as inventory grows → natural position limit
        self.inventory_skew_per_btc = 0.005


class BullishFeatureVector:
    def __init__(self):
        self.vpin           = 0.5
        self.obi            = 0.0
        self.ofi            = 0.0
        self.microprice_ret = 0.0   # microprice return (not raw price)
        self.spread_bps     = 2.0
        self.realized_vol   = 0.01
        self.stat_arb_z     = 0.0
        self.combined_alpha = 0.0
        self.regime         = 0


class BullishMetrics:
    def __init__(self):
        self.total_trades   = 0
        self.winning_trades = 0
        self.realized_pnl   = 0.0
        self.max_drawdown   = 0.0
        self.gross_edge_bps = 0.0
        self.maker_rebates  = 0.0


class TradeRecord:
    def __init__(self):
        self.timestamp_ns = 0
        self.side         = Side.BID
        self.entry_price  = 0
        self.exit_price   = 0
        self.quantity     = 0.0
        self.pnl          = 0.0
        self.is_maker     = True


# ── Signal weights ───────────────────────────────────────────────────
BULLISH_WEIGHTS = {
    "w_obi":         +0.35,
    "w_ofi":         +0.28,
    "w_microprice":  +0.22,
    "w_stat_arb":    +0.10,
    "w_vol":         +0.05,
    "w_bias":        +0.002,
}


class BullishStrategyEngine:
    """
    Long-side specialist HFT engine.

    Only trades BID (buy) direction. Posts maker limit orders at
    best_bid_price to capture the -0.5 bps maker rebate.
    Exits via limit ask at best_ask when alpha decays or take-profit hit.
    """

    ENGINE_ID = "BULLISH_v1"

    def __init__(self, config: BullishConfig = None):
        self.config        = config or BullishConfig()
        self.mode          = EngineMode.BACKTEST

        # Position & PnL state
        self.pos           = 0.0          # current BTC long position
        self._entry_price  = 0            # fixed-point avg entry
        self._cash         = self.config.initial_capital
        self._equity       = self.config.initial_capital
        self._peak_equity  = self.config.initial_capital

        # Signal state
        self._ticks         = 0
        self._halted        = False
        self._last_trade_ns = 0
        self._fv            = BullishFeatureVector()
        self._metrics       = BullishMetrics()
        self._records       = []

        # VPIN accumulator
        self._vpin_bvol    = 0.0
        self._vpin_buyvol  = 0.0
        self._vpin_window  = []
        self._VPIN_BUCKET  = 50.0

        # OBI / microprice state
        self._prev_micro   = 0.0
        self._prev_bid_qty = 0
        self._prev_ask_qty = 0

        print(f"[{self.ENGINE_ID}] Initialised | capital=${self.config.initial_capital:,.0f}"
              f" | size={self.config.order_size_btc} BTC | threshold={self.config.alpha_entry_threshold}")

    # ── Public interface ─────────────────────────────────────────────

    def equity(self):
        return self._equity

    def position(self):
        return self.pos

    def last_features(self):
        return self._fv

    def metrics(self):
        return self._metrics

    def trade_journal(self):
        return list(self._records)

    def set_mode(self, mode: EngineMode):
        self.mode = mode

    def flatten(self, book):
        """Emergency close: sell entire long position at best bid (taker)."""
        if self.pos > 0 and book.best_bid_price > 0:
            self._execute_close(book.best_bid_price, self.pos, is_maker=False)
            print(f"[{self.ENGINE_ID}] FLATTEN: sold {self.pos:.4f} BTC @ ${book.best_bid_price/1e8:.2f}")

    # ── Market data handlers ─────────────────────────────────────────

    def on_book_update(self, book):
        """Update OBI, microprice, spread from new book snapshot."""
        if book.best_bid_price <= 0 or book.best_ask_price <= 0:
            return

        mid = (book.best_bid_price + book.best_ask_price) / 2.0

        # Mark-to-market
        if self.pos > 0:
            self._equity = self._cash + self.pos * (mid - self._entry_price) / 1e8

        # OBI
        bq = book.best_bid_qty
        aq = book.best_ask_qty
        tq = bq + aq
        if tq > 0:
            self._fv.obi = (bq - aq) / tq

            # OFI = change in bid qty - change in ask qty
            ofi = (bq - self._prev_bid_qty) - (aq - self._prev_ask_qty)
            self._fv.ofi = float(ofi) / max(tq, 1.0)
            self._prev_bid_qty = bq
            self._prev_ask_qty = aq

            # Microprice return
            mp = (book.best_bid_price * aq + book.best_ask_price * bq) / tq
            mp_btc = mp / 1e8
            if self._prev_micro > 0:
                self._fv.microprice_ret = (mp_btc - self._prev_micro) / self._prev_micro * 1e4
            self._prev_micro = mp_btc

        self._fv.spread_bps = (book.best_ask_price - book.best_bid_price) / book.best_bid_price * 10000.0

    def on_trade(self, trade, book):
        """Process a new trade tick through the bullish pipeline."""
        self._ticks += 1
        if self._ticks < self.config.min_warmup_ticks:
            return
        if self._halted:
            return

        # Daily loss limit
        loss = self.config.initial_capital - self._equity
        if loss >= self.config.daily_loss_limit_usd:
            self._halted = True
            print(f"[{self.ENGINE_ID}][RISK] Daily loss ${loss:,.0f} >= limit ${self.config.daily_loss_limit_usd:,.0f}. HALTED.")
            return

        # Execution cooldown
        now_ns = int(time.time() * 1e9)
        if now_ns - self._last_trade_ns < self.config.execution_cooldown_ns:
            return

        # Update VPIN
        self._update_vpin(trade)
        vpin = self._fv.vpin

        # VPIN kill gate — long side is more exposed to informed sellers
        if vpin > self.config.vpin_halt_threshold:
            return

        # Hard spread circuit-breaker — 3D surface red-zone gate
        if getattr(self.config, 'max_spread_bps_cutoff', 3.0) > 0:
            if self._fv.spread_bps > self.config.max_spread_bps_cutoff:
                return

        # Compute directional alpha (long-only weights)
        w = BULLISH_WEIGHTS
        raw_alpha = (
            w["w_obi"]        * self._fv.obi +
            w["w_ofi"]        * self._fv.ofi +
            w["w_microprice"] * self._fv.microprice_ret +
            w["w_stat_arb"]   * self._fv.stat_arb_z +
            w["w_vol"]        * self._fv.realized_vol +
            w["w_bias"]
        )
        alpha = max(-1.0, min(1.0, raw_alpha))
        self._fv.combined_alpha = alpha

        # Regime multiplier
        regime = self._fv.regime
        if regime == 1:    # high-vol chaos: step back
            regime_mult = 1.5
        elif regime == 2:  # mean-reversion: buy dips aggressively
            regime_mult = 0.7
        elif regime == 3:  # crisis: halt
            return
        else:
            regime_mult = 1.0

        spread_bps = self._fv.spread_bps
        spread_addon = spread_bps * self.config.spread_alpha_mult

        # Inventory skew: raise entry threshold as position grows
        inventory_skew = self.pos * self.config.inventory_skew_per_btc

        entry_thr = (self.config.alpha_entry_threshold * regime_mult
                     + spread_addon + inventory_skew)

        # ── EXIT: close long on alpha decay ──────────────────────────
        if self.pos > 0:
            unrealised_bps = 0.0
            if self._entry_price > 0 and book.best_bid_price > 0:
                unrealised_bps = ((book.best_bid_price - self._entry_price)
                                  / self._entry_price * 10_000.0)

            # Exit if alpha decayed below 0.01 AND either:
            #   (a) we've hit the take-profit target, or
            #   (b) alpha has flipped negative (adverse signal)
            if alpha < 0.01:
                if (unrealised_bps >= self.config.min_take_profit_bps
                        or alpha < -0.02):
                    if book.best_ask_price > 0:
                        # Post limit sell at best_ask (maker)
                        self._execute_close(book.best_ask_price, self.pos, is_maker=True)
                    return

        # ── ENTRY: open long if alpha strong enough ───────────────────
        if alpha > entry_thr:
            remaining_capacity = self.config.max_position_btc - self.pos
            if remaining_capacity < 0.01:
                return  # Already at max

            qty = min(self.config.order_size_btc, remaining_capacity)
            if book.best_bid_price > 0:
                # Post limit buy at best_bid (maker)
                self._execute_open(book.best_bid_price, qty, is_maker=True)

    # ── Internal execution ───────────────────────────────────────────

    def _execute_open(self, price_int, qty, is_maker=True):
        fee_pct = self.config.maker_fee_pct if is_maker else self.config.taker_fee_pct
        fee     = price_int / 1e8 * qty * abs(fee_pct)
        # Maker rebate is income, taker fee is cost
        if is_maker:
            self._cash    += price_int / 1e8 * qty * abs(fee_pct)  # rebate credit
            self._metrics.maker_rebates += price_int / 1e8 * qty * abs(fee_pct)
        else:
            self._cash    -= fee

        # VWAP entry
        old_notional      = self._entry_price * self.pos / 1e8
        new_notional      = price_int / 1e8 * qty
        self.pos         += qty
        if self.pos > 0:
            self._entry_price = int((old_notional + new_notional) / self.pos * 1e8)

        now_ns            = int(time.time() * 1e9)
        self._last_trade_ns = now_ns

        if self.mode == EngineMode.BACKTEST:
            r = TradeRecord()
            r.timestamp_ns = now_ns
            r.side         = Side.BID
            r.entry_price  = price_int
            r.exit_price   = 0
            r.quantity     = qty
            r.is_maker     = is_maker
            self._records.append(r)

        self._metrics.total_trades += 1

    def _execute_close(self, price_int, qty, is_maker=True):
        fee_pct  = self.config.maker_fee_pct if is_maker else self.config.taker_fee_pct
        fee      = price_int / 1e8 * qty * abs(fee_pct)

        pnl      = (price_int - self._entry_price) / 1e8 * qty
        if is_maker:
            pnl  += price_int / 1e8 * qty * abs(self.config.maker_fee_pct)  # rebate
            self._metrics.maker_rebates += price_int / 1e8 * qty * abs(self.config.maker_fee_pct)
        else:
            pnl  -= fee

        self._cash             += pnl
        self._equity            = self._cash
        self.pos               -= qty
        if self.pos < 1e-8:
            self.pos            = 0.0
            self._entry_price   = 0

        self._metrics.realized_pnl += pnl
        if pnl > 0:
            self._metrics.winning_trades += 1

        # Track edge in bps
        if self._entry_price > 0:
            edge_bps = (price_int - self._entry_price) / self._entry_price * 10_000.0
            self._metrics.gross_edge_bps += edge_bps

        dd = self._peak_equity - self._equity
        if dd > self._metrics.max_drawdown:
            self._metrics.max_drawdown = dd
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        now_ns = int(time.time() * 1e9)
        self._last_trade_ns = now_ns

        if self.mode == EngineMode.BACKTEST:
            r = TradeRecord()
            r.timestamp_ns = now_ns
            r.side         = Side.ASK
            r.entry_price  = self._entry_price
            r.exit_price   = price_int
            r.quantity     = qty
            r.pnl          = pnl
            r.is_maker     = is_maker
            self._records.append(r)

        self._metrics.total_trades += 1

    def _update_vpin(self, trade):
        vol_btc = abs(trade.qty) / 1e8
        self._vpin_bvol += vol_btc
        if trade.side == Side.BID:
            self._vpin_buyvol += vol_btc

        if self._vpin_bvol >= self._VPIN_BUCKET:
            sell_vol = self._vpin_bvol - self._vpin_buyvol
            imb = abs(self._vpin_buyvol - sell_vol) / self._vpin_bvol
            self._vpin_window.append(imb)
            if len(self._vpin_window) > 50:
                self._vpin_window.pop(0)
            self._vpin_bvol   = 0.0
            self._vpin_buyvol = 0.0

        if self._vpin_window:
            self._fv.vpin = float(sum(self._vpin_window) / len(self._vpin_window))
