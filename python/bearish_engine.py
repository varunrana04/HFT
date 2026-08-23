"""
bearish_engine.py — Directional Short-Side Specialist Engine
=============================================================
Optimized exclusively for SELL-side maker limit orders.

Architecture rationale:
  The short side operates under fundamentally different microstructure
  dynamics than the long side:

  1. VPIN is the PRIMARY signal here, not a kill gate.
     High VPIN on the SELL side means informed sellers are distributing
     inventory into buy orders. This is the most reliable short signal
     in crypto market microstructure (Easley, Lopez de Prado, O'Hara).

  2. Order Book Collapse detection:
     When the bid side of the book thins faster than the ask side
     (negative OBI trend), this predicts short-term downward pressure.
     The Bearish Engine specifically monitors the rate of change of OBI.

  3. Spread widening as entry signal:
     Unlike the long side (where wide spreads are friction), widening
     spreads on the bearish engine are an ENTRY signal: market makers
     are pulling their bids, signaling they expect the price to fall.

  4. Stat-arb Z-score inversion:
     High positive stat-arb z-score means price is ABOVE the mean →
     reversion downward is expected → bearish entry.

  5. Regime behaviour:
     Regime 1 (high-vol chaos): INCREASE short aggression — panic sells
     create the most profitable short-side opportunities.
     Regime 2 (mean-reversion): Standard fade-the-rally sizing.
     Regime 3 (crisis): Moderate position only (avoid short squeezes).

  6. VPIN kill gate is HIGHER (0.85):
     The short side WANTS high VPIN — that IS the signal. We only halt
     when VPIN is so extreme (>0.85) it indicates a short squeeze risk
     where informed BUY orders are overwhelming the sell side.

Signal weights (short-only optimized):
  w_vpin       = -0.38  (high toxicity → bearish, strongest predictor)
  w_obi        = -0.28  (negative OBI → bid-side weakness → bearish)
  w_stat_arb   = -0.18  (price above mean → revert down)
  w_spread     = -0.10  (widening spread → makers pulling bids → bearish)
  w_ofi        = -0.06  (sell flow dominance)
  w_vol        = +0.00  (excluded: rising vol is symmetric)
  w_microprice =  0.00  (excluded: micro below mid already in OFI)
  bias         = -0.002 (slight short bias for regime 1 = high-vol)
"""

import time
from enum import Enum


class Side(Enum):
    BID = 0
    ASK = 1


class EngineMode(Enum):
    LIVE     = 0
    BACKTEST = 1


class BearishConfig:
    """Configuration specific to the short-side specialist engine."""
    def __init__(self):
        self.initial_capital       = 10_000_000.0
        self.order_size_btc        = 1.0
        self.max_position_btc      = 2.0
        self.alpha_entry_threshold = 0.045
        self.spread_alpha_mult     = -0.02        # Wide spread BOOSTS short entry (inverted)
        self.min_take_profit_bps   = 3.5
        self.maker_fee_pct         = -0.00005
        self.taker_fee_pct         =  0.00015
        self.vpin_halt_threshold   = 0.80         # was 0.85 — tightened; extreme VPIN = squeeze risk
        self.daily_loss_limit_usd  = 12_000.0
        self.min_warmup_ticks      = 1000
        self.execution_cooldown_ns = 600_000_000
        self.max_spread_bps_cutoff = 4.5          # shorts tolerate wider spread but cap at 4.5 bps

        # OBI trend window: compute rate of change of OBI over this many ticks
        self.obi_trend_window      = 20

        # Inventory skew: increase threshold as short grows (short squeeze protection)
        self.inventory_skew_per_btc = 0.008       # Steeper than long side


class BearishFeatureVector:
    def __init__(self):
        self.vpin           = 0.5
        self.obi            = 0.0
        self.obi_delta      = 0.0    # Rate of OBI change — book collapse signal
        self.ofi            = 0.0
        self.spread_bps     = 2.0
        self.spread_delta   = 0.0    # Spread widening rate
        self.realized_vol   = 0.01
        self.stat_arb_z     = 0.0
        self.combined_alpha = 0.0
        self.regime         = 0


class BearishMetrics:
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
        self.side         = Side.ASK
        self.entry_price  = 0
        self.exit_price   = 0
        self.quantity     = 0.0
        self.pnl          = 0.0
        self.is_maker     = True


BEARISH_WEIGHTS = {
    "w_vpin":    -0.38,
    "w_obi":     -0.28,
    "w_stat_arb":-0.18,
    "w_spread":  -0.10,
    "w_ofi":     -0.06,
    "w_bias":    -0.002,
}


class BearishStrategyEngine:
    """
    Short-side specialist HFT engine.

    Only trades ASK (sell/short) direction. Posts maker limit orders at
    best_ask_price to capture the -0.5 bps maker rebate.
    Covers (buys back) via limit bid at best_bid when alpha decays or
    take-profit hit.

    Primary alpha source: VPIN + OBI collapse + spread widening.
    """

    ENGINE_ID = "BEARISH_v1"

    def __init__(self, config: BearishConfig = None):
        self.config        = config or BearishConfig()
        self.mode          = EngineMode.BACKTEST

        self.pos           = 0.0          # negative = short position
        self._entry_price  = 0
        self._cash         = self.config.initial_capital
        self._equity       = self.config.initial_capital
        self._peak_equity  = self.config.initial_capital

        self._ticks         = 0
        self._halted        = False
        self._last_trade_ns = 0
        self._fv            = BearishFeatureVector()
        self._metrics       = BearishMetrics()
        self._records       = []

        # VPIN state
        self._vpin_bvol    = 0.0
        self._vpin_buyvol  = 0.0
        self._vpin_window  = []
        self._VPIN_BUCKET  = 50.0

        # OBI / spread tracking for delta computation
        self._obi_history    = []
        self._spread_history = []

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
        """Emergency cover: buy back entire short at best ask (taker)."""
        if self.pos < 0 and book.best_ask_price > 0:
            cover_qty = abs(self.pos)
            self._execute_close(book.best_ask_price, cover_qty, is_maker=False)
            print(f"[{self.ENGINE_ID}] FLATTEN: covered {cover_qty:.4f} BTC @ ${book.best_ask_price/1e8:.2f}")

    # ── Market data handlers ─────────────────────────────────────────

    def on_book_update(self, book):
        if book.best_bid_price <= 0 or book.best_ask_price <= 0:
            return

        mid = (book.best_bid_price + book.best_ask_price) / 2.0

        # Mark-to-market for short position
        if self.pos < 0:
            # Short PnL: entry_price - current_mid (profit when price falls)
            self._equity = self._cash + abs(self.pos) * (self._entry_price - mid) / 1e8

        bq = book.best_bid_qty
        aq = book.best_ask_qty
        tq = bq + aq
        if tq > 0:
            obi = (bq - aq) / tq
            self._fv.obi = obi

            # OFI
            ofi = (bq - self._prev_bid_qty) - (aq - self._prev_ask_qty)
            self._fv.ofi = float(ofi) / max(tq, 1.0)
            self._prev_bid_qty = bq
            self._prev_ask_qty = aq

            # OBI delta (book collapse signal)
            self._obi_history.append(obi)
            if len(self._obi_history) > self.config.obi_trend_window:
                self._obi_history.pop(0)
            if len(self._obi_history) >= 2:
                self._fv.obi_delta = self._obi_history[-1] - self._obi_history[0]

        spread_bps = (book.best_ask_price - book.best_bid_price) / book.best_bid_price * 10000.0
        self._fv.spread_bps = spread_bps

        # Spread delta (widening = bearish signal)
        self._spread_history.append(spread_bps)
        if len(self._spread_history) > self.config.obi_trend_window:
            self._spread_history.pop(0)
        if len(self._spread_history) >= 2:
            self._fv.spread_delta = self._spread_history[-1] - self._spread_history[0]

    def on_trade(self, trade, book):
        self._ticks += 1
        if self._ticks < self.config.min_warmup_ticks:
            return
        if self._halted:
            return

        loss = self.config.initial_capital - self._equity
        if loss >= self.config.daily_loss_limit_usd:
            self._halted = True
            print(f"[{self.ENGINE_ID}][RISK] Daily loss ${loss:,.0f} >= limit. HALTED.")
            return

        now_ns = int(time.time() * 1e9)
        if now_ns - self._last_trade_ns < self.config.execution_cooldown_ns:
            return

        self._update_vpin(trade)
        vpin = self._fv.vpin

        # VPIN kill gate — high VPIN is the alpha, halt only on extreme (short squeeze risk)
        if vpin > self.config.vpin_halt_threshold:
            return

        # Hard spread circuit-breaker — cap even short entries above max_spread_bps_cutoff
        if getattr(self.config, 'max_spread_bps_cutoff', 4.5) > 0:
            if self._fv.spread_bps > self.config.max_spread_bps_cutoff:
                return

        # Bearish alpha: VPIN + OBI collapse + spread widening + stat-arb above mean
        w = BEARISH_WEIGHTS
        # Note: all weights are negative so a HIGH vpin/spread gives a NEGATIVE alpha
        # which is the SHORT signal. alpha < -threshold = enter short.
        raw_alpha = (
            w["w_vpin"]    * (vpin - 0.5) +           # centred: >0.5 = bearish
            w["w_obi"]     * self._fv.obi +            # negative OBI = bid thinning
            w["w_obi"]     * self._fv.obi_delta * 0.5 + # rapid collapse amplifier
            w["w_stat_arb"]* self._fv.stat_arb_z +    # above mean = revert down
            w["w_spread"]  * (self._fv.spread_delta) + # widening spread
            w["w_ofi"]     * self._fv.ofi +
            w["w_bias"]
        )
        alpha = max(-1.0, min(1.0, raw_alpha))
        self._fv.combined_alpha = alpha

        # Regime behaviour (inverted vs long side)
        regime = self._fv.regime
        if regime == 1:    # high-vol chaos: INCREASE short aggression
            regime_mult = 0.7   # tighter threshold = more aggressive
        elif regime == 2:  # mean-reversion: standard
            regime_mult = 1.0
        elif regime == 3:  # crisis: small short only (short squeeze risk)
            regime_mult = 2.0   # much wider threshold = very conservative
        else:
            regime_mult = 1.0

        spread_bps   = self._fv.spread_bps
        # Wide spread BOOSTS short entry (inverted: spread_alpha_mult is negative)
        spread_addon = spread_bps * self.config.spread_alpha_mult

        # Inventory skew: make it harder to add to an already-large short
        inventory_skew = abs(self.pos) * self.config.inventory_skew_per_btc

        entry_thr = (self.config.alpha_entry_threshold * regime_mult
                     + spread_addon + inventory_skew)

        # ── EXIT: cover short on alpha decay ─────────────────────────
        if self.pos < 0:
            unrealised_bps = 0.0
            if self._entry_price > 0 and book.best_ask_price > 0:
                unrealised_bps = ((self._entry_price - book.best_ask_price)
                                  / self._entry_price * 10_000.0)

            if alpha > -0.01:  # alpha faded back toward zero
                if (unrealised_bps >= self.config.min_take_profit_bps
                        or alpha > 0.02):
                    if book.best_bid_price > 0:
                        # Post limit buy at best_bid (maker cover)
                        self._execute_close(book.best_bid_price, abs(self.pos), is_maker=True)
                    return

        # ── ENTRY: open short if alpha negative enough ────────────────
        if alpha < -entry_thr:
            current_short = abs(self.pos)
            remaining_capacity = self.config.max_position_btc - current_short
            if remaining_capacity < 0.01:
                return

            qty = min(self.config.order_size_btc, remaining_capacity)
            if book.best_ask_price > 0:
                # Post limit sell at best_ask (maker short)
                self._execute_open(book.best_ask_price, qty, is_maker=True)

    # ── Internal execution ───────────────────────────────────────────

    def _execute_open(self, price_int, qty, is_maker=True):
        fee_pct = self.config.maker_fee_pct if is_maker else self.config.taker_fee_pct
        if is_maker:
            self._cash += price_int / 1e8 * qty * abs(fee_pct)  # rebate
            self._metrics.maker_rebates += price_int / 1e8 * qty * abs(fee_pct)
        else:
            self._cash -= price_int / 1e8 * qty * abs(fee_pct)

        # VWAP short entry
        old_notional  = self._entry_price * abs(self.pos) / 1e8
        new_notional  = price_int / 1e8 * qty
        self.pos     -= qty   # position goes more negative
        if abs(self.pos) > 0:
            self._entry_price = int((old_notional + new_notional) / abs(self.pos) * 1e8)

        now_ns = int(time.time() * 1e9)
        self._last_trade_ns = now_ns

        if self.mode == EngineMode.BACKTEST:
            r = TradeRecord()
            r.timestamp_ns = now_ns
            r.side         = Side.ASK
            r.entry_price  = price_int
            r.exit_price   = 0
            r.quantity     = qty
            r.is_maker     = is_maker
            self._records.append(r)

        self._metrics.total_trades += 1

    def _execute_close(self, price_int, qty, is_maker=True):
        fee_pct = self.config.maker_fee_pct if is_maker else self.config.taker_fee_pct
        if is_maker:
            self._cash += price_int / 1e8 * qty * abs(fee_pct)  # rebate
            self._metrics.maker_rebates += price_int / 1e8 * qty * abs(fee_pct)
        else:
            self._cash -= price_int / 1e8 * qty * abs(fee_pct)

        # Short cover PnL: entry - cover price (profit when price falls)
        pnl  = (self._entry_price - price_int) / 1e8 * qty

        self._cash           += pnl
        self._equity          = self._cash
        self.pos             += qty   # reduce short
        if abs(self.pos) < 1e-8:
            self.pos          = 0.0
            self._entry_price = 0

        self._metrics.realized_pnl += pnl
        if pnl > 0:
            self._metrics.winning_trades += 1

        if self._entry_price > 0:
            edge_bps = (self._entry_price - price_int) / self._entry_price * 10_000.0
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
            r.side         = Side.BID
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
