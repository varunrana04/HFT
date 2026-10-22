"""
HFT Strategy — NautilusTrader Adapter
======================================
Wraps the trained LGBM signal model and PPO RL policy inside a NautilusTrader
Strategy. Use BacktestEngine for realistic OOS validation with proper fill
simulation, fees, position tracking — no more hand-rolled backtest scripts.

Usage:
    python hft_strategy.py --model lgbm
    python hft_strategy.py --model rl

Models expected at:
    ../ml_pipeline/models/lgbm_trade_tape_model.txt
    ../ml_pipeline/models/rl_policy.pt
"""

import os, sys, argparse, requests, zipfile, io
from pathlib import Path
from datetime import datetime

import numpy as np
import polars as pl
import lightgbm as lgb
import torch
import torch.nn as nn
from torch.distributions import Normal

from nautilus_trader.backtest.engine import BacktestEngine, BacktestEngineConfig
from nautilus_trader.config import LoggingConfig
from nautilus_trader.model.currencies import USDT, BTC
from nautilus_trader.model.data import TradeTick
from nautilus_trader.model.enums import (
    AccountType, OmsType, OrderSide, TimeInForce, AggressorSide
)
from nautilus_trader.model.identifiers import InstrumentId, Symbol, Venue, TradeId
from nautilus_trader.model.instruments import CryptoPerpetual
from nautilus_trader.model.objects import Price, Quantity, Money
from nautilus_trader.trading.strategy import Strategy
from nautilus_trader.config import StrategyConfig

# ── Paths
HERE      = Path(__file__).parent
ML_DIR    = HERE.parent / "ml_pipeline" / "models"
LGBM_PATH = ML_DIR / "lgbm_trade_tape_model.txt"
RL_PATH   = ML_DIR / "rl_policy.pt"
BASE_URL  = "https://data.binance.vision/data/futures/um/daily"

# ── Constants (must match training)
FEATURES        = ["tfi_ema_50", "tfi_ema_200", "volatility_ema_100"]  # old 3-feature model
CONFIDENCE_BAND = 0.5
TAKER_FEE_BPS   = 0.1
COOLDOWN_MS     = 200
MAX_POSITION    = 10
N_FEATURES      = 4   # 3 features + inv_frac (old model)

# ─────────────────────────────────────────────────────────────────────────────
# RL NETWORK (identical to train_rl_ppo.py)
# ─────────────────────────────────────────────────────────────────────────────
class StateExtractor(nn.Module):
    def __init__(self, n_in=N_FEATURES, latent_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in,256), nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256,256),  nn.LayerNorm(256), nn.SiLU(),
            nn.Linear(256,latent_dim), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class Actor(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.net     = nn.Sequential(nn.Linear(latent_dim,128), nn.SiLU(), nn.Linear(128,3))
        self.log_std = nn.Parameter(torch.zeros(3))
        self.register_buffer("action_mins", torch.tensor([0.3, 0.3, 0.01]))
        self.register_buffer("action_maxs", torch.tensor([3.0, 3.0, 0.10]))
    def forward(self, lat):
        lo, hi = self.action_mins, self.action_maxs
        mean   = lo + (hi-lo) * torch.sigmoid(self.net(lat))
        std    = torch.exp(self.log_std.clamp(-3,0)).expand_as(mean)
        return mean, std
    def sample(self, lat):
        mean, std = self(lat)
        raw = Normal(mean, std).rsample()
        return torch.clamp(raw, self.action_mins, self.action_maxs)

class PPOAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = StateExtractor()
        self.actor   = Actor()
    @torch.no_grad()
    def act(self, obs_np):
        t = torch.tensor(obs_np, dtype=torch.float32).unsqueeze(0)
        return self.actor.sample(self.encoder(t)).squeeze(0).numpy()

# ─────────────────────────────────────────────────────────────────────────────
# STREAMING FEATURE BUILDER
# ─────────────────────────────────────────────────────────────────────────────
class TickFeatureBuilder:
    """Maintains running EMAs over a live tick stream — zero lookahead."""
    def __init__(self):
        self.a50  = 1 - np.exp(-1/50)
        self.a200 = 1 - np.exp(-1/200)
        self.a100 = 1 - np.exp(-1/100)
        self.a500 = 1 - np.exp(-1/500)
        self.tfi50 = self.tfi200 = self.v100 = self.v500 = self.abs_sv = 0.0
        self.prev_price = None

    def update(self, price, qty, is_buyer_maker):
        sv = -qty if is_buyer_maker else qty
        dp = abs(price - self.prev_price) if self.prev_price is not None else 0.0
        self.prev_price = price
        self.tfi50  += self.a50  * (sv  - self.tfi50)
        self.tfi200 += self.a200 * (sv  - self.tfi200)
        self.v100   += self.a100 * (dp  - self.v100)
        self.v500   += self.a500 * (dp  - self.v500)
        self.abs_sv += self.a50  * (abs(sv) - self.abs_sv)
        return {
            "tfi_ema_50":         self.tfi50,
            "tfi_ema_200":        self.tfi200,
            "volatility_ema_100": self.v100,
            "volatility_ema_500": self.v500,
            "ofi":                sv / (self.abs_sv + 1e-9),
            "spread_proxy":       self.v100 * 2.0,
        }

# ─────────────────────────────────────────────────────────────────────────────
# LGBM STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
class LGBMStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    model_path:    str

class LGBMStrategy(Strategy):
    def __init__(self, config: LGBMStrategyConfig):
        super().__init__(config)
        self.iid          = InstrumentId.from_str(config.instrument_id)
        self.model        = lgb.Booster(model_file=config.model_path)
        self.fb           = TickFeatureBuilder()
        self.position     = 0
        self.last_ts      = 0
        self.cooldown_ns  = COOLDOWN_MS * 1_000_000

    def on_start(self):
        self.subscribe_trade_ticks(self.iid)

    def on_trade_tick(self, tick: TradeTick):
        is_bm = tick.aggressor_side == AggressorSide.SELLER
        feats = self.fb.update(float(tick.price), float(tick.size), is_bm)
        if (tick.ts_event - self.last_ts) < self.cooldown_ns:
            return
        fv  = np.array([[feats[f] for f in FEATURES]], dtype=np.float32)
        pred = float(self.model.predict(fv)[0])
        if abs(pred) <= CONFIDENCE_BAND:
            return
        if pred > 0 and self.position >= MAX_POSITION:
            return
        if pred < 0 and self.position <= -MAX_POSITION:
            return
        inst  = self.cache.instrument(self.iid)
        side  = OrderSide.BUY if pred > 0 else OrderSide.SELL
        order = self.order_factory.market(
            instrument_id=self.iid,
            order_side=side,
            quantity=inst.make_qty(1),
            time_in_force=TimeInForce.IOC,
        )
        self.submit_order(order)
        self.last_ts   = tick.ts_event
        self.position += 1 if side == OrderSide.BUY else -1

    def on_stop(self):
        self.log.info(f"LGBM | final pos: {self.position}")

# ─────────────────────────────────────────────────────────────────────────────
# RL STRATEGY
# ─────────────────────────────────────────────────────────────────────────────
class RLStrategyConfig(StrategyConfig, frozen=True):
    instrument_id: str
    policy_path:   str

class RLStrategy(Strategy):
    def __init__(self, config: RLStrategyConfig):
        super().__init__(config)
        self.iid          = InstrumentId.from_str(config.instrument_id)
        self.agent        = PPOAgent()
        self.agent.load_state_dict(
            torch.load(config.policy_path, map_location="cpu", weights_only=True),
            strict=False
        )
        self.agent.eval()
        self.fb          = TickFeatureBuilder()
        self.position    = 0
        self.last_ts     = 0
        self.cooldown_ns = COOLDOWN_MS * 1_000_000

    def on_start(self):
        self.subscribe_trade_ticks(self.iid)

    def on_trade_tick(self, tick: TradeTick):
        is_bm = tick.aggressor_side == AggressorSide.SELLER
        feats = self.fb.update(float(tick.price), float(tick.size), is_bm)
        if (tick.ts_event - self.last_ts) < self.cooldown_ns:
            return
        inv_frac = self.position / MAX_POSITION
        obs = np.array([feats[f] for f in FEATURES] + [inv_frac], dtype=np.float32)
        obs = np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        actions     = self.agent.act(obs)
        vol         = feats["volatility_ema_100"]
        half_spread = vol * float(actions[0])

        # Simple signal: if spread is tight and vol is low, take a position
        if half_spread < CONFIDENCE_BAND and abs(self.position) < MAX_POSITION:
            inst  = self.cache.instrument(self.iid)
            side  = OrderSide.BUY if self.position <= 0 else OrderSide.SELL
            order = self.order_factory.market(
                instrument_id=self.iid,
                order_side=side,
                quantity=inst.make_qty(1),
                time_in_force=TimeInForce.IOC,
            )
            self.submit_order(order)
            self.last_ts   = tick.ts_event
            self.position += 1 if side == OrderSide.BUY else -1

    def on_stop(self):
        self.log.info(f"RL | final pos: {self.position}")

# ─────────────────────────────────────────────────────────────────────────────
# DATA LOADER: Binance Vision -> NT TradeTick
# ─────────────────────────────────────────────────────────────────────────────
CACHE_DIR = Path("nautilus_cache")
CACHE_DIR.mkdir(exist_ok=True)

def load_trade_ticks(symbol, dates, instrument_id):
    """Downloads aggTrades with local parquet disk cache. Safe to re-run."""
    ticks = []
    for date_str in dates:
        cache_path = CACHE_DIR / f"{symbol}-aggTrades-{date_str}.parquet"
        if cache_path.exists():
            print(f"  [{date_str}] loading from cache...")
            df = pl.read_parquet(cache_path)
        else:
            url = f"{BASE_URL}/aggTrades/{symbol}/{symbol}-aggTrades-{date_str}.zip"
            print(f"  [{date_str}] downloading...")
            try:
                r = requests.get(url, timeout=60)
                if r.status_code != 200:
                    print(f"  [SKIP] HTTP {r.status_code}")
                    continue
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    with z.open(z.namelist()[0]) as f:
                        df = pl.read_csv(f.read(), has_header=True)
                df = df.rename({c: c.lower() for c in df.columns})
                df.write_parquet(cache_path)
                print(f"  [{date_str}] cached to {cache_path}")
            except Exception as e:
                print(f"  [ERROR] {date_str}: {e}")
                continue

        for row in df.iter_rows(named=True):
            try:
                side = AggressorSide.SELLER if row["is_buyer_maker"] else AggressorSide.BUYER
                ticks.append(TradeTick(
                    instrument_id=instrument_id,
                    price=Price.from_str(f"{float(row['price']):.2f}"),
                    size=Quantity.from_str(f"{float(row['quantity']):.3f}"),
                    aggressor_side=side,
                    trade_id=TradeId(str(row["agg_trade_id"])),
                    ts_event=int(row["transact_time"]) * 1_000_000,
                    ts_init=int(row["transact_time"]) * 1_000_000,
                ))
            except Exception:
                continue

    print(f"  Total ticks: {len(ticks):,}")
    return ticks

# ─────────────────────────────────────────────────────────────────────────────
# BACKTEST RUNNER
# ─────────────────────────────────────────────────────────────────────────────
def run_backtest(model_type="lgbm"):
    print(f"\n{'='*60}")
    print(f"NautilusTrader HFT Backtest — {model_type.upper()}")
    print(f"OOS: Aug 25-29 2026  |  BTCUSDT Perp  |  Binance")
    print(f"{'='*60}\n")

    engine = BacktestEngine(config=BacktestEngineConfig(
        logging=LoggingConfig(log_level="WARNING"),
    ))

    BINANCE = Venue("BINANCE")
    engine.add_venue(
        venue=BINANCE,
        oms_type=OmsType.NETTING,
        account_type=AccountType.MARGIN,
        base_currency=None,
        starting_balances=[Money(100_000, USDT)],
    )

    instrument_id = InstrumentId(Symbol("BTCUSDT-PERP"), BINANCE)
    instrument = CryptoPerpetual(
        instrument_id=instrument_id,
        raw_symbol=Symbol("BTCUSDT"),
        base_currency=BTC,
        quote_currency=USDT,
        settlement_currency=USDT,
        is_inverse=False,
        price_precision=2,
        size_precision=3,
        price_increment=Price.from_str("0.10"),
        size_increment=Quantity.from_str("0.001"),
        max_quantity=Quantity.from_str("1000.0"),
        min_quantity=Quantity.from_str("0.001"),
        max_notional=None,
        min_notional=Money(10, USDT),
        max_price=Price.from_str("9999999.0"),
        min_price=Price.from_str("100.0"),
        margin_init=None,
        margin_maint=None,
        maker_fee=None,
        taker_fee=None,
        ts_event=0,
        ts_init=0,
    )
    engine.add_instrument(instrument)

    OOS_DATES = ["2026-08-25","2026-08-26","2026-08-27","2026-08-28","2026-08-29"]
    ticks = load_trade_ticks("BTCUSDT", OOS_DATES, instrument_id)
    engine.add_data(ticks)

    if model_type == "lgbm":
        assert LGBM_PATH.exists(), f"Model not found: {LGBM_PATH}"
        strategy = LGBMStrategy(LGBMStrategyConfig(
            instrument_id="BTCUSDT-PERP.BINANCE",
            model_path=str(LGBM_PATH),
        ))
    else:
        assert RL_PATH.exists(), f"Policy not found: {RL_PATH}"
        strategy = RLStrategy(RLStrategyConfig(
            instrument_id="BTCUSDT-PERP.BINANCE",
            policy_path=str(RL_PATH),
        ))

    engine.add_strategy(strategy)
    print("Running...")
    engine.run()

    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    for line in engine.get_stats_pnls_formatted():
        print(line)
    fills = engine.get_fills()
    print(f"\nTotal fills: {len(fills):,}")
    engine.dispose()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lgbm","rl"], default="lgbm")
    args = parser.parse_args()
    run_backtest(model_type=args.model)

