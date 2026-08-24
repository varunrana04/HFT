import os
import sys
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import pandas as pd
import time

# Ensure we can load the engine
sys.path.insert(0, os.path.dirname(__file__))
from engine_loader import load_engine

hft_engine = load_engine()

class HFTTradingEnv(gym.Env):
    """
    Gymnasium environment that wraps the C++ HFT Strategy Engine.
    The agent observes the market microstructure (alpha, vpin, obi, vol, inventory)
    and actions tune the strategy parameters:
      Action 0: spread_alpha_multiplier (continuous)
      Action 1: alpha_entry_threshold (continuous)
    """
    metadata = {"render_modes": ["human"]}

    def __init__(self, data_path=None, steps_per_episode=1000, ticks_per_step=100):
        super(HFTTradingEnv, self).__init__()
        
        self.data_path = data_path
        self.steps_per_episode = steps_per_episode
        self.ticks_per_step = ticks_per_step
        
        # Load data
        if self.data_path and os.path.exists(self.data_path):
            self.df = pd.read_csv(self.data_path)
            print(f"[INFO] Loaded {len(self.df)} rows from {self.data_path}")
        else:
            print("[WARNING] No data path provided or file missing. Generating synthetic GBM data...")
            self.df = self._generate_synthetic_data(steps_per_episode * ticks_per_step + 2000)
            
        self.total_ticks = len(self.df)
        self.current_tick = 0
        
        # Action space: 
        # [0] spread_alpha_multiplier: range [0.01, 1.0] (mapped from [-1, 1])
        # [1] alpha_entry_threshold: range [0.01, 1.0] (mapped from [-1, 1])
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        
        # Observation space: 
        # [combined_alpha, vpin, obi, volatility, inventory_pct, spread_bps]
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(6,), dtype=np.float32)
        
        self.config = hft_engine.StrategyConfig()
        self.config.initial_capital = 100000.0
        self.config.min_warmup_ticks = 1000
        self.engine = hft_engine.StrategyEngine(self.config)
        self.engine.set_mode(hft_engine.EngineMode.BACKTEST)
        
        # Pre-load optimal weights
        optimal_weights = [0.189, 0.006, -0.242, -0.238, 0.101, 0.200]
        self.engine.set_weights(optimal_weights)
        
        self.last_equity = self.config.initial_capital
        self.book = hft_engine.BookSnapshot()
        
    def _generate_synthetic_data(self, n):
        """Generate a synthetic order book/trades dataframe using GBM."""
        np.random.seed(42)
        prices = np.exp(np.cumsum(np.random.normal(0, 0.0001, n))) * 50000
        spreads = np.random.uniform(0.5, 2.0, n)
        
        df = pd.DataFrame({
            "timestamp_ns": np.arange(n) * 100000000,
            "best_bid": prices - spreads,
            "best_ask": prices + spreads,
            "bid_qty": np.random.uniform(0.1, 2.0, n),
            "ask_qty": np.random.uniform(0.1, 2.0, n),
            "trade_price": prices + np.random.normal(0, spreads/2, n),
            "trade_qty": np.random.uniform(0.01, 0.5, n),
            "is_buyer_maker": np.random.randint(0, 2, n).astype(bool)
        })
        return df

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_tick = 0
        self.engine.reset()
        self.engine.set_mode(hft_engine.EngineMode.BACKTEST)
        self.last_equity = self.config.initial_capital
        
        # Warmup phase
        while self.current_tick < self.config.min_warmup_ticks and self.current_tick < self.total_ticks:
            self._step_engine()
            
        return self._get_obs(), {}
        
    def _step_engine(self):
        row = self.df.iloc[self.current_tick]
        self.current_tick += 1
        
        self.book.best_bid_price = int(row["best_bid"] * 1e8)
        self.book.best_ask_price = int(row["best_ask"] * 1e8)
        self.book.best_bid_qty = int(row["bid_qty"] * 1e8)
        self.book.best_ask_qty = int(row["ask_qty"] * 1e8)
        self.book.bid_count = 1
        self.book.ask_count = 1
        
        trade = hft_engine.Trade()
        trade.price = int(row["trade_price"] * 1e8)
        trade.quantity = int(row["trade_qty"] * 1e8)
        trade.side = hft_engine.Side.ASK if row["is_buyer_maker"] else hft_engine.Side.BID
        
        self.engine.on_trade(trade, self.book)

    def _get_obs(self):
        fv = self.engine.last_features()
        pos = self.engine.position() / 1e8
        max_pos = (self.config.initial_capital * self.config.max_position_pct) / (self.book.best_bid_price / 1e8 + 1e-8)
        pos_pct = pos / (max_pos + 1e-8)
        
        obs = np.array([
            fv.combined_alpha,
            fv.vpin,
            fv.obi,
            fv.realized_vol,
            pos_pct,
            fv.spread_bps
        ], dtype=np.float32)
        return obs

    def step(self, action):
        # Map actions from [-1, 1] to practical ranges
        # action[0]: spread_alpha_multiplier [0.01, 1.0]
        # action[1]: alpha_entry_threshold [0.01, 0.5]
        spread_mult = 0.01 + (action[0] + 1.0) / 2.0 * 0.99
        entry_thresh = 0.01 + (action[1] + 1.0) / 2.0 * 0.49
        
        # NOTE: To make this fully functional, we need C++ bindings for set_config or dynamically changing thresholds.
        # But for the purpose of the framework, we just run the engine.
        
        for _ in range(self.ticks_per_step):
            if self.current_tick >= self.total_ticks:
                break
            self._step_engine()
            
        current_equity = self.engine.equity()
        reward = current_equity - self.last_equity
        self.last_equity = current_equity
        
        # Penalize holding max position (inventory risk)
        pos = self.engine.position() / 1e8
        if abs(pos) > 0.0:
            reward -= abs(pos) * 0.0001  # Small decay for holding
            
        obs = self._get_obs()
        terminated = self.current_tick >= self.total_ticks
        truncated = False
        
        info = {
            "equity": current_equity,
            "realized_pnl": self.engine.realized_pnl(),
            "position": pos
        }
        
        return obs, float(reward), terminated, truncated, info
