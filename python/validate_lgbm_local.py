"""
Local LGBM Holdout Validation
Uses the holdout_l2.parquet already downloaded to:
  python/ml_pipeline/data/holdout_l2.parquet
"""
import os, json, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
warnings.filterwarnings("ignore")

BASE_DIR   = r"C:\Users\Varun\Downloads\HFT"
MODEL_PATH = os.path.join(BASE_DIR, "python", "models", "lgbm_signal_model.txt")
DATA_PATH  = os.path.join(BASE_DIR, "python", "ml_pipeline", "data", "holdout_l2.parquet")
OUT_DIR    = os.path.join(BASE_DIR, "python", "models")

FEATURES = [
    "tfi_ema_50", "tfi_ema_200",
    "volatility_ema_100", "volatility_ema_500",
    "book_imbalance", "spread_bps", "microprice", "spread_ema",
]
TARGET        = "label_fwd_ret_bps"
CONF_BAND     = 0.5   # bps — minimum |pred| to trade
TAKER_FEE_BPS = 0.1
COOLDOWN_MS   = 200

print("=" * 60)
print("HFT LGBM v4 — Local Holdout Validation (Aug 2026 OOS)")
print("=" * 60)

assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
assert os.path.exists(DATA_PATH),  f"Holdout not found: {DATA_PATH}"

print(f"\nLoading model from: {MODEL_PATH}")
model = lgb.Booster(model_file=MODEL_PATH)
print(f"  Trees: {model.num_trees()}, Features: {model.num_feature()}")
print(f"  Model features: {model.feature_name()}")

print(f"\nLoading holdout data from: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"  Rows: {len(df):,}, Columns: {list(df.columns)}")

# Validate feature alignment
missing = [f for f in FEATURES if f not in df.columns]
if missing:
    print(f"\n[ERROR] Missing features in holdout: {missing}")
    print("Available columns:", list(df.columns))
    raise SystemExit(1)

df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
X   = df[FEATURES].astype(np.float32)
y   = df[TARGET].values.astype(np.float32)
print(f"  Clean rows: {len(df):,}")

print("\nRunning inference...")
preds = model.predict(X.values)

# Backtest with cooldown
signal_mask = np.abs(preds) > CONF_BAND
times       = df["transact_time"].values if "transact_time" in df.columns else np.arange(len(df)) * 100
directions  = np.sign(preds)

trades = []
last_t = -np.inf
for i in np.where(signal_mask)[0]:
    t = times[i]
    if t - last_t < COOLDOWN_MS:
        continue
    pnl = float(directions[i] * y[i] - TAKER_FEE_BPS)
    trades.append({"time": t, "pred_bps": float(preds[i]),
                   "fwd_bps": float(y[i]), "direction": int(directions[i]), "pnl_bps": pnl})
    last_t = t

if not trades:
    print("[ERROR] No trades fired — check CONF_BAND or data.")
    raise SystemExit(1)

trade_df = pd.DataFrame(trades)
pnl_vec  = trade_df["pnl_bps"].values
cum_pnl  = trade_df["pnl_bps"].cumsum()
max_dd   = float((cum_pnl - cum_pnl.cummax()).min())
sharpe   = float((pnl_vec.mean() / (pnl_vec.std() + 1e-9)) *
                 np.sqrt(252 * 6.5 * 3600 / COOLDOWN_MS * 1000))
win_rate = float((pnl_vec > 0).mean() * 100)
dir_acc  = float((np.sign(preds[signal_mask]) == np.sign(y[signal_mask])).mean() * 100)
pf       = float(pnl_vec[pnl_vec > 0].sum() / (abs(pnl_vec[pnl_vec < 0].sum()) + 1e-9))

print(f"\n{'='*60}")
print(f"RESULTS — LGBM v4 L2  (OOS Holdout)")
print(f"{'='*60}")
print(f"  Rows total          : {len(df):,}")
print(f"  Signals fired       : {signal_mask.sum():,}  ({100*signal_mask.sum()/len(df):.2f}%)")
print(f"  Trades (cooldown)   : {len(trade_df):,}")
print(f"  Win rate            : {win_rate:.2f}%")
print(f"  Directional acc     : {dir_acc:.2f}%")
print(f"  Total PnL           : {pnl_vec.sum():,.2f} bps")
print(f"  Avg PnL / trade     : {pnl_vec.mean():.4f} bps")
print(f"  Max drawdown        : {max_dd:,.2f} bps")
print(f"  Sharpe (ann.)       : {sharpe:.3f}")
print(f"  Profit factor       : {pf:.3f}")
print(f"{'='*60}")

# Save metrics
metrics = {
    "model": "lgbm_v4_l2",
    "dataset": "holdout_l2_local",
    "total_trades": len(trade_df),
    "total_pnl_bps": float(pnl_vec.sum()),
    "win_rate_pct": win_rate,
    "dir_acc_pct": dir_acc,
    "avg_pnl_per_trade_bps": float(pnl_vec.mean()),
    "max_dd_bps": max_dd,
    "sharpe": sharpe,
    "profit_factor": pf,
}
out_path = os.path.join(OUT_DIR, "holdout_validation_metrics.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to: {out_path}")

trade_df.to_csv(os.path.join(OUT_DIR, "holdout_trades.csv"), index=False)
print("Trade log saved.")
