"""
Pure-Python LGBM Holdout Validation
Uses our custom C++ flat-array LGBM inference — no lightgbm Python package needed.
Reads the .txt model file directly (same format the C++ engine uses).
"""
import os, json, warnings, re, time
import numpy as np
import pandas as pd
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
CONF_BAND     = 0.5
TAKER_FEE_BPS = 0.1
COOLDOWN_MS   = 200

# ── Pure Python LGBM Tree Evaluator ─────────────────────────────────────────
class LGBMPureTree:
    """Parse and evaluate a LightGBM .txt model without the lgbm DLL."""

    def __init__(self, model_path):
        self.trees = []
        self.num_class = 1
        self._parse(model_path)
        print(f"  Parsed {len(self.trees)} trees, {self._n_features} features")
        print(f"  Features: {self._feature_names}")

    def _parse(self, path):
        with open(path, "r") as f:
            content = f.read()

        # Extract feature names
        m = re.search(r"feature_names=(.+)", content)
        self._feature_names = m.group(1).strip().split() if m else []
        self._n_features = len(self._feature_names)

        # Parse each tree block
        tree_blocks = re.split(r"\nTree=\d+\n", content)
        tree_blocks = tree_blocks[1:]  # skip header

        for block in tree_blocks:
            tree = self._parse_tree(block)
            if tree is not None:
                self.trees.append(tree)

    def _parse_ints(self, block, key):
        m = re.search(rf"^{key}=(.+)$", block, re.MULTILINE)
        return list(map(int, m.group(1).split())) if m else []

    def _parse_floats(self, block, key):
        m = re.search(rf"^{key}=(.+)$", block, re.MULTILINE)
        return list(map(float, m.group(1).split())) if m else []

    def _parse_tree(self, block):
        try:
            split_feature  = self._parse_ints(block, "split_feature")
            left_child     = self._parse_ints(block, "left_child")
            right_child    = self._parse_ints(block, "right_child")
            thresholds     = self._parse_floats(block, "threshold")
            leaf_values    = self._parse_floats(block, "leaf_value")
            if not split_feature:
                return None
            return {
                "split_feature": split_feature,
                "left_child":    left_child,
                "right_child":   right_child,
                "threshold":     thresholds,
                "leaf_value":    leaf_values,
            }
        except Exception:
            return None

    def _predict_one_tree(self, tree, x):
        node = 0
        sf   = tree["split_feature"]
        thr  = tree["threshold"]
        lc   = tree["left_child"]
        rc   = tree["right_child"]
        lv   = tree["leaf_value"]
        while True:
            feat_val = x[sf[node]]
            if feat_val <= thr[node]:
                nxt = lc[node]
            else:
                nxt = rc[node]
            if nxt < 0:
                return lv[-(nxt + 1)]
            node = nxt

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict on (N, F) array. Returns (N,) predictions."""
        N = len(X)
        preds = np.zeros(N, dtype=np.float64)
        t0 = time.time()
        for i, row in enumerate(X):
            s = 0.0
            for tree in self.trees:
                s += self._predict_one_tree(tree, row)
            preds[i] = s
            if i % 50000 == 0 and i > 0:
                elapsed = time.time() - t0
                rate = i / elapsed
                remaining = (N - i) / rate
                print(f"  Progress: {i:,}/{N:,} ({100*i/N:.1f}%) — ETA {remaining:.0f}s")
        return preds


# ── Main ──────────────────────────────────────────────────────────────────────
print("=" * 60)
print("HFT LGBM v4 — Pure-Python Holdout Validation (Aug 2026 OOS)")
print("(No lgbm DLL required — pure Python tree eval)")
print("=" * 60)

assert os.path.exists(MODEL_PATH), f"Model not found: {MODEL_PATH}"
assert os.path.exists(DATA_PATH),  f"Holdout not found: {DATA_PATH}"

print(f"\nParsing model: {MODEL_PATH}")
model = LGBMPureTree(MODEL_PATH)

print(f"\nLoading holdout data: {DATA_PATH}")
df = pd.read_parquet(DATA_PATH)
print(f"  Rows: {len(df):,}, Columns: {list(df.columns)}")

missing = [f for f in FEATURES if f not in df.columns]
if missing:
    raise ValueError(f"Missing features: {missing}\nAvailable: {list(df.columns)}")

df = df.dropna(subset=FEATURES + [TARGET]).reset_index(drop=True)
X = df[FEATURES].astype(np.float64).values
y = df[TARGET].values.astype(np.float64)
print(f"  Clean rows: {len(df):,}")

# Subsample for speed — pure Python is ~100x slower than lgbm DLL
# Use every 10th row; results are still statistically valid
STEP = 10
X_s = X[::STEP]
y_s = y[::STEP]
df_s = df.iloc[::STEP].reset_index(drop=True)
print(f"  Subsampled (1/{STEP}): {len(X_s):,} rows")

print("\nRunning inference (pure Python — may take a few minutes)...")
t0 = time.time()
preds = model.predict(X_s)
print(f"  Done in {time.time()-t0:.1f}s")

# Backtest
signal_mask = np.abs(preds) > CONF_BAND
times       = df_s["transact_time"].values if "transact_time" in df_s.columns else np.arange(len(df_s)) * STEP * 100
directions  = np.sign(preds)

trades = []
last_t = -np.inf
for i in np.where(signal_mask)[0]:
    t = times[i]
    if t - last_t < COOLDOWN_MS:
        continue
    pnl = float(directions[i] * y_s[i] - TAKER_FEE_BPS)
    trades.append({"time": t, "pred_bps": float(preds[i]),
                   "fwd_bps": float(y_s[i]), "direction": int(directions[i]), "pnl_bps": pnl})
    last_t = t

if not trades:
    print("[ERROR] No trades fired — confidence band too wide or bad model")
    raise SystemExit(1)

trade_df = pd.DataFrame(trades)
pnl_vec  = trade_df["pnl_bps"].values
cum_pnl  = trade_df["pnl_bps"].cumsum()
max_dd   = float((cum_pnl - cum_pnl.cummax()).min())
sharpe   = float((pnl_vec.mean() / (pnl_vec.std() + 1e-9)) *
                 np.sqrt(252 * 6.5 * 3600 / COOLDOWN_MS * 1000))
win_rate = float((pnl_vec > 0).mean() * 100)
dir_acc  = float((np.sign(preds[signal_mask]) == np.sign(y_s[signal_mask])).mean() * 100)
pf       = float(pnl_vec[pnl_vec > 0].sum() / (abs(pnl_vec[pnl_vec < 0].sum()) + 1e-9))

print(f"\n{'='*60}")
print(f"RESULTS — LGBM v4 L2  (OOS Holdout, 1/{STEP} sample)")
print(f"{'='*60}")
print(f"  Rows sampled        : {len(X_s):,}")
print(f"  Signals fired       : {signal_mask.sum():,}  ({100*signal_mask.sum()/len(X_s):.2f}%)")
print(f"  Trades (cooldown)   : {len(trade_df):,}")
print(f"  Win rate            : {win_rate:.2f}%")
print(f"  Directional acc     : {dir_acc:.2f}%")
print(f"  Total PnL           : {pnl_vec.sum():,.2f} bps")
print(f"  Avg PnL / trade     : {pnl_vec.mean():.4f} bps")
print(f"  Max drawdown        : {max_dd:,.2f} bps")
print(f"  Sharpe (ann.)       : {sharpe:.3f}")
print(f"  Profit factor       : {pf:.3f}")
print(f"{'='*60}")

metrics = {
    "model": "lgbm_v4_l2", "dataset": "holdout_l2_pure_python",
    "subsample_step": STEP, "total_trades": len(trade_df),
    "total_pnl_bps": float(pnl_vec.sum()), "win_rate_pct": win_rate,
    "dir_acc_pct": dir_acc, "avg_pnl_per_trade_bps": float(pnl_vec.mean()),
    "max_dd_bps": max_dd, "sharpe": sharpe, "profit_factor": pf,
}
out_path = os.path.join(OUT_DIR, "holdout_validation_metrics.json")
with open(out_path, "w") as f:
    json.dump(metrics, f, indent=2)
print(f"\nMetrics saved to: {out_path}")
trade_df.to_csv(os.path.join(OUT_DIR, "holdout_trades.csv"), index=False)
print("Trade log saved. Done.")
