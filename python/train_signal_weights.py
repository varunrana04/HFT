import os, struct, time, math, requests
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

SYMBOL       = "BTCUSDT"
INTERVAL     = "1m"
KLINE_LIMIT  = 1500
N_REQUESTS   = 4
FORWARD_BARS = 5
BASE_URL     = "https://fapi.binance.com"
WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "signal_weights.bin")
# 7 weight names matching the engine alpha formula exactly
WEIGHT_NAMES = ["w_obi","w_vpin","w_vol","w_spread","w_ofi","w_microprice","w_bias"]

def fetch_klines(symbol, interval, limit, end_time=None):
    p = {"symbol": symbol, "interval": interval, "limit": limit}
    if end_time: p["endTime"] = end_time
    r = requests.get(f"{BASE_URL}/fapi/v1/klines", params=p, timeout=15)
    r.raise_for_status()
    return r.json()

print("Fetching klines...")
all_klines, end_time = [], None
for i in range(N_REQUESTS):
    batch = fetch_klines(SYMBOL, INTERVAL, KLINE_LIMIT, end_time)
    if not batch: break
    all_klines = batch + all_klines
    end_time = int(batch[0][0]) - 1
    print(f"  {len(all_klines)} bars")
    time.sleep(0.3)

K       = np.array(all_klines, dtype=object)
opens   = K[:,1].astype(float)
highs   = K[:,2].astype(float)
lows    = K[:,3].astype(float)
prices  = K[:,4].astype(float)   # close
volumes = K[:,5].astype(float)   # total vol
tvol    = K[:,9].astype(float)   # taker buy base vol

taker_sell = volumes - tvol
N = len(prices)

# Feature 1: OBI proxy  (±1)
obi = (tvol - taker_sell) / (volumes + 1e-8)

# Feature 2: VPIN proxy centered at 0.5  → engine uses (vpin - 0.5)
vpin_raw = np.zeros(N)
for i in range(N):
    w = np.abs(obi[max(0,i-20):i+1])
    vpin_raw[i] = float(np.mean(w)) if len(w) else 0.5
vpin_centered = vpin_raw - 0.5

# Feature 3: realized vol (20-bar log-return std × sqrt(100))
log_rets = np.zeros(N)
log_rets[1:] = np.log(prices[1:] / (prices[:-1] + 1e-8))
vol = np.zeros(N)
for i in range(N):
    w = log_rets[max(0,i-20):i+1]
    vol[i] = float(np.std(w)*np.sqrt(100)) if len(w)>1 else 0.01

# Feature 4: spread_bps proxy centered at 2  → engine uses (spread - 2.0)
spread = (highs - lows) / (prices + 1e-8) * 10000.0
spread_centered = spread - 2.0

# Feature 5: OFI (sign-adjusted volume delta)
ofi = np.zeros(N)
for i in range(1, N):
    d = volumes[i] - volumes[i-1]
    sign = 1.0 if prices[i] >= prices[i-1] else -1.0
    ofi[i] = sign * d / (volumes[i] + 1e-8)

# Feature 6: micro_ret * 1e4 (return in bps)  → engine multiplies by 1e4
micro_ret_bps = (prices - opens) / (opens + 1e-8) * 1e4

# Feature 7: bias = 1 always
bias = np.ones(N)

X_raw = np.column_stack([obi, vpin_centered, vol, spread_centered, ofi, micro_ret_bps, bias])

# Label: 5-bar forward log return
fwd = np.zeros(N)
fwd[:-FORWARD_BARS] = np.log(prices[FORWARD_BARS:] / (prices[:-FORWARD_BARS] + 1e-8))

# Trim warmup (200 bars) and trailing
X = X_raw[200:-FORWARD_BARS]
y = fwd[200:-FORWARD_BARS]

print(f"Training on {len(X)} samples, 7 features...")

# Scale features (except bias col 6 which is constant), fit Ridge
# We use Ridge with fit_intercept=False (bias is already in column 6)
scaler = StandardScaler(with_std=True)
X_scaled = scaler.fit_transform(X)
ridge = Ridge(alpha=0.5, fit_intercept=False)
ridge.fit(X_scaled, y)

# Absorb scaler: w_raw[i] = coef[i] / scale[i]
# This gives weights to apply directly to unscaled features
coef = ridge.coef_
scale = scaler.scale_
w_absorbed = coef / (scale + 1e-12)

# Clip to prevent instability
w_absorbed = np.clip(w_absorbed, -3.0, 3.0)

r2 = ridge.score(X_scaled, y)
print(f"\nR2: {r2:.4f}  (expected ~0.01 for 1-min data)")
print("\nFinal weights:")
for name, wi in zip(WEIGHT_NAMES, w_absorbed):
    bar = "+" * int(max(wi,0)*30) if wi>0 else "-" * int(abs(min(wi,0))*30)
    print(f"  {name:14s}: {wi:+.4f}  {bar}")

# Save as 7 × float64 little-endian
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
with open(WEIGHTS_PATH, "wb") as f:
    f.write(struct.pack("<7d", *w_absorbed))
print(f"\nSaved: {WEIGHTS_PATH} ({os.path.getsize(WEIGHTS_PATH)} bytes)")
