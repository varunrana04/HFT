import os, struct, time, requests, pickle
import numpy as np
from hmmlearn import hmm
from sklearn.preprocessing import StandardScaler

SYMBOL   = "BTCUSDT"
BASE_URL = "https://fapi.binance.com"
PKL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models", "hmm_regime.pkl")

def fetch_klines(limit=1500, end_time=None):
    p = {"symbol": SYMBOL, "interval": "5m", "limit": limit}
    if end_time: p["endTime"] = end_time
    r = requests.get(f"{BASE_URL}/fapi/v1/klines", params=p, timeout=15)
    r.raise_for_status()
    return r.json()

print("Fetching 5m klines for HMM training...")
all_klines, end_time = [], None
for i in range(6):  # 6 x 1500 = 9000 bars = ~31 days of 5m data
    batch = fetch_klines(1500, end_time)
    if not batch: break
    all_klines = batch + all_klines
    end_time = int(batch[0][0]) - 1
    print(f"  {len(all_klines)} bars")
    time.sleep(0.3)

K = np.array(all_klines, dtype=object)
highs   = K[:,2].astype(float)
lows    = K[:,3].astype(float)
closes  = K[:,4].astype(float)
tvol    = K[:,9].astype(float)
volumes = K[:,5].astype(float)
N = len(closes)

# Feature 1: realized volatility (20-bar log-return std x sqrt(100))
log_rets = np.zeros(N)
log_rets[1:] = np.log(closes[1:] / (closes[:-1] + 1e-8))
vol = np.array([np.std(log_rets[max(0,i-20):i+1])*np.sqrt(100) if i>0 else 0.01 for i in range(N)])

# Feature 2: spread_bps proxy (high-low range)
spread = (highs - lows) / (closes + 1e-8) * 10000.0

X_raw = np.column_stack([vol, spread])
X_raw = X_raw[50:]  # skip warmup

scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

print(f"Training HMM on {len(X)} samples (4 states)...")
model = hmm.GaussianHMM(n_components=4, covariance_type="full", n_iter=200, random_state=42)
model.fit(X)

states = model.predict(X)
for s in range(4):
    mask = states == s
    if mask.sum() > 0:
        sv = vol[50:][mask].mean()
        ss = spread[50:][mask].mean()
        print(f"  State {s}: vol={sv:.4f} spread={ss:.1f}bps  n={mask.sum()}")

# Order states by vol: 0=low-vol, 1=mid-vol, 2=high-vol, 3=crisis
state_vols = [vol[50:][states == s].mean() if (states==s).sum()>0 else 0 for s in range(4)]
order = np.argsort(state_vols)
mapping = {old: new for new, old in enumerate(order)}
print(f"State remapping (vol-sorted): {mapping}")

# Save
os.makedirs(os.path.dirname(PKL_PATH), exist_ok=True)
payload = {"model": model, "mean": scaler.mean_, "std": scaler.scale_, "state_mapping": mapping}
with open(PKL_PATH, "wb") as f:
    pickle.dump(payload, f)
print(f"Saved: {PKL_PATH}  ({os.path.getsize(PKL_PATH)} bytes)")
