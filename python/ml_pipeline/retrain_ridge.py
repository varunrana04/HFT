"""
retrain_ridge.py — Train Ridge regression on features_dump_clean.csv
and export signal_weights.bin with directional (signed) coefficients.
Bypasses LightGBM entirely to avoid Windows OpenMP deadlock on large data.
"""
import os, sys, struct, time, shutil
import numpy as np
os.environ['OMP_NUM_THREADS'] = '1'
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

ROOT = os.path.join(os.path.dirname(__file__), '..')
DATA = os.path.join(ROOT, 'data', 'features_dump_clean.csv')
OUT  = os.path.join(ROOT, 'models', 'signal_weights.bin')
OUT2 = os.path.join(ROOT, 'models', 'production', 'signal_weights.bin')

SIGNALS = ['microprice', 'ofi', 'vpin', 'spread_bps', 'realized_vol', 'stat_arb_zscore']
TARGET  = 'forward_return_100'

print('=' * 60)
print('  Ridge Signal Weight Training')
print('=' * 60)

t0 = time.time()
cols = SIGNALS + [TARGET, 'is_warmed_up']
df = pd.read_csv(DATA, usecols=cols, nrows=1_000_000)
df = df[df['is_warmed_up'] == 1].dropna()
print(f'  Rows loaded   : {len(df):,}  ({time.time()-t0:.1f}s)')

X = df[SIGNALS].values.astype(np.float64)
y = df[TARGET].values.astype(np.float64)

split = int(len(X) * 0.70)
X_tr, y_tr = X[:split], y[:split]
X_te, y_te = X[split:], y[split:]

scaler = StandardScaler()
X_tr_s = scaler.fit_transform(X_tr)
X_te_s = scaler.transform(X_te)

t1 = time.time()
ridge = Ridge(alpha=1.0, fit_intercept=True)
ridge.fit(X_tr_s, y_tr)
print(f'  Ridge fit time: {time.time()-t1:.1f}s  (n={len(X_tr):,})')

coef = ridge.coef_.astype(np.float64)
bias = float(ridge.intercept_)

print('\n  Directional coefficients (raw):')
for n, c in zip(SIGNALS, coef):
    print(f'    {n:>20}: {c:+.6f}')
print(f'    {"bias":>20}: {bias:+.6f}')

total_abs = np.sum(np.abs(coef))
w_norm = coef / total_abs
norm_bias = bias / total_abs

print('\n  L1-normalised weights (→ signal_weights.bin):')
for n, w in zip(SIGNALS, w_norm):
    arrow = 'BULLISH ↑' if w > 0.01 else ('BEARISH ↓' if w < -0.01 else 'neutral')
    print(f'    {n:>20}: {w:+.4f}   {arrow}')

y_pred = ridge.predict(X_te_s)
ss_res = np.sum((y_te - y_pred) ** 2)
ss_tot = np.sum((y_te - y_te.mean()) ** 2)
r2 = 1.0 - ss_res / max(ss_tot, 1e-30)
pearson_r = float(np.corrcoef(y_pred, y_te)[0, 1])
positions  = np.sign(y_pred)
hit_rate   = float(np.mean(np.sign(y_pred) == np.sign(y_te)))

print(f'\n  OOS R^2        : {r2:.6f}')
print(f'  OOS Pearson r  : {pearson_r:.6f}')
print(f'  Directional hit: {hit_rate*100:.2f}%')

strat_rets = positions * y_te
active     = strat_rets[positions != 0]
sharpe_oos = float(np.mean(active) / (np.std(active) + 1e-15) * np.sqrt(252*24*3600))
print(f'  OOS Sharpe     : {sharpe_oos:.4f}')

os.makedirs(os.path.dirname(OUT), exist_ok=True)
with open(OUT, 'wb') as f:
    w_norm.tofile(f)
    np.array([norm_bias], dtype=np.float64).tofile(f)

os.makedirs(os.path.dirname(OUT2), exist_ok=True)
shutil.copy(OUT, OUT2)

print(f'\n  [+] {OUT}  ({os.path.getsize(OUT)} bytes)')
print(f'  [+] {OUT2}')
print('=' * 60)
