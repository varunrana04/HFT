"""
alpha_surface_3d.py — 3D Alpha Edge Surface: Alpha(Spread, Volatility)
Uses real feature data to show where the strategy's edge originates.
"""
import os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

ROOT    = os.path.join(os.path.dirname(__file__), '..')
OUT_DIR = os.path.join(ROOT, 'reports', 'charts')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load features data ───────────────────────────────────────
# Use tardis_features.csv (smaller) for fast surface computation
FEAT = os.path.join(ROOT, 'data', 'tardis_features.csv')
CLEAN = os.path.join(ROOT, 'data', 'features_dump_clean.csv')

COLS = ['spread_bps', 'realized_vol', 'combined_alpha',
        'vpin', 'ofi', 'microprice', 'stat_arb_zscore', 'forward_return_100', 'is_warmed_up']

for path in [FEAT, CLEAN]:
    try:
        available = pd.read_csv(path, nrows=1).columns.tolist()
        use_cols  = [c for c in COLS if c in available]
        df = pd.read_csv(path, usecols=use_cols, nrows=200_000)
        if 'is_warmed_up' in df.columns:
            df = df[df['is_warmed_up'] == 1]
        df = df.dropna()
        if len(df) > 1000:
            print(f"Loaded {len(df):,} rows from {os.path.basename(path)}")
            break
    except Exception as e:
        print(f"Could not load {path}: {e}")

# ── Compute directional weights (from trained signal_weights.bin) ─────
SIGNALS = ['microprice', 'ofi', 'vpin', 'spread_bps', 'realized_vol', 'stat_arb_zscore']
BIN_PATH = os.path.join(ROOT, 'models', 'signal_weights.bin')
if os.path.exists(BIN_PATH):
    import struct
    with open(BIN_PATH, 'rb') as f:
        weights = list(struct.unpack('<6d', f.read(48)))
    print(f"Loaded weights: {[round(w,3) for w in weights]}")
else:
    weights = [0.189, 0.006, -0.242, -0.238, 0.101, 0.200]
    print(f"Fallback weights: {weights}")

# Compute alpha score for each row
for i, sig in enumerate(SIGNALS):
    if sig in df.columns:
        df[sig] = pd.to_numeric(df[sig], errors='coerce')
df = df.dropna()

alpha_col = np.zeros(len(df))
for i, sig in enumerate(SIGNALS):
    if sig in df.columns:
        alpha_col += weights[i] * df[sig].values
alpha_col = np.clip(alpha_col, -1.0, 1.0)
df['computed_alpha'] = alpha_col

# ── Build 2D grid: spread_bps × realized_vol → mean |alpha| ──
spread_col  = df['spread_bps'].values
vol_col     = df['realized_vol'].values
alpha_abs   = np.abs(df['computed_alpha'].values)

# Use forward_return if available for PnL surface
use_pnl = 'forward_return_100' in df.columns
if use_pnl:
    fwd = df['forward_return_100'].values
    edge_signal = np.sign(alpha_col) * fwd  # strategy PnL proxy
else:
    edge_signal = alpha_abs

# Grid boundaries (clip outliers)
sp_lo, sp_hi = np.percentile(spread_col, 2), np.percentile(spread_col, 98)
vl_lo, vl_hi = np.percentile(vol_col, 2),    np.percentile(vol_col, 98)
GRID = 30

sp_bins = np.linspace(sp_lo, sp_hi, GRID + 1)
vl_bins = np.linspace(vl_lo, vl_hi, GRID + 1)

Z = np.full((GRID, GRID), np.nan)
for i in range(GRID):
    for j in range(GRID):
        mask = (
            (spread_col >= sp_bins[i])  & (spread_col < sp_bins[i+1]) &
            (vol_col    >= vl_bins[j])  & (vol_col    < vl_bins[j+1])
        )
        if mask.sum() >= 5:
            Z[i, j] = np.mean(edge_signal[mask]) * 10000.0  # in bps

# Fill nans with neighbour interpolation for clean surface
from scipy.ndimage import generic_filter
def fill_nan(Z):
    def fn(v):
        v2 = v[~np.isnan(v)]
        return np.mean(v2) if len(v2) > 0 else 0.0
    return generic_filter(Z, fn, size=3)

Z_filled = fill_nan(Z)
SP, VL = np.meshgrid(
    0.5*(sp_bins[:-1]+sp_bins[1:]),
    0.5*(vl_bins[:-1]+vl_bins[1:]),
    indexing='ij'
)

# ── Plot ─────────────────────────────────────────────────────
fig = plt.figure(figsize=(14, 9))
fig.patch.set_facecolor('#0d1117')
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor('#161b22')

surf = ax.plot_surface(SP, VL, Z_filled,
    cmap=cm.RdYlGn, linewidth=0, antialiased=True, alpha=0.92)

cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10, pad=0.08)
cbar.set_label('Edge Signal (bps)', color='#c9d1d9', fontsize=9)
cbar.ax.yaxis.set_tick_params(color='#8b949e')
plt.setp(cbar.ax.yaxis.get_ticklabels(), color='#c9d1d9')

ax.set_xlabel('Spread (bps)', color='#c9d1d9', fontsize=10, labelpad=8)
ax.set_ylabel('Realized Vol', color='#c9d1d9', fontsize=10, labelpad=8)
ax.set_zlabel('Alpha Edge (bps)', color='#c9d1d9', fontsize=10, labelpad=8)
ax.tick_params(colors='#8b949e', labelsize=8)

label_suffix = "(× Forward Return)" if use_pnl else "(× |Alpha|)"
ax.set_title(
    f'Alpha Edge Surface — Spread × Volatility {label_suffix}\n'
    f'Green = positive edge region  |  Red = adverse / toxic region\n'
    f'Ridge weights: {[round(w,3) for w in weights]}',
    color='#f0f6fc', fontsize=11, fontweight='bold', pad=12)

ax.view_init(elev=28, azim=-55)

OUT = os.path.join(OUT_DIR, 'alpha_surface_3d.png')
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f"[+] Saved: {OUT}")

# Print key insight: where is edge concentrated?
max_idx = np.unravel_index(np.nanargmax(Z_filled), Z_filled.shape)
min_idx = np.unravel_index(np.nanargmin(Z_filled), Z_filled.shape)
print(f"\nPeak edge at:  spread={SP[max_idx]:.2f} bps, vol={VL[max_idx]:.5f}  → {Z_filled[max_idx]:+.3f} bps")
print(f"Worst edge at: spread={SP[min_idx]:.2f} bps, vol={VL[min_idx]:.5f}  → {Z_filled[min_idx]:+.3f} bps")
