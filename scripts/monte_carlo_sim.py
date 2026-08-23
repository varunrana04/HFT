"""
monte_carlo_sim.py — HFT Engine Monte Carlo Validation
Block-bootstraps daily PnL from live paper trade session.
10,000 simulations of 1-month (21 days) and 1-year (252 days) equity paths.
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

ROOT    = os.path.join(os.path.dirname(__file__), '..')
OUT_DIR = os.path.join(ROOT, 'reports', 'charts')
os.makedirs(OUT_DIR, exist_ok=True)

# ── Load all live sessions ──────────────────────────────────────
import glob
files = sorted(glob.glob(os.path.join(ROOT, 'paper_trades_*.csv')))
pnls_raw_list = []
equity0 = 10_000_000.0 # Default starting equity
found_data = False

for CSV in files:
    try:
        df = pd.read_csv(CSV)
        df.columns = [c.strip().lower() for c in df.columns]
        if 'equity' not in df.columns or len(df) < 50: continue
        df['equity'] = pd.to_numeric(df['equity'], errors='coerce')
        df = df.dropna(subset=['equity']).reset_index(drop=True)
        if not found_data:
            equity0 = float(df['equity'].iloc[0])
            found_data = True
        
        # calculate per-fill pnl and scale it to 1 BTC size
        pnl = np.diff(df['equity'].values) * 1000.0
        pnls_raw_list.extend(pnl.tolist())
    except Exception as e:
        pass

pnls_raw = np.array(pnls_raw_list, dtype=float)
if len(pnls_raw) == 0:
    print("ERROR: No valid paper trade fills found.")
    sys.exit(1)


# Aggregate fills into synthetic days (session: ~143 fills/hr × 8.8 hrs)
FILLS_PER_DAY = 143
n_days  = len(pnls_raw) // FILLS_PER_DAY
daily   = np.array([pnls_raw[i*FILLS_PER_DAY:(i+1)*FILLS_PER_DAY].sum()
                    for i in range(n_days)])

ann_sh = np.mean(daily) / (np.std(daily) + 1e-15) * np.sqrt(252)
print(f"Synthetic trading days  : {n_days}")
print(f"Mean daily PnL          : ${np.mean(daily):+,.2f}  (1 BTC scale)")
print(f"Std  daily PnL          : ${np.std(daily):,.2f}")
print(f"Annualised daily Sharpe : {ann_sh:+.3f}")

N_SIM = 10_000
N_1M  = 21
N_1Y  = 252
np.random.seed(42)

def block_bootstrap(daily_pnl, n_target, n_sim, block=3):
    n   = len(daily_pnl)
    blk = min(block, max(1, n - 1))
    nbl = n_target // blk + 2
    s   = np.random.randint(0, max(1, n - blk + 1), (n_sim, nbl))
    idx = np.clip(s[:, :, None] + np.arange(blk)[None, None, :], 0, n - 1)
    draws = daily_pnl[idx].reshape(n_sim, -1)[:, :n_target]
    return equity0 + np.concatenate(
        [np.zeros((n_sim, 1)), np.cumsum(draws, axis=1)], axis=1)

print("Running 1-month MC ...")
p1m = block_bootstrap(daily, N_1M, N_SIM)
print("Running 1-year  MC ...")
p1y = block_bootstrap(daily, N_1Y, N_SIM)

def pcts(p):
    return np.percentile(p, [1, 5, 25, 50, 75, 95, 99], axis=0)

def ror(paths, frac):
    thr = paths[:, 0:1] * (1.0 - frac)
    return float(np.mean(np.any(paths < thr, axis=1)))

def mdd(paths):
    pk = np.maximum.accumulate(paths, axis=1)
    return np.max((pk - paths) / np.maximum(pk, 1.0), axis=1)

q1m, q1y = pcts(p1m), pcts(p1y)
f1m, f1y = p1m[:, -1], p1y[:, -1]
d1m, d1y = mdd(p1m),   mdd(p1y)

print("\n=== Monte Carlo Risk Report ===")
for label, finals, dds, r5, r10 in [
    ("1-Month",  f1m, d1m, ror(p1m,.05), ror(p1m,.10)),
    ("1-Year",   f1y, d1y, ror(p1y,.05), ror(p1y,.10)),
]:
    print(f"  {label} median equity   : ${np.median(finals):>14,.0f}")
    print(f"  {label} P5 / P95        : ${np.percentile(finals,5):>12,.0f}  /  ${np.percentile(finals,95):>12,.0f}")
    print(f"  {label} Risk of Ruin 5% : {r5*100:.3f}%   10%: {r10*100:.3f}%")
    print(f"  {label} Median Max DD   : {np.median(dds)*100:.3f}%")
    print()

# ── Fan Chart ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.patch.set_facecolor('#0d1117')

SPECS = [
    (0, '#ff4444', 0.30, 0.8, 'P1'),
    (1, '#ff8800', 0.50, 1.0, 'P5'),
    (3, '#00cc44', 1.00, 2.0, 'P50'),
    (5, '#ff8800', 0.50, 1.0, 'P95'),
    (6, '#ff4444', 0.30, 0.8, 'P99'),
]

for ax, q, n_d, label, finals, dds, r5, r10 in [
    (axes[0], q1m, N_1M, '1 Month',  f1m, d1m, ror(p1m,.05), ror(p1m,.10)),
    (axes[1], q1y, N_1Y, '1 Year',   f1y, d1y, ror(p1y,.05), ror(p1y,.10)),
]:
    ax.set_facecolor('#161b22')
    for sp in ax.spines.values(): sp.set_edgecolor('#30363d')
    x = np.arange(n_d + 1)

    ax.fill_between(x, q[0], q[6], color='#4488ff', alpha=0.07)
    ax.fill_between(x, q[1], q[5], color='#4488ff', alpha=0.12)
    ax.fill_between(x, q[2], q[4], color='#4488ff', alpha=0.20)

    for idx, color, alpha, lw, lbl in SPECS:
        ax.plot(x, q[idx], color=color, lw=lw, alpha=alpha, label=lbl)

    ax.axhline(equity0, color='#8b949e', lw=0.8, ls='--', label='Initial')
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda v, _: f'${v/1e6:.3f}M'))
    ax.set_xlabel('Trading Days', color='#c9d1d9', fontsize=10)
    ax.set_ylabel('Portfolio Equity', color='#c9d1d9', fontsize=10)
    ax.tick_params(colors='#8b949e')
    ax.set_title(
        f'Monte Carlo Fan — {label}  (N={N_SIM:,})\n'
        f'RoR 5%: {r5*100:.3f}%  |  RoR 10%: {r10*100:.3f}%\n'
        f'Median: ${np.median(finals)/1e6:.3f}M  |  P5: ${np.percentile(finals,5)/1e6:.3f}M'
        f'  |  Median MaxDD: {np.median(dds)*100:.2f}%',
        color='#f0f6fc', fontsize=10, fontweight='bold')
    ax.legend(fontsize=8, loc='upper left', framealpha=0.3,
              labelcolor='#c9d1d9', facecolor='#161b22', edgecolor='#30363d')

fig.suptitle(
    f'HFT Engine — Monte Carlo Equity Simulation  '
    f'($10M Portfolio, 1 BTC/trade)\n'
    f'Block bootstrap · {n_days} synthetic days · '
    f'Mean daily PnL ${np.mean(daily):+,.0f} · '
    f'Ann. Sharpe {ann_sh:+.2f}',
    color='#f0f6fc', fontsize=12, fontweight='bold', y=1.01)

plt.tight_layout()
OUT = os.path.join(OUT_DIR, 'monte_carlo_fan.png')
plt.savefig(OUT, dpi=150, bbox_inches='tight', facecolor='#0d1117')
print(f'[+] Saved: {OUT}')
