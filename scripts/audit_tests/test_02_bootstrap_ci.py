import numpy as np
import pandas as pd
import sys
import os

def block_bootstrap_sharpe(pnls, num_bootstraps=1000, block_size=10):
    np.random.seed(42)
    n = len(pnls)
    if n < block_size:
        block_size = max(1, n // 2)
    
    num_blocks = n // block_size + 1
    bootstrapped_sharpes = []
    
    for _ in range(num_bootstraps):
        start_indices = np.random.randint(0, n - block_size + 1, num_blocks)
        sampled_pnls = np.concatenate([pnls[i:i+block_size] for i in start_indices])[:n]
        
        mean_pnl = np.mean(sampled_pnls)
        std_pnl = np.std(sampled_pnls)
        if std_pnl > 0:
            # Annualize assuming 10,000 trades/year for HFT benchmark
            sharpe = (mean_pnl / std_pnl) * np.sqrt(10000)
            bootstrapped_sharpes.append(sharpe)
            
    if not bootstrapped_sharpes:
        return 0.0, 0.0, 0.0
        
    bootstrapped_sharpes = np.array(bootstrapped_sharpes)
    mean_sharpe = np.mean(bootstrapped_sharpes)
    ci_lower = np.percentile(bootstrapped_sharpes, 2.5)
    ci_upper = np.percentile(bootstrapped_sharpes, 97.5)
    return mean_sharpe, ci_lower, ci_upper

def test_bootstrap_ci(journal_path):
    if not os.path.exists(journal_path):
        print(f"Error: {journal_path} not found.")
        return
        
    df = pd.read_csv(journal_path)
    if len(df) == 0:
        print("No trades found in the journal.")
        return
    df = df[df["pnl"] != 0.0]
    pnls = df["pnl"].values
    mean_sharpe, ci_lower, ci_upper = block_bootstrap_sharpe(pnls)
    
    print(f"=== Block Bootstrap Sharpe CI (N={len(df)}) ===")
    print(f"Robust Bootstrap Mean: {mean_sharpe:.4f}")
    print(f"95% CI: [{ci_lower:.4f}, {ci_upper:.4f}]")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/tardis_trade_journal.csv"
    test_bootstrap_ci(path)
