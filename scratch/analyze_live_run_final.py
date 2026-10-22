import pandas as pd
import sys

csv_file = sys.argv[1]
try:
    df = pd.read_csv(csv_file)
    if df.empty:
        print(f"File {csv_file} is empty (only headers).")
        sys.exit(0)
    
    num_trades = len(df)
    total_pnl = df["PnL"].sum()
    win_rate = (df["PnL"] > 0).mean() * 100 if num_trades > 0 else 0
    total_slippage = df["Slippage"].sum() if "Slippage" in df.columns else 0
    total_qty = df["Qty"].sum()
    gross_fees = (df["ExitPrice"] * df["Qty"] * 0.00015).sum() # approx fee
    
    print(f"Results for {csv_file}:")
    print(f"Total Trades: {num_trades}")
    print(f"Total PnL: {total_pnl:.4f}")
    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Total Volume (Qty): {total_qty:.4f}")
    print(f"Estimated Fees: {gross_fees:.4f}")
    print(f"Net PnL (w/ fees): {total_pnl - gross_fees:.4f}")
    
    print("\nFirst 5 trades:")
    print(df.head(5).to_string())
    
except Exception as e:
    print(f"Error reading CSV: {e}")
