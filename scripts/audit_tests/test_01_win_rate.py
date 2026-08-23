import csv
import sys
import os

def calculate_win_rate(journal_path):
    if not os.path.exists(journal_path):
        print(f"Error: {journal_path} not found.")
        return
        
    total_trades = 0
    winning_trades = 0
    losing_trades = 0
    
    with open(journal_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_trades += 1
            pnl = float(row["pnl"])
            if pnl > 0:
                winning_trades += 1
            elif pnl < 0:
                losing_trades += 1
                
    if total_trades > 0:
        win_rate = winning_trades / total_trades
        print("=== Win Rate Raw Counts ===")
        print(f"Total Trades:   {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades:  {losing_trades}")
        print(f"Calculated Win Rate: {winning_trades} / {total_trades} = {win_rate:.6f} ({win_rate * 100:.2f}%)")
    else:
        print("No trades found in the journal.")

if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/tardis_trade_journal.csv"
    calculate_win_rate(path)
