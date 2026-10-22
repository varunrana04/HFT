import requests, csv, os, time

SYMBOL   = "BTCUSDT"
INTERVAL = "1m"
LIMIT    = 1000
DAYS     = 30

os.makedirs("data", exist_ok=True)
out = "data/btcusdt_1m.csv"

end_ts   = int(time.time() * 1000)
start_ts = end_ts - DAYS * 86_400_000
rows     = []
url = "https://fapi.binance.com/fapi/v1/klines"
cur = start_ts
while cur < end_ts:
    r = requests.get(url, params=dict(symbol=SYMBOL, interval=INTERVAL,
                                      startTime=cur, limit=LIMIT), timeout=15)
    data = r.json()
    if not data or not isinstance(data, list):
        break
    rows.extend(data)
    cur = int(data[-1][6]) + 1
    print(f"\r  {len(rows):,} rows...", end="", flush=True)
    if len(data) < LIMIT:
        break

with open(out, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["open_time","open","high","low","close","volume","close_time","quote_vol","trades","taker_buy_vol","taker_buy_quote_vol","ignore"])
    w.writerows(rows)
print(f"\nDownloaded {len(rows):,} rows to {out}")
