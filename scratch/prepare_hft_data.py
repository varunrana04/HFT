import urllib.request
import zipfile
import os
import pandas as pd
import numpy as np

def download_and_extract(url, extract_to):
    filename = url.split('/')[-1]
    filepath = os.path.join(extract_to, filename)
    print(f"Downloading {url}...")
    try:
        urllib.request.urlretrieve(url, filepath)
        print(f"Extracting {filepath}...")
        with zipfile.ZipFile(filepath, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        os.remove(filepath)
        return filepath.replace('.zip', '.csv')
    except Exception as e:
        print(f"Failed to download {url}: {e}")
        return None

def main():
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Binance Vision URLs for a specific day (2024-01-15)
    # AggTrades (ticks)
    agg_url = "https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-2024-01-15.zip"
    
    # Downloading aggTrades
    agg_file = download_and_extract(agg_url, data_dir)
    if not agg_file:
        return
        
    print(f"Data ready at {agg_file}")
    
    # Check the data format
    df = pd.read_csv(agg_file, header=None, names=["agg_trade_id", "price", "qty", "first_trade_id", "last_trade_id", "timestamp", "is_buyer_maker"])
    print("AggTrades Sample (Rows: ", len(df), "):")
    print(df.head())
    
    # We will need to map this into hftbacktest's NumPy format (Timestamp, Event Type, Order Side, Price, Qty)
    print("\nNext step: Convert this raw tick data into hftbacktest .npz format.")

if __name__ == "__main__":
    main()
