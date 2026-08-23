import json
import os
import urllib.request
import gzip
import shutil

def download_tardis_data():
    os.makedirs('data/raw', exist_ok=True)
    
    data_types = ['book_snapshot_25', 'trades']
    
    for dt in data_types:
        url = f"https://api.tardis.dev/v1/data-urls/binance?dataType={dt}&symbol=BTCUSDT&from=2024-01-01&to=2024-01-02"
        print(f"Fetching URLs for {dt}...")
        
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req) as response:
            urls_data = json.loads(response.read().decode())
        
        items = urls_data if isinstance(urls_data, list) else urls_data.get('data', [])
        if not items and hasattr(urls_data, 'keys'):
            items = [urls_data] if 'url' in urls_data else items

        for item in items:
            file_url = item['url']
            filename = file_url.split('?')[0].split('/')[-1]
            if not filename.endswith('.gz'):
                filename = f"binance_{dt}_2024-01-01_BTCUSDT.csv.gz"
                
            gz_path = f"data/raw/{filename}"
            csv_path = gz_path.replace('.gz', '')
            
            if not os.path.exists(csv_path):
                print(f"Downloading {filename}...")
                urllib.request.urlretrieve(file_url, gz_path)
                
                print(f"Extracting {filename}...")
                with gzip.open(gz_path, 'rb') as f_in:
                    with open(csv_path, 'wb') as f_out:
                        shutil.copyfileobj(f_in, f_out)
                
                print(f"Cleaning up {gz_path}...")
                os.remove(gz_path)
            else:
                print(f"{csv_path} already exists. Skipping.")

if __name__ == '__main__':
    download_tardis_data()
