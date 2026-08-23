import time
import urllib.request
import argparse
from datetime import datetime

def ping_server(url: str, interval_mins: int):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Starting Railway Keep-Alive Pinger...")
    print(f"Target: {url}")
    print(f"Interval: {interval_mins} minutes")
    
    interval_secs = interval_mins * 60
    
    while True:
        try:
            req = urllib.request.Request(url, headers={'User-Agent': 'HFT-Keep-Alive/1.0'})
            with urllib.request.urlopen(req, timeout=10) as response:
                status = response.getcode()
                body = response.read().decode('utf-8')
                print(f"[{datetime.now().strftime('%H:%M:%S')}] Ping Success! Status: {status} | Response: {body}")
        except Exception as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}] Ping Failed! Error: {e}")
            
        time.sleep(interval_secs)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Keep a Railway app awake by pinging it.")
    parser.add_argument("--url", type=str, required=True, help="The URL to ping (e.g., https://your-app.up.railway.app/health)")
    parser.add_argument("--interval", type=int, default=10, help="Ping interval in minutes (default: 10)")
    
    args = parser.parse_args()
    ping_server(args.url, args.interval)
