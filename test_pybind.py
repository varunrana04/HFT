import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
from engine_loader import load_engine

hft_engine = load_engine()
import time

ws = hft_engine.BinanceWs("btcusdt")
ws.start_live_feed(None) # Pass None for engine just to test connection
time.sleep(5)
ws.stop()
