import asyncio
import time
from fastapi import FastAPI
from contextlib import asynccontextmanager
import uvicorn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'python'))
from engine_loader import load_engine

hft_engine = load_engine()
cpp_gateway = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global cpp_gateway
    print("Init gateway...")
    cpp_gateway = hft_engine.BinanceWs("btcusdt")
    cpp_gateway.start_live_feed(None)
    yield
    print("Stopping gateway...")
    cpp_gateway.stop()

app = FastAPI(lifespan=lifespan)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
