import sys

with open('python/live_paper_trade.py', 'r') as f:
    content = f.read()

# 1. Fix python_bybit_ws
old_bybit_ws = """async def python_bybit_ws():
    \"\"\"Pure-Python WebSocket fallback with sequence-gap detection and exponential backoff.\"\"\"
    global gateway_latency_ns

    ws_base = os.environ.get('BYBIT_WS_BASE_URL', 'wss://fstream.bybit.com')
    url = f"{ws_base}/stream?streams=btcusdt@trade/btcusdt@depth20@100ms"

    # Sequence gap tracking (Bybit Futures depth stream)
    last_update_id: int = 0
    backoff: float = 1.0

    while True:
        try:
            print(f"[PythonWs] Connecting to {url}...")
            async with websockets.connect(url, ping_interval=30, ping_timeout=10, max_size=None) as ws:
                print("[PythonWs] Connected and subscribed.")
                backoff = 1.0  # reset on successful connect

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        if "data" not in data:
                            continue

                        d = data["data"]
                        stream = data.get("stream", "")

                        if "@trade" in stream:
                            trade = hft_engine.Trade()
                            trade.timestamp_ns = int(d["T"]) * 1_000_000
                            trade.price = int(float(d["p"]) * 1e8)
                            trade.qty   = int(float(d["q"]) * 1e8)  # must be .qty not .quantity
                            trade.side  = hft_engine.Side.ASK if d["m"] else hft_engine.Side.BID
                            engine.on_trade(trade, latest_book)

                        elif "@depth20" in stream:
                            u  = d.get("u", 0)
                            pu = d.get("pu", 0)

                            if last_update_id != 0 and pu != last_update_id:
                                print(
                                    f"[PythonWs] Sequence gap: expected pu={last_update_id} "
                                    f"got pu={pu}. Reconnecting..."
                                )
                                last_update_id = 0
                                break

                            last_update_id = u
                            latest_book.timestamp_ns = int(time.time() * 1e9)
                            bids = d.get("b", [])
                            asks = d.get("a", [])

                            if bids:
                                latest_book.best_bid_price = int(float(bids[0][0]) * 1e8)
                                latest_book.best_bid_qty   = int(float(bids[0][1]) * 1e8)
                                latest_book.bid_count      = len(bids)
                            if asks:
                                latest_book.best_ask_price = int(float(asks[0][0]) * 1e8)
                                latest_book.best_ask_qty   = int(float(asks[0][1]) * 1e8)
                                latest_book.ask_count      = len(asks)

                            if latest_book.is_valid():
                                # Weighted OBI from full 20-level ladder
                                # Weight = 1/(level+1) so top of book has most influence
                                bid_wvol = sum(
                                    float(b[1]) / (i + 1)
                                    for i, b in enumerate(bids[:20]) if float(b[1]) > 0
                                )
                                ask_wvol = sum(
                                    float(a[1]) / (i + 1)
                                    for i, a in enumerate(asks[:20]) if float(a[1]) > 0
                                )
                                ladder_obi = (bid_wvol - ask_wvol) / (bid_wvol + ask_wvol + 1e-8)

                                _t0 = time.perf_counter_ns()
                                engine.on_book_update(latest_book)
                                gateway_latency_ns = time.perf_counter_ns() - _t0

                                # Override single-level OBI with full-ladder OBI
                                engine._last_features.obi = ladder_obi

                    except Exception as loop_err:
                        print(f"[PythonWs] Message loop error: {loop_err}")

        except asyncio.CancelledError:
            break
        except Exception as e:
            import math
            jitter = 0.5 * backoff * (1 + (time.time() % 1))  # ~0-100% jitter
            wait = backoff + jitter
            print(f"[PythonWs] Disconnected: {e}. Reconnecting in {wait:.1f}s...")
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, 60.0)  # cap at 60 seconds"""

new_bybit_ws = """async def python_bybit_ws():
    \"\"\"Bybit Public WebSocket (V5) for Orderbook and Trades.\"\"\"
    global gateway_latency_ns
    
    url = "wss://stream-testnet.bybit.com/v5/public/linear"
    symbol = os.environ.get("BYBIT_SYMBOL", "BTCUSDT").upper()
    backoff: float = 1.0

    while True:
        try:
            print(f"[PythonWs] Connecting to {url}...")
            async with websockets.connect(url, ping_interval=20, ping_timeout=10, max_size=None) as ws:
                print("[PythonWs] Connected and subscribed.")
                backoff = 1.0
                
                sub_msg = {
                    "op": "subscribe",
                    "args": [f"orderbook.50.{symbol}", f"publicTrade.{symbol}"]
                }
                await ws.send(json.dumps(sub_msg))

                async for msg in ws:
                    try:
                        data = json.loads(msg)
                        topic = data.get("topic", "")
                        
                        if "publicTrade" in topic:
                            for d in data.get("data", []):
                                trade = hft_engine.Trade()
                                trade.timestamp_ns = int(d["T"]) * 1_000_000
                                trade.price = int(float(d["p"]) * 1e8)
                                trade.qty   = int(float(d["v"]) * 1e8)
                                trade.side  = hft_engine.Side.BID if d["S"] == "Buy" else hft_engine.Side.ASK
                                engine.on_trade(trade, latest_book)

                        elif "orderbook" in topic:
                            d = data.get("data", {})
                            bids = d.get("b", [])
                            asks = d.get("a", [])
                            
                            if not bids and not asks:
                                continue
                                
                            latest_book.timestamp_ns = int(time.time() * 1e9)

                            if bids:
                                latest_book.best_bid_price = int(float(bids[0][0]) * 1e8)
                                latest_book.best_bid_qty   = int(float(bids[0][1]) * 1e8)
                                latest_book.bid_count      = len(bids)
                            if asks:
                                latest_book.best_ask_price = int(float(asks[0][0]) * 1e8)
                                latest_book.best_ask_qty   = int(float(asks[0][1]) * 1e8)
                                latest_book.ask_count      = len(asks)

                            if latest_book.is_valid():
                                bid_wvol = sum(float(b[1]) / (i + 1) for i, b in enumerate(bids[:20]) if float(b[1]) > 0)
                                ask_wvol = sum(float(a[1]) / (i + 1) for i, a in enumerate(asks[:20]) if float(a[1]) > 0)
                                ladder_obi = (bid_wvol - ask_wvol) / (bid_wvol + ask_wvol + 1e-8)

                                _t0 = time.perf_counter_ns()
                                engine.on_book_update(latest_book)
                                gateway_latency_ns = time.perf_counter_ns() - _t0
                                engine._last_features.obi = ladder_obi

                    except Exception as loop_err:
                        pass

        except asyncio.CancelledError:
            break
        except Exception as e:
            wait = backoff + 0.5
            print(f"[PythonWs] Disconnected: {e}. Reconnecting in {wait:.1f}s...")
            await asyncio.sleep(wait)
            backoff = min(backoff * 2, 60.0)"""

content = content.replace(old_bybit_ws, new_bybit_ws)

# 2. Fix user_data_loop (Execution Reports)
old_ud_loop_start = """async def user_data_loop(gateway: BybitOrderGateway):
    \"\"\"
    Connects to the Bybit User Data Stream and processes execution reports.
    Updates the C++ engine via on_fill().
    \"\"\"
    print("[EXEC_WS] Starting Bybit User Data Stream Loop...")
    
    backoff = 2.0
    while True:
        try:
            # We use Bybit's stream_user_data generator directly if implemented
            async for data in gateway.stream_user_data():
                try:
                    e_type = data.get("e")"""
                    
old_ud_loop_full = """async def user_data_loop(gateway: BybitOrderGateway):
    \"\"\"
    Connects to the Bybit User Data Stream and processes execution reports.
    Updates the C++ engine via on_fill().
    \"\"\"
    print("[EXEC_WS] Starting Bybit User Data Stream Loop...")
    
    backoff = 2.0
    while True:
        try:
            # We use Bybit's stream_user_data generator directly if implemented
            async for data in gateway.stream_user_data():
                try:
                    e_type = data.get("e")
                    if e_type == "ORDER_TRADE_UPDATE":
                        o = data.get("o", {})
                        status = o.get("X")  # FILLED, CANCELED, EXPIRED, etc.
                        client_id = o.get("c", "")
                        
                        if status == "FILLED":
                            fill_qty = float(o.get("l", 0.0))
                            fill_price = float(o.get("L", 0.0))
                            realized_pnl = float(o.get("rp", 0.0))
                            fee = float(o.get("n", 0.0))
                            side = o.get("S")
                            
                            is_maker = o.get("m", False)
                            
                            print(f"[FILL] {side} {fill_qty} @ {fill_price} (Maker: {is_maker}) PnL: {realized_pnl}")
                            
                            if client_id in pending_orders:
                                pending_orders.remove(client_id)
                                
                            engine.on_fill(
                                hft_engine.Side.ASK if side == "SELL" else hft_engine.Side.BID,
                                int(fill_price * 1e8),
                                int(fill_qty * 1e8)
                            )
                            engine.set_realized_pnl(engine.realized_pnl() + realized_pnl)
                            
                        elif status in ("CANCELED", "EXPIRED", "REJECTED"):
                            if client_id in pending_orders:
                                pending_orders.remove(client_id)
                                
                except Exception as e:
                    print(f"[EXEC_WS] Error parsing: {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[EXEC_WS] Disconnected: {e}. Reconnecting...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)"""

new_ud_loop = """async def user_data_loop(gateway: BybitOrderGateway):
    \"\"\"
    Connects to the Bybit User Data Stream and processes execution reports.
    Updates the C++ engine via on_fill().
    \"\"\"
    print("[EXEC_WS] Starting Bybit User Data Stream Loop...")
    
    backoff = 2.0
    while True:
        try:
            async for exc in gateway.stream_user_data():
                try:
                    status = exc.get("orderStatus")
                    client_id = exc.get("orderLinkId", "")
                    
                    if status == "Filled" or status == "PartiallyFilled":
                        fill_qty = float(exc.get("execQty", 0.0))
                        fill_price = float(exc.get("execPrice", 0.0))
                        fee = float(exc.get("execFee", 0.0))
                        side = exc.get("side")
                        
                        is_maker = exc.get("isMaker", False)
                        
                        print(f"[FILL] {side} {fill_qty} @ {fill_price} (Maker: {is_maker})")
                        
                        if client_id in pending_orders:
                            pending_orders.remove(client_id)
                            
                        engine.on_fill(
                            hft_engine.Side.ASK if side == "Sell" else hft_engine.Side.BID,
                            int(fill_price * 1e8),
                            int(fill_qty * 1e8)
                        )
                        
                    elif status in ("Cancelled", "Deactivated", "Rejected"):
                        if client_id in pending_orders:
                            pending_orders.remove(client_id)
                            
                except Exception as e:
                    print(f"[EXEC_WS] Error parsing: {e}")
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"[EXEC_WS] Disconnected: {e}. Reconnecting...")
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 60.0)"""
content = content.replace(old_ud_loop_full, new_ud_loop)

# 3. Fix funding_rate_loop (Bybit uses /v5/market/tickers)
old_funding = """async def funding_rate_loop():
    \"\"\"
    Polls Bybit /fapi/v1/premiumIndex every 30 seconds to keep the global
    funding rate tracker updated in the engine.
    \"\"\"
    SYMBOL  = os.environ.get("BYBIT_SYMBOL", "BTCUSDT").upper()
    WS_BASE = os.environ.get("BYBIT_BASE_URL", "https://fapi.bybit.com")
    
    POLL_S  = 30.0  # Bybit updates funding rate info every ~5 s; 30 s is plenty
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                url = f"{WS_BASE}/fapi/v1/premiumIndex?symbol={SYMBOL}"
                async with session.get(url, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        fr = float(data.get("lastFundingRate", 0.0))
                        engine.update_funding_rate(fr)
            except asyncio.CancelledError:
                break
            except Exception as e:
                pass
            await asyncio.sleep(POLL_S)"""

new_funding = """async def funding_rate_loop():
    SYMBOL  = os.environ.get("BYBIT_SYMBOL", "BTCUSDT").upper()
    WS_BASE = os.environ.get("BYBIT_BASE_URL", "https://api-testnet.bybit.com")
    
    POLL_S  = 30.0
    
    async with aiohttp.ClientSession() as session:
        while True:
            try:
                url = f"{WS_BASE}/v5/market/tickers?category=linear&symbol={SYMBOL}"
                async with session.get(url, timeout=5.0) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        fr = float(data.get("result", {}).get("list", [{}])[0].get("fundingRate", 0.0))
                        engine.update_funding_rate(fr)
            except asyncio.CancelledError:
                break
            except Exception as e:
                pass
            await asyncio.sleep(POLL_S)"""
content = content.replace(old_funding, new_funding)

with open('python/live_paper_trade.py', 'w') as f:
    f.write(content)
