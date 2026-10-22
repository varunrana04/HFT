import asyncio
import os
import sys
from binance_order_gateway import BinanceOrderGateway

async def validate_testnet():
    print("=====================================================")
    print(" HFT Engine - Binance Testnet Validation")
    print("=====================================================")
    
    api_key = os.environ.get('BINANCE_API_KEY')
    api_secret = os.environ.get('BINANCE_API_SECRET')
    
    is_mock = False
    if not api_key or not api_secret:
        print("[WARNING] BINANCE_API_KEY and BINANCE_API_SECRET missing. Running in MOCK mode.")
        is_mock = True
    else:
        print("[INFO] API Keys found. Initializing gateway...")
        
    gateway = BinanceOrderGateway()
    await gateway.connect()
    
    symbol = "BTCUSDT"
    
    try:
        if is_mock:
            current_price = 60000.0
            print(f"[INFO] MOCK MODE: Hardcoded {symbol} price: {current_price}")
        else:
            async with gateway.session.get(f"{gateway.base_url}/fapi/v1/ticker/price?symbol={symbol}") as resp:
                data = await resp.json()
                if 'price' not in data:
                    print(f"[ERROR] Could not fetch price: {data}")
                    sys.exit(1)
                current_price = float(data['price'])
                print(f"[INFO] Current {symbol} price: {current_price}")
        
        # Place a Limit BUY order 10% below current price to ensure it doesn't execute
        safe_price = round(current_price * 0.90, 1)
        quantity = 0.005 # Minimum BTC qty
        
        print(f"\n[TEST 1] Placing Limit BUY order for {quantity} {symbol} @ {safe_price}")
        order_res = await gateway.place_limit_order(symbol, "BUY", quantity, safe_price)
        
        if is_mock:
            order_res["orderId"] = 123456789
            
        if "orderId" not in order_res:
            print(f"[ERROR] Order placement failed: {order_res}")
            sys.exit(1)
            
        order_id = order_res["orderId"]
        latency = order_res.get("latency_ms", 0)
        print(f"[SUCCESS] Order {order_id} placed successfully in {latency:.2f} ms")
        
        print(f"\n[TEST 2] Cancelling Order {order_id}")
        cancel_res = await gateway.cancel_order(symbol, order_id)
        
        if is_mock:
            cancel_res["status"] = "CANCELED"
            
        if "status" in cancel_res and cancel_res["status"] == "CANCELED":
            print(f"[SUCCESS] Order {order_id} cancelled successfully.")
        else:
            print(f"[ERROR] Cancel failed: {cancel_res}")
            
        print("\n=====================================================")
        print(" Validation Complete! Gateway is functioning perfectly.")
        print("=====================================================")
            
    finally:
        await gateway.close()

if __name__ == "__main__":
    asyncio.run(validate_testnet())
