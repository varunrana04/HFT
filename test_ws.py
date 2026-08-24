import asyncio
import websockets

async def test():
    try:
        async with websockets.connect('wss://fstream.binance.com/ws') as ws:
            print('Connected!')
            await ws.send('{"method":"SUBSCRIBE","params":["btcusdt@trade"],"id":1}')
            msg = await ws.recv()
            print('Received:', msg)
    except Exception as e:
        print("Error:", e)

asyncio.run(test())
