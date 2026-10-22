import numpy as np
from numba import njit
from hftbacktest import HftBacktest, BUY_EVENT, SELL_EVENT, GTX, NONE, BUY, SELL

@njit
def avellaneda_stoikov_market_maker(hbt, stat):
    # MM Parameters
    half_spread = 0.5    # Minimum half spread in ticks
    grid_interval = 1.0  # Tick size
    skew_mult = 1.5      # Skew aggressively based on inventory
    max_position = 10    # Max allowed position
    
    # Track order IDs for cancellation
    bid_order_id = -1
    ask_order_id = -1
    
    while hbt.elapse(100_000_000):  # advance by 100ms
        
        # 1. Clear filled/canceled orders from our tracking
        hbt.clear_inactive_orders()
        
        # 2. Get current market state
        mid = (hbt.best_bid + hbt.best_ask) / 2.0
        pos = hbt.position
        
        # 3. Calculate Avellaneda-Stoikov style reservation price & spread
        # Reservation price shifts away from mid based on our inventory
        reservation_price = mid - (pos * skew_mult)
        
        # Calculate quoted bid/ask
        bid_price = np.floor((reservation_price - half_spread) / grid_interval) * grid_interval
        ask_price = np.ceil((reservation_price + half_spread) / grid_interval) * grid_interval
        
        # 4. Market Making Logic: Place or update bids
        if pos < max_position:
            # We can still buy. Check if we already have a bid at the right price
            active_bids = [order for order in hbt.orders.values() if order.side == BUY and order.cancellable]
            if len(active_bids) == 0:
                hbt.submit_buy_order(bid_order_id, bid_price, 1.0, GTX)
                bid_order_id -= 1
        
        # 5. Market Making Logic: Place or update asks
        if pos > -max_position:
            active_asks = [order for order in hbt.orders.values() if order.side == SELL and order.cancellable]
            if len(active_asks) == 0:
                hbt.submit_sell_order(ask_order_id, ask_price, 1.0, GTX)
                ask_order_id -= 1
        
        # Record stats every interval
        stat.record(hbt)

print("Avellaneda-Stoikov HFT Strategy Code Generated.")
print("To run on real data, we must convert Binance L2 snapshots to hftbacktest .npz format.")
