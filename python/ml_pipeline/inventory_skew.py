import numpy as np


def inventory_skew_bps(inv_frac: float, gamma: float, vol: float) -> float:
    """
    AS reservation skew: r = s - q*gamma*sigma^2*(T-t), expressed in bps to
    match how alpha-skew is already computed in strategy_engine.cpp
    (reservation_price = fair_value_anchor + alpha*max_skew).

    No literal (T-t): there's no session end on a 24/7 perp -- folded into
    gamma as a single tunable constant instead of a real time-to-horizon.

    inv_frac: current inventory / max_position, in [-1, 1]. positive = long.
    vol: realized_vol

    Returns skew in bps, negative for long inventory (pulls reservation
    price below mid -> ask side tightens toward mid, bid side widens away).
    """
    return -inv_frac * gamma * (vol ** 2) * 10000.0


def compute_quotes(mid_price: float, inv_frac: float, gamma: float, vol: float,
                    half_spread_bps: float, breach_threshold: float = 0.6,
                    emergency_bid_mult: float = 3.0, emergency_ask_mult: float = 0.3
                    ) -> tuple[float, float]:
    """
    mid_price: current fair-value anchor (use the microprice, not naive mid).
    half_spread_bps: bandit's chosen width.
    Returns (bid_price, ask_price).
    """
    skew_bps = inventory_skew_bps(inv_frac, gamma, vol)
    reservation = mid_price * (1.0 + skew_bps / 10000.0)

    breached = abs(inv_frac) > breach_threshold
    if breached and inv_frac > 0:       # over-long: defend bid, clear ask
        bid_mult, ask_mult = emergency_bid_mult, emergency_ask_mult
    elif breached and inv_frac < 0:     # over-short: defend ask, clear bid
        bid_mult, ask_mult = emergency_ask_mult, emergency_bid_mult
    else:
        bid_mult = ask_mult = 1.0        # normal-mode width applies symmetrically here

    bid = reservation * (1.0 - bid_mult * half_spread_bps / 10000.0)
    ask = reservation * (1.0 + ask_mult * half_spread_bps / 10000.0)
    return bid, ask
