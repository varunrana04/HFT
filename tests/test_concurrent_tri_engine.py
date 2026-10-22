"""
test_concurrent_tri_engine.py
Verifies that all 3 specialist engines (Bullish, Bearish, Mix) run concurrently,
maintain simultaneous active quotes, execute fills in parallel, and aggregate
the portfolio composite correctly.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import paper_trade_rl_live as rl
from tri_engine_manager import TriEngineManager, ActiveEngineMode


def test_concurrent_execution():
    print("Testing Simultaneous Concurrent Tri-Engine Execution...")

    agent = rl.RLPaperAgent()
    mgr = TriEngineManager("BTC", agent)

    # 1. Verify default mode is CONCURRENT
    assert mgr.active_mode == ActiveEngineMode.CONCURRENT, f"Expected CONCURRENT mode, got {mgr.active_mode}"

    # 2. Simulate orderbook tick
    mgr.on_book_tick(65000.0, 65002.0, 10.0, 10.0)

    # 3. Simulate sequential trade ticks to build alphas
    ts = 1700000000000
    fills_all = []

    # Series of alternating buys and sells
    for i in range(30):
        ts += 500
        px = 65001.0 + (i % 3) * 0.5
        is_buy = (i % 2 == 0)
        f = mgr.on_trade_tick(px, 1.5, is_buy, ts)
        fills_all.extend(f)

    # 4. Verify get_concurrent_quotes returns all 3 engines and composite
    quotes = mgr.get_concurrent_quotes()
    assert "bullish" in quotes, "Missing bullish quotes"
    assert "bearish" in quotes, "Missing bearish quotes"
    assert "mix" in quotes, "Missing mix quotes"
    assert "composite" in quotes, "Missing composite quotes"

    comp = quotes["composite"]
    bull = quotes["bullish"]
    bear = quotes["bearish"]
    mix = quotes["mix"]

    print(f"Bullish Engine: Bid={bull['bid']}, Ask={bull['ask']}, Inv={bull['inventory']}, PnL={bull['pnl']}")
    print(f"Bearish Engine: Bid={bear['bid']}, Ask={bear['ask']}, Inv={bear['inventory']}, PnL={bear['pnl']}")
    print(f"Mix Engine:     Bid={mix['bid']}, Ask={mix['ask']}, Inv={mix['inventory']}, PnL={mix['pnl']}")
    print(f"Composite Book: BestBid={comp['best_bid']}, BestAsk={comp['best_ask']}")
    print(f"Portfolio Net Inventory: {comp['net_inventory']} (Gross: {comp['gross_inventory']}, Hedge Eff: {comp['hedge_efficiency_pct']}%)")
    print(f"Portfolio Total PnL: ${comp['total_pnl']}")

    # 5. Verify portfolio math
    expected_net_inv = round(bull["inventory"] + bear["inventory"] + mix["inventory"], 2)
    assert comp["net_inventory"] == expected_net_inv, f"Net inv mismatch: {comp['net_inventory']} vs {expected_net_inv}"

    expected_total_pnl = round(bull["pnl"] + bear["pnl"] + mix["pnl"], 2)
    assert abs(comp["total_pnl"] - expected_total_pnl) < 0.05, f"Total PnL mismatch: {comp['total_pnl']} vs {expected_total_pnl}"

    # 6. Verify get_active_quotes in CONCURRENT mode returns best composite quotes
    best_b, best_a = mgr.get_active_quotes()
    assert best_b == comp["best_bid"], f"Expected {comp['best_bid']}, got {best_b}"
    assert best_a == comp["best_ask"], f"Expected {comp['best_ask']}, got {best_a}"

    # 7. Verify summary state structure
    summary = mgr.get_summary_state()
    assert summary["active_mode"] == "concurrent"
    assert "composite" in summary
    assert "concurrent_quotes" in summary

    print("\n[PASS] ALL 3 ENGINES RUNNING SIMULTANEOUSLY & CONCURRENTLY VERIFIED!")


def test_concurrent_fills_and_hedging():
    print("\nTesting Concurrent Multi-Engine Fills & Natural Inventory Hedging...")
    agent = rl.RLPaperAgent()
    mgr = TriEngineManager("BTC", agent)

    # Directly set resting quotes for testing concurrent execution
    mgr.bullish_engine.my_bid = 65000.0
    mgr.bearish_engine.my_ask = 65002.0
    mgr.mix_engine.my_bid = 64999.0
    mgr.mix_engine.my_ask = 65003.0

    quotes = mgr.get_concurrent_quotes()
    comp = quotes["composite"]

    # Best bid should be bullish 65000.0, best ask bearish 65002.0
    assert comp["best_bid"] == 65000.0, f"Expected 65000.0, got {comp['best_bid']}"
    assert comp["best_ask"] == 65002.0, f"Expected 65002.0, got {comp['best_ask']}"

    # Simulate price dip hitting Bullish bid
    f1 = mgr.bullish_engine.check_fill(65000.0, 1000)
    assert f1 is not None and f1["side"] == "BUY" and f1["engine"] == "BULLISH"
    assert mgr.bullish_engine.inventory == 1.0

    # Simulate price spike hitting Bearish ask
    f2 = mgr.bearish_engine.check_fill(65002.0, 1500)
    assert f2 is not None and f2["side"] == "SELL" and f2["engine"] == "BEARISH"
    assert mgr.bearish_engine.inventory == -1.0

    # Verify Natural Portfolio Hedging
    q_post = mgr.get_concurrent_quotes()
    post_comp = q_post["composite"]

    print(f"Bullish Inv: {q_post['bullish']['inventory']} | Bearish Inv: {q_post['bearish']['inventory']}")
    print(f"Net Portfolio Inv: {post_comp['net_inventory']} (Hedge Efficiency: {post_comp['hedge_efficiency_pct']}%)")

    assert post_comp["net_inventory"] == 0.0, f"Expected net 0.0 inventory hedge, got {post_comp['net_inventory']}"
    assert post_comp["gross_inventory"] == 2.0, f"Expected gross 2.0, got {post_comp['gross_inventory']}"
    assert post_comp["hedge_efficiency_pct"] == 100.0, f"Expected 100% hedge efficiency, got {post_comp['hedge_efficiency_pct']}"

    print("[PASS] NATURAL INVENTORY HEDGING ACROSS CONCURRENT ENGINES VERIFIED!")


if __name__ == "__main__":
    test_concurrent_execution()
    test_concurrent_fills_and_hedging()
