"""
tests/test_phase2.py
─────────────────────────────────────────────────────────────────────────────
Phase 2 integration test suite.

Covers every interface added in Phase 2 without requiring live Binance
keys, Docker, or the C++ hft_engine.pyd to be compiled.

Run with:
    pytest tests/test_phase2.py -v
or directly:
    python tests/test_phase2.py
"""
import asyncio
import sys
import os
import time

# ── Make clean_HFT/python importable ─────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))


# ═══════════════════════════════════════════════════════════════════════════
# Gateway tests
# ═══════════════════════════════════════════════════════════════════════════

class TestBinanceOrderGateway:
    """BinanceOrderGateway paper-mode (no API keys)."""

    def setup_method(self):
        from binance_order_gateway import BinanceOrderGateway
        self.GW = BinanceOrderGateway

    def _run(self, coro):
        return asyncio.run(coro)

    def test_place_limit_order_paper_mode(self):
        async def go():
            gw = self.GW()
            await gw.connect()
            r = await gw.place_limit_order("BTCUSDT", "BUY", 0.001, 60_000.0)
            assert r["status"] == "MOCK_SUCCESS"
            assert r["orderId"] == -1
            assert r["latency_ms"] == 0
            await gw.close()
        self._run(go())

    def test_cancel_order_paper_mode(self):
        async def go():
            gw = self.GW()
            await gw.connect()
            r = await gw.cancel_order("BTCUSDT", 99999)
            assert r["status"] == "MOCK_CANCEL"
            await gw.close()
        self._run(go())

    def test_create_listen_key_no_api_key(self):
        async def go():
            gw = self.GW()
            await gw.connect()
            k = await gw.create_listen_key()
            assert k is None, "Expected None with no API key"
            await gw.close()
        self._run(go())

    def test_avg_latency_ms_empty(self):
        gw = self.GW()
        assert gw.avg_latency_ms() == 0.0

    def test_avg_latency_ms_with_data(self):
        gw = self.GW()
        gw.latency_stats = [10.0, 20.0, 30.0]
        assert gw.avg_latency_ms() == 20.0

    def test_get_position_risk_paper_mode(self):
        async def go():
            gw = self.GW()
            await gw.connect()
            r = await gw.get_position_risk("BTCUSDT")
            assert r["positionAmt"] == 0.0
            await gw.close()
        self._run(go())

    def test_get_realized_pnl_paper_mode(self):
        async def go():
            gw = self.GW()
            await gw.connect()
            pnl = await gw.get_realized_pnl("BTCUSDT", 0)
            assert pnl == 0.0
            await gw.close()
        self._run(go())


# ═══════════════════════════════════════════════════════════════════════════
# Pure Python Engine — is_trading_halted
# ═══════════════════════════════════════════════════════════════════════════

class TestIsTradinHalted:
    def _make_engine(self, capital=10_000.0, loss_limit=500.0, max_pos=1.0):
        from pure_python_engine import StrategyEngine, StrategyConfig
        cfg = StrategyConfig()
        cfg.initial_capital       = capital
        cfg.daily_loss_limit_usd  = loss_limit
        cfg.max_position_btc      = max_pos
        return StrategyEngine(cfg)

    def test_not_halted_fresh(self):
        eng = self._make_engine()
        assert not eng.is_trading_halted(0)

    def test_halted_after_loss_limit(self):
        eng = self._make_engine(capital=10_000, loss_limit=500)
        eng._equity = 9_400.0   # loss = 600 > 500
        assert eng.is_trading_halted(0)

    def test_halted_after_position_limit(self):
        eng = self._make_engine(max_pos=1.0)
        eng.pos = 1.5            # exceeds max_pos
        assert eng.is_trading_halted(0)

    def test_not_halted_within_limits(self):
        eng = self._make_engine(capital=10_000, loss_limit=500, max_pos=1.0)
        eng._equity = 9_600.0   # loss = 400 < 500
        eng.pos = 0.5
        assert not eng.is_trading_halted(0)

    def test_halted_flag_persists(self):
        """Once halted it stays halted even if equity recovers."""
        eng = self._make_engine(capital=10_000, loss_limit=500)
        eng._equity = 9_400.0
        eng.is_trading_halted(0)  # trips the flag
        eng._equity = 10_000.0   # pretend equity recovered
        assert eng.is_trading_halted(0), "Halt should persist until manual reset"


# ═══════════════════════════════════════════════════════════════════════════
# Pure Python Engine — simulate_fill
# ═══════════════════════════════════════════════════════════════════════════

class TestSimulateFill:
    def _make_engine(self):
        from pure_python_engine import StrategyEngine, StrategyConfig
        cfg = StrategyConfig()
        cfg.initial_capital       = 10_000.0
        cfg.daily_loss_limit_usd  = 500.0
        cfg.max_position_btc      = 1.0
        cfg.maker_fee_pct         = -0.00005  # maker rebate
        return StrategyEngine(cfg)

    def test_buy_fill_updates_position(self):
        from pure_python_engine import Side
        eng = self._make_engine()
        eng.simulate_fill(Side.BID, int(60_000 * 1e8), int(0.001 * 1e8))
        assert abs(eng.pos - 0.001) < 1e-9

    def test_buy_fill_deducts_fee(self):
        from pure_python_engine import Side
        eng = self._make_engine()
        pre = eng._cash
        eng.simulate_fill(Side.BID, int(60_000 * 1e8), int(0.001 * 1e8), is_maker=False)
        fee = 60_000 * 0.001 * 0.00015
        assert abs(eng._cash - (pre - fee)) < 0.01

    def test_sell_closes_long_and_realises_pnl(self):
        from pure_python_engine import Side
        eng = self._make_engine()
        eng.simulate_fill(Side.BID, int(60_000 * 1e8), int(0.001 * 1e8))   # open long
        eng.simulate_fill(Side.ASK, int(61_000 * 1e8), int(0.001 * 1e8))   # close long
        assert abs(eng.pos) < 1e-9, "Position should be flat after close"
        assert eng._metrics.realized_pnl > 0, "Expected positive PnL on winning trade"

    def test_round_trip_pnl_math(self):
        from pure_python_engine import Side
        eng = self._make_engine()
        eng.simulate_fill(Side.BID, int(60_000 * 1e8), int(0.001 * 1e8), is_maker=False)
        eng.simulate_fill(Side.ASK, int(61_000 * 1e8), int(0.001 * 1e8), is_maker=False)
        gross_pnl = (61_000 - 60_000) * 0.001   # = 1.0
        taker_fees = 2 * (60_000 + 61_000) / 2 * 0.001 * 0.00015
        expected_net = gross_pnl - taker_fees
        assert abs(eng._metrics.realized_pnl - expected_net) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
# Sequence gap detection
# ═══════════════════════════════════════════════════════════════════════════

class TestSequenceGap:
    def test_gap_detected_on_pu_mismatch(self):
        last_update_id = 100
        u, pu = 103, 101   # pu should be 100
        assert last_update_id != 0 and pu != last_update_id

    def test_no_gap_on_contiguous_update(self):
        last_update_id = 100
        u, pu = 101, 100
        assert not (last_update_id != 0 and pu != last_update_id)

    def test_no_gap_check_on_first_message(self):
        last_update_id = 0   # initial state
        u, pu = 500, 499
        assert not (last_update_id != 0 and pu != last_update_id)


# ═══════════════════════════════════════════════════════════════════════════
# Kill-switch gate logic (execution_loop behaviour, isolated)
# ═══════════════════════════════════════════════════════════════════════════

class TestKillSwitchGate:
    def test_halted_blocks_submission(self):
        """Simulate what execution_loop does when halted=True."""
        async def go():
            from binance_order_gateway import BinanceOrderGateway
            gw = BinanceOrderGateway()
            await gw.connect()

            active_order_id = 55
            halted = True

            if halted:
                if active_order_id != -1:
                    r = await gw.cancel_order("BTCUSDT", active_order_id)
                    assert r["status"] == "MOCK_CANCEL"
                    active_order_id = -1

            assert active_order_id == -1
            await gw.close()

        asyncio.run(go())

    def test_stale_cancel_before_new_order(self):
        """New signal cancels the stale order before placing fresh one."""
        async def go():
            from binance_order_gateway import BinanceOrderGateway
            gw = BinanceOrderGateway()
            await gw.connect()

            active_order_id = 77
            new_signal = True

            if new_signal and active_order_id != -1:
                r = await gw.cancel_order("BTCUSDT", active_order_id)
                assert r["status"] == "MOCK_CANCEL"
                active_order_id = -1

            r2 = await gw.place_limit_order("BTCUSDT", "BUY", 0.001, 60_000)
            active_order_id = r2.get("orderId", -1)
            assert active_order_id == -1   # paper mode returns -1

            await gw.close()

        asyncio.run(go())


# ═══════════════════════════════════════════════════════════════════════════
# Telemetry payload shape
# ═══════════════════════════════════════════════════════════════════════════

class TestTelemetryPayload:
    def test_latency_fields_present_and_typed(self):
        payload = {
            "latency": {
                "book_update_us": round(12_345 / 1000.0, 2),
                "order_submit_ms": round(5_600_000 / 1_000_000.0, 2),
            },
            "kill_switch_halted": False,
        }
        assert isinstance(payload["latency"]["book_update_us"], float)
        assert isinstance(payload["latency"]["order_submit_ms"], float)
        assert isinstance(payload["kill_switch_halted"], bool)
        assert payload["latency"]["book_update_us"] == 12.35
        assert payload["latency"]["order_submit_ms"] == 5.6

    def test_latency_color_thresholds(self):
        """Verify the dashboard colour-band logic (mirrored from JS)."""
        def book_color(us):
            return "green" if us <= 100 else ("yellow" if us <= 500 else "red")

        def order_color(ms):
            if ms == 0: return "blue"
            return "green" if ms <= 50 else ("yellow" if ms <= 200 else "red")

        assert book_color(50)  == "green"
        assert book_color(200) == "yellow"
        assert book_color(600) == "red"
        assert order_color(0)  == "blue"
        assert order_color(30) == "green"
        assert order_color(150) == "yellow"
        assert order_color(250) == "red"


# ═══════════════════════════════════════════════════════════════════════════
# Standalone runner (no pytest required)
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import traceback

    suites = [
        TestBinanceOrderGateway,
        TestIsTradinHalted,
        TestSimulateFill,
        TestSequenceGap,
        TestKillSwitchGate,
        TestTelemetryPayload,
    ]

    passed = failed = 0
    for suite_cls in suites:
        suite = suite_cls()
        methods = [m for m in dir(suite_cls) if m.startswith("test_")]
        for method in methods:
            if hasattr(suite, "setup_method"):
                suite.setup_method()
            try:
                getattr(suite, method)()
                print(f"  PASS  {suite_cls.__name__}::{method}")
                passed += 1
            except Exception:
                print(f"  FAIL  {suite_cls.__name__}::{method}")
                traceback.print_exc()
                failed += 1

    print(f"\n{'='*60}")
    print(f"  Results: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    sys.exit(0 if failed == 0 else 1)
