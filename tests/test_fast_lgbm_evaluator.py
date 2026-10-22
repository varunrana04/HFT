"""
test_fast_lgbm_evaluator.py — Unit Tests for Flat-Array LightGBM Evaluator
"""
import os
import sys
from pathlib import Path
import numpy as np
import pytest

_THIS_DIR = Path(__file__).resolve().parent
_SRC_DIR = _THIS_DIR.parent / "python"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from fast_lgbm_evaluator import FastLGBMEvaluator


def test_fast_lgbm_load_and_predict():
    model_path = _SRC_DIR / "models" / "lgbm_signal_model.txt"
    assert model_path.exists(), f"Model file missing: {model_path}"

    evaluator = FastLGBMEvaluator(str(model_path))
    assert evaluator.is_loaded
    assert evaluator.num_trees > 0
    print(f"Loaded {evaluator.num_trees} trees successfully.")

    # 8 features: tfi_50, tfi_200, vol_100, vol_500, book_imbalance, spread_bps, microprice, spread_ema
    sample_feat = np.array([0.5, 0.2, 0.05, 0.08, 0.35, 1.8, 84250.0, 1.75], dtype=np.float64)
    pred_bps = evaluator.predict_single(sample_feat)
    
    assert isinstance(pred_bps, float)
    assert not np.isnan(pred_bps)
    assert not np.isinf(pred_bps)
    print(f"Predicted forward return: {pred_bps:.4f} bps")


def test_fast_lgbm_batch_prediction():
    model_path = _SRC_DIR / "models" / "lgbm_signal_model.txt"
    evaluator = FastLGBMEvaluator(str(model_path))
    
    X = np.random.randn(20, 8).astype(np.float64)
    X[:, 5] = np.abs(X[:, 5]) + 1.0  # positive spread
    X[:, 6] = 84000.0 + X[:, 6] * 100.0  # realistic microprice
    
    batch_preds = evaluator.predict_batch(X)
    assert len(batch_preds) == 20
    assert not np.any(np.isnan(batch_preds))


def test_fast_lgbm_latency_benchmark():
    model_path = _SRC_DIR / "models" / "lgbm_signal_model.txt"
    # Full ensemble (1600 trees)
    evaluator_full = FastLGBMEvaluator(str(model_path))
    bench_full = evaluator_full.benchmark_latency(iterations=100)
    assert bench_full["total_trees"] == 1600
    assert bench_full["ops_per_sec"] > 0
    print(f"Full 1600 trees: {bench_full['avg_latency_us']} µs/tick | {bench_full['ops_per_sec']:,} ops/sec")

    # Fast pruned sub-ensemble (top 50 trees for ultra-low latency)
    evaluator_fast = FastLGBMEvaluator(str(model_path), max_trees=50)
    bench_fast = evaluator_fast.benchmark_latency(iterations=1000)
    assert bench_fast["ops_per_sec"] > 1000
    print(f"Fast 50 trees: {bench_fast['avg_latency_us']} µs/tick | {bench_fast['ops_per_sec']:,} ops/sec")
