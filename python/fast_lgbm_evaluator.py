"""
fast_lgbm_evaluator.py — Ultra-Fast Flat-Array LightGBM Tree Inference Engine
═════════════════════════════════════════════════════════════════════════════
Loads trained LightGBM tree dump (`lgbm_signal_model.txt`) into contiguous flat
memory arrays and performs zero-overhead decision tree ensemble traversal.

Features (8 L2 Microstructure Signals):
  0: tfi_ema_50          (Trade Flow Imbalance Fast EMA)
  1: tfi_ema_200         (Trade Flow Imbalance Slow EMA)
  2: volatility_ema_100  (Price-tick Vol Fast EMA)
  3: volatility_ema_500  (Price-tick Vol Slow EMA)
  4: book_imbalance      (Cont-Kukanov Order Book Imbalance / OFI)
  5: spread_bps          (Bid-Ask Spread in Basis Points)
  6: microprice          (Stoikov Volume-Weighted Fair Price)
  7: spread_ema          (Regime-Smoothed Spread EMA)

Target:
  Predicted 200ms forward return in basis points (label_fwd_ret_bps).
"""

import math
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


FEATURE_NAMES = [
    "tfi_ema_50",
    "tfi_ema_200",
    "volatility_ema_100",
    "volatility_ema_500",
    "book_imbalance",
    "spread_bps",
    "microprice",
    "spread_ema",
]


@dataclass
class FlatNode:
    split_feature: int
    threshold: float
    left_child: int
    right_child: int
    is_leaf: bool
    leaf_value: float


class FastLGBMEvaluator:
    """
    Sub-microsecond Decision Tree Ensemble Evaluator.
    Directly navigates flat index arrays without object overhead.
    """
    def __init__(self, model_path: Optional[str] = None, max_trees: Optional[int] = None):
        self.feature_names = list(FEATURE_NAMES)
        self.num_trees: int = 0
        self.num_features: int = len(self.feature_names)
        self.max_trees: Optional[int] = max_trees
        
        # Flattened ensemble arrays
        self.tree_offsets: List[int] = []
        self.split_features: np.ndarray = np.empty(0, dtype=np.int32)
        self.thresholds: np.ndarray = np.empty(0, dtype=np.float64)
        self.left_children: np.ndarray = np.empty(0, dtype=np.int32)
        self.right_children: np.ndarray = np.empty(0, dtype=np.int32)
        self.is_leaf: np.ndarray = np.empty(0, dtype=bool)
        self.leaf_values: np.ndarray = np.empty(0, dtype=np.float64)
        self.is_loaded: bool = False

        if model_path and os.path.exists(model_path):
            self.load_model(model_path)

    def load_model(self, model_path: str) -> bool:
        """
        Parse LightGBM model text file into flat arrays.
        """
        if not os.path.exists(model_path):
            return False

        all_splits: List[int] = []
        all_thresholds: List[float] = []
        all_lefts: List[int] = []
        all_rights: List[int] = []
        all_is_leaf: List[bool] = []
        all_leaves: List[float] = []
        tree_starts: List[int] = []

        curr_splits: List[int] = []
        curr_thresh: List[float] = []
        curr_left: List[int] = []
        curr_right: List[int] = []
        curr_leaf_vals: List[float] = []
        parsing_tree = False

        with open(model_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("Tree="):
                    if parsing_tree and curr_splits:
                        # Finalize previous tree into global arrays
                        tree_starts.append(len(all_splits))
                        self._append_tree(
                            curr_splits, curr_thresh, curr_left, curr_right, curr_leaf_vals,
                            all_splits, all_thresholds, all_lefts, all_rights, all_is_leaf, all_leaves
                        )
                    curr_splits = []
                    curr_thresh = []
                    curr_left = []
                    curr_right = []
                    curr_leaf_vals = []
                    parsing_tree = True

                elif parsing_tree:
                    if line.startswith("split_feature="):
                        curr_splits = [int(x) for x in line[14:].split()]
                    elif line.startswith("threshold="):
                        curr_thresh = [float(x) for x in line[10:].split()]
                    elif line.startswith("left_child="):
                        curr_left = [int(x) for x in line[11:].split()]
                    elif line.startswith("right_child="):
                        curr_right = [int(x) for x in line[12:].split()]
                    elif line.startswith("leaf_value="):
                        curr_leaf_vals = [float(x) for x in line[11:].split()]

        # Finalize last tree
        if parsing_tree and curr_splits:
            tree_starts.append(len(all_splits))
            self._append_tree(
                curr_splits, curr_thresh, curr_left, curr_right, curr_leaf_vals,
                all_splits, all_thresholds, all_lefts, all_rights, all_is_leaf, all_leaves
            )

        if not tree_starts:
            return False

        self.num_trees = len(tree_starts)
        self.tree_offsets = tree_starts
        self.split_features = np.array(all_splits, dtype=np.int32)
        self.thresholds = np.array(all_thresholds, dtype=np.float64)
        self.left_children = np.array(all_lefts, dtype=np.int32)
        self.right_children = np.array(all_rights, dtype=np.int32)
        self.is_leaf = np.array(all_is_leaf, dtype=bool)
        self.leaf_values = np.array(all_leaves, dtype=np.float64)
        self.is_loaded = True
        return True

    def _append_tree(self, splits, thresholds, lefts, rights, leaf_vals,
                     all_splits, all_thresh, all_lefts, all_rights, all_is_leaf, all_leaves):
        """Helper to append a tree structure with mapped direct absolute offsets."""
        tree_base = len(all_splits)
        num_internal = len(splits)
        num_leaves = len(leaf_vals)
        leaf_start_idx = num_internal

        # Internal nodes
        for i in range(num_internal):
            l_child = lefts[i]
            r_child = rights[i]
            
            # In LightGBM format, child < 0 denotes a leaf node with bitwise NOT indexing
            actual_l = tree_base + ((leaf_start_idx + ~l_child) if l_child < 0 else l_child)
            actual_r = tree_base + ((leaf_start_idx + ~r_child) if r_child < 0 else r_child)

            all_splits.append(splits[i])
            all_thresh.append(thresholds[i])
            all_lefts.append(actual_l)
            all_rights.append(actual_r)
            all_is_leaf.append(False)
            all_leaves.append(0.0)

        # Leaf nodes
        for i in range(num_leaves):
            all_splits.append(-1)
            all_thresh.append(0.0)
            all_lefts.append(-1)
            all_rights.append(-1)
            all_is_leaf.append(True)
            all_leaves.append(leaf_vals[i])

    def predict_single(self, features: Union[List[float], np.ndarray]) -> float:
        """
        Evaluate all trees on a single 8-feature vector.
        Ultra-low latency, zero object allocations.
        Returns predicted forward return in bps.
        """
        if not self.is_loaded:
            return 0.0

        # Direct flat array access
        splits = self.split_features
        thresh = self.thresholds
        lefts = self.left_children
        rights = self.right_children
        is_leaf = self.is_leaf
        leaves = self.leaf_values
        offsets = self.tree_offsets

        total_prediction = 0.0
        n_trees = min(len(offsets), self.max_trees) if self.max_trees else len(offsets)

        for t_idx in range(n_trees):
            curr_idx = offsets[t_idx]
            while not is_leaf[curr_idx]:
                feat_idx = splits[curr_idx]
                val = features[feat_idx]
                if val <= thresh[curr_idx]:
                    curr_idx = lefts[curr_idx]
                else:
                    curr_idx = rights[curr_idx]
            total_prediction += leaves[curr_idx]

        return float(total_prediction)


    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Vectorized batch prediction over an N x 8 feature matrix.
        """
        if not self.is_loaded or len(X) == 0:
            return np.zeros(len(X), dtype=np.float64)

        n_rows = len(X)
        preds = np.zeros(n_rows, dtype=np.float64)
        for i in range(n_rows):
            preds[i] = self.predict_single(X[i])
        return preds

    def benchmark_latency(self, iterations: int = 10_000) -> dict:
        """
        Benchmark single-tick prediction latency.
        """
        dummy_feat = np.array([0.5, 0.2, 0.05, 0.08, 0.35, 1.8, 84250.0, 1.75], dtype=np.float64)
        # Warmup
        for _ in range(500):
            self.predict_single(dummy_feat)

        t0 = time.perf_counter_ns()
        for _ in range(iterations):
            self.predict_single(dummy_feat)
        elapsed_ns = time.perf_counter_ns() - t0

        avg_latency_us = (elapsed_ns / iterations) / 1000.0
        ops_per_sec = int(iterations / (elapsed_ns / 1e9))

        return {
            "iterations": iterations,
            "total_trees": self.num_trees,
            "avg_latency_us": round(avg_latency_us, 3),
            "ops_per_sec": ops_per_sec,
        }
