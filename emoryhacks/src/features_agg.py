from typing import Dict

import numpy as np


def aggregate_statistics(frame_features: Dict[str, np.ndarray]) -> Dict[str, float]:
    """
    Aggregate frame-level features to compact ML-ready statistics (~44 dims target):
    For each multivariate stream, compute mean and std per dimension.
    """
    stats: Dict[str, float] = {}
    for key, arr in frame_features.items():
        if arr.ndim == 1:
            arr = arr.reshape(-1, 1)
        # robust guard
        if arr.size == 0:
            continue
        mean = np.mean(arr, axis=0)
        std = np.std(arr, axis=0)
        for i, (m, s) in enumerate(zip(mean, std)):
            stats[f"{key}_mean_{i}"] = float(m)
            stats[f"{key}_std_{i}"] = float(s)
    return stats


def aggregate_all(
    frame_features: Dict[str, np.ndarray],
    global_features: Dict[str, float],
) -> Dict[str, float]:
    """
    Merge aggregated frame statistics with global engineered features.
    """
    stats = aggregate_statistics(frame_features)
    stats.update(global_features)
    return stats

