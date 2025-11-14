from pathlib import Path
from typing import Dict, List, Tuple

import json
import random


def stratified_subject_folds(
    subject_to_label: Dict[str, int], k: int = 5, seed: int = 42
) -> List[Tuple[List[str], List[str]]]:
    """
    Create k stratified folds of subjects preserving label balance.
    Returns list of (train_subjects, val_subjects) tuples.
    """
    rng = random.Random(seed)
    by_label: Dict[int, List[str]] = {}
    for sid, y in subject_to_label.items():
        by_label.setdefault(y, []).append(sid)
    for lst in by_label.values():
        rng.shuffle(lst)
    folds = [([], []) for _ in range(k)]
    # Round-robin assign subjects per class to val folds
    for y, lst in by_label.items():
        for i, sid in enumerate(lst):
            val_idx = i % k
            for fidx in range(k):
                if fidx == val_idx:
                    folds[fidx][1].append(sid)
                else:
                    folds[fidx][0].append(sid)
    return folds


def save_folds(folds: List[Tuple[List[str], List[str]]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (train_s, val_s) in enumerate(folds):
        obj = {"train_subjects": train_s, "val_subjects": val_s}
        (out_dir / f"fold_{i+1}.json").write_text(json.dumps(obj, indent=2))


