
import os
from typing import List, Tuple
import time
import numpy as np
import psutil

# ---------------------------
# Timing helper
# ---------------------------
def measure_time(func, *args, **kwargs):
    """
    Helper để đo thời gian chạy một hàm.
    Trả về: (kết quả, thời_gian_chạy)
    """
    start = time.perf_counter()
    result = func(*args, **kwargs)
    end = time.perf_counter()
    elapsed = end - start
    return result, elapsed

def measure_memory(func, *args, **kwargs):
    process = psutil.Process(os.getpid())
    mem_before = process.memory_info().rss
    result, elapsed = measure_time(func, *args, **kwargs)
    mem_after = process.memory_info().rss
    peak = max(mem_before, mem_after)
    return (result, elapsed), peak / (1024*1024)  # MB

# ---------------------------
# Evaluation helpers (if ground-truth provided)
# ---------------------------
def compute_precision_recall(pred_clusters: List[List[int]], ground_truth_pairs: List[Tuple[str, str]], ids: List[str]):
    """
    ground_truth_pairs: list of (pathA, pathB) which are true duplicates
    This is a simple pairwise precision/recall approximation:
      - For each predicted cluster, produce all pairs inside cluster
      - Count true positives among them
    """
    id_to_idx = {p: i for i, p in enumerate(ids)}
    gt_set = set()
    for a, b in ground_truth_pairs:
        if a in id_to_idx and b in id_to_idx:
            i, j = id_to_idx[a], id_to_idx[b]
            if i < j:
                gt_set.add((i, j))
            else:
                gt_set.add((j, i))
    pred_pairs = set()
    for cl in pred_clusters:
        for i in range(len(cl)):
            for j in range(i + 1, len(cl)):
                a, b = cl[i], cl[j]
                if a < b:
                    pred_pairs.add((a, b))
                else:
                    pred_pairs.add((b, a))
    tp = len(pred_pairs & gt_set)
    fp = len(pred_pairs - gt_set)
    fn = len(gt_set - pred_pairs)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}

# ---------------------------
# Full evaluation pipeline
# ---------------------------
def evaluate_pipeline(pred_clusters, ground_truth_pairs, ids, timings, memory_usage):
    """
    timings: dict chứa tên bước -> thời gian chạy
    memory_usage: dict chứa tên bước -> MB sử dụng
    """
    metrics = compute_precision_recall(pred_clusters, ground_truth_pairs, ids)
    report = {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
        "timings": timings,
        "memory": memory_usage
    }
    return report
