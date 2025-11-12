#!/usr/bin/env python3
"""
Image Deduplication Pipeline

A complete pipeline for detecting and removing duplicate images using deep learning
feature extraction and similarity search algorithms.

Usage:
    Basic (recommended defaults):
        python run_pipeline.py --dataset data/raw
        
    Custom configuration:
        python run_pipeline.py --dataset data/raw --extractor resnet --method faiss --threshold 30
        python run_pipeline.py --dataset data/raw --method simhash --hamming-threshold 5
        
    For detailed help:
        python run_pipeline.py --help

Default behavior:
    - Feature extractor: EfficientNet-B0 (fast, accurate)
    - Search method: FAISS (exact nearest neighbor)
    - Clustering threshold: 50 (Euclidean distance)
"""
import os
import sys

# Fix for macOS PyTorch segfault - MUST be set before importing torch
if sys.platform == "darwin":
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"  # Fix OpenMP duplicate library error

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import re
import argparse
import pandas as pd
import shutil
import time
import hashlib
import json
import logging
from pathlib import Path
from typing import List, Tuple, Dict
import numpy as np
from PIL import Image
from src.evaluator import evaluate_pipeline, measure_memory, measure_time
import imagehash
from src.similarity_search.faiss_search import build_faiss_index, search_faiss_index
from src.utils.image_utils import choose_representatives
from src.utils.io_utils import group_exact_duplicates, list_images_recursive
from src.utils.ground_truth_utils import extract_labels, generate_ground_truth


# Optional heavy deps
try:
    import faiss
except Exception:
    faiss = None

try:
    import cv2
except Exception:
    cv2 = None

try:
    import psutil
except Exception:
    psutil = None

# Dynamic import helpers
import importlib
import threading

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------
# Helper: run function and sample peak RSS + time
# ---------------------------
def run_with_time_and_peak_rss(fn, *args, sample_interval: float = 0.01, **kwargs):
    """
    Run fn(*args, **kwargs) while sampling process RSS every sample_interval seconds.
    Returns: (result, elapsed_seconds, peak_rss_in_MB).
    If psutil is not installed, returns mem = 0.0.
    """
    # If psutil isn't available, just time the call and return mem=0.0
    if psutil is None:
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        elapsed = time.perf_counter() - start
        return result, elapsed, 0.0

    samples = []
    stop_event = threading.Event()
    proc = psutil.Process(os.getpid())

    def sampler():
        try:
            while not stop_event.is_set():
                try:
                    samples.append(proc.memory_info().rss)
                except Exception:
                    samples.append(0)
                stop_event.wait(sample_interval)
        except Exception:
            # ensure sampler never crashes the main thread
            samples.append(0)

    th = threading.Thread(target=sampler, daemon=True)
    th.start()

    start = time.perf_counter()
    try:
        result = fn(*args, **kwargs)
    finally:
        elapsed = time.perf_counter() - start
        stop_event.set()
        th.join()

    # If no samples were recorded, read current RSS as fallback
    peak = max(samples) if samples else proc.memory_info().rss
    peak_mb = peak / (1024 * 1024)
    return result, elapsed, peak_mb



# ---------------------------
# Feature extraction wrapper
# ---------------------------
def get_extractor_instance(name: str, device: str = "cpu", **kwargs):
    """
    Dynamically import extractor class from src.feature_extraction.<name>_extractor
    and instantiate it. The exact class name convention expected:
      - resnet_extractor.py  -> class ResNetExtractor
      - vit_extractor.py     -> class ViTExtractor
      - efficientnet_extractor.py -> class EfficientNetExtractor
    You can adapt the mapping below to your actual class names.
    """
    mapping = {
        "resnet": ("feature_extraction.resnet_extractor", "ResNetExtractor"),
        "vit": ("feature_extraction.vit_extractor", "ViTExtractor"),
        "efficientnet": ("feature_extraction.efficientnet_extractor", "EfficientNetExtractor"),
        "convnexttiny": ("feature_extraction.convnexttiny_extractor", "ConvNeXtTinyExtractor"),
        # add more mappings if you have them
    }
    if name not in mapping:
        raise ValueError(f"Unknown extractor: {name}. Available: {list(mapping.keys())}")
    module_name, class_name = mapping[name]
    try:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        inst = cls(device=device, **kwargs)  # many extractors accept device and config
        return inst
    except Exception as e:
        logging.exception(f"Failed to import extractor {name} ({module_name}.{class_name}): {e}")
        raise

def extract_features_with_extractor(extractor, image_paths: List[str], batch_size: int = 32, out_features=None):
    from math import ceil
    from PIL import Image
    import torch
    
    # Fix for macOS segfault: disable MKL threading
    if sys.platform == "darwin":
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["OMP_NUM_THREADS"] = "1"
        torch.set_num_threads(1)

    features_list = []
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        imgs = [Image.open(p).convert("RGB") for p in batch_paths]
        feats = extractor.extract_features_batch(imgs)  # <-- gọi đúng API BaseExtractor
        features_list.append(feats)

    features = np.vstack(features_list)
    if out_features:
        outdir = os.path.dirname(out_features) or "."
        os.makedirs(outdir, exist_ok=True)
        np.save(out_features, features)
    return features, image_paths


# ---------------------------
# Clustering via union-find on kNN pairs
# ---------------------------
def cluster_from_knn(I: np.ndarray, D: np.ndarray, threshold: float) -> List[List[int]]:
    """I, D shape (N, k). cluster indices where distance < threshold (single-link)."""
    N, k = I.shape
    parent = list(range(N))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(N):
        for neigh_idx, dist in zip(I[i], D[i]):
            if neigh_idx == -1:
                continue
            if dist <= threshold:
                union(i, neigh_idx)

    clusters = {}
    for i in range(N):
        r = find(i)
        clusters.setdefault(r, []).append(i)
    return list(clusters.values())


# ---------------------------
# Main orchestration
# ---------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Image Deduplication Pipeline - Find and remove duplicate images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Basic usage (recommended defaults):
    python run_pipeline.py --dataset data/raw
    
  Using ResNet50 feature extractor:
    python run_pipeline.py --dataset data/raw --extractor resnet --threshold 30
    
  Using SimHash LSH (fast, approximate):
    python run_pipeline.py --dataset data/raw --method simhash --hamming-threshold 5
    
  Load pre-extracted features:
    python run_pipeline.py --dataset data/raw --load-features features.npy --build-index

Output:
  - data/processed/image_labels.csv: Image paths with labels
  - data/processed/ground_truth_pairs.json: True duplicate pairs
  - data/processed/features.npy: Extracted feature vectors
  - data/processed/evaluation_full.json: Performance metrics
  - data/processed/<representatives>: One image per cluster

For more information, visit: https://github.com/tanphong-sudo/image-deduplication-project
        """
    )
    
    parser.add_argument("--dataset", default="data/raw", 
                        help="Path to image folder (default: data/raw)")
    parser.add_argument("--out-dir", default="data/processed", 
                        help="Output folder (default: data/processed)")
    
    parser.add_argument("--extractor", choices=["resnet", "vit", "efficientnet", "convnexttiny"], 
                        default="efficientnet",
                        help="Feature extractor: efficientnet (default, fast), resnet (accurate), vit (modern)")
    
    parser.add_argument("--method", choices=["exact", "faiss", "simhash", "minhash"], 
                        default="faiss",
                        help="Search method: faiss (default, exact), simhash (fast, approximate)")
    
    parser.add_argument("--build-index", action="store_true", 
                        help="Build search index (auto-enabled for faiss/simhash)")
    parser.add_argument("--load-features", default=None, 
                        help="Load features from .npy file (skip extraction)")
    parser.add_argument("--save-features", default=None, 
                        help="Save extracted features to .npy file")
    
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for feature extraction (default: 16)")
    parser.add_argument("--use-gpu", action="store_true",
                        help="Use GPU for feature extraction (if available)")
    
    parser.add_argument("--k", type=int, default=50, 
                        help="Number of nearest neighbors (default: 50)")
    parser.add_argument("--threshold", type=float, default=50.0, 
                        help="Distance threshold for clustering (default: 50)")
    
    parser.add_argument("--nlist", type=int, default=1024, 
                        help="FAISS IVF clusters (default: 1024)")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="flat",
                        help="FAISS index type (default: flat)")
    
    parser.add_argument("--simhash-bits", type=int, default=64,
                        help="SimHash bits (default: 64)")
    parser.add_argument("--hamming-threshold", type=int, default=5,
                        help="SimHash Hamming threshold (default: 5, try 5-6 for best recall)")
    
    parser.add_argument("--pca-dim", type=int, default=None,
                        help="PCA dimension reduction (optional)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    
    args = parser.parse_args()
    
    print("\n" + "="*70)
    print("  IMAGE DEDUPLICATION PIPELINE")
    print("="*70)
    print(f" Dataset:            {args.dataset}")
    print(f" Feature Extractor:  {args.extractor}")
    print(f" Search Method:      {args.method}")
    print(f" Threshold:          {args.threshold}")
    if args.method == "simhash":
        print(f" Hamming Threshold:  {args.hamming_threshold}")
    print(f" Output Directory:   {args.out_dir}")
    print("="*70 + "\n")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. list images
    image_paths = list_images_recursive(args.dataset)
    if len(image_paths) == 0:
        logging.error("No images found in dataset")
        sys.exit(1)
    logging.info(f"Found {len(image_paths)} images")

    # extract labels from filenames and generate ground-truth.json
    csv_path = extract_labels(image_paths, out_dir)
    gt_path = generate_ground_truth(csv_path, out_dir)
    

    # 2. exact duplicate step
    if args.method == "exact":
        groups = group_exact_duplicates(image_paths)
        logging.info(f"Found {len(groups)} exact duplicate groups")
        # save result
        with open(out_dir / "exact_duplicates.json", "w") as f:
            json.dump(groups, f, indent=2)
        return

    # For methods requiring features (faiss / simhash / minhash) we need features
    timings = {}
    memory_usage = {}
    if args.load_features:
        logging.info(f"Loading features from {args.load_features}")
        features = np.load(args.load_features)
        ids = list(np.loadtxt(str(Path(args.load_features).with_suffix(".ids.txt")), dtype=str))
        timings["feature_extraction"] = 0.0
        memory_usage["feature_extraction"] = 0.0
    else:
        # instantiate extractor
        device = "cuda" if args.use_gpu else "cpu"
        logging.info(f"Using extractor={args.extractor} device={device}")
        # đo memory + time cho feature extraction
        (result, elapsed, mem_mb) = run_with_time_and_peak_rss(
            extract_features_with_extractor, get_extractor_instance(args.extractor, device=device), image_paths, args.batch_size, args.save_features
        )
        features, ids = result
        timings["feature_extraction"] = elapsed
        memory_usage["feature_extraction"] = mem_mb

        # save ids file
        ids_path = out_dir / "features.ids.txt"
        with open(ids_path, "w") as f:
            for p in ids:
                f.write(p + "\n")

    # match dtype
    features = features.astype(np.float32)
    N, d = features.shape
    logging.info(f"Features loaded: N={N}, d={d}")

    # 4. build index (FAISS) or simhash/minhash
    clusters_idx = []
    reps = []
    if args.method == "faiss":
        if faiss is None:
            logging.error("faiss not available. install faiss-cpu or faiss-gpu")
            sys.exit(1)
        index_path = out_dir / f"faiss_index_{args.index_type}.index"
        
        if args.build_index or not index_path.exists():
            logging.info(f"Building FAISS index type={args.index_type} nlist={args.nlist}")
            index = build_faiss_index(features, index_type=args.index_type, nlist=args.nlist)
        else:
            logging.info(f"Loading existing FAISS index from {index_path}")
            index = faiss.read_index(str(index_path))

        k = args.k
        logging.info("Running kNN self-search (FAISS)")
        (res, elapsed, mem_mb) = run_with_time_and_peak_rss(search_faiss_index, index, features, k, 10)
        D, I = res
        timings["faiss_search"] = elapsed
        memory_usage["faiss_search"] = mem_mb

        if args.threshold is None:
            med = float(np.median(D[:, 1])) if D.shape[1] > 1 else float(np.median(D[:, 0]))
            args.threshold = med * 0.5
            logging.info(f"No threshold provided: using heuristic threshold={args.threshold:.4f}")
        clusters_idx = cluster_from_knn(I, D, threshold=args.threshold)
        clusters = [[ids[i] for i in cl] for cl in clusters_idx]
        reps = choose_representatives(clusters_idx, ids)
        with open(out_dir / "faiss_clusters.json", "w") as f:
            json.dump({"clusters": clusters, "representatives": reps}, f, indent=2)
        logging.info(f"Saved clusters (count={len(clusters)}) to faiss_clusters.json")

    elif args.method == "simhash":
        try:
            from src.similarity_search.simhash_search import SimHashSearch
            logging.info("Using SimHashSearch (C++ backend via Pybind11)")

            dim = features.shape[1]
            num_tables = 8
            searcher = SimHashSearch(dim=dim, num_bits=args.simhash_bits, num_tables=num_tables)
            searcher.add_batch(features, np.arange(len(features)))

            logging.info(f"Running SimHash search (tables={num_tables}, hamming_threshold={args.hamming_threshold})...")
            def _simhash_search():
                N = len(features)
                k = args.k
                I = np.full((N, k), -1, dtype=int)
                D = np.full((N, k), np.inf, dtype=float)

                for i, vec in enumerate(features):
                    neighbors = searcher.query(vec, k=k, max_candidates=2000, hamming_threshold=args.hamming_threshold)
                    for j, (nid, dist) in enumerate(neighbors):
                        I[i, j] = nid
                        D[i, j] = dist
                return D, I

            (res, elapsed, mem_mb) = run_with_time_and_peak_rss(_simhash_search)
            D, I = res
            timings["simhash_search"] = elapsed
            memory_usage["simhash_search"] = mem_mb

            # Use --threshold if provided, otherwise use --hamming-threshold
            threshold = args.threshold if args.threshold is not None else args.hamming_threshold
            logging.info(f"Clustering with threshold={threshold}")
            clusters_idx = cluster_from_knn(I, D, threshold=threshold)
            clusters = [[ids[i] for i in cl] for cl in clusters_idx]
            reps = choose_representatives(clusters_idx, ids)
            with open(out_dir / "simhash_clusters.json", "w") as f:
                json.dump({"clusters": clusters, "representatives": reps}, f, indent=2)
            logging.info(f"Saved simhash clusters ({len(clusters)})")
        except Exception as e:
            logging.exception("SimHash failed: %s", e)
            sys.exit(1)

    elif args.method == "minhash":
        try:
            mm = importlib.import_module("src.similarity_search.minhash_search")
            index_path = Path(out_dir) / "minhash_index.bin"
            
            # Build index if needed
            if args.build_index or not index_path.exists():
                logging.info("Building MinHash index...")
                mm.build_index(features, ids, str(index_path))
            
            def _minhash_search():
                return mm.search_index(str(index_path), features, k=args.k)

            (res, elapsed, mem_mb) = run_with_time_and_peak_rss(_minhash_search)
            D, I = res
            timings["minhash_search"] = elapsed
            memory_usage["minhash_search"] = mem_mb

            med = np.median(D[D > 0])
            args.threshold = med * 0.75
            logging.info(f"Auto threshold based on median distance = {args.threshold:.3f}")

            clusters_idx = cluster_from_knn(I, D, threshold=args.threshold if args.threshold is not None else 0.3)
            clusters = [[ids[i] for i in cl] for cl in clusters_idx]
            reps = choose_representatives(clusters_idx, ids)
            with open(out_dir / "minhash_clusters.json", "w") as f:
                json.dump({"clusters": clusters, "representatives": reps}, f, indent=2)
            logging.info("Saved minhash clusters")
        except Exception as e:
            logging.exception("MinHash module failed: %s", e)
            sys.exit(1)
    else:
        logging.error("Unsupported method (should be handled above)")
        sys.exit(1)
    
     # copy representatives outward for inspection
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    # reps is Dict[cluster_id, image_path]
    for cluster_id, rep_path in reps.items():
        try:
            if os.path.exists(rep_path):
                dst = processed_dir / Path(rep_path).name
                shutil.copy(rep_path, dst)
                logging.info(f"Copied representative for cluster {cluster_id}: {Path(rep_path).name}")
            else:
                logging.warning(f"Representative file not found: {rep_path}")
        except Exception as e:
            logging.warning(f"Failed to copy representative {rep_path}: {e}")

    # optionally, compute evaluation if you have ground-truth pairs
    gt_path = Path(args.out_dir) / "ground_truth_pairs.json"
    if gt_path.exists():
        with open(gt_path, "r") as f:
            gt_pairs = json.load(f)
        gt_pairs = [(a, b) for a, b in gt_pairs]
    else:
        gt_pairs = []

    report = evaluate_pipeline(clusters_idx, gt_pairs, ids, timings, memory_usage)

    with open(out_dir / "evaluation_full.json", "w") as f:
        json.dump(report, f, indent=2)

    logging.info(f"Full evaluation saved to {out_dir}/evaluation_full.json")
    logging.info("Pipeline finished.")
    
    # Display results using view_results.py
    try:
        import subprocess
        view_results_path = Path(__file__).parent / "view_results.py"
        if view_results_path.exists():
            subprocess.run([sys.executable, str(view_results_path)], check=False)
        else:
            logging.warning("view_results.py not found, skipping results display")
    except Exception as e:
        logging.warning(f"Could not display results: {e}")

if __name__ == "__main__":
    main()
