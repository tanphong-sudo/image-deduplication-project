#!/usr/bin/env python3
"""
run_pipeline.py
Skeleton pipeline orchestrator for image deduplication project.

Usage examples:
  python run_pipeline.py --dataset /path/to/images --extractor resnet --method faiss --build-index --use-gpu
  python run_pipeline.py --dataset /path/to/images --method phash --phash-threshold 8
  python run_pipeline.py --dataset /path/to/images --extractor vit --method simhash --simhash-bits 64 --hamming-threshold 10
"""
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import argparse
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
    parser = argparse.ArgumentParser(description="Run image deduplication pipeline.")
    parser.add_argument("--dataset", required=True, help="Path to image folder")
    parser.add_argument("--out-dir", default="output", help="Output folder to store features/indices/results")
    parser.add_argument("--extractor", choices=["resnet", "vit", "efficientnet", "convnexttiny"], default="resnet",
                        help="Which feature extractor to use")
    parser.add_argument("--method", choices=["exact", "faiss", "simhash", "minhash"], default="faiss",
                        help="Which duplicate detection / search method to run")
    parser.add_argument("--build-index", action="store_true", help="Build index (if method is faiss/simhash)")
    parser.add_argument("--load-features", default=None, help="Path to npy file of features to load (skip extract)")
    parser.add_argument("--save-features", default=None, help="Path to save features (npy)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--use-gpu", action="store_true")
    parser.add_argument("--k", type=int, default=5, help="k for kNN search")
    parser.add_argument("--threshold", type=float, default=None, help="distance threshold for clustering (L2 for FAISS)")
    parser.add_argument("--nlist", type=int, default=1024, help="FAISS nlist (if IVF)")
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="flat")
    parser.add_argument("--simhash-bits", type=int, default=64)
    parser.add_argument("--hamming-threshold", type=int, default=10)
    parser.add_argument("--pca-dim", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. list images
    image_paths = list_images_recursive(args.dataset)
    if len(image_paths) == 0:
        logging.error("No images found in dataset")
        sys.exit(1)
    logging.info(f"Found {len(image_paths)} images")

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
        if args.build_index:
            logging.info(f"Building FAISS index type={args.index_type} nlist={args.nlist}")
            index = build_faiss_index(features, index_type=args.index_type, nlist=args.nlist)
        else:
            if index_path.exists():
                index = faiss.read_index(str(index_path))
            else:
                logging.error("Index not found. Use --build-index")
                sys.exit(1)

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
            searcher = SimHashSearch(dim=dim, num_bits=args.simhash_bits, num_tables=4)
            searcher.add_batch(features, np.arange(len(features)))

            logging.info("Running SimHash search...")
            N = len(features)
            k = args.k
            I = np.full((N, k), -1, dtype=int)
            D = np.full((N, k), np.inf, dtype=float)

            for i, vec in enumerate(features):
                neighbors = searcher.query(vec, k=k)
                for j, (nid, dist) in enumerate(neighbors):
                    I[i, j] = nid
                    D[i, j] = dist

            clusters_idx = cluster_from_knn(I, D, threshold=args.hamming_threshold)
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
            # expected interface similar to simhash wrapper
            if args.build_index:
                mm.build_index(features, ids, str(Path(out_dir) / "minhash_index.bin"))
            D, I = mm.search_index(str(Path(out_dir) / "minhash_index.bin"), features, k=args.k)
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
    for rep in reps:
        try:
            rep_path = ids[rep] if isinstance(rep, int) and rep < len(ids) else rep
            dst = processed_dir / Path(rep_path).name
            shutil.copy(rep_path, dst)
        except Exception as e:
            logging.warning(f"Failed to copy representative {rep}: {e}")

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

if __name__ == "__main__":
    main()
