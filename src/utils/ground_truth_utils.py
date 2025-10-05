import re
import json
import pandas as pd
from itertools import combinations
from pathlib import Path
import logging


def extract_labels(image_paths, out_dir):
    """
    Extract object labels from image filenames (e.g. obj10__5.png -> label 'obj10')
    and save to CSV.
    """
    labels = []
    pattern = re.compile(r'^(obj\d+)_+\d+', re.IGNORECASE)

    for path in image_paths:
        fname = Path(path).stem
        match = pattern.match(fname)
        label = match.group(1) if match else "unknown"
        labels.append(label)

    logging.info(f"Extracted {len(set(labels))} unique labels.")

    df = pd.DataFrame({"path": image_paths, "label": labels})
    csv_path = Path(out_dir) / "image_labels.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")

    logging.info(f"Saved image paths and labels to {csv_path}")
    return csv_path


def generate_ground_truth(csv_path, out_dir):
    """
    Generate ground-truth pairs (same-label image pairs) from image_labels.csv
    and save as JSON. Skip if file already exists.
    """
    json_path = Path(out_dir) / "ground_truth_pairs.json"
    if json_path.exists():
        logging.info(f"Ground-truth file already exists at {json_path}, skipping generation.")
        return json_path

    df = pd.read_csv(csv_path)

    gt_pairs = []
    for label, group in df.groupby("label"):
        paths = group["path"].tolist()
        for a, b in combinations(paths, 2):
            gt_pairs.append([a, b])

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(gt_pairs, f, indent=2)

    logging.info(f"Saved {len(gt_pairs)} ground-truth pairs to {json_path}")
    return json_path
