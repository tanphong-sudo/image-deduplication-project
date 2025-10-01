import cv2
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict


# ---------------------------
# Select representative (sharpness + resolution)
# ---------------------------
def image_sharpness_cv2(path: str):
    if cv2 is None:
        # fallback: use variance of grayscale pixel intensities via PIL
        im = Image.open(path).convert("L")
        arr = np.asarray(im).astype(np.float32)
        return float(np.var(arr))
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0.0
    return float(cv2.Laplacian(img, cv2.CV_64F).var())


def choose_representatives(clusters: List[List[int]], ids: List[str]) -> Dict[int, str]:
    reps = {}
    for cluster_id, members in enumerate(clusters):
        best = None
        best_score = -1.0
        for idx in members:
            p = ids[idx]
            try:
                sharp = image_sharpness_cv2(p)
                with Image.open(p) as im:
                    w, h = im.size
                score = (w * h) * 0.7 + sharp * 1.0  # weighting heuristic
            except Exception:
                score = 0.0
            if score > best_score:
                best_score = score
                best = p
        reps[cluster_id] = best
    return reps