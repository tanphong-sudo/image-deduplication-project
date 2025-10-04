from pathlib import Path
import os
import hashlib
from typing import List, Tuple, Dict

# ---------------------------
# Helpers: IO, hashes, utils
# ---------------------------
def list_images_recursive(dataset_dir: str, exts=(".jpg", ".jpeg", ".png", ".bmp", ".webp")) -> List[str]:
    p = Path(dataset_dir)
    files = [str(f) for f in p.rglob("*") if f.suffix.lower() in exts]
    files.sort()
    return files


def md5_of_file(path: str, block_size=65536) -> str:
    h = hashlib.md5() 
    with open(path, "rb") as f: # open file in binary mode
        for block in iter(lambda: f.read(block_size), b""): # read file in chunks
            h.update(block)
    return h.hexdigest() 


def group_exact_duplicates(paths: List[str]) -> Dict[str, List[str]]:
    """Return mapping md5->list(paths)."""
    d = {}
    for p in paths:
        key = md5_of_file(p)
        d.setdefault(key, []).append(p)
    return {k: v for k, v in d.items() if len(v) > 1} # only keep duplicates