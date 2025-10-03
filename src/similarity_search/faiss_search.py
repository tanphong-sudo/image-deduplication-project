try:
    import faiss
except Exception:
    faiss = None
import numpy as np

# ---------------------------
# FAISS helpers
# ---------------------------
def build_faiss_index(features: np.ndarray, index_path: str = None, index_type: str = "flat", pca_dim: int = None, nlist: int = 1024):
    """
    Build a faiss index.
    index_type: "flat", "ivf", "hnsw", "ivfpq"
    returns index object (and optionally writes to index_path)
    """
    if faiss is None:
        raise RuntimeError("faiss not installed or unavailable")

    d = features.shape[1]
    xb = features.astype(np.float32)
    if index_type == "flat": # exact search
        index = faiss.IndexFlatL2(d)
        index.add(xb)
    elif index_type == "ivf": # approximate search
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist, faiss.METRIC_L2)
        index.train(xb)
        index.add(xb)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(d, 32)  # M=32 default
        index.hnsw.efConstruction = 200
        index.add(xb)
    else:
        raise ValueError(f"Unknown faiss index_type: {index_type}")

    if index_path:
        faiss.write_index(index, index_path)
    return index


def search_faiss_index(index, queries: np.ndarray, k=5, nprobe: int = 10):
    if hasattr(index, "nprobe"):
        index.nprobe = nprobe
    D, I = index.search(queries.astype(np.float32), k)
    return D, I