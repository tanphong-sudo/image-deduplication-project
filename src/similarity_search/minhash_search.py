try:
    from datasketch import MinHash, MinHashLSH
except Exception:
    MinHash = None
    MinHashLSH = None
import numpy as np
import pickle

# ---------------------------
# MinHash LSH helpers
# ---------------------------
def build_index(features: np.ndarray, ids, index_path: str = None, num_perm: int = 128, threshold: float = 0.5):
    """
    Build MinHash LSH index.
    returns (lsh, minhashes) tuple
    """
    if MinHashLSH is None:
        raise RuntimeError("datasketch not installed. pip install datasketch")
    
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    
    for i, (feat, img_id) in enumerate(zip(features, ids)):
        m = MinHash(num_perm=num_perm)
        for j, val in enumerate(feat):
            m.update(f"{j}:{val:.6f}".encode('utf8'))
        minhashes[img_id] = m
        lsh.insert(img_id, m)
    
    if index_path:
        with open(index_path, 'wb') as f:
            pickle.dump({'lsh': lsh, 'minhashes': minhashes, 'ids': ids}, f)
    
    return lsh, minhashes


def search_index(index_path: str, query_features: np.ndarray, k: int = 10):
    """
    Search MinHash LSH index.
    Returns D, I like FAISS.
    """
    with open(index_path, 'rb') as f:
        data = pickle.load(f)
    
    lsh = data['lsh']
    minhashes = data['minhashes']
    ids = data['ids']
    id_to_idx = {img_id: idx for idx, img_id in enumerate(ids)}
    num_perm = list(minhashes.values())[0].hashvalues.size
    
    N = query_features.shape[0]
    D = np.full((N, k), 1.0, dtype=np.float32)
    I = np.full((N, k), -1, dtype=np.int32)
    
    for i, feat in enumerate(query_features):
        m = MinHash(num_perm=num_perm)
        for j, val in enumerate(feat):
            m.update(f"{j}:{val:.6f}".encode('utf8'))
        
        candidates = lsh.query(m)
        similarities = []
        for cand_id in candidates:
            if cand_id in minhashes:
                jaccard = m.jaccard(minhashes[cand_id])
                similarities.append((cand_id, jaccard))
        
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        for j, (cand_id, sim) in enumerate(similarities[:k]):
            I[i, j] = id_to_idx[cand_id]
            D[i, j] = 1.0 - sim
    
    return D, I