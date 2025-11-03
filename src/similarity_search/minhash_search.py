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
def build_index(features: np.ndarray, ids, index_path: str = None, num_perm: int = 128, threshold: float = 0.3):
    """
    Build MinHash LSH index.
    Uses top-k feature dimensions as discrete set for MinHash.
    """
    if MinHashLSH is None:
        raise RuntimeError("datasketch not installed. pip install datasketch")
    
    # Lower threshold for better recall with dense vectors
    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    minhashes = {}
    
    # Convert dense vectors to sets: use top 10% dimensions
    top_k = max(10, int(features.shape[1] * 0.1))
    
    for i, (feat, img_id) in enumerate(zip(features, ids)):
        m = MinHash(num_perm=num_perm)
        # Get top-k dimension indices
        top_indices = np.argsort(np.abs(feat))[-top_k:]
        for idx in top_indices:
            m.update(str(int(idx)).encode('utf8'))
        minhashes[img_id] = m
        lsh.insert(img_id, m)
    
    if index_path:
        with open(index_path, 'wb') as f:
            pickle.dump({'lsh': lsh, 'minhashes': minhashes, 'ids': ids, 'top_k': top_k}, f)
    
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
    top_k = data.get('top_k', max(10, int(query_features.shape[1] * 0.1)))
    id_to_idx = {img_id: idx for idx, img_id in enumerate(ids)}
    num_perm = list(minhashes.values())[0].hashvalues.size
    
    N = query_features.shape[0]
    D = np.full((N, k), 1.0, dtype=np.float32)
    I = np.full((N, k), -1, dtype=np.int32)
    
    for i, feat in enumerate(query_features):
        m = MinHash(num_perm=num_perm)
        # Get top-k dimensions for query
        top_indices = np.argsort(np.abs(feat))[-top_k:]
        for idx in top_indices:
            m.update(str(int(idx)).encode('utf8'))
        
        # Query LSH - this returns candidates with Jaccard > threshold
        candidates = lsh.query(m)
        
        # If no candidates from LSH, compute all similarities (fallback)
        if len(candidates) == 0:
            similarities = []
            for cand_id in list(minhashes.keys())[:100]:  # Top 100 to avoid too slow
                jaccard = m.jaccard(minhashes[cand_id])
                similarities.append((cand_id, jaccard))
        else:
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