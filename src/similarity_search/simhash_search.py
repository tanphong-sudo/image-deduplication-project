
import numpy as np
from typing import List, Tuple, Dict, Optional
import sys
import os

# Import C++ module
try:
    # Add parent directory to path for imports
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
    from lsh_cpp_module import SimHashLSH as SimHashLSH_CPP, Simhash as Simhash_CPP
    CPP_AVAILABLE = True
except ImportError as e:
    CPP_AVAILABLE = False
    _error_msg = (
        f"\n{'='*70}\n"
        f"ERROR: C++ LSH module not available: {e}\n"
        f"{'='*70}\n"
        f"Please compile the module:\n"
        f"  cd src/lsh_cpp_module\n"
        f"  python setup.py build_ext --inplace\n"
        f"{'='*70}\n"
    )
    raise ImportError(_error_msg)


class SimHashSearch:
    """
    SimHash-based similarity search for image deduplication.
    
    This is the main interface for using SimHash LSH in the project.
    Uses the high-performance C++ implementation via Pybind11.
    
    Examples
    --------
    >>> # Initialize search index
    >>> searcher = SimHashSearch(dim=512, num_bits=64, num_tables=4)
    >>> 
    >>> # Add image features
    >>> features = np.random.randn(1000, 512).astype('float32')
    >>> ids = np.arange(1000)
    >>> searcher.add_batch(features, ids)
    >>> 
    >>> # Find similar images
    >>> query = features[0]
    >>> results = searcher.query(query, k=10)
    """
    
    def __init__(self, dim: int, num_bits: int = 64, num_tables: int = 4):
        """
        Initialize SimHash search index using C++ backend.
        
        Parameters
        ----------
        dim : int
            Dimensionality of feature vectors
        num_bits : int, default=64
            Number of bits in hash signature
        num_tables : int, default=4
            Number of hash tables (more tables = higher recall)
        """
        if not CPP_AVAILABLE:
            raise RuntimeError("C++ module not available. Please compile it first.")
            
        self.dim = dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        
        # Use C++ implementation
        self.backend = SimHashLSH_CPP(dim, num_bits, num_tables)
        print(f"✓ Using C++ SimHash LSH (dim={dim}, bits={num_bits}, tables={num_tables})")
    
    def add(self, vector: np.ndarray, id: int):
        """
        Add a single vector to the index.
        
        Parameters
        ----------
        vector : np.ndarray
            Feature vector (1D array)
        id : int
            Unique identifier for this vector
        """
        self.backend.add(vector, id)
    
    def add_batch(self, vectors: np.ndarray, ids: np.ndarray):
        """
        Add multiple vectors to the index (batch operation - faster).
        
        Parameters
        ----------
        vectors : np.ndarray
            Feature vectors (2D array: n_samples x n_features)
        ids : np.ndarray
            Unique identifiers (1D array: n_samples)
        """
        if not isinstance(ids, np.ndarray):
            ids = np.array(ids, dtype=np.int32)
        
        self.backend.add_batch(vectors, ids)
    
    def query(self, query_vector: np.ndarray, k: int = 10,
              max_candidates: int = 1000, hamming_threshold: int = 0) -> List[Tuple[int, float]]:
        """
        Find k nearest neighbors.
        
        Parameters
        ----------
        query_vector : np.ndarray
            Query feature vector (1D array)
        k : int, default=10
            Number of nearest neighbors to return
        max_candidates : int, default=1000
            Maximum number of candidates to examine
        hamming_threshold : int, default=0
            Hamming distance threshold for multi-probing (0 = exact hash match only)
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (id, distance) tuples, sorted by distance
        """
        return self.backend.query(query_vector, k, max_candidates, hamming_threshold)
    
    def query_radius(self, query_vector: np.ndarray, threshold: float,
                     max_candidates: int = 2000, hamming_threshold: int = 0) -> List[Tuple[int, float]]:
        """
        Find all neighbors within a distance threshold.
        
        Parameters
        ----------
        query_vector : np.ndarray
            Query feature vector
        threshold : float
            Maximum distance threshold
        max_candidates : int, default=2000
            Maximum number of candidates to examine
        hamming_threshold : int, default=0
            Hamming distance threshold for multi-probing (0 = exact hash match only)
            
        Returns
        -------
        List[Tuple[int, float]]
            List of (id, distance) tuples where distance <= threshold
        """
        return self.backend.query_radius(query_vector, threshold, max_candidates, hamming_threshold)
    
    def get_stats(self) -> Dict[str, int]:
        """
        Get statistics about the index.
        
        Returns
        -------
        Dict[str, int]
            Dictionary containing index statistics
        """
        return self.backend.get_stats()
    
    def clear(self):
        """Clear all data from the index."""
        self.backend.clear()
    
    def __repr__(self):
        stats = self.get_stats()
        return (f"<SimHashSearch backend=C++ "
                f"vectors={stats.get('num_vectors', 0)} "
                f"dim={self.dim} bits={self.num_bits} tables={self.num_tables}>")
