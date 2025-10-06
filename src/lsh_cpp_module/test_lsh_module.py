import sys
import os
import unittest
import numpy as np

# Thêm thư mục src vào path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Import Python implementation
from similarity_search.simhash_search import SimHashLSH_Python, Simhash_Python

# Try import C++ module (có thể chưa compile)
try:
    from lsh_cpp_module import SimHashLSH, Simhash
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("Warning: C++ module not available. Compile it first with: cd src/lsh_cpp_module && python setup.py build_ext --inplace")


class TestSimhashText(unittest.TestCase):
    """Test cases cho Simhash text fingerprinting."""
    
    def test_python_simhash_basic(self):
        """Test basic Simhash Python implementation."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy dog"
        text3 = "A completely different sentence"
        
        sh1 = Simhash_Python(text1)
        sh2 = Simhash_Python(text2)
        sh3 = Simhash_Python(text3)
        
        # Same text should have same fingerprint
        self.assertEqual(sh1.get_fingerprint(), sh2.get_fingerprint())
        
        # Different text should have different fingerprint
        self.assertNotEqual(sh1.get_fingerprint(), sh3.get_fingerprint())
        
        # Distance between identical texts should be 0
        self.assertEqual(sh1.hamming_distance(sh2), 0)
        
        # Distance between different texts should be > 0
        self.assertGreater(sh1.hamming_distance(sh3), 0)
    
    def test_python_simhash_similarity(self):
        """Test Simhash similarity detection."""
        text1 = "The quick brown fox"
        text2 = "The quick brown foxes"  # Very similar
        text3 = "Completely unrelated content"
        
        sh1 = Simhash_Python(text1)
        sh2 = Simhash_Python(text2)
        sh3 = Simhash_Python(text3)
        
        dist_similar = sh1.hamming_distance(sh2)
        dist_different = sh1.hamming_distance(sh3)
        
        # Similar texts should have smaller distance
        self.assertLess(dist_similar, dist_different)
    
    @unittest.skipIf(not CPP_AVAILABLE, "C++ module not available")
    def test_cpp_simhash_basic(self):
        """Test basic Simhash C++ implementation."""
        text1 = "The quick brown fox jumps over the lazy dog"
        text2 = "The quick brown fox jumps over the lazy dog"
        
        sh1 = Simhash(text1)
        sh2 = Simhash(text2)
        
        self.assertEqual(sh1.get_fingerprint(), sh2.get_fingerprint())
        self.assertEqual(sh1.hamming_distance(sh2), 0)


class TestSimHashLSH_Python(unittest.TestCase):
    """Test cases cho SimHashLSH Python implementation."""
    
    def setUp(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.dim = 128
        self.num_vectors = 100
        self.vectors = np.random.randn(self.num_vectors, self.dim).astype(np.float32)
        self.ids = np.arange(self.num_vectors)
    
    def test_initialization(self):
        """Test LSH initialization."""
        lsh = SimHashLSH_Python(dim=self.dim, num_bits=64, num_tables=4)
        
        self.assertEqual(lsh.dim, self.dim)
        self.assertEqual(lsh.num_bits, 64)
        self.assertEqual(lsh.num_tables, 4)
        
        stats = lsh.get_stats()
        self.assertEqual(stats['num_vectors'], 0)
    
    def test_add_single_vector(self):
        """Test adding single vector."""
        lsh = SimHashLSH_Python(dim=self.dim)
        
        vec = np.random.randn(self.dim).astype(np.float32)
        lsh.add(vec, id=0)
        
        stats = lsh.get_stats()
        self.assertEqual(stats['num_vectors'], 1)
    
    def test_add_batch(self):
        """Test adding batch of vectors."""
        lsh = SimHashLSH_Python(dim=self.dim)
        
        lsh.add_batch(self.vectors, self.ids)
        
        stats = lsh.get_stats()
        self.assertEqual(stats['num_vectors'], self.num_vectors)
    
    def test_query_knn(self):
        """Test k-nearest neighbors query."""
        lsh = SimHashLSH_Python(dim=self.dim, num_bits=64, num_tables=4)
        lsh.add_batch(self.vectors, self.ids)
        
        # Query với chính vector đầu tiên
        query_vec = self.vectors[0]
        results = lsh.query(query_vec, k=5)
        
        # Kết quả phải có ít nhất 1 vector
        self.assertGreater(len(results), 0)
        
        # Vector gần nhất phải là chính nó (id=0, distance~0)
        top_id, top_dist = results[0]
        self.assertEqual(top_id, 0)
        self.assertAlmostEqual(top_dist, 0.0, places=5)
    
    def test_query_radius(self):
        """Test radius query."""
        lsh = SimHashLSH_Python(dim=self.dim)
        lsh.add_batch(self.vectors, self.ids)
        
        query_vec = self.vectors[0]
        threshold = 0.1
        results = lsh.query_radius(query_vec, threshold)
        
        # Phải tìm được ít nhất chính nó
        self.assertGreater(len(results), 0)
        
        # Tất cả kết quả phải có distance <= threshold
        for id, dist in results:
            self.assertLessEqual(dist, threshold)
    
    def test_dimension_mismatch(self):
        """Test error handling for dimension mismatch."""
        lsh = SimHashLSH_Python(dim=self.dim)
        
        wrong_vec = np.random.randn(self.dim + 10).astype(np.float32)
        
        with self.assertRaises(ValueError):
            lsh.add(wrong_vec, id=0)
        
        with self.assertRaises(ValueError):
            lsh.query(wrong_vec, k=5)
    
    def test_clear(self):
        """Test clearing the index."""
        lsh = SimHashLSH_Python(dim=self.dim)
        lsh.add_batch(self.vectors, self.ids)
        
        self.assertEqual(lsh.get_stats()['num_vectors'], self.num_vectors)
        
        lsh.clear()
        self.assertEqual(lsh.get_stats()['num_vectors'], 0)


@unittest.skipIf(not CPP_AVAILABLE, "C++ module not available")
class TestSimHashLSH_CPP(unittest.TestCase):
    """Test cases cho SimHashLSH C++ implementation."""
    
    def setUp(self):
        """Setup test fixtures."""
        np.random.seed(42)
        self.dim = 128
        self.num_vectors = 100
        self.vectors = np.random.randn(self.num_vectors, self.dim).astype(np.float32)
        self.ids = np.arange(self.num_vectors, dtype=np.int32)
    
    def test_initialization(self):
        """Test LSH C++ initialization."""
        lsh = SimHashLSH(dim=self.dim, num_bits=64, num_tables=4)
        
        stats = lsh.get_stats()
        self.assertEqual(stats['num_vectors'], 0)
        self.assertEqual(stats['num_tables'], 4)
        self.assertEqual(stats['num_bits'], 64)
    
    def test_add_single_vector(self):
        """Test adding single vector to C++ module."""
        lsh = SimHashLSH(dim=self.dim)
        
        vec = np.random.randn(self.dim).astype(np.float32)
        lsh.add(vec, id=0)
        
        stats = lsh.get_stats()
        self.assertEqual(stats['num_vectors'], 1)
    
    def test_add_batch(self):
        """Test adding batch of vectors to C++ module."""
        lsh = SimHashLSH(dim=self.dim)
        
        lsh.add_batch(self.vectors, self.ids)
        
        stats = lsh.get_stats()
        self.assertEqual(stats['num_vectors'], self.num_vectors)
    
    def test_query_knn(self):
        """Test k-nearest neighbors query in C++."""
        lsh = SimHashLSH(dim=self.dim, num_bits=64, num_tables=4)
        lsh.add_batch(self.vectors, self.ids)
        
        query_vec = self.vectors[0]
        results = lsh.query(query_vec, k=5)
        
        self.assertGreater(len(results), 0)
        
        # Check top result
        top_id, top_dist = results[0]
        self.assertEqual(top_id, 0)
        self.assertAlmostEqual(top_dist, 0.0, places=5)
    
    def test_query_radius(self):
        """Test radius query in C++."""
        lsh = SimHashLSH(dim=self.dim)
        lsh.add_batch(self.vectors, self.ids)
        
        query_vec = self.vectors[0]
        threshold = 0.1
        results = lsh.query_radius(query_vec, threshold)
        
        self.assertGreater(len(results), 0)
        
        for id, dist in results:
            self.assertLessEqual(dist, threshold)


@unittest.skipIf(not CPP_AVAILABLE, "C++ module not available")
class TestPythonCPPConsistency(unittest.TestCase):
    """Test consistency between Python and C++ implementations."""
    
    def setUp(self):
        """Setup test fixtures."""
        # Use same seed để có reproducible results
        np.random.seed(42)
        self.dim = 64
        self.num_vectors = 50
        self.vectors = np.random.randn(self.num_vectors, self.dim).astype(np.float32)
        self.ids = np.arange(self.num_vectors, dtype=np.int32)
    
    def test_same_results(self):
        """
        Test that Python and C++ implementations produce similar results.
        Note: Không expect kết quả hoàn toàn giống nhau vì random seed khác nhau,
        nhưng top results nên tương tự.
        """
        # Create both implementations với cùng parameters
        lsh_py = SimHashLSH_Python(dim=self.dim, num_bits=32, num_tables=2)
        lsh_cpp = SimHashLSH(dim=self.dim, num_bits=32, num_tables=2)
        
        # Add same data
        lsh_py.add_batch(self.vectors, self.ids)
        lsh_cpp.add_batch(self.vectors, self.ids)
        
        # Query
        query_vec = self.vectors[0]
        results_py = lsh_py.query(query_vec, k=5)
        results_cpp = lsh_cpp.query(query_vec, k=5)
        
        # Both should find the query vector itself as closest
        self.assertEqual(results_py[0][0], 0)
        self.assertEqual(results_cpp[0][0], 0)
        
        # Distances should be very close to 0
        self.assertLess(results_py[0][1], 0.001)
        self.assertLess(results_cpp[0][1], 0.001)


def run_tests():
    """Run all tests."""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    print("="*70)
    print("Running LSH Module Unit Tests")
    print("="*70)
    print()
    
    if not CPP_AVAILABLE:
        print("WARNING: C++ module not compiled yet!")
        print("To compile: cd src/lsh_cpp_module && python setup.py build_ext --inplace")
        print()
    
    run_tests()
