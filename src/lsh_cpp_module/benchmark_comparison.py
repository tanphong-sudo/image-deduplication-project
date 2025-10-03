import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
import sys
import os
from tabulate import tabulate

# Setup path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

# Try import C++ module
try:
    from lsh_cpp_module import SimHashLSH
    CPP_AVAILABLE = True
except ImportError:
    CPP_AVAILABLE = False
    print("❌ ERROR: C++ module not available. Compile it first!")
    print("   cd src/lsh_cpp_module && python setup.py build_ext --inplace")
    sys.exit(1)

# Try import Python LSH libraries for comparison
PYTHON_LIB = None

try:
    from datasketch import MinHashLSH, MinHash
    PYTHON_LIB = "datasketch"
    print("✓ Using datasketch (MinHash LSH) for comparison")
except ImportError:
    try:
        # Fallback: try pure Python simhash if available
        import simhash as simhash_lib
        PYTHON_LIB = "simhash"
        print("✓ Using simhash-py library for comparison")
    except ImportError:
        print("⚠ Warning: No Python LSH library found for comparison")
        print("   Install: pip install datasketch")
        print("   Falling back to naive LSH implementation")
        PYTHON_LIB = "naive"


class PythonLSHWrapper:
    """
    Wrapper for Python LSH libraries to match our C++ interface.
    Supports datasketch (MinHash) and naive LSH implementation.
    """
    
    def __init__(self, dim: int, num_bits: int = 64, num_tables: int = 4):
        self.dim = dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.database = {}
        
        if PYTHON_LIB == "datasketch":
            # Use MinHashLSH from datasketch
            self.lsh = MinHashLSH(threshold=0.5, num_perm=num_bits)
        else:
            # Naive LSH: simple random projection
            np.random.seed(42)
            self.projection_matrices = []
            for _ in range(num_tables):
                matrix = np.random.randn(num_bits, dim).astype(np.float32)
                self.projection_matrices.append(matrix)
            self.hash_tables = [{} for _ in range(num_tables)]
    
    def add(self, vector: np.ndarray, id: int):
        """Add single vector."""
        self.database[id] = vector
        
        if PYTHON_LIB == "datasketch":
            # Convert vector to MinHash
            m = MinHash(num_perm=self.num_bits)
            # Hash vector elements (simple approach)
            for i, val in enumerate(vector):
                m.update(f"{i}:{val}".encode('utf8'))
            self.lsh.insert(str(id), m)
        else:
            # Naive LSH
            for t in range(self.num_tables):
                hash_val = self._hash_vector(vector, t)
                if hash_val not in self.hash_tables[t]:
                    self.hash_tables[t][hash_val] = []
                self.hash_tables[t][hash_val].append(id)
    
    def add_batch(self, vectors: np.ndarray, ids: np.ndarray):
        """Add multiple vectors."""
        for vec, id in zip(vectors, ids):
            self.add(vec, int(id))
    
    def query(self, query_vector: np.ndarray, k: int = 10, 
              max_candidates: int = 1000) -> List[Tuple[int, float]]:
        """Query k nearest neighbors."""
        if PYTHON_LIB == "datasketch":
            # MinHash query
            m = MinHash(num_perm=self.num_bits)
            for i, val in enumerate(query_vector):
                m.update(f"{i}:{val}".encode('utf8'))
            candidates = self.lsh.query(m)
            
            # Compute distances
            results = []
            for cand_id in candidates[:max_candidates]:
                cand_id = int(cand_id)
                dist = np.linalg.norm(query_vector - self.database[cand_id])
                results.append((cand_id, dist))
        else:
            # Naive LSH query
            candidates_set = set()
            for t in range(self.num_tables):
                hash_val = self._hash_vector(query_vector, t)
                if hash_val in self.hash_tables[t]:
                    for cand_id in self.hash_tables[t][hash_val]:
                        candidates_set.add(cand_id)
                        if len(candidates_set) >= max_candidates:
                            break
                if len(candidates_set) >= max_candidates:
                    break
            
            results = []
            for cand_id in candidates_set:
                dist = np.linalg.norm(query_vector - self.database[cand_id])
                results.append((cand_id, dist))
        
        results.sort(key=lambda x: x[1])
        return results[:k]
    
    def query_radius(self, query_vector: np.ndarray, threshold: float,
                     max_candidates: int = 2000) -> List[Tuple[int, float]]:
        """Query within radius."""
        all_results = self.query(query_vector, k=max_candidates, max_candidates=max_candidates)
        return [(id, dist) for id, dist in all_results if dist <= threshold]
    
    def get_stats(self) -> Dict[str, int]:
        """Get statistics."""
        return {
            'num_vectors': len(self.database),
            'num_tables': self.num_tables,
            'num_bits': self.num_bits,
            'dimension': self.dim
        }
    
    def clear(self):
        """Clear index."""
        self.database.clear()
        if PYTHON_LIB == "datasketch":
            self.lsh = MinHashLSH(threshold=0.5, num_perm=self.num_bits)
        else:
            self.hash_tables = [{} for _ in range(self.num_tables)]
    
    def _hash_vector(self, vector: np.ndarray, table_idx: int) -> int:
        """Hash vector (for naive LSH)."""
        projections = self.projection_matrices[table_idx] @ vector
        binary = ''.join('1' if p > 0 else '0' for p in projections)
        return int(binary, 2)


class BenchmarkRunner:
    """Class để chạy các benchmark tests."""
    
    def __init__(self, dim: int = 512, num_bits: int = 64, num_tables: int = 4):
        self.dim = dim
        self.num_bits = num_bits
        self.num_tables = num_tables
        self.results = {}
    
    def generate_data(self, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """Generate random test data."""
        np.random.seed(42)
        vectors = np.random.randn(n_samples, self.dim).astype(np.float32)
        # Normalize vectors (common preprocessing)
        vectors = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-8)
        ids = np.arange(n_samples, dtype=np.int32)
        return vectors, ids
    
    def benchmark_initialization(self, n_runs: int = 100) -> Dict[str, float]:
        """Benchmark initialization time."""
        print(f"\n{'='*70}")
        print(f"Benchmark 1: Initialization (dim={self.dim}, {n_runs} runs)")
        print(f"{'='*70}")
        
        # Python Library
        start = time.time()
        for _ in range(n_runs):
            lsh_py = PythonLSHWrapper(dim=self.dim, num_bits=self.num_bits, 
                                      num_tables=self.num_tables)
        time_py = time.time() - start
        
        # C++
        start = time.time()
        for _ in range(n_runs):
            lsh_cpp = SimHashLSH(dim=self.dim, num_bits=self.num_bits,
                                 num_tables=self.num_tables)
        time_cpp = time.time() - start
        
        speedup = time_py / time_cpp
        
        print(f"Python Library ({PYTHON_LIB}): {time_py:.4f}s ({time_py/n_runs*1000:.2f}ms per init)")
        print(f"C++ (Custom):                  {time_cpp:.4f}s ({time_cpp/n_runs*1000:.2f}ms per init)")
        print(f"Speedup: {speedup:.2f}x")
        
        return {
            'python': time_py,
            'cpp': time_cpp,
            'speedup': speedup
        }
    
    def benchmark_insertion(self, n_samples_list: List[int]) -> Dict[str, List[float]]:
        """Benchmark insertion time for different dataset sizes."""
        print(f"\n{'='*70}")
        print(f"Benchmark 2: Batch Insertion")
        print(f"{'='*70}")
        
        times_py = []
        times_cpp = []
        speedups = []
        
        for n_samples in n_samples_list:
            print(f"\nDataset size: {n_samples}")
            
            vectors, ids = self.generate_data(n_samples)
            
            # Python Library
            lsh_py = PythonLSHWrapper(dim=self.dim, num_bits=self.num_bits,
                                      num_tables=self.num_tables)
            start = time.time()
            lsh_py.add_batch(vectors, ids)
            time_py = time.time() - start
            
            # C++
            lsh_cpp = SimHashLSH(dim=self.dim, num_bits=self.num_bits,
                                 num_tables=self.num_tables)
            start = time.time()
            lsh_cpp.add_batch(vectors, ids)
            time_cpp = time.time() - start
            
            speedup = time_py / time_cpp
            
            times_py.append(time_py)
            times_cpp.append(time_cpp)
            speedups.append(speedup)
            
            print(f"  Python Library ({PYTHON_LIB}): {time_py:.4f}s")
            print(f"  C++ (Custom):                  {time_cpp:.4f}s")
            print(f"  Speedup: {speedup:.2f}x")
        
        return {
            'python': times_py,
            'cpp': times_cpp,
            'speedups': speedups,
            'n_samples': n_samples_list
        }
    
    def benchmark_query(self, n_samples: int = 10000, n_queries: int = 100,
                        k: int = 10) -> Dict[str, float]:
        """Benchmark query time."""
        print(f"\n{'='*70}")
        print(f"Benchmark 3: Query ({n_queries} queries on {n_samples} vectors)")
        print(f"{'='*70}")
        
        vectors, ids = self.generate_data(n_samples)
        
        # Prepare query vectors
        query_vectors = vectors[:n_queries]
        
        # Python Library
        lsh_py = PythonLSHWrapper(dim=self.dim, num_bits=self.num_bits,
                                  num_tables=self.num_tables)
        lsh_py.add_batch(vectors, ids)
        
        start = time.time()
        for query_vec in query_vectors:
            results = lsh_py.query(query_vec, k=k)
        time_py = time.time() - start
        
        # C++
        lsh_cpp = SimHashLSH(dim=self.dim, num_bits=self.num_bits,
                             num_tables=self.num_tables)
        lsh_cpp.add_batch(vectors, ids)
        
        start = time.time()
        for query_vec in query_vectors:
            results = lsh_cpp.query(query_vec, k=k)
        time_cpp = time.time() - start
        
        speedup = time_py / time_cpp
        
        print(f"Python Library ({PYTHON_LIB}): {time_py:.4f}s ({time_py/n_queries*1000:.2f}ms per query)")
        print(f"C++ (Custom):                  {time_cpp:.4f}s ({time_cpp/n_queries*1000:.2f}ms per query)")
        print(f"Speedup: {speedup:.2f}x")
        
        return {
            'python': time_py,
            'cpp': time_cpp,
            'speedup': speedup,
            'python_per_query': time_py / n_queries,
            'cpp_per_query': time_cpp / n_queries
        }
    
    def benchmark_memory(self, n_samples: int = 10000) -> Dict[str, int]:
        """
        Benchmark memory usage (approximate).
        Note: Đây là estimation, không phải actual memory profiling.
        """
        print(f"\n{'='*70}")
        print(f"Benchmark 4: Memory Usage Estimation ({n_samples} vectors)")
        print(f"{'='*70}")
        
        vectors, ids = self.generate_data(n_samples)
        
        # Python Library (for reference, though memory usage similar)
        lsh_py = PythonLSHWrapper(dim=self.dim, num_bits=self.num_bits,
                                  num_tables=self.num_tables)
        lsh_py.add_batch(vectors, ids)
        
        # Estimate memory
        # Database: n_samples * dim * 4 bytes (float32)
        db_memory = n_samples * self.dim * 4
        
        # Projection matrices: num_tables * num_bits * dim * 4 bytes
        proj_memory = self.num_tables * self.num_bits * self.dim * 4
        
        # Hash tables (estimate): n_samples * 8 bytes per entry (pointer/id)
        # Times num_tables
        hash_memory = n_samples * 8 * self.num_tables
        
        total_memory = db_memory + proj_memory + hash_memory
        
        print(f"Database: {db_memory / 1024 / 1024:.2f} MB")
        print(f"Projection matrices: {proj_memory / 1024 / 1024:.2f} MB")
        print(f"Hash tables: {hash_memory / 1024 / 1024:.2f} MB")
        print(f"Total (estimated): {total_memory / 1024 / 1024:.2f} MB")
        
        return {
            'database_mb': db_memory / 1024 / 1024,
            'projection_mb': proj_memory / 1024 / 1024,
            'hashtable_mb': hash_memory / 1024 / 1024,
            'total_mb': total_memory / 1024 / 1024
        }
    
    def run_all_benchmarks(self):
        """Run all benchmark tests."""
        print("\n" + "="*70)
        print(f"SimHash LSH Performance Benchmark: Python ({PYTHON_LIB}) vs C++ (Custom)")
        print("="*70)
        print(f"Configuration: dim={self.dim}, num_bits={self.num_bits}, num_tables={self.num_tables}")
        
        # 1. Initialization
        self.results['initialization'] = self.benchmark_initialization(n_runs=100)
        
        # 2. Insertion
        n_samples_list = [1000, 5000, 10000, 20000]
        self.results['insertion'] = self.benchmark_insertion(n_samples_list)
        
        # 3. Query
        self.results['query'] = self.benchmark_query(n_samples=10000, n_queries=100)
        
        # 4. Memory
        self.results['memory'] = self.benchmark_memory(n_samples=10000)
        
        return self.results
    
    def plot_results(self, save_path: str = "benchmark_results.png"):
        """Plot benchmark results."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'SimHash LSH Performance: Python Libraries ({PYTHON_LIB}) vs C++ (Custom)', fontsize=16, fontweight='bold')
        
        # Plot 1: Insertion time vs dataset size
        ax = axes[0, 0]
        insertion = self.results['insertion']
        ax.plot(insertion['n_samples'], insertion['python'], 'o-', label=f'Python ({PYTHON_LIB})', linewidth=2, markersize=8)
        ax.plot(insertion['n_samples'], insertion['cpp'], 's-', label='C++ (Custom)', linewidth=2, markersize=8)
        ax.set_xlabel('Number of Vectors', fontsize=12)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Batch Insertion Time', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Speedup for insertion
        ax = axes[0, 1]
        ax.bar(range(len(insertion['n_samples'])), insertion['speedups'], color='green', alpha=0.7)
        ax.set_xlabel('Dataset Size', fontsize=12)
        ax.set_ylabel('Speedup (Python/C++)', fontsize=12)
        ax.set_title(f'C++ (Custom) Speedup over Python ({PYTHON_LIB})', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(insertion['n_samples'])))
        ax.set_xticklabels([str(n) for n in insertion['n_samples']])
        ax.axhline(y=1, color='r', linestyle='--', label='1x (no speedup)')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for i, v in enumerate(insertion['speedups']):
            ax.text(i, v + 0.1, f'{v:.2f}x', ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Query performance comparison
        ax = axes[1, 0]
        query = self.results['query']
        categories = ['Initialization\n(100 runs)', 'Query\n(100 queries)']
        py_times = [self.results['initialization']['python'], query['python']]
        cpp_times = [self.results['initialization']['cpp'], query['cpp']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax.bar(x - width/2, py_times, width, label=f'Python ({PYTHON_LIB})', color='skyblue', alpha=0.8)
        bars2 = ax.bar(x + width/2, cpp_times, width, label='C++ (Custom)', color='orange', alpha=0.8)
        
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Operation Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(categories, fontsize=10)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}s', ha='center', va='bottom', fontsize=9)
        
        # Plot 4: Overall speedup summary
        ax = axes[1, 1]
        operations = ['Init', 'Query']
        speedups = [
            self.results['initialization']['speedup'],
            query['speedup']
        ]
        
        colors = ['#2ecc71' if s > 1 else '#e74c3c' for s in speedups]
        bars = ax.barh(operations, speedups, color=colors, alpha=0.7)
        ax.set_xlabel('Speedup Factor (Python/C++)', fontsize=12)
        ax.set_title('Overall Speedup Summary', fontsize=14, fontweight='bold')
        ax.axvline(x=1, color='black', linestyle='--', linewidth=2, label='1x (baseline)')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, speedup) in enumerate(zip(bars, speedups)):
            ax.text(speedup + 0.2, bar.get_y() + bar.get_height()/2,
                   f'{speedup:.2f}x', va='center', fontweight='bold', fontsize=11)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n✓ Benchmark plot saved to: {save_path}")
        
        return fig
    
    def generate_report(self, save_path: str = "benchmark_report.txt"):
        """Generate detailed text report."""
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SimHash LSH Performance Benchmark Report\n")
            f.write(f"Python Library ({PYTHON_LIB}) vs C++ (Custom Implementation)\n")
            f.write("="*70 + "\n\n")
            
            f.write(f"Configuration:\n")
            f.write(f"  - Vector dimension: {self.dim}\n")
            f.write(f"  - Hash bits: {self.num_bits}\n")
            f.write(f"  - Number of tables: {self.num_tables}\n")
            f.write(f"  - Python LSH library: {PYTHON_LIB}\n\n")
            
            # Summary table
            f.write("PERFORMANCE SUMMARY\n")
            f.write("-"*70 + "\n")
            
            table_data = [
                ["Operation", f"Python/{PYTHON_LIB} (s)", "C++/Custom (s)", "Speedup"],
                ["-"*15, "-"*20, "-"*15, "-"*10],
                ["Initialization (100x)", 
                 f"{self.results['initialization']['python']:.4f}",
                 f"{self.results['initialization']['cpp']:.4f}",
                 f"{self.results['initialization']['speedup']:.2f}x"],
                ["Query (100 queries)",
                 f"{self.results['query']['python']:.4f}",
                 f"{self.results['query']['cpp']:.4f}",
                 f"{self.results['query']['speedup']:.2f}x"],
            ]
            
            f.write(tabulate(table_data, headers='firstrow', tablefmt='grid'))
            f.write("\n\n")
            
            # Insertion details
            f.write("INSERTION BENCHMARK\n")
            f.write("-"*70 + "\n")
            insertion = self.results['insertion']
            insert_table = [
                ["Dataset Size", f"Python/{PYTHON_LIB} (s)", "C++/Custom (s)", "Speedup"]
            ]
            for i, n in enumerate(insertion['n_samples']):
                insert_table.append([
                    str(n),
                    f"{insertion['python'][i]:.4f}",
                    f"{insertion['cpp'][i]:.4f}",
                    f"{insertion['speedups'][i]:.2f}x"
                ])
            
            f.write(tabulate(insert_table, headers='firstrow', tablefmt='grid'))
            f.write("\n\n")
            
            # Memory estimation
            f.write("MEMORY ESTIMATION (10,000 vectors)\n")
            f.write("-"*70 + "\n")
            mem = self.results['memory']
            f.write(f"  Database:           {mem['database_mb']:.2f} MB\n")
            f.write(f"  Projection matrices: {mem['projection_mb']:.2f} MB\n")
            f.write(f"  Hash tables:        {mem['hashtable_mb']:.2f} MB\n")
            f.write(f"  Total:              {mem['total_mb']:.2f} MB\n\n")
            
            # Conclusion
            f.write("CONCLUSION\n")
            f.write("-"*70 + "\n")
            avg_speedup = np.mean([
                self.results['initialization']['speedup'],
                self.results['query']['speedup']
            ])
            f.write(f"Average speedup: {avg_speedup:.2f}x\n")
            f.write(f"\nC++ (custom) implementation vs Python library ({PYTHON_LIB}):\n")
            f.write(f"The custom C++ implementation shows performance improvements,\n")
            f.write(f"especially for insertion operations with large datasets.\n")
            f.write(f"The C++ version is recommended for production use with\n")
            f.write(f"large-scale image deduplication tasks where performance is critical.\n")
        
        print(f"✓ Benchmark report saved to: {save_path}")


def main():
    """Main function."""
    print("\n" + "🚀 "*20)
    print("Starting Performance Benchmark...")
    print(f"Comparing: Python Library ({PYTHON_LIB}) vs C++ (Custom)")
    print("🚀 "*20 + "\n")
    
    # Create benchmark runner
    benchmark = BenchmarkRunner(dim=512, num_bits=64, num_tables=4)
    
    # Run all benchmarks
    results = benchmark.run_all_benchmarks()
    
    # Generate visualizations
    print("\n" + "-"*70)
    print("Generating visualizations...")
    benchmark.plot_results(save_path="lsh_benchmark_results.png")
    
    # Generate report
    print("Generating detailed report...")
    benchmark.generate_report(save_path="lsh_benchmark_report.txt")
    
    print("\n" + "✅ "*20)
    print("Benchmark completed successfully!")
    print("✅ "*20 + "\n")
    
    print("Output files:")
    print("  📊 lsh_benchmark_results.png - Visualization plots")
    print("  📄 lsh_benchmark_report.txt - Detailed text report")


if __name__ == '__main__':
    # Check if required packages are installed
    try:
        import matplotlib
        import seaborn
        import tabulate as tab
    except ImportError as e:
        print(f"Error: Missing required package: {e}")
        print("Install with: pip install matplotlib seaborn tabulate")
        sys.exit(1)
    
    main()
