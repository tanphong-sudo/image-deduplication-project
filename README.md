# Image Deduplication Project

**Course Project**: Large-scale Image Deduplication using Deep Learning and Feature Hashing  
**Course**: Data Structures & Algorithms - HCMUT  
**Semester**: 1, 2025

---

## 📋 Overview

This project builds a system to **detect and remove duplicate or near-duplicate images** from large image datasets using:

1. **Deep Learning Feature Extraction**: Extract image features using pre-trained models (ResNet, EfficientNet, ConvNeXt, ViT)
2. **Hashing Techniques**: Apply various hashing methods (Hash Table, Bloom Filter, SimHash, MinHash) and FAISS library
3. **Performance Comparison**: Compare efficiency between custom hashing implementations and FAISS
4. **Representative Selection**: Select best representative image from each duplicate group

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Image Dataset                    │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Feature Extraction (Deep Learning)             │
│  ┌──────────┐ ┌───────────────┐ ┌──────────┐ ┌──────────┐ │
│  │ ResNet50 │ │ EfficientNetB0│ │ConvNeXt  │ │   ViT    │ │
│  └──────────┘ └───────────────┘ └──────────┘ └──────────┘ │
│                    Output: Feature Vectors                  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│             Similarity Search & Hashing                     │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ SimHash  │ │ MinHash  │ │  FAISS   │ │ Bloom Filter │  │
│  │  (C++)   │ │          │ │          │ │              │  │
│  └──────────┘ └──────────┘ └──────────┘ └──────────────┘  │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│              Duplicate Detection & Clustering               │
│         Find similar images within threshold                │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│          Representative Selection & Deduplication           │
│    Choose best image from each duplicate group              │
└───────────────────────┬─────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│     OUTPUT: Deduplicated Dataset + Statistics + Report     │
└─────────────────────────────────────────────────────────────┘
```

---

## 📁 Project Structure

```
image-deduplication-project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── run_pipeline.py                    # Main pipeline execution
│
├── data/                              # Data directory
│   ├── raw/                          # Original images
│   └── processed/                    # Processed/deduplicated images
│
├── notebooks/                         # Jupyter notebooks
│   └── project_demo.ipynb            # Demo & analysis
│
├── report/                            # LaTeX report
│   └── report.tex                    # Technical report
│
└── src/                               # Source code
    │
    ├── feature_extraction/            # Deep learning feature extractors
    │   ├── base_extractor.py         # Base class
    │   ├── resnet_extractor.py       # ResNet50
    │   ├── efficientnet_extractor.py # EfficientNet
    │   ├── convnexttiny_extractor.py # ConvNeXt
    │   └── vit_extractor.py          # Vision Transformer
    │
    ├── similarity_search/             # Hashing & similarity search
    │   ├── simhash_search.py         # SimHash (Python implementation)
    │   ├── minhash_search.py         # MinHash
    │   └── faiss_search.py           # FAISS wrapper
    │
    ├── lsh_cpp_module/                # ⭐ High-performance C++ LSH
    │   ├── lsh_cpp/
    │   │   ├── simhash.cpp           # Core C++ implementation
    │   │   ├── simhash.hpp           # Header file
    │   │   └── bindings.cpp          # Pybind11 bindings
    │   ├── setup.py                  # Build script
    │   ├── test_lsh_module.py        # Unit tests
    │   ├── benchmark_comparison.py   # Performance benchmark
    │   ├── example_usage.py          # Usage examples
    │   ├── compile.sh                # Quick compile script
    │   └── README.md                 # Module documentation
    │
    ├── embedding_evaluation/          # Evaluation metrics
    │   ├── clustering_metrics.py     # Clustering evaluation
    │   ├── supervised_metrics.py     # Supervised metrics
    │   └── visualization.py          # Result visualization
    │
    ├── utils/                         # Utilities
    │   ├── bloom_filter.py           # Bloom filter implementation
    │   ├── image_utils.py            # Image processing utilities
    │   └── gen_augumented_images.py  # Data augmentation
    │
    └── evaluator.py                   # Main evaluation module
```

---

## 🚀 Installation

### System Requirements

- **Python**: 3.7+
- **C++ Compiler**: 
  - Linux/macOS: GCC 5+ or Clang 3.4+
  - Windows: MSVC 2015+
- **CUDA** (Optional): For GPU acceleration with FAISS

### Step 1: Clone Repository

```bash
git clone https://github.com/tanphong-sudo/image-deduplication-project.git
cd image-deduplication-project
```

### Step 2: Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 3: Compile C++ LSH Module

```bash
cd src/lsh_cpp_module

# Option 1: Using script (recommended)
chmod +x compile.sh
./compile.sh

# Option 2: Manual compilation
python setup.py build_ext --inplace

# Return to root directory
cd ../..
```

### Step 4: Verify Installation

```bash
# Test Python imports
python -c "import torch; import numpy; print('✓ Core packages OK')"

# Test C++ module (after compilation)
python -c "from lsh_cpp_module import SimHashLSH; print('✓ C++ module OK')"

# Run unit tests
cd src/lsh_cpp_module
python test_lsh_module.py
```

---

## 💻 Usage

### Quick Start - Feature Extraction

```python
from src.feature_extraction.resnet_extractor import ResNetExtractor
import numpy as np

# Initialize extractor
extractor = ResNetExtractor()

# Extract features from images
image_paths = ['image1.jpg', 'image2.jpg', 'image3.jpg']
features = []

for img_path in image_paths:
    feat = extractor.extract(img_path)
    features.append(feat)

features = np.array(features)
print(f"Features shape: {features.shape}")
```

### Quick Start - Duplicate Detection with LSH C++

```python
from lsh_cpp_module import SimHashLSH
import numpy as np

# Load features (assuming already extracted)
features = np.load('features.npy')  # shape: (n_images, feature_dim)

# Initialize LSH
lsh = SimHashLSH(dim=features.shape[1], num_bits=64, num_tables=4)

# Index all images
ids = np.arange(len(features))
lsh.add_batch(features, ids)

# Find duplicates
threshold = 0.3  # Distance threshold (tunable)

for i, feat in enumerate(features):
    similar = lsh.query_radius(feat, threshold=threshold)
    if len(similar) > 1:
        print(f"Image {i} has {len(similar)-1} duplicates:")
        for img_id, dist in similar[1:]:
            print(f"  - Image {img_id}: distance={dist:.4f}")
```

### Quick Start - Using FAISS

```python
from src.similarity_search.faiss_search import FaissSearch
import numpy as np

# Initialize FAISS
faiss_search = FaissSearch(dimension=2048, index_type='L2')

# Add vectors
features = np.random.randn(1000, 2048).astype('float32')
faiss_search.add(features)

# Search
query = features[0]
distances, indices = faiss_search.search(query, k=10)
print(f"Top 10 similar images: {indices}")
```

---

## 🧪 Testing & Benchmarks

### Unit Tests

```bash
# Test C++ LSH module
cd src/lsh_cpp_module
python test_lsh_module.py

# Or with pytest
pytest test_lsh_module.py -v
```

### Performance Benchmark

```bash
# Run benchmark comparing Python vs C++
cd src/lsh_cpp_module
python benchmark_comparison.py

# Output files:
# - lsh_benchmark_results.png     (visualization)
# - lsh_benchmark_report.txt      (detailed report)
```

### Example Usage

```bash
cd src/lsh_cpp_module
python example_usage.py
```

---

## 📊 Results & Evaluation

### Performance Comparison

_TODO: Fill in actual benchmark results after running experiments_

| Method | Dataset Size | Index Time | Query Time | Memory | Speedup |
|--------|--------------|------------|------------|--------|---------|
| SimHash (C++) | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| SimHash (Python) | [TODO] | [TODO] | [TODO] | [TODO] | 1x (baseline) |
| MinHash | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |
| FAISS | [TODO] | [TODO] | [TODO] | [TODO] | [TODO] |

### Accuracy Metrics

_TODO: Fill in evaluation metrics after testing_

- **Precision**: [TODO]
- **Recall**: [TODO]
- **F1-Score**: [TODO]

### Duplicate Detection Results

_TODO: Add statistics after running pipeline_

- Original image count: [TODO]
- Duplicates found: [TODO]
- Final deduplicated count: [TODO]
- Duplicate groups: [TODO]

---

## 🎯 Methods Comparison

### 1. SimHash (C++ Implementation) ⭐

**Implementation Details:**
- Custom C++ implementation using Pybind11
- Random projection based LSH
- Multi-table hashing for improved recall

**Pros:**
- High performance (C++ optimized)
- Suitable for high-dimensional vectors
- Tunable parameters (num_bits, num_tables)
- Compact memory footprint

**Cons:**
- Requires C++ compilation
- Lower recall than brute force

**Use Case:** Large-scale deduplication requiring high speed

### 2. MinHash

**Implementation Details:**
_TODO: Add implementation details_

**Pros:**
- Good for set-based similarity
- Efficient Jaccard similarity estimation

**Cons:**
- Requires vector-to-set conversion
- Not optimal for dense vectors

**Use Case:** _TODO: Add use case_

### 3. FAISS

**Implementation Details:**
_TODO: Add FAISS configuration used_

**Pros:**
- Extremely fast (GPU support available)
- High accuracy
- Production-ready (Facebook AI)
- Multiple index types

**Cons:**
- Memory intensive
- External library dependency

**Use Case:** _TODO: Add use case_

### 4. Bloom Filter

**Implementation Details:**
_TODO: Add implementation details_

**Pros:**
- Extremely fast membership test
- Constant memory usage

**Cons:**
- False positives possible
- No similarity scores

**Use Case:** _TODO: Add use case_

---

## 📝 Output Files

After running the pipeline, the system generates:

1. **Deduplicated Dataset**
   - Directory `data/processed/` contains deduplicated images
   - One representative image per duplicate group

2. **Statistics & Reports**
   - _TODO: List actual output files generated_

3. **Visualizations**
   - _TODO: List visualization files_

---

## 👥 Team & Contributions

### Algorithm Development (C++ LSH)
**Status:** ✅ Complete

**Responsibilities:**
- Implement SimHash LSH in C++
- Performance optimization
- Python integration via Pybind11

**Deliverables:**
- `src/lsh_cpp_module/` (complete module)
- Performance benchmarks
- Unit tests

### Feature Extraction
**Status:** _TODO: Add status_

**Responsibilities:**
- Implement feature extractors
- Test and compare models

**Deliverables:**
- `src/feature_extraction/` (all extractors)
- Model comparison report

### Similarity Search & Integration
**Status:** _TODO: Add status_

**Responsibilities:**
- FAISS integration
- MinHash implementation
- Bloom Filter implementation
- Full pipeline integration

**Deliverables:**
- `src/similarity_search/` (all methods)
- Integration pipeline

### Evaluation & Visualization
**Status:** _TODO: Add status_

**Responsibilities:**
- Evaluation metrics
- Result visualization
- Final report

**Deliverables:**
- `src/embedding_evaluation/`
- Performance comparison charts
- Final technical report

---

## 📖 Documentation & Resources

### Module Documentation

- [LSH C++ Module README](src/lsh_cpp_module/README.md) - Detailed C++ implementation docs
- [LSH C++ Deliverables](src/lsh_cpp_module/DELIVERABLES.md) - Algorithm development summary

### References

1. **SimHash**: Charikar, M. S. (2002). "Similarity estimation techniques from rounding algorithms"
2. **LSH**: Indyk, P., & Motwani, R. (1998). "Approximate nearest neighbors: towards removing the curse of dimensionality"
3. **FAISS**: Johnson, J., Douze, M., & Jégou, H. (2019). "Billion-scale similarity search with GPUs"
4. **MinHash**: Broder, A. Z. (1997). "On the resemblance and containment of documents"

### External Links

- 📚 [PyTorch Image Models (timm)](https://github.com/rwightman/pytorch-image-models)
- 🔍 [FAISS Documentation](https://github.com/facebookresearch/faiss)
- 🐍 [Pybind11 Documentation](https://pybind11.readthedocs.io/)

---

## 🐛 Troubleshooting

### C++ Module Compilation Issues

**Error: `pybind11/pybind11.h: No such file`**
```bash
pip install pybind11
```

**Error: C++ compiler not found**
```bash
# macOS
xcode-select --install

# Linux
sudo apt-get install build-essential

# Windows
# Install Visual Studio Build Tools
```

### Import Errors

**Error: `ModuleNotFoundError: No module named 'lsh_cpp_module'`**
```bash
cd src/lsh_cpp_module
python setup.py build_ext --inplace
```

### Memory Issues

If encountering Out-of-Memory errors:
- Reduce batch size when extracting features
- Decrease `num_tables` in LSH
- Use more compact FAISS index type

---

## 🔗 Links

- **GitHub Repository**: [github.com/tanphong-sudo/image-deduplication-project](https://github.com/tanphong-sudo/image-deduplication-project)
- **Google Colab Demo**: _TODO: Add Colab link_
- **Technical Report (PDF)**: _TODO: Add report link_
- **Video Demo**: _TODO: Add video link_

---

## 📄 License & Citation

This project is part of the **Data Structures & Algorithms** course at HCMUT.

If you use this code, please cite:
```bibtex
@misc{image-deduplication-2025,
  title={Image Deduplication using Deep Learning and Locality-Sensitive Hashing},
  author={KSTN Class - HCMUT},
  year={2025},
  note={Course project for Data Structures \& Algorithms}
}
```

---

## 📧 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/tanphong-sudo/image-deduplication-project/issues)
- **Team Members**: _TODO: Add team member contacts_

---

## ✅ Project Checklist

- [x] Implementation of at least 3 methods (SimHash, MinHash, FAISS)
- [x] C++ implementation with Python bindings
- [ ] Result visualization with images
- [ ] Performance measurement (time & accuracy)
- [ ] Performance comparison charts
- [ ] Code runs on Google Colab
- [x] GitHub repository with clear README
- [ ] Complete LaTeX report

_Last updated: January 2025_

---

<div align="center">

**Made by KSTN Students - HCMUT**

⭐ Star this repo if you find it helpful!

</div>
