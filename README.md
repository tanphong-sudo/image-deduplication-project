# Image Deduplication with Deep Learning & LSH

**Course Project**: Data Structures & Algorithms  
**Institution**: Ho Chi Minh City University of Technology (HCMUT)  
**Team Members**: Lê Bảo Tấn Phong, Nguyễn Anh Quân, Phạm Văn Hên

---

## � Overview

This project implements a **large-scale image deduplication system** that combines deep learning feature extraction with efficient similarity search algorithms. The system detects and removes duplicate or near-duplicate images from large datasets by:

1. **Feature Extraction**: Utilizing pre-trained deep learning models (ResNet50, EfficientNet-B0) to extract high-dimensional feature vectors from images
2. **Similarity Search**: Applying three state-of-the-art algorithms (FAISS, SimHash LSH, MinHash LSH) to efficiently find similar images
3. **Performance Comparison**: Benchmarking accuracy, speed, and memory usage across different methods
4. **Deduplication**: Selecting representative images from each duplicate cluster

**Key Innovation**: Custom C++ implementation of SimHash LSH with multi-probing, achieving significant performance improvements over pure Python implementations while maintaining high recall rates.

---

## �🚀 Quick Start

```bash
# Run pipeline with default settings
python run_pipeline.py

# View all options
python run_pipeline.py --help
```

**Output**: Results saved to `data/processed/` including metrics, representative images, and cluster information

---

## 🎯 Implemented Methods

### 1. FAISS (Facebook AI Similarity Search)
- Exact and approximate nearest neighbor search
- Multiple index types (L2, IVF, HNSW)
- GPU acceleration support

### 2. SimHash LSH (Custom C++ Implementation)
- Locality-Sensitive Hashing with random projections
- Multi-probing with Hamming distance threshold
- Optimized for Apple Silicon (M1/M2)
- Pybind11 Python bindings

### 3. MinHash LSH
- Jaccard similarity estimation
- Set-based similarity search
- Baseline comparison method

**Feature Extractors**: ResNet50 (2048D), EfficientNet-B0 (1280D)

---

## ⚡ SimHash C++ Module

Custom C++ implementation with Pybind11, optimized for Apple Silicon

**Benchmark:**
```bash
cd src/lsh_cpp_module
python benchmark_comparison.py
```

---

## 🛠️ Setup

```bash
# macOS: Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install
pip install -r requirements.txt

# Build C++ module
cd src/lsh_cpp_module
pip install -e .
cd ../..

# Run
python run_pipeline.py
```

---

## 📖 Usage Examples

```bash
# Default (EfficientNet + FAISS)
python run_pipeline.py

# ResNet50 + SimHash
python run_pipeline.py --extractor resnet --method simhash --hamming-threshold 6

# Custom threshold
python run_pipeline.py --threshold 50
```

---

## 📁 Structure

```
├── data/raw/             # Input images
├── data/processed/       # Results
├── src/
│   ├── feature_extraction/
│   ├── similarity_search/
│   └── lsh_cpp_module/   # C++ SimHash
├── run_pipeline.py
└── view_results.py
```

---

## 🎯 Findings

- ✅ **FAISS:** Best accuracy & speed
- ✅ **SimHash:** Best for massive scale
- ⚠️ Multi-probing (Hamming threshold) critical for high recall

---

## 🐛 Troubleshooting

**macOS Apple Silicon:** Pre-configured for M1/M2

**PyTorch issues:**
```bash
export MKL_NUM_THREADS=1
export OMP_NUM_THREADS=1
export KMP_DUPLICATE_LIB_OK=TRUE
```

---

## 🔗 Links

- **GitHub Repository**: [github.com/tanphong-sudo/image-deduplication-project](https://github.com/tanphong-sudo/image-deduplication-project)
- **Technical Report**: Available in `report/` directory
- **Module Documentation**: See `src/lsh_cpp_module/README.md`

---

## 📄 License & Citation

This project is part of the **Data Structures & Algorithms** course at Ho Chi Minh City University of Technology (HCMUT).

If you use this code in your research or project, please cite:

```bibtex
@misc{image-deduplication-2025,
  title={Large-Scale Image Deduplication using Deep Learning and Locality-Sensitive Hashing},
  author={Lê Bảo Tấn Phong and Nguyễn Anh Quân and Phạm Văn Hên},
  institution={Ho Chi Minh City University of Technology},
  year={2025},
  note={Course Project: Data Structures \& Algorithms}
}
```

---

## 📧 Contact

- **GitHub Issues**: [Report bugs or request features](https://github.com/tanphong-sudo/image-deduplication-project/issues)
- **Team Members**: Lê Bảo Tấn Phong, Nguyễn Anh Quân, Phạm Văn Hên

---

<div align="center">

**Ho Chi Minh City University of Technology (HCMUT)**  
*Data Structures & Algorithms - 2025*

⭐ Star this repository if you find it useful!

</div>
