#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "simhash.cpp"

namespace py = pybind11;

/**
 * @brief Module bindings cho Pybind11 để expose C++ classes sang Python.
 * Module này tạo bridge giữa C++ high-performance code và Python ecosystem.
 */
PYBIND11_MODULE(lsh_cpp_module, m) {
    m.doc() = "High-performance LSH (SimHash) implementation in C++ with Python bindings";

    // ==================== Simhash Class (cho text) ====================
    py::class_<Simhash>(m, "Simhash", "Simhash fingerprinting for text documents")
        .def(py::init<const std::string&>(), 
             py::arg("text"),
             "Initialize Simhash with a text string")
        
        .def("get_fingerprint", &Simhash::getFingerprint,
             "Get the 64-bit fingerprint value")
        
        .def("hamming_distance", 
             py::overload_cast<const Simhash&>(&Simhash::hammingDistance, py::const_),
             py::arg("other"),
             "Calculate Hamming distance with another Simhash object")
        
        .def_static("hamming_distance_static",
                    py::overload_cast<uint64_t, uint64_t>(&Simhash::hammingDistance),
                    py::arg("hash1"), py::arg("hash2"),
                    "Calculate Hamming distance between two hash values")
        
        .def("__repr__", [](const Simhash &sh) {
            return "<Simhash fingerprint=" + std::to_string(sh.getFingerprint()) + ">";
        });

    // ==================== SimHashLSH Class (cho image vectors) ====================
    py::class_<SimHashLSH>(m, "SimHashLSH", 
                           "Locality-Sensitive Hashing using SimHash for high-dimensional vectors")
        .def(py::init<int, int, int>(),
             py::arg("dim"),
             py::arg("num_bits") = 64,
             py::arg("num_tables") = 4,
             R"doc(
Initialize SimHashLSH index.

Parameters
----------
dim : int
    Dimensionality of input vectors
num_bits : int, optional (default=64)
    Number of bits in hash signature
num_tables : int, optional (default=4)
    Number of hash tables (more tables = higher recall but more memory)

Example
-------
>>> lsh = SimHashLSH(dim=512, num_bits=64, num_tables=4)
)doc")
        
        .def("add", 
             [](SimHashLSH &self, py::array_t<float> vec, int id) {
                 auto buf = vec.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Vector must be 1-dimensional");
                 }
                 std::vector<float> vec_cpp(buf.shape[0]);
                 std::memcpy(vec_cpp.data(), buf.ptr, buf.shape[0] * sizeof(float));
                 self.add(vec_cpp, id);
             },
             py::arg("vector"), py::arg("id"),
             R"doc(
Add a single vector to the LSH index.

Parameters
----------
vector : numpy.ndarray
    Feature vector (1D array)
id : int
    Unique identifier for this vector
)doc")
        
        .def("add_batch",
             [](SimHashLSH &self, py::array_t<float> vectors, py::array_t<int> ids) {
                 auto vec_buf = vectors.request();
                 auto id_buf = ids.request();
                 
                 if (vec_buf.ndim != 2) {
                     throw std::runtime_error("Vectors must be 2-dimensional (n_samples x n_features)");
                 }
                 if (id_buf.ndim != 1) {
                     throw std::runtime_error("IDs must be 1-dimensional");
                 }
                 if (vec_buf.shape[0] != id_buf.shape[0]) {
                     throw std::runtime_error("Number of vectors and IDs must match");
                 }
                 
                 int n_samples = vec_buf.shape[0];
                 int n_features = vec_buf.shape[1];
                 
                 std::vector<std::vector<float>> vecs_cpp;
                 std::vector<int> ids_cpp;
                 
                 float* vec_ptr = static_cast<float*>(vec_buf.ptr);
                 int* id_ptr = static_cast<int*>(id_buf.ptr);
                 
                 for (int i = 0; i < n_samples; ++i) {
                     std::vector<float> vec(n_features);
                     std::memcpy(vec.data(), vec_ptr + i * n_features, n_features * sizeof(float));
                     vecs_cpp.push_back(vec);
                     ids_cpp.push_back(id_ptr[i]);
                 }
                 
                 self.add_batch(vecs_cpp, ids_cpp);
             },
             py::arg("vectors"), py::arg("ids"),
             R"doc(
Add multiple vectors to the LSH index (batch operation - faster).

Parameters
----------
vectors : numpy.ndarray
    Feature vectors (2D array: n_samples x n_features)
ids : numpy.ndarray
    Unique identifiers (1D array: n_samples)
)doc")
        
        .def("query",
             [](SimHashLSH &self, py::array_t<float> query_vec, int k, int max_candidates, int hamming_threshold) {
                 auto buf = query_vec.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Query vector must be 1-dimensional");
                 }
                 std::vector<float> vec_cpp(buf.shape[0]);
                 std::memcpy(vec_cpp.data(), buf.ptr, buf.shape[0] * sizeof(float));
                 
                 auto results = self.query(vec_cpp, k, max_candidates, hamming_threshold);
                 
                 // Convert to Python list of tuples
                 py::list py_results;
                 for (const auto& pair : results) {
                     py_results.append(py::make_tuple(pair.first, pair.second));
                 }
                 return py_results;
             },
             py::arg("query_vector"), 
             py::arg("k") = 10,
             py::arg("max_candidates") = 1000,
             py::arg("hamming_threshold") = 0,
             R"doc(
Find k nearest neighbors for a query vector.

Parameters
----------
query_vector : numpy.ndarray
    Query feature vector (1D array)
k : int, optional (default=10)
    Number of nearest neighbors to return
max_candidates : int, optional (default=1000)
    Maximum number of candidates to examine

Returns
-------
list of tuples
    List of (id, distance) tuples, sorted by distance
)doc")
        
        .def("query_radius",
             [](SimHashLSH &self, py::array_t<float> query_vec, float threshold, int max_candidates, int hamming_threshold) {
                 auto buf = query_vec.request();
                 if (buf.ndim != 1) {
                     throw std::runtime_error("Query vector must be 1-dimensional");
                 }
                 std::vector<float> vec_cpp(buf.shape[0]);
                 std::memcpy(vec_cpp.data(), buf.ptr, buf.shape[0] * sizeof(float));
                 
                 auto results = self.query_radius(vec_cpp, threshold, max_candidates, hamming_threshold);
                 
                 py::list py_results;
                 for (const auto& pair : results) {
                     py_results.append(py::make_tuple(pair.first, pair.second));
                 }
                 return py_results;
             },
             py::arg("query_vector"),
             py::arg("threshold"),
             py::arg("max_candidates") = 2000,
             py::arg("hamming_threshold") = 0,
             R"doc(
Find all neighbors within a distance threshold.

Parameters
----------
query_vector : numpy.ndarray
    Query feature vector (1D array)
threshold : float
    Maximum distance threshold
max_candidates : int, optional (default=2000)
    Maximum number of candidates to examine

Returns
-------
list of tuples
    List of (id, distance) tuples where distance <= threshold
)doc")
        
        .def("get_stats", &SimHashLSH::get_stats,
             "Get statistics about the LSH index")
        
        .def("clear", &SimHashLSH::clear,
             "Clear all data from the index")
        
        .def("__repr__", [](const SimHashLSH &lsh) {
            auto stats = lsh.get_stats();
            return "<SimHashLSH num_vectors=" + std::to_string(stats.at("num_vectors")) + 
                   " num_tables=" + std::to_string(stats.at("num_tables")) + 
                   " num_bits=" + std::to_string(stats.at("num_bits")) + ">";
        });
}
