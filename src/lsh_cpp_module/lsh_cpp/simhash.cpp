#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <functional>
#include <cstdint>
#include <bitset>
#include <cctype>

// Sử dụng hàm popcount có sẵn của GCC/Clang để đếm bit hiệu quả nhất.
// Nếu dùng trình biên dịch khác (như MSVC), sẽ dùng hàm thay thế.
#if defined(__GNUC__) || defined(__clang__)
#define popcount __builtin_popcountll
#else
inline int popcount(uint64_t n) {
    int count = 0;
    while (n > 0) {
        n &= (n - 1);
        count++;
    }
    return count;
}
#endif
/**
 * @brief Lớp SimHashLSH áp dụng Locality-Sensitive Hashing cho vectors đặc trưng ảnh.
 * Sử dụng Random Projection (SimHash variant) để băm vectors cao chiều thành các bit strings ngắn.
 */
class SimHashLSH {
public:
    /**
     * @brief Constructor khởi tạo LSH với các tham số.
     * @param dim Số chiều của vector đặc trưng đầu vào.
     * @param num_bits Số bits trong mỗi hash signature (độ dài hash code).
     * @param num_tables Số bảng băm (nhiều bảng tăng recall).
     */
    SimHashLSH(int dim, int num_bits = 64, int num_tables = 4) 
        : dim(dim), num_bits(num_bits), num_tables(num_tables) {
        
        // Khởi tạo random projection matrices cho mỗi bảng băm
        // Mỗi bảng có ma trận riêng để tăng độ đa dạng
        std::srand(42); // Seed cố định để reproducible
        
        for (int t = 0; t < num_tables; ++t) {
            std::vector<std::vector<float>> table_matrix;
            for (int b = 0; b < num_bits; ++b) {
                std::vector<float> hyperplane(dim);
                // Tạo random hyperplane từ phân phối chuẩn
                for (int d = 0; d < dim; ++d) {
                    // Box-Muller transform để tạo số ngẫu nhiên Gaussian
                    float u1 = (float)rand() / RAND_MAX;
                    float u2 = (float)rand() / RAND_MAX;
                    hyperplane[d] = std::sqrt(-2.0f * std::log(u1)) * std::cos(2.0f * M_PI * u2);
                }
                table_matrix.push_back(hyperplane);
            }
            projection_matrices.push_back(table_matrix);
        }
        
        // Khởi tạo hash tables
        hash_tables.resize(num_tables);
    }

    /**
     * @brief Thêm một vector vào LSH index.
     * @param vec Vector đặc trưng đầu vào.
     * @param id ID duy nhất của vector (ví dụ: index của ảnh).
     */
    void add(const std::vector<float>& vec, int id) {
        if (vec.size() != dim) {
            throw std::invalid_argument("Vector dimension mismatch!");
        }
        
        // Lưu vector vào database
        database[id] = vec;
        
        // Băm vector vào tất cả các bảng
        for (int t = 0; t < num_tables; ++t) {
            uint64_t hash_val = hash_vector(vec, t);
            hash_tables[t][hash_val].push_back(id);
        }
    }

    /**
     * @brief Thêm batch vectors vào LSH index (hiệu quả hơn với dữ liệu lớn).
     * @param vectors Danh sách các vectors.
     * @param ids Danh sách các IDs tương ứng.
     */
    void add_batch(const std::vector<std::vector<float>>& vectors, 
                   const std::vector<int>& ids) {
        if (vectors.size() != ids.size()) {
            throw std::invalid_argument("Vectors and IDs must have same length!");
        }
        
        for (size_t i = 0; i < vectors.size(); ++i) {
            add(vectors[i], ids[i]);
        }
    }

    /**
     * @brief Tìm kiếm k nearest neighbors của một query vector.
     * @param query_vec Vector query.
     * @param k Số lượng neighbors cần tìm.
     * @param max_candidates Số lượng candidates tối đa để kiểm tra (tránh quét toàn bộ).
     * @return Vector các pairs (id, distance) được sắp xếp theo khoảng cách.
     */
    std::vector<std::pair<int, float>> query(const std::vector<float>& query_vec, 
                                              int k = 10, 
                                              int max_candidates = 1000) {
        if (query_vec.size() != dim) {
            throw std::invalid_argument("Query vector dimension mismatch!");
        }
        
        // Thu thập candidates từ tất cả các bảng băm
        std::map<int, bool> candidates_set;
        
        for (int t = 0; t < num_tables && candidates_set.size() < max_candidates; ++t) {
            uint64_t hash_val = hash_vector(query_vec, t);
            
            // Lấy tất cả IDs trong cùng bucket
            if (hash_tables[t].count(hash_val)) {
                for (int id : hash_tables[t][hash_val]) {
                    candidates_set[id] = true;
                    if (candidates_set.size() >= max_candidates) break;
                }
            }
        }
        
        // Tính khoảng cách thực tế cho tất cả candidates
        std::vector<std::pair<int, float>> results;
        for (const auto& pair : candidates_set) {
            int id = pair.first;
            float dist = euclidean_distance(query_vec, database[id]);
            results.push_back({id, dist});
        }
        
        // Sắp xếp theo khoảng cách
        std::sort(results.begin(), results.end(), 
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        // Trả về top k
        if (results.size() > k) {
            results.resize(k);
        }
        
        return results;
    }

    /**
     * @brief Tìm tất cả vectors trong vòng threshold distance.
     * @param query_vec Vector query.
     * @param threshold Ngưỡng khoảng cách.
     * @return Vector các pairs (id, distance) có khoảng cách <= threshold.
     */
    std::vector<std::pair<int, float>> query_radius(const std::vector<float>& query_vec,
                                                     float threshold,
                                                     int max_candidates = 2000) {
        if (query_vec.size() != dim) {
            throw std::invalid_argument("Query vector dimension mismatch!");
        }
        
        std::map<int, bool> candidates_set;
        
        for (int t = 0; t < num_tables && candidates_set.size() < max_candidates; ++t) {
            uint64_t hash_val = hash_vector(query_vec, t);
            
            if (hash_tables[t].count(hash_val)) {
                for (int id : hash_tables[t][hash_val]) {
                    candidates_set[id] = true;
                    if (candidates_set.size() >= max_candidates) break;
                }
            }
        }
        
        std::vector<std::pair<int, float>> results;
        for (const auto& pair : candidates_set) {
            int id = pair.first;
            float dist = euclidean_distance(query_vec, database[id]);
            if (dist <= threshold) {
                results.push_back({id, dist});
            }
        }
        
        std::sort(results.begin(), results.end(),
                  [](const auto& a, const auto& b) { return a.second < b.second; });
        
        return results;
    }

    /**
     * @brief Lấy thống kê về LSH index.
     */
    std::map<std::string, int> get_stats() const {
        std::map<std::string, int> stats;
        stats["num_vectors"] = database.size();
        stats["num_tables"] = num_tables;
        stats["num_bits"] = num_bits;
        stats["dimension"] = dim;
        
        // Tính số bucket trung bình
        int total_buckets = 0;
        for (const auto& table : hash_tables) {
            total_buckets += table.size();
        }
        stats["total_buckets"] = total_buckets;
        stats["avg_buckets_per_table"] = num_tables > 0 ? total_buckets / num_tables : 0;
        
        return stats;
    }

    /**
     * @brief Xóa toàn bộ index.
     */
    void clear() {
        database.clear();
        for (auto& table : hash_tables) {
            table.clear();
        }
    }

private:
    int dim;           // Số chiều của vector
    int num_bits;      // Số bits trong hash signature
    int num_tables;    // Số bảng băm
    
    // Random projection matrices: [num_tables][num_bits][dim]
    std::vector<std::vector<std::vector<float>>> projection_matrices;
    
    // Hash tables: [num_tables][hash_value] -> list of IDs
    std::vector<std::map<uint64_t, std::vector<int>>> hash_tables;
    
    // Database lưu trữ vectors gốc: id -> vector
    std::map<int, std::vector<float>> database;

    /**
     * @brief Băm một vector thành 64-bit hash code sử dụng random projection.
     * @param vec Vector đầu vào.
     * @param table_idx Index của bảng băm (để chọn projection matrix).
     * @return Hash code 64-bit.
     */
    uint64_t hash_vector(const std::vector<float>& vec, int table_idx) const {
        uint64_t hash_code = 0;
        const auto& proj_matrix = projection_matrices[table_idx];
        
        for (int b = 0; b < num_bits; ++b) {
            // Tính dot product giữa vector và hyperplane
            float dot_product = 0.0f;
            for (int d = 0; d < dim; ++d) {
                dot_product += vec[d] * proj_matrix[b][d];
            }
            
            // Nếu dot product > 0, set bit tương ứng
            if (dot_product > 0) {
                hash_code |= (1ULL << b);
            }
        }
        
        return hash_code;
    }

    /**
     * @brief Tính khoảng cách Euclidean giữa hai vectors.
     */
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) const {
        float sum = 0.0f;
        for (size_t i = 0; i < a.size(); ++i) {
            float diff = a[i] - b[i];
            sum += diff * diff;
        }
        return std::sqrt(sum);
    }
};

// note: Cải tiến Kỹ thuật Trích xuất Đặc trưng (Feature Extraction)
