#pragma once
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
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
 * @brief Lớp hiện thực thuật toán Simhash để tạo vân tay cho văn bản.
 * Thuật toán này hiệu quả trong việc phát hiện các văn bản gần giống nhau.
 */
class Simhash {
public:
    // Hằng số xác định độ dài bit của Simhash (64-bit là tiêu chuẩn)
    static constexpr int HASH_BITS = 64;

    /**
     * @brief Khởi tạo và tính toán Simhash cho một đoạn văn bản.
     * @param text Văn bản đầu vào.
     */
    explicit Simhash(const std::string& text) : fingerprint(0) {
        generate(text);
    }

    /**
     * @brief Lấy giá trị vân tay Simhash 64-bit.
     */
    uint64_t getFingerprint() const {
        return fingerprint;
    }

    /**
     * @brief Tính khoảng cách Hamming giữa Simhash này và một Simhash khác.
     * @param other Đối tượng Simhash khác để so sánh.
     * @return Khoảng cách Hamming (số bit khác nhau, càng nhỏ càng giống).
     */
    int hammingDistance(const Simhash& other) const {
        return hammingDistance(this->fingerprint, other.fingerprint);
    }

    /**
     * @brief (Static) Tính khoảng cách Hamming giữa hai giá trị vân tay 64-bit.
     * @param hash1 Vân tay thứ nhất.
     * @param hash2 Vân tay thứ hai.
     * @return Khoảng cách Hamming.
     */
    static int hammingDistance(uint64_t hash1, uint64_t hash2) {
        // Phép XOR sẽ cho ra 1 ở những bit khác nhau.
        // Sau đó chỉ cần đếm số bit 1 (popcount).
        return popcount(hash1 ^ hash2);
    }

private:
    uint64_t fingerprint;

    /**
     * @brief Hàm lõi để thực hiện toàn bộ quá trình tạo vân tay Simhash.
     * @param text Văn bản đầu vào.
     */
    void generate(const std::string& text) {
        // Bước 1: Phân tách & Trọng số (Sử dụng map để đếm tần suất)
        std::map<std::string, int> word_weights;
        std::stringstream ss(text);
        std::string word;
        while (ss >> word) {
            // Đơn giản hóa: chuyển thành chữ thường để không phân biệt hoa/thường
            for (char& c : word) {
                c = std::tolower(c);
            }
            word_weights[word]++;
        }

        // Bước 2 & 3: Băm và Vector hóa
        std::vector<int> v(HASH_BITS, 0);
        std::hash<std::string> hasher;

        for (const auto& pair : word_weights) {
            const std::string& current_word = pair.first;
            const int weight = pair.second;
            uint64_t word_hash = hasher(current_word);

            for (int i = 0; i < HASH_BITS; ++i) {
                if ((word_hash >> i) & 1) {
                    v[i] += weight;
                } else {
                    v[i] -= weight;
                }
            }
        }

        // Bước 4: Tạo vân tay
        uint64_t final_hash = 0;
        // Dùng 1ULL (unsigned long long) để đảm bảo phép dịch bit an toàn trên hệ 64-bit
        for (int i = 0; i < HASH_BITS; ++i) {
            if (v[i] > 0) {
                final_hash |= (1ULL << i);
            }
        }
        this->fingerprint = final_hash;
    }
};

/**
 * @brief Lớp SimHashLSH áp dụng Locality-Sensitive Hashing cho vectors đặc trưng ảnh.
 * Sử dụng Random Projection (SimHash variant) để băm vectors cao chiều thành các bit strings ngắn.
 */
class SimHashLSH {
private:
    /**
     * @brief Tính Hamming distance giữa hai hash values 64-bit.
     * @param h1 Hash value thứ nhất.
     * @param h2 Hash value thứ hai.
     * @return Số bits khác nhau giữa h1 và h2.
     */
    inline int hamming_distance(uint64_t h1, uint64_t h2) const {
        uint64_t xor_val = h1 ^ h2;
        return __builtin_popcountll(xor_val);
    }

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
     * @param hamming_threshold Ngưỡng Hamming distance cho multi-probing (0 = exact match only).
     * @return Vector các pairs (id, distance) được sắp xếp theo khoảng cách.
     */
    std::vector<std::pair<int, float>> query(const std::vector<float>& query_vec, 
                                              int k = 10, 
                                              int max_candidates = 1000,
                                              int hamming_threshold = 0) {
        if (query_vec.size() != dim) {
            throw std::invalid_argument("Query vector dimension mismatch!");
        }
        
        // Thu thập candidates từ tất cả các bảng băm
        std::map<int, bool> candidates_set;
        
        for (int t = 0; t < num_tables && candidates_set.size() < max_candidates; ++t) {
            uint64_t query_hash = hash_vector(query_vec, t);
            
            // Multi-probing: tìm buckets trong vòng Hamming distance threshold
            for (const auto& bucket_pair : hash_tables[t]) {
                uint64_t bucket_hash = bucket_pair.first;
                int ham_dist = hamming_distance(query_hash, bucket_hash);
                
                // Nếu Hamming distance <= threshold, lấy candidates từ bucket này
                if (ham_dist <= hamming_threshold) {
                    for (int id : bucket_pair.second) {
                        candidates_set[id] = true;
                        if (candidates_set.size() >= max_candidates) break;
                    }
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
     * @param threshold Ngưỡng khoảng cách Euclidean.
     * @param max_candidates Số lượng candidates tối đa.
     * @param hamming_threshold Ngưỡng Hamming distance cho multi-probing (0 = exact match only).
     * @return Vector các pairs (id, distance) có khoảng cách <= threshold.
     */
    std::vector<std::pair<int, float>> query_radius(const std::vector<float>& query_vec,
                                                     float threshold,
                                                     int max_candidates = 2000,
                                                     int hamming_threshold = 0) {
        if (query_vec.size() != dim) {
            throw std::invalid_argument("Query vector dimension mismatch!");
        }
        
        std::map<int, bool> candidates_set;
        
        for (int t = 0; t < num_tables && candidates_set.size() < max_candidates; ++t) {
            uint64_t query_hash = hash_vector(query_vec, t);
            
            // Multi-probing: tìm buckets trong vòng Hamming distance threshold
            for (const auto& bucket_pair : hash_tables[t]) {
                uint64_t bucket_hash = bucket_pair.first;
                int ham_dist = hamming_distance(query_hash, bucket_hash);
                
                // Nếu Hamming distance <= threshold, lấy candidates từ bucket này
                if (ham_dist <= hamming_threshold) {
                    for (int id : bucket_pair.second) {
                        candidates_set[id] = true;
                        if (candidates_set.size() >= max_candidates) break;
                    }
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
