#ifndef SIMHASH_HPP
#define SIMHASH_HPP

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <sstream>
#include <functional>
#include <cstdint>
#include <bitset>
#include <cctype>
#include <cmath>
#include <algorithm>
#include <stdexcept>

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

#ifndef M_PI
#define M_PI 3.14159265358979323846
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
    explicit Simhash(const std::string& text);

    /**
     * @brief Lấy giá trị vân tay Simhash 64-bit.
     */
    uint64_t getFingerprint() const;

    /**
     * @brief Tính khoảng cách Hamming giữa Simhash này và một Simhash khác.
     * @param other Đối tượng Simhash khác để so sánh.
     * @return Khoảng cách Hamming (số bit khác nhau, càng nhỏ càng giống).
     */
    int hammingDistance(const Simhash& other) const;

    /**
     * @brief (Static) Tính khoảng cách Hamming giữa hai giá trị vân tay 64-bit.
     * @param hash1 Vân tay thứ nhất.
     * @param hash2 Vân tay thứ hai.
     * @return Khoảng cách Hamming.
     */
    static int hammingDistance(uint64_t hash1, uint64_t hash2);

private:
    uint64_t fingerprint;

    /**
     * @brief Hàm lõi để thực hiện toàn bộ quá trình tạo vân tay Simhash.
     * @param text Văn bản đầu vào.
     */
    void generate(const std::string& text);
};


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
    SimHashLSH(int dim, int num_bits = 64, int num_tables = 4);

    /**
     * @brief Thêm một vector vào LSH index.
     * @param vec Vector đặc trưng đầu vào.
     * @param id ID duy nhất của vector (ví dụ: index của ảnh).
     */
    void add(const std::vector<float>& vec, int id);

    /**
     * @brief Thêm batch vectors vào LSH index (hiệu quả hơn với dữ liệu lớn).
     * @param vectors Danh sách các vectors.
     * @param ids Danh sách các IDs tương ứng.
     */
    void add_batch(const std::vector<std::vector<float>>& vectors, 
                   const std::vector<int>& ids);

    /**
     * @brief Tìm kiếm k nearest neighbors của một query vector.
     * @param query_vec Vector query.
     * @param k Số lượng neighbors cần tìm.
     * @param max_candidates Số lượng candidates tối đa để kiểm tra (tránh quét toàn bộ).
     * @return Vector các pairs (id, distance) được sắp xếp theo khoảng cách.
     */
    std::vector<std::pair<int, float>> query(const std::vector<float>& query_vec, 
                                              int k = 10, 
                                              int max_candidates = 1000);

    /**
     * @brief Tìm tất cả vectors trong vòng threshold distance.
     * @param query_vec Vector query.
     * @param threshold Ngưỡng khoảng cách.
     * @return Vector các pairs (id, distance) có khoảng cách <= threshold.
     */
    std::vector<std::pair<int, float>> query_radius(const std::vector<float>& query_vec,
                                                     float threshold,
                                                     int max_candidates = 2000);

    /**
     * @brief Lấy thống kê về LSH index.
     */
    std::map<std::string, int> get_stats() const;

    /**
     * @brief Xóa toàn bộ index.
     */
    void clear();

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
    uint64_t hash_vector(const std::vector<float>& vec, int table_idx) const;

    /**
     * @brief Tính khoảng cách Euclidean giữa hai vectors.
     */
    float euclidean_distance(const std::vector<float>& a, const std::vector<float>& b) const;
};

#endif // SIMHASH_HPP
