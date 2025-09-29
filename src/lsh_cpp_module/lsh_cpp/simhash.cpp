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

// note: Cải tiến Kỹ thuật Trích xuất Đặc trưng (Feature Extraction)
