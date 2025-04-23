#include <immintrin.h>
#include <iostream>
#include <cstdlib>
#include <chrono>
#include <malloc.h>

using namespace std::chrono;

constexpr size_t SIZE = 1024 * 1024 * 100;
constexpr int REPEATS = 10;

#pragma optimize("g", off)

void add_with_avx2() {
    std::cout << "AVX2 is available\n";
}

void add_with_avx512() {
    std::cout << "AVX512 is available\n";
}

void add_avx512(const float* a, const float* b, float* c, size_t size) {
    size_t i = 0;
    for (; i + 16 <= size; i += 16) {
        __m512 va = _mm512_load_ps(a + i);
        __m512 vb = _mm512_load_ps(b + i);
        __m512 vc = _mm512_add_ps(va, vb);
        _mm512_store_ps(c + i, vc);
    }
    for (; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

void add_avx2(const float* a, const float* b, float* c, size_t size) {
    size_t i = 0;
    for (; i + 8 <= size; i += 8) {
        __m256 va = _mm256_load_ps(a + i);
        __m256 vb = _mm256_load_ps(b + i);
        __m256 vc = _mm256_add_ps(va, vb);
        _mm256_store_ps(c + i, vc);
    }
    for (; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

void add_scalar(const float* a, const float* b, float* c, size_t size) {
    for (size_t i = 0; i < size; ++i) {
        c[i] = a[i] + b[i];
    }
}

int main() {
#ifdef __AVX512F__
    add_with_avx512();
#elif defined(__AVX2__)
    add_with_avx2();
#else
    std::cout << "No AVX support\n";
#endif

    float* a = static_cast<float*>(_aligned_malloc(SIZE * sizeof(float), 64));
    float* b = static_cast<float*>(_aligned_malloc(SIZE * sizeof(float), 64));
    float* c_avx512 = static_cast<float*>(_aligned_malloc(SIZE * sizeof(float), 64));
    float* c_avx2 = static_cast<float*>(_aligned_malloc(SIZE * sizeof(float), 64));
    float* c_scalar = static_cast<float*>(_aligned_malloc(SIZE * sizeof(float), 64));

    for (size_t i = 0; i < SIZE; ++i) {
        a[i] = static_cast<float>(rand()) / RAND_MAX;
        b[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // warm-up
    for (int i = 0; i < 2; ++i) add_avx512(a, b, c_avx512, SIZE);
    for (int i = 0; i < 2; ++i) add_avx2(a, b, c_avx2, SIZE);
    for (int i = 0; i < 2; ++i) add_scalar(a, b, c_scalar, SIZE);

    auto start_avx512 = high_resolution_clock::now();
    for (int i = 0; i < REPEATS; ++i) add_avx512(a, b, c_avx512, SIZE);
    auto end_avx512 = high_resolution_clock::now();

    auto start_avx2 = high_resolution_clock::now();
    for (int i = 0; i < REPEATS; ++i) add_avx2(a, b, c_avx2, SIZE);
    auto end_avx2 = high_resolution_clock::now();

    auto start_scalar = high_resolution_clock::now();
    for (int i = 0; i < REPEATS; ++i) add_scalar(a, b, c_scalar, SIZE);
    auto end_scalar = high_resolution_clock::now();

    duration<double> elapsed_avx512 = (end_avx512 - start_avx512) / REPEATS;
    duration<double> elapsed_avx2 = (end_avx2 - start_avx2) / REPEATS;
    duration<double> elapsed_scalar = (end_scalar - start_scalar) / REPEATS;

    std::cout << "AVX-512 avg time: " << duration_cast<milliseconds>(elapsed_avx512).count() << " ms\n";
    std::cout << "AVX2    avg time: " << duration_cast<milliseconds>(elapsed_avx2).count() << " ms\n";
    std::cout << "Scalar  avg time: " << duration_cast<milliseconds>(elapsed_scalar).count() << " ms\n";
    std::cout << "Sample output: c_avx512[0] = " << c_avx512[0]
              << ", c_avx2[0] = " << c_avx2[0]
              << ", c_scalar[0] = " << c_scalar[0] << "\n";

    _aligned_free(a);
    _aligned_free(b);
    _aligned_free(c_avx512);
    _aligned_free(c_avx2);
    _aligned_free(c_scalar);
    return 0;
}