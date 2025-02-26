/*
Compilation:
1. Clone or download the ggml repository (to the same folder as this file): https://github.com/ggml-org/ggml
2. Ensure that you've saved CMakeLists.txt next to this file.
3. Run: cmake -B build
4. Run: cmake --build build --config Release

For CUDA, include `-D GGML_CUDA=1` in step 3.
*/

#include "ggml.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <vector>
#include <iostream>
#include <chrono>

ggml_backend_t backend = NULL;
ggml_gallocr_t allocr = NULL;

using namespace std::chrono;

// #define dtype half
// #define dtype_ggml GGML_TYPE_F16
// #define dtype_ggml_t ggml_fp16_t
// #define dtype_cast_in ggml_fp32_to_fp16
// #define dtype_cast_out ggml_fp16_to_fp32

#define dtype float
#define dtype_ggml GGML_TYPE_F32
#define dtype_ggml_t float
#define dtype_cast_in
#define dtype_cast_out

void init_backend() {
#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    backend = ggml_backend_cuda_init(0); // init device 0
    if (!backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

    if (!backend) {
        backend = ggml_backend_cpu_init();
    }
}

void init_mem_allocator() {
    allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
}

void predict() {
#ifdef GGML_USE_CUDA
    // Record initial VRAM state
    size_t free_mem_start, total_mem;
    ggml_backend_cuda_get_device_memory(0, &free_mem_start, &total_mem);
#endif

    // create a context
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(params);

    const int N = 1024 * 1024 * 500;

    // 1. Define the tensor variables
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, dtype_ggml, N);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, dtype_ggml, N);

    // 2. Define the computation graph
    struct ggml_tensor* result = ggml_add(ctx, a, b);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // 3. Allocate memory for the tensor variables, and assign the data
    ggml_gallocr_alloc_graph(allocr, gf);

    // Convert data
    std::vector<dtype_ggml_t> a_data(N);
    std::vector<dtype_ggml_t> b_data(N);

    for (size_t i = 0; i < N; ++i) {
        a_data[i] = dtype_cast_in(1.5f);
        b_data[i] = dtype_cast_in(3.5f);
    }
    size_t size = N * ggml_type_size(dtype_ggml);
    printf("host mem usage: %lld MB\n", 2 * size / (1024 * 1024)); // one per array

    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));

    // 4. Run the computation, and read the result
    auto start = high_resolution_clock::now();
    ggml_backend_graph_compute(backend, gf);
    auto stop = high_resolution_clock::now();
    std::cout<<"Time taken: "<<duration_cast<milliseconds>(stop - start).count()<<" ms"<<std::endl;

#ifdef GGML_USE_CUDA
    // Record final VRAM state
    size_t free_mem_end;
    ggml_backend_cuda_get_device_memory(0, &free_mem_end, &total_mem);

    // Calculate peak VRAM usage
    size_t peak_usage = free_mem_start - free_mem_end;
    printf("Peak VRAM usage: %f MB\n", peak_usage / (1024.0 * 1024.0));
#endif

    struct ggml_tensor* result_node = ggml_graph_node(gf, -1);  // get the last node in the graph

    std::cout<<"Output type: "<<result_node->type<<std::endl;

    int n = ggml_nelements(result_node); // create an array to store the result data
    std::vector<dtype_ggml_t> result_data(n);

    // copy the data from the backend memory into the result array
    ggml_backend_tensor_get(result_node, result_data.data(), 0, ggml_nbytes(result_node));

    // print the data
    for (int i = 0; i < 10; i++) {
        std::cout<<dtype_cast_out(result_data[i])<<", ";
    }
    std::cout<<std::endl;

    // free the resources
    ggml_free(ctx);
}

int main(int argc, char* argv[]) {
    init_backend();
    init_mem_allocator();

    predict();

    // free the resources
    ggml_gallocr_free(allocr);
    ggml_backend_free(backend);

    return 0;
}
