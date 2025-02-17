#include "ggml.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <vector>
#include <iostream>

ggml_backend_t backend = NULL;
ggml_gallocr_t allocr = NULL;

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
    // create a context
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // 1. Define the tensor variables
    struct ggml_tensor* a = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    struct ggml_tensor* b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);
    struct ggml_tensor* c = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 3);

    // 2. Define the computation graph
    struct ggml_tensor* result = ggml_add(ctx, a, ggml_add(ctx, b, c));

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // 3. Allocate memory for the tensor variables, and assign the data
    ggml_gallocr_alloc_graph(allocr, gf);

    std::vector<float> a_data = {1, 2, 3};
    std::vector<float> b_data = {10, 20, 30};
    std::vector<float> c_data = {100, 200, 300};
    ggml_backend_tensor_set(a, a_data.data(), 0, ggml_nbytes(a));
    ggml_backend_tensor_set(b, b_data.data(), 0, ggml_nbytes(b));
    ggml_backend_tensor_set(c, c_data.data(), 0, ggml_nbytes(c));

    // 4. Run the computation, and read the result
    ggml_backend_graph_compute(backend, gf);

    struct ggml_tensor* result_node = ggml_graph_node(gf, -1);  // get the last node in the graph

    int n = ggml_nelements(result_node); // create an array to store the result data
    std::vector<float> result_data(n);

    // copy the data from the backend memory into the result array
    ggml_backend_tensor_get(result_node, result_data.data(), 0, ggml_nbytes(result_node));

    // print the data
    for (int i = 0; i < n; i++) {
        std::cout<<result_data[i]<<", ";
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
