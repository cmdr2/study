#include "ggml.h"
#include "ggml-cpu.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <vector>
#include <iostream>

struct simple_model {
    struct ggml_tensor * a;
    struct ggml_tensor * b;

    ggml_backend_t backend = NULL;

    ggml_backend_buffer_t buffer;

    struct ggml_context * ctx;
};

int main(int argc, char* argv[]) {
    simple_model model;

#ifdef GGML_USE_CUDA
    fprintf(stderr, "%s: using CUDA backend\n", __func__);
    model.backend = ggml_backend_cuda_init(0); // init device 0
    if (!model.backend) {
        fprintf(stderr, "%s: ggml_backend_cuda_init() failed\n", __func__);
    }
#endif

    if (!model.backend) {
        model.backend = ggml_backend_cpu_init();
    }

    int num_tensors = 2;

    struct ggml_init_params params {
            /*.mem_size   =*/ ggml_tensor_overhead() * num_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
    };

    // create context
    model.ctx = ggml_init(params);

    // create tensors
    model.a = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 3);
    model.b = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, 3);

    // create a backend buffer (backend memory) and alloc the tensors from the context
    model.buffer = ggml_backend_alloc_ctx_tensors(model.ctx, model.backend);

    // load data from cpu memory to backend buffer
    std::vector<float> a = {0, 1, 2};
    std::vector<float> b = {10, 20, 30};
    ggml_backend_tensor_set(model.a, a.data(), 0, ggml_nbytes(model.a));
    ggml_backend_tensor_set(model.b, b.data(), 0, ggml_nbytes(model.b));

    // build the compute graph to perform a matrix multiplication
    static size_t buf_size = ggml_tensor_overhead()*GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params0 = {
        /*.mem_size   =*/ buf_size,
        /*.mem_buffer =*/ buf.data(),
        /*.no_alloc   =*/ true, // the tensors will be allocated later by ggml_allocr_alloc_graph()
    };

    // create a temporary context to build the graph
    struct ggml_context * ctx0 = ggml_init(params0);

    struct ggml_cgraph  * gf = ggml_new_graph(ctx0);

    // result
    struct ggml_tensor * result_node = ggml_add(ctx0, model.a, model.b);

    // build operations nodes
    ggml_build_forward_expand(gf, result_node);

    // delete the temporary context used to build the graph
    ggml_free(ctx0);

    // allocate tensors
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(model.backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_graph_compute(model.backend, gf);

    // in this case, the output tensor is the last one in the graph
    struct ggml_tensor* result = ggml_graph_node(gf, -1);

    // create a array to print result
    int n = ggml_nelements(result);
    std::vector<float> out_data(n);

    // bring the data from the backend memory
    ggml_backend_tensor_get(result, out_data.data(), 0, ggml_nbytes(result));

    for (int i = 0; i < n; i++) {
        std::cout<<out_data[i]<<", ";
    }
    std::cout<<std::endl;

    // release backend memory used for computation
    ggml_gallocr_free(allocr);

    // free memory
    ggml_free(model.ctx);

    // release backend memory and free backend
    ggml_backend_buffer_free(model.buffer);
    ggml_backend_free(model.backend);
    return 0;
}