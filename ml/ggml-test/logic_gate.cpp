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

ggml_backend_t backend = NULL;
ggml_gallocr_t allocr = NULL;

struct logic_gate_model {
    struct ggml_tensor* fc1_weight;
    struct ggml_tensor* fc1_bias;
    struct ggml_tensor* fc2_weight;
    struct ggml_tensor* fc2_bias;
    struct ggml_context* ctx;

    struct model_config {
        int32_t n_input = 2;
        int32_t n_hidden = 16;
        int32_t n_output = 1;
    } config;
};

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

void load_weights(struct logic_gate_model &model) {
    // create a context (for weights)
    int num_weight_tensors = 4; // since we store four tensors in the model
    struct ggml_init_params weights_ctx_params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * num_weight_tensors,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    model.ctx = ggml_init(weights_ctx_params);

    // 1. Define the tensor variables required for model weights
    model.fc1_weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.config.n_input, model.config.n_hidden);
    model.fc1_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, model.config.n_hidden);
    model.fc2_weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.config.n_hidden, model.config.n_output);
    model.fc2_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, model.config.n_output);

    // 2. Allocate memory for the model weight tensors, and assign the model data
    ggml_backend_alloc_ctx_tensors(model.ctx, backend);

    std::vector<float> fc1_weights = {0.3823428750038147, -0.025383293628692627, -0.9573087692260742, -0.9572989344596863, 0.16090649366378784, -0.5280575752258301, -1.108095645904541, -1.1080050468444824, 0.6486688852310181, -0.1373618096113205, -1.6423455476760864, 1.6422196626663208, 0.49841493368148804, 0.1416090577840805, -0.322096586227417, -0.19233638048171997, 0.29381388425827026, 0.3779483437538147, -0.2726233899593353, 0.5532194375991821, -1.3715006113052368, -1.3713363409042358, 0.572895348072052, 0.032060544937849045, -0.5058584213256836, -0.3485799729824066, 1.7513960599899292, 2.1723506450653076, -0.5591199994087219, -0.40017884969711304, -0.6257259845733643, -0.6248390078544617};
    std::vector<float> fc1_biases = {-0.43759301, 0.95727664, -0.16652977, 1.10802257, 0.89978647, -0.00009338, -0.64021438, -0.27616167, -0.68894398, 0.45800728, 1.37129271, 1.02515948, -0.40761885, -1.75141144, -0.24035817, -0.33963186};
    std::vector<float> fc2_weights = {0.00021207332611083984, -1.458432912826538, -0.032215267419815063, -1.7775063514709473, 0.8093816041946411, 2.174016237258911, 0.17627593874931335, -0.06556382775306702, -0.1391684114933014, -0.2268752008676529, -2.22202730178833, 0.8318433165550232, 0.2408897578716278, -3.270573854446411, 0.03849664330482483, 0.14314663410186768};
    std::vector<float> fc2_biases = {0.87267971};

    ggml_backend_tensor_set(model.fc1_weight, fc1_weights.data(), 0, ggml_nbytes(model.fc1_weight));
    ggml_backend_tensor_set(model.fc1_bias, fc1_biases.data(), 0, ggml_nbytes(model.fc1_bias));
    ggml_backend_tensor_set(model.fc2_weight, fc2_weights.data(), 0, ggml_nbytes(model.fc2_weight));
    ggml_backend_tensor_set(model.fc2_bias, fc2_biases.data(), 0, ggml_nbytes(model.fc2_bias));
}

void predict(struct logic_gate_model &model, std::vector<float> input) {
    // create a context
    struct ggml_init_params params = {
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    };
    struct ggml_context* ctx = ggml_init(params);

    // 1. Define the tensor variables
    struct ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.config.n_input);

    // 2. Define the computation graph (add 'weight' to each calculation)
    struct ggml_tensor* fc1 = ggml_add(ctx, ggml_mul_mat(ctx, model.fc1_weight, x), model.fc1_bias);
    struct ggml_tensor* fc2 = ggml_add(ctx, ggml_mul_mat(ctx, model.fc2_weight, ggml_relu(ctx, fc1)), model.fc2_bias);
    struct ggml_tensor* result = ggml_hardsigmoid(ctx, fc2);

    struct ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // 3. Allocate memory for the tensor variables, and assign the data
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));

    // 4. Run the computation, and read the result
    ggml_backend_graph_compute(backend, gf);

    struct ggml_tensor* result_node = ggml_graph_node(gf, -1);  // get the last node in the graph

    int n = ggml_nelements(result_node); // create an array to store the result data
    std::vector<float> result_data(n);

    // copy the data from the backend memory into the result array
    ggml_backend_tensor_get(result_node, result_data.data(), 0, ggml_nbytes(result_node));

    // print the data
    std::cout<<input[0]<<", "<<input[1]<<", got "<<result_data[0]<<std::endl;

    // free the resources
    ggml_free(ctx);
}

int main(int argc, char* argv[]) {
    init_backend();
    init_mem_allocator();

    struct logic_gate_model model;

    load_weights(model);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::vector<float> input = {float(i), float(j)};
            predict(model, input);
        }
    }

    // free the resources
    ggml_gallocr_free(allocr);
    ggml_free(model.ctx);
    ggml_backend_free(backend);

    return 0;
}
