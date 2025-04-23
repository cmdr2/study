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
    ggml_tensor* fc1_weight;
    ggml_tensor* fc1_bias;
    ggml_tensor* fc2_weight;
    ggml_tensor* fc2_bias;
    ggml_context* params_ctx;

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

void load_weights(logic_gate_model &model) {
    // create a context (for weights)
    int num_weight_tensors = 4; // since we store four tensors in the model
    model.params_ctx = ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * num_weight_tensors,
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    });

    // 1. Define the tensor variables required for model weights
    model.fc1_weight = ggml_new_tensor_2d(model.params_ctx, GGML_TYPE_F32, model.config.n_input, model.config.n_hidden);
    model.fc1_bias = ggml_new_tensor_1d(model.params_ctx, GGML_TYPE_F32, model.config.n_hidden);
    model.fc2_weight = ggml_new_tensor_2d(model.params_ctx, GGML_TYPE_F32, model.config.n_hidden, model.config.n_output);
    model.fc2_bias = ggml_new_tensor_1d(model.params_ctx, GGML_TYPE_F32, model.config.n_output);

    // 2. Allocate memory for the model weight tensors, and assign the model data
    ggml_backend_alloc_ctx_tensors(model.params_ctx, backend);

    std::vector<float> fc1_weight = { 0.22488207, -0.39456311, 0.32581645, -0.56285965, 2.41329503, -2.41322660, -0.37499088, 0.08395171, 0.21755114, 0.80772698, 0.25437704, 1.57216692, -0.43496752, 0.22240390, 0.46247596, -0.02229351, 0.32341745, 0.25361675, -0.20483392, 0.26918083, -0.91469419, 1.23764634, 0.15310341, -0.67303509, 1.77088165, 1.77059495, -0.11867817, -0.37374884, 0.79170924, -1.17232382, 0.07894109, -0.41966945 };
    std::vector<float> fc1_bias = { -0.35652003, -0.67564911, 0.00009615, -0.62946773, 0.27859268, 0.01491952, 0.52390707, -0.47604990, -0.25365347, 0.21269353, 0.00003640, -0.44338676, -1.77084744, 0.82772928, 1.17234588, 0.77097332 };
    std::vector<float> fc2_weight = { 0.13858399, -0.20547047, 3.41583562, 0.15011564, 0.56532770, 1.40391135, 0.00871399, 0.24152395, -0.39389160, 0.16984159, 1.34791148, -0.12602532, -3.02119160, -0.68023020, -1.64424217, -0.63815284 };
    std::vector<float> fc2_bias = { -0.55232018 };

    ggml_backend_tensor_set(model.fc1_weight, fc1_weight.data(), 0, ggml_nbytes(model.fc1_weight));
    ggml_backend_tensor_set(model.fc1_bias, fc1_bias.data(), 0, ggml_nbytes(model.fc1_bias));
    ggml_backend_tensor_set(model.fc2_weight, fc2_weight.data(), 0, ggml_nbytes(model.fc2_weight));
    ggml_backend_tensor_set(model.fc2_bias, fc2_bias.data(), 0, ggml_nbytes(model.fc2_bias));
}

void predict(logic_gate_model &model, std::vector<float> input) {
    // create a context
    ggml_context* ctx = ggml_init({
        /*.mem_size   =*/ ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead(),
        /*.mem_buffer =*/ NULL,
        /*.no_alloc   =*/ true,
    });

    // 1. Define the tensor variables
    ggml_tensor* x = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, model.config.n_input);

    // 2. Define the computation graph
    ggml_tensor* fc1 = ggml_add(ctx, ggml_mul_mat(ctx, model.fc1_weight, x), model.fc1_bias);  // multiply the weights, and add the bias
    ggml_tensor* fc1_relu = ggml_relu(ctx, fc1);
    ggml_tensor* fc2 = ggml_add(ctx, ggml_mul_mat(ctx, model.fc2_weight, fc1_relu), model.fc2_bias);
    ggml_tensor* result = ggml_hardsigmoid(ctx, fc2);

    ggml_cgraph* gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, result);

    // 3. Allocate memory for the tensor variables, and assign the data
    ggml_gallocr_alloc_graph(allocr, gf);

    ggml_backend_tensor_set(x, input.data(), 0, ggml_nbytes(x));

    // 4. Run the computation, and read the result
    ggml_backend_graph_compute(backend, gf);

    ggml_tensor* result_node = ggml_graph_node(gf, -1);  // get the last node in the graph

    int64_t n = ggml_nelements(result_node); // create an array to store the result data
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

    logic_gate_model model;

    load_weights(model);

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::vector<float> input = {float(i), float(j)};
            predict(model, input);
        }
    }

    // free the resources
    ggml_gallocr_free(allocr);
    ggml_free(model.params_ctx);
    ggml_backend_free(backend);

    return 0;
}
