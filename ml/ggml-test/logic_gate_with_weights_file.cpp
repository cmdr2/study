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
#include <string>
#include <iostream>
#include <unordered_map>

#include "safetensors.hpp"

ggml_backend_t backend = NULL;
ggml_gallocr_t allocr = NULL;

struct logic_gate_model {
    ggml_tensor* fc1_weight;
    ggml_tensor* fc1_bias;
    ggml_tensor* fc2_weight;
    ggml_tensor* fc2_bias;
    ggml_context* params_ctx;

    std::unordered_map<std::string, struct ggml_tensor*> tensor_map;

    struct model_config {
        int32_t n_input = 2;
        int32_t n_hidden = 16;
        int32_t n_output = 1;
    } config;

    logic_gate_model() {
        // create a context (for weights)
        int num_weight_tensors = 4; // since we store four tensors in the model
        params_ctx = ggml_init({
            /*.mem_size   =*/ ggml_tensor_overhead() * num_weight_tensors,
            /*.mem_buffer =*/ NULL,
            /*.no_alloc   =*/ true,
        });

        // Define the tensor variables required for model weights
        fc1_weight = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F32, config.n_input, config.n_hidden);
        fc1_bias = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, config.n_hidden);
        fc2_weight = ggml_new_tensor_2d(params_ctx, GGML_TYPE_F32, config.n_hidden, config.n_output);
        fc2_bias = ggml_new_tensor_1d(params_ctx, GGML_TYPE_F32, config.n_output);

        // names of the parameters as written by the training code
        tensor_map["fc1.weight"] = fc1_weight;
        tensor_map["fc1.bias"] = fc1_bias;
        tensor_map["fc2.weight"] = fc2_weight;
        tensor_map["fc2.bias"] = fc2_bias;

        ggml_backend_alloc_ctx_tensors(params_ctx, backend);
    }

    ~logic_gate_model() {
        ggml_free(params_ctx);
    }

    void load_weights(std::string filename) {
        auto tensors = tensor_map;
        safetensors::load_from_file(filename, [&tensors](const std::string& key, const std::string& dtype, const std::vector<uint64_t>& shape, const std::vector<uint8_t>& tensor_data) {
            std::cout<<"Read tensor: "<<key<<", size: "<<tensor_data.size()<<" bytes"<<std::endl;

            auto it = tensors.find(key);
            if (it != tensors.end()) {
                ggml_tensor* tensor = it->second;
                ggml_backend_tensor_set(tensor, tensor_data.data(), 0, ggml_nbytes(tensor));
            } else {
                std::cout<<"Unknown key: "<<key<<std::endl;
            }
        });
    }

    ggml_tensor* forward(ggml_context *ctx, ggml_tensor *x) {
        ggml_tensor* fc1 = ggml_add(ctx, ggml_mul_mat(ctx, fc1_weight, x), fc1_bias);  // multiply the weights, and add the bias
        ggml_tensor* fc1_relu = ggml_relu(ctx, fc1);
        ggml_tensor* fc2 = ggml_add(ctx, ggml_mul_mat(ctx, fc2_weight, fc1_relu), fc2_bias);
        return ggml_hardsigmoid(ctx, fc2);
    }
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
    ggml_tensor* result = model.forward(ctx, x);

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
    model.load_weights("model.sft");

    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            std::vector<float> input = {float(i), float(j)};
            predict(model, input);
        }
    }

    // free the resources
    ggml_gallocr_free(allocr);
    ggml_backend_free(backend);

    return 0;
}
