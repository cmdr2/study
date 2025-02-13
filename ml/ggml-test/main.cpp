#include "ggml.h"
#include "ggml-cpu.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"

#ifdef GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <iostream>
#include <vector>

struct gate_model_hparams {
    int32_t n_input = 2;
    int32_t n_hidden = 16;
    int32_t n_output = 1;
};

struct gate_model {
    gate_model_hparams hparams;
    struct ggml_tensor* fc1_weight;
    struct ggml_tensor* fc1_bias;
    struct ggml_tensor* fc2_weight;
    struct ggml_tensor* fc2_bias;
    struct ggml_context* ctx;

    struct ggml_tensor* x;
};

bool load(gate_model &model, ggml_backend_t backend) {
    { // make context
        size_t ctx_size = 0;
        // ctx_size += (model.hparams.n_input + 1) * model.hparams.n_hidden * ggml_type_size(GGML_TYPE_F32); // weight + bias
        // ctx_size += (model.hparams.n_hidden + 1) * model.hparams.n_output * ggml_type_size(GGML_TYPE_F32); // weight + bias
        ctx_size += 5 * ggml_tensor_overhead(); // 4 tensors + 1

        std::cout<<ctx_size<<" bytes for context"<<std::endl;

        struct ggml_init_params params = {
            /* mem size */ ctx_size,
            /* mem_buffer */ NULL,
            /* no_alloc */ true, // the tensors will be allocated later by ggml_backend_alloc_ctx_tensors()
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            std::cerr<<"ggml_init() failed!"<<std::endl;
            return false;
        }
    }

    { // load weights
        // TODO - read this from 'model.pth' instead
        std::vector<float> fc1_weights = {
            0.3823428750038147, -0.025383293628692627,
            -0.9573087692260742, -0.9572989344596863,
            0.16090649366378784, -0.5280575752258301,
            -1.108095645904541, -1.1080050468444824,
            0.6486688852310181, -0.1373618096113205,
            -1.6423455476760864, 1.6422196626663208,
            0.49841493368148804, 0.1416090577840805,
            -0.322096586227417, -0.19233638048171997,
            0.29381388425827026, 0.3779483437538147,
            -0.2726233899593353, 0.5532194375991821,
            -1.3715006113052368, -1.3713363409042358,
            0.572895348072052, 0.032060544937849045,
            -0.5058584213256836, -0.3485799729824066,
            1.7513960599899292, 2.1723506450653076,
            -0.5591199994087219, -0.40017884969711304,
            -0.6257259845733643, -0.6248390078544617,
        };
        std::vector<float> fc1_biases = {-0.43759301, 0.95727664, -0.16652977, 1.10802257, 0.89978647, -0.00009338, -0.64021438, -0.27616167, -0.68894398, 0.45800728, 1.37129271, 1.02515948, -0.40761885, -1.75141144, -0.24035817, -0.33963186};
        std::vector<float> fc2_weights = {
            0.00021207332611083984,
            -1.458432912826538,
            -0.032215267419815063,
            -1.7775063514709473,
            0.8093816041946411,
            2.174016237258911,
            0.17627593874931335,
            -0.06556382775306702,
            -0.1391684114933014,
            -0.2268752008676529,
            -2.22202730178833,
            0.8318433165550232,
            0.2408897578716278,
            -3.270573854446411,
            0.03849664330482483,
            0.14314663410186768,
        };
        std::vector<float> fc2_biases = {0.87267971};

        model.fc1_weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.hparams.n_input, model.hparams.n_hidden);
        model.fc1_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, model.hparams.n_hidden);
        model.fc2_weight = ggml_new_tensor_2d(model.ctx, GGML_TYPE_F32, model.hparams.n_hidden, model.hparams.n_output);
        model.fc2_bias = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, model.hparams.n_output);
        // std::copy(fc2_biases.begin(), fc2_biases.end(), static_cast<float*>(model.fc2_bias->data)); // TODO - how about ggml_backend_tensor_set instead?
        // ggml_set_name(model.fc2_bias, "model.fc2_bias");

        model.x = ggml_new_tensor_1d(model.ctx, GGML_TYPE_F32, model.hparams.n_input);

        ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(model.ctx, backend);

        ggml_backend_tensor_set(model.fc1_weight, fc1_weights.data(), 0, ggml_nbytes(model.fc1_weight));
        ggml_backend_tensor_set(model.fc1_bias, fc1_biases.data(), 0, ggml_nbytes(model.fc1_bias));
        ggml_backend_tensor_set(model.fc2_weight, fc2_weights.data(), 0, ggml_nbytes(model.fc2_weight));
        ggml_backend_tensor_set(model.fc2_bias, fc2_biases.data(), 0, ggml_nbytes(model.fc2_bias));
    }

    return true;
}

float predict(const gate_model &model, ggml_backend_t backend, std::vector<float> input) {
    // static size_t buf_size = 0; //model.hparams.n_input + 1024*1024; // TODO - wtf is this? and why is it static?
    // buf_size += 1 * ggml_tensor_overhead(); // 1 tensor for input
    static size_t buf_size = ggml_tensor_overhead() * GGML_DEFAULT_GRAPH_SIZE + ggml_graph_overhead();
    static std::vector<uint8_t> buf(buf_size);

    struct ggml_init_params params = {
        /* mem size*/ buf_size,
        /* buf*/ buf.data(),
        /* no_alloc */ true, // the tensors will be allocated later by ggml_gallocr_alloc_graph(),
    };

    struct ggml_context* input_ctx = ggml_init(params);
    struct ggml_cgraph* gf = ggml_new_graph(input_ctx);

    // struct ggml_tensor* x = ggml_new_tensor_1d(input_ctx, GGML_TYPE_F32, model.hparams.n_input);
    ggml_backend_tensor_set(model.x, input.data(), 0, ggml_nbytes(model.x));

    // define the calc
    ggml_tensor* fc1 = ggml_add(input_ctx, ggml_mul_mat(input_ctx, model.fc1_weight, model.x), model.fc1_bias);
    ggml_tensor* fc2 = ggml_add(input_ctx, ggml_mul_mat(input_ctx, model.fc2_weight, ggml_relu(input_ctx, fc1)), model.fc2_bias);

    ggml_tensor* probs = ggml_hardsigmoid(input_ctx, fc2);
    ggml_set_name(probs, "probs");

    ggml_build_forward_expand(gf, probs);
    
    ggml_gallocr_t allocr = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    ggml_gallocr_alloc_graph(allocr, gf);

    // compute
    ggml_backend_graph_compute(backend, gf);

    // float output[1] = {0};

    struct ggml_tensor* result = ggml_graph_node(gf, -1);
    std::vector<float> result_data(ggml_nelements(result));

    ggml_backend_tensor_get(result, result_data.data(), 0, ggml_nbytes(result));
    // ggml_backend_tensor_get(probs, output, 0, ggml_nbytes(probs));

    // ggml_free(input_ctx);

    return result_data[0];
}

int main(int argc, char* argv[]) {
    // backend
    ggml_backend_t backend = NULL;
#ifdef GGML_USE_CUDA
    std::cout<<"Using CUDA"<<std::endl;
    backend = ggml_backend_cuda_init(0); // device 0
    if (!backend) {
        std::cerr<<"Error initializing CUDA backend!"<<std::endl;
    }
#endif

    if (!backend) {
        std::cout<<"Using CPU"<<std::endl;
        backend = ggml_backend_cpu_init();
    }

    // model
    gate_model model;

    load(model, backend);

    std::cout<<"Loaded model.."<<std::endl;

    for (int i = 0; i <= 1; i++) {
        for (int j = 0; j <= 1; j++) {
            std::vector<float> input = {float(i), float(j)};
            const float pred = predict(model, backend, input);
            std::cout<<i<<","<<j<<" got "<<pred<<std::endl;
        }
    }

    return 0;
}
