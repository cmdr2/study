#include <iostream>
#include <string>
#include <vector>

#include "safetensors.hpp"

int main(int argc, char* argv[]) {
    if (argc < 2) {
        std::cerr<<"Usage: "<<argv[0]<<" <model_file>"<<std::endl;
        return 1;
    }

    std::string filename = argv[1];

    std::cout<<"Loading "<<filename<<std::endl;

    safetensors::load_from_file(filename, [](const std::string& key, const std::string& dtype, const std::vector<uint64_t> shape, const std::vector<uint8_t> tensor_data) {
        std::cout<<key<<", dtype: "<<dtype<<", shape: ";
        for (int i = 0; i < shape.size(); i++) {
            std::cout<<shape[i]<<", ";
        }
        std::cout<<tensor_data.size()<<" bytes"<<std::endl;
    });

    return 0;
}