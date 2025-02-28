// see also: https://www.pyspur.dev/blog/introduction_cuda_programming

#include <cuda_fp16.h>
#include <iostream>

__global__ void add_half_arrays(const __half* a, const __half* b, __half* c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
        // c[idx] = __hadd(a[idx], b[idx]);
    }
}

void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main() {
    const int N = 1024;
    size_t size = N * sizeof(__half);

    // Host arrays
    __half* h_a = new __half[N];
    __half* h_b = new __half[N];
    __half* h_c = new __half[N];

    // Initialize input arrays
    for (int i = 0; i < N; ++i) {
        h_a[i] = __float2half(1.5f);
        h_b[i] = __float2half(2.5f);
    }

    // Device arrays
    __half *d_a, *d_b, *d_c;
    check_cuda_error(cudaMalloc(&d_a, size), "Failed to allocate d_a");
    check_cuda_error(cudaMalloc(&d_b, size), "Failed to allocate d_b");
    check_cuda_error(cudaMalloc(&d_c, size), "Failed to allocate d_c");

    // Copy data to device
    check_cuda_error(cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice), "Failed to copy h_a to d_a");
    check_cuda_error(cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice), "Failed to copy h_b to d_b");

    // Launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    add_half_arrays<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    check_cuda_error(cudaGetLastError(), "Kernel launch failed");

    // Copy result back to host
    check_cuda_error(cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost), "Failed to copy d_c to h_c");

    // Verify result
    for (int i = 0; i < 10; ++i) {
        std::cout << "h_c[" << i << "] = " << __half2float(h_c[i]) << std::endl;
    }

    // Cleanup
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
