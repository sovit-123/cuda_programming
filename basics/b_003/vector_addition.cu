#include <iostream>
#include <cuda_runtime.h>

#define N 10000

__global__ void vector_addition(float *a, float *b, float *output, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // for(int i=0; i<n; i++) {
        // output[i] = a[i] + b[i];
    // }
    if (idx < n) {
        output[idx] = a[idx] +b[idx];
    }
}

int main() {
    float *a, *b, *host_output;
    float *d_a, *d_b;
    float *output;

    a = (float*)malloc(sizeof(float) * N);
    b = (float*)malloc(sizeof(float) * N);
    host_output = (float*)malloc(sizeof(float) * N);

    // Allocate GPU memory.
    cudaMalloc((void**)&d_a, sizeof(float) * N);
    cudaMalloc((void**)&d_b, sizeof(float) * N);
    cudaMalloc((void**)&output, sizeof(float) * N);

    // Initialize vectors.
    for(int i=0; i<N; i++) {
        a[i] = 1.0f;
        b[i] = 9.0f;
    }

    // Transfer data from host to CPU
    cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, sizeof(float) * N, cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    vector_addition<<<numBlocks, blockSize>>>(d_a, d_b, output, N);

    cudaDeviceSynchronize();

    cudaMemcpy(host_output, output, sizeof(float) * N, cudaMemcpyDeviceToHost);

    // std::cout << "Output: " << host_output << std::endl;
    // Print output
    for(int i=0; i<N; i++) {
        std::cout << "Output[" << i << "]: " << host_output[i] << std::endl;
    }

    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(output);

    return 0;
}