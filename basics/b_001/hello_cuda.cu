#include <iostream>
#include <cuda_runtime.h>

// CUDA Kernel function
__global__ void helloGPU() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    std::cout << "Launching CUDA kernel...\n";

    // Launch kernel with 5 threads
    helloGPU<<<1, 5>>>();
    
    // Wait for GPU to finish
    cudaDeviceSynchronize();

    std::cout << "CUDA kernel execution finished...\n";
    
    return 0;
}
