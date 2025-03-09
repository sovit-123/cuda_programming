#include <iostream>
#include <cuda_runtime.h>

__global__ void add(float a, float b, float *output) {
    *output = a + b;

    printf("Outputs: %f\n", *output);
}

int main() {
    float a = 9999;
    float b = 1000;
    // Initialize as pointer because CUDA function can only have void 
    // return type and with pointer the value of output will not be 
    // updated anywhere other than the add function
    float *output = 0;

    float host_output = 0;

    cudaMalloc(&output, sizeof(float));  // Allocate GPU memory

    add<<<1, 1>>>(a, b, output);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // main function runs on CPU and the add on GPU. If we print the following
    // we get segmentation fault.
    // CPU is host and GPU is device.
    // std::cout << "Output in main: " << *output << std::endl;
    
    // We first have to copy the value of output to host memory.
    cudaMemcpy(&host_output, output, sizeof(float), cudaMemcpyDeviceToHost);

    std::cout << "Output in main: " << host_output << std::endl;

    cudaFree(output);

    return 0;
}