/**
Single-precision A*X Plus Y.
Here A is a constant, and X & Y are vectors.

Reference:
https://developer.nvidia.com/blog/easy-introduction-cuda-c-and-c/
*/

#include <iostream>
#include <cuda_runtime.h>

__global__ void saxpy(int n, float a, float *x, float *y) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < n) {
        y[i] = a*x[i] + y[i];
    }
}

int main() {
    int N = 1<<20;
    float *x, *y, *d_x, *d_y;

    x = (float*)malloc(N*sizeof(float));
    y = (float*)malloc(N*sizeof(float));

    // Allocate GPU memory for pointers pointing to device.
    cudaMalloc(&d_x, N*sizeof(float));
    cudaMalloc(&d_y, N*sizeof(float));

    for (int i = 0; i < N; i++) {
        x[i] = 1.0f;
        y[i] = 2.0f;
    }

    // Copy the data from host to device.
    cudaMemcpy(d_x, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Perform SAXPY.
    saxpy<<<(N+255)/256, 256>>>(N, 2.0f, d_x, d_y);

    cudaMemcpy(y, d_y, N*sizeof(float), cudaMemcpyDeviceToHost);

    float maxError = 0.0f;

    for (int i = 0; i < N; i++) {
        maxError = max(maxError, abs(y[i]-4.0f));
    }

    std::cout << "Max error: " << maxError << std::endl;
    
    cudaFree(d_x);
    cudaFree(d_y);
    free(x);
    free(y);

    return 0;
}