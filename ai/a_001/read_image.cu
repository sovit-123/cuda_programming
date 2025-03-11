/**
This program reads an image using OpenCV, transfers the image matrix to CUDA,
inverts the colors, and then transfers back to the host.

USAGE: 
*/

#include <opencv2/opencv.hpp>
#include <iostream>
#include <cuda_runtime.h>

__global__ void invertColors(unsigned char *img, int width, int height, int channels) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        int idx = (y * width + x) * channels;

        // Invert colors (BGR format)
        img[idx]     = 255 - img[idx];     // Blue
        img[idx + 1] = 255 - img[idx + 1]; // Green
        img[idx + 2] = 255 - img[idx + 2]; // Red
    }
}

int main() {
    // Read image with OpenCV
    cv::Mat img = cv::imread("image.jpg");

    if (img.empty()) {
        std::cerr << "Error: Could not load image\n";
        return -1;
    }

    int width = img.cols;
    int height = img.rows;
    int channels = img.channels();
    int imgSize = width * height * channels * sizeof(unsigned char);

    // Allocate memory on GPU
    unsigned char *d_img;
    cudaMalloc((void**)&d_img, imgSize);

    // Copy image data from CPU (host) to GPU (device)
    cudaMemcpy(d_img, img.data, imgSize, cudaMemcpyHostToDevice);

    // Define CUDA block and grid sizes
    dim3 blockSize(16, 16);
    dim3 gridSize((width + blockSize.x - 1) / blockSize.x, (height + blockSize.y - 1) / blockSize.y);

    // Run CUDA kernel
    invertColors<<<gridSize, blockSize>>>(d_img, width, height, channels);

    // Copy modified image back to CPU (host)
    cudaMemcpy(img.data, d_img, imgSize, cudaMemcpyDeviceToHost);

    // Free GPU memory
    cudaFree(d_img);

    // Display the modified image
    cv::imshow("Inverted Image", img);
    cv::waitKey(0);

    return 0;
}