#include "pch.cuh"
#include <random>

#define N 25600000

__global__ void histgram(const int *hist_data, int *bin_data, int data_size) {
    // this case will work when numbers of threads = numbers of hist data
    unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = global_tid; i < data_size; i += gridDim.x * blockDim.x) {
        atomicAdd(&bin_data[hist_data[global_tid]], 1);  // operations on global mem space is slower, we can use shard mem to optimize it
    }
}

bool checkResults(const unsigned int *input, unsigned int *groundTruth, int size) {
    for (int i = 0; i < size; i++){
        if (input[i] != groundTruth[i]) {
            printf("Check Failed, out[i]=%d, gt[i]=%d\n", input[i], groundTruth[i]);
            return false;
        }
    }
    return true;
}

int main() {
    unsigned int *h_hist, *h_bin;
    unsigned int *d_hist, *d_bin;

    // get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 256;  // 256 threads/block
    int gridSize = std::min(((N + blockSize - 1) / blockSize), deviceProp.maxGridSize[0]);  // keep gridSize because we are not reallocate data to block we are reallocate threads

    // allocate CPU data memory
    h_hist = (unsigned int*)malloc(N * sizeof(unsigned int));
    h_bin = (unsigned int*)malloc(256 * sizeof(unsigned int));  // we only count numbers from 0 to 255

    // init data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::vector<double> weights(blockSize, 1.0);
    for (int i = 0; i < blockSize / 5; ++i) {
        weights[i] = 5.0;
    }
    std::discrete_distribution<> dist(weights.begin(), weights.end());

    unsigned int *groundTruth = (unsigned int*)calloc(256, sizeof(unsigned int));;
    for (int i = 0; i < N; ++i) {
        h_hist[i] = dist(gen);
        groundTruth[h_hist[i]] += 1;
    }
    for (int i = 0; i < 256; i++) {
        std::cout << i << ": " << groundTruth[i] << std::endl;
    }

    //allocate GPU memory
    cudaMalloc((void**)&d_hist, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_bin, 256 * sizeof(unsigned int));

    // copy data to GPU
    cudaMemcpy(d_hist, h_hist, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // call kernel function
    dim3 Grid(gridSize);  // number of blocks
    dim3 Block(blockSize);  // number of threads
    float millisecond;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    histgram<<<Grid, Block>>>(d_hist, d_bin, N);  // first get result of each block
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);

    // copy GPU result to CPU
    cudaMemcpy(h_bin, d_bin, 256 * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;

    // Check Result
    checkResults(h_bin,groundTruth, 256);

    // free resource
    cudaFree(d_hist);
    cudaFree(d_bin);

    free(h_hist);
    free(h_bin);
    free(groundTruth);
}
