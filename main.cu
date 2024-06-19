#include "pch.cuh"
#define N 25600000

__global__ void histgram(int *hist_data, int *bin_data, int data_size) {
    __shared__ int cache[256];
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;
    cache[tid] = 0;
    __syncthreads();

    for (int i = global_tid; i < data_size; i += gridDim.x * blockDim.x) {
        int val = hist_data[i];// 每个单线程计算全局内存中的若干个值
        atomicAdd(&cache[val], 1);
    }
    __syncthreads();
    atomicAdd(&bin_data[tid], cache[tid]);
}

bool checkResults(const int *input, int *groundTruth, int size) {
    for (int i = 0; i < size; i++){
        if (input[i] != groundTruth[i]) {
            printf("Check Failed, out[i]=%d, gt[i]=%d\n", input[i], groundTruth[i]);
            return false;
        }
    }
    return true;
}

int main() {
    int *h_hist, *h_bin;
    int *d_hist, *d_bin;

    // get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 256;  // 256 threads/block
    int gridSize = std::min(((N + blockSize - 1) / blockSize), deviceProp.maxGridSize[0]);  // keep gridSize because we are not reallocate data to block we are reallocate threads

    // allocate CPU data memory
    h_hist = (int*)malloc(N * sizeof(int));
    h_bin = (int*)malloc(256 * sizeof(int));  // each block stores temporary block results

    // init data
    for (int i = 0; i < N; i++) {
        h_hist[i] = i % 256;
    }

    //allocate GPU memory
    cudaMalloc((void**)&d_hist, N * sizeof(int));
    cudaMalloc((void**)&d_bin, 256 * sizeof(int));

    // copy data to GPU
    cudaMemcpy(d_hist, h_hist, N * sizeof(int), cudaMemcpyHostToDevice);

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
    cudaMemcpy(h_bin, d_bin, 256 * sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;

    // Check Result
    int *groudtruth = (int*)malloc(256 * sizeof(int));;
    for(int j = 0; j < 256; j++){
        groudtruth[j] = 100000;
    }
    checkResults(h_bin,groudtruth, 256);

    // free resource
    cudaFree(d_hist);
    cudaFree(d_bin);

    free(h_hist);
    free(h_bin);
}
