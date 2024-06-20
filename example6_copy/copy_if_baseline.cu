#include "pch.cuh"
#include <random>
#define N 25600000

__global__ void copy_if_baseline(const unsigned int *input, unsigned int *dst, unsigned int *NCopy, const unsigned int data_size) {
    const unsigned int global_tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_tid < data_size && input[global_tid] > 0) {
        dst[atomicAdd(NCopy, 1)] = input[global_tid];
    }
}


int copy_if_cpu(const unsigned int *input, unsigned int *dst, const unsigned int data_size){
    int index = 0;
    for (auto i = 0; i < data_size; i++) {
        if (input[i] > 0) {
            dst[index++] = input[i];
        }
    }
    return index;  // how many values copied
}

bool checkResults(const unsigned int *dst_cpu, const unsigned int *dst_gpu, const unsigned int data_size) {
    for (unsigned int i = 0 ; i < data_size; i++) {
        if (dst_cpu[i] != dst_gpu[i]) {
            std::cout << "Check Failed: res=" << dst_cpu[i] << " Ground Truth=" << dst_gpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    unsigned int *h_in, *h_dst, *h_dst_cpu, *h_NCopy;
    unsigned int *d_in, *d_dst, *d_NCopy;

    // get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 256;  // 256 threads/block
    int gridSize = std::min(((N + blockSize - 1) / blockSize), deviceProp.maxGridSize[0]);

    // allocate CPU data memory
    h_in = (unsigned int*)malloc(N * sizeof(unsigned int));
    h_dst = (unsigned int*)malloc(N * sizeof(unsigned int));
    h_dst_cpu = (unsigned int*)malloc(N * sizeof(unsigned int));
    h_NCopy = (unsigned int*)malloc(sizeof(unsigned int));

    // init data
    *h_NCopy = 0;
    for(int i = 0; i < N; i++){
        // all set to 1
        h_in[i] = 1;
    }

    //allocate GPU memory
    cudaMalloc((void**)&d_in, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_dst, N * sizeof(unsigned int));
    cudaMalloc((void**)&d_NCopy, sizeof(unsigned int));

    // copy data to GPU
    cudaMemcpy(d_in, h_in, N * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // call kernel function
    dim3 Grid(gridSize);  // number of blocks
    dim3 Block(blockSize);  // number of threads
    float millisecond;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    copy_if_baseline<<<Grid, Block>>>(d_in, d_dst, d_NCopy, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);

    // copy GPU result to CPU
    cudaMemcpy(h_dst, d_dst, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_NCopy, d_NCopy, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;

    // Check Result
    *h_NCopy = copy_if_cpu(h_in, h_dst_cpu, N);
    checkResults(h_dst_cpu, h_dst, N);
    if (*h_NCopy != N) {
        std::cout << "Wrong Result:" << " CPU copied: " << N << " GPU copied: " << *h_NCopy << std::endl;
    }
    else{
        std::cout << "Right Result:" << " CPU copied: " << N << " GPU copied: " << *h_NCopy << std::endl;
    }

    // free resource
    cudaFree(d_in);
    cudaFree(d_dst);
    cudaFree(d_NCopy);

    free(h_in);
    free(h_dst);
    free(h_dst_cpu);
    free(h_NCopy);
}
