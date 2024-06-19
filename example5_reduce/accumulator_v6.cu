#include "pch.cuh"
#define N 25600000

template<int blockSize>
__device__ void warpReduce(float *input) {
    // fully extension of for-loop savine 2 instructions: > and >>=1 overhead
    if (blockSize >= 1024) {
        if (threadIdx.x < 512) {
            input[threadIdx.x] += input[threadIdx.x + 512];
        }
        __syncthreads();
    }
    if (blockSize >= 512) {
        if (threadIdx.x < 256) {
            input[threadIdx.x] += input[threadIdx.x + 256];
        }
        __syncthreads();
    }
    if (blockSize >= 256) {
        if (threadIdx.x < 128) {
            input[threadIdx.x] += input[threadIdx.x + 128];
        }
        __syncthreads();
    }
    if (blockSize >= 128) {
        if (threadIdx.x < 64) {
            input[threadIdx.x] += input[threadIdx.x + 64];
        }
        __syncthreads();
    }

    // final warp extension
    if (threadIdx.x < 32) {
        float x = input[threadIdx.x];
        x += input[threadIdx.x + 32]; __syncwarp();
        input[threadIdx.x] = x; __syncwarp();

        x += input[threadIdx.x + 16]; __syncwarp();
        input[threadIdx.x] = x; __syncwarp();

        x += input[threadIdx.x + 8]; __syncwarp();
        input[threadIdx.x] = x; __syncwarp();

        x += input[threadIdx.x + 4]; __syncwarp();
        input[threadIdx.x] = x; __syncwarp();

        x += input[threadIdx.x + 2]; __syncwarp();
        input[threadIdx.x] = x; __syncwarp();

        x += input[threadIdx.x + 1]; __syncwarp();
        input[threadIdx.x] = x; __syncwarp();
    }
}

template<int blockSize>
__global__ void accumulator_v6(const float *input, float *output, int size) {
    // Example for using shared memory, each block has individual shared memory space.

    /* Make kernel function can handle all data when gridSize is smaller than total number of float data
     * For example, when you have 5N float number to process, and only 4 blocks, each block will handle N numbers and left N data.
     * Now we can let block0 handle 2N data to make sure the last N data will not be left. */
    __shared__ float shared_float_256[blockSize]; // init an array stores 64 float numbers on shared memory for blockSize(128) threads to use
    unsigned int tid = threadIdx.x;  // thread id in this block
    unsigned int global_tid = blockDim.x * blockIdx.x + threadIdx.x;  // thread id in global thread pool
    unsigned int steps = gridDim.x * blockDim.x;  // let one block handle multiple blocks data

    // first assign data to this block by adding additional data by steps
    float sum = 0.0f;
    for (int32_t i = global_tid; i < size; i += steps) {
        sum += input[i];
    }
    shared_float_256[tid] = sum;  // curr block handle pre-added data
    __syncthreads();

    // By reduce blockSize to blockSize / 2, actually we just reassigned the first iteration in v2 method to each thread, so in this v3
    warpReduce<blockSize>(shared_float_256);

    if (tid == 0) output[blockIdx.x] = shared_float_256[0];
}

void checkResults(const float *input, float *groundTruth) {
    if (*input != *groundTruth) {
        std::cout << "Wrong Results! " << *input << std::endl;
    }
    else{
        std::cout << "[GPU] Sum = " << *input << std::endl;
    }

}

void accumulator_baseline_cpu(const float *input, float *output) {
    double sum = 0.0f;
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    *output = sum;
}

int main() {
    float *hin, *hout, *hout_cpu;
    float *din, *dout, *dout_block;

    // get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 256;  // 256 threads/block
//    int gridSize = std::min(((N + blockSize - 1) / blockSize), deviceProp.maxGridSize[0]);  // keep gridSize because we are not reallocate data to block we are reallocate threads
    int gridSize = 500;

    // allocate CPU data memory
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(sizeof(float));  // each block stores temporary block results
    hout_cpu = (float*)malloc(sizeof(float));

    // init data
    for (int i = 0; i < N; i++) {
        hin[i] = 1.0f;
    }

    //allocate GPU memory
    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMalloc((void**)&dout, sizeof(float));
    cudaMalloc((void**)&dout_block, gridSize * sizeof(float));

    // copy data to GPU
    cudaMemcpy(din, hin, N * sizeof(float), cudaMemcpyHostToDevice);

    // call kernel function
    dim3 Grid(gridSize);  // number of blocks
    dim3 Block(blockSize);  // number of threads
    float millisecond;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    accumulator_v6<blockSize><<<Grid, Block>>>(din, dout_block, N);  // first get result of each block
    accumulator_v6<blockSize><<<1, Block>>>(dout_block, dout, gridSize);  // add result of all blocks together
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);

    // copy GPU result to CPU
    cudaMemcpy(hout, dout, sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;


    // call CPU function
    accumulator_baseline_cpu(hin, hout_cpu);
    std::cout << "[CPU] Sum = " << *hout_cpu << std::endl;
    checkResults(hout,hout_cpu);

    // free resource
    cudaFree(din);
    cudaFree(dout);

    free(hin);
    free(hout);
    free(hout_cpu);

}
