#include "pch.cuh"
#define N 25600000

template<int blockSize>
__global__ void accumulator_v2(const float *input, float *output) {
    // Example for using shared memory, each block has individual shared memory space.
    __shared__ float shared_float_256[blockSize]; // init an array stores 256 float numbers on shared memory for blockSize(256) threads to use
    unsigned int tid = threadIdx.x;  // thread id in this block
    unsigned int global_tid = blockDim.x * blockIdx.x + threadIdx.x;  // thread id in global thread pool
    shared_float_256[tid] = input[global_tid];  // load corresponding float value into block memory
    __syncthreads();

    /*
    In accumulator_v1, every iteration reduces the number of working threads in each warp by half.
    In v1, the threads' workload is consolidated into the first few warps, minimizing idle threads
    and reducing synchronization overhead. For step=1, the first 4 warps are fully loaded while the
    last 4 warps are idle, lowering the synchronization cost.

    In v2, the optimization aims to eliminate bank conflicts. At step=1, different groups of threads
    within a warp simultaneously access different slots of the same bank. No bank conflict occurs
    because the first group of threads, accessing 32 floats, reaches the 128-byte memory transaction
    limit, distributing them into different transactions. Bank conflicts only occur within the same
    memory transaction.

    At step=8, each thread accesses two floats (8 bytes). Threads 0-15 are in the same memory
    transaction. Here, thread0 and thread2, in the same transaction, access different slots of the
    same bank, causing a bank conflict.

    In v2, the optimization does not affect step=1. However, for step=2 and above, different threads
    in the same transaction accessing different slots of the same bank cause bank conflicts. v2's
    optimization effectively prevents this.
    */
    for (unsigned int index = blockDim.x / 2; index > 0; index >>= 1) {
        /* iter1: warp0(row0) = row0 + row3, warp1(row1) = row1 + row4, warp2(row2) = row2 + row5, warp3(row3) = row3 + row7
         * iter2: warp0(row0) = row0 + row2, warp1(row1) = row1 + row3
         * iter3: warp0(row0) = row0 + row1
         * iter4: index = 16, for the next iteration, only row1 will be operated and (16/8/4/2/1 threads in warp0 will be used)*/
        if (tid < index) {
            shared_float_256[tid] += shared_float_256[tid + index];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = shared_float_256[0];
}

void checkResults(const float *input, int size, float *groundTruth) {
    double sum = 0.0f;
    for (int i = 0; i < size; i++) {
        sum += input[i];
    }

    if (sum != *groundTruth) {
        std::cout << "Wrong Results! " << sum << std::endl;
    }
    else{
        std::cout << "[GPU] Sum = " << sum << std::endl;
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
    float *din, *dout;

    // get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 256;  // 256 threads/block
    int gridSize = std::min((N / blockSize) + 1, deviceProp.maxGridSize[0]);  // total blocks need to process all data

    // allocate CPU data memory
    hin = (float*)malloc(N * sizeof(float));
    hout = (float*)malloc(gridSize * sizeof(float));  // each block stores temporary block results
    hout_cpu = (float*)malloc(sizeof(float));

    // init data
    for (int i = 0; i < N; i++) {
        hin[i] = 1.0f;
    }

    //allocate GPU memory
    cudaMalloc((void**)&din, N * sizeof(float));
    cudaMalloc((void**)&dout, gridSize * sizeof(float));

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
    accumulator_v2<blockSize><<<Grid, Block>>>(din, dout);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);

    // copy GPU result to CPU
    cudaMemcpy(hout, dout, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;


    // call CPU function
    accumulator_baseline_cpu(hin, hout_cpu);
    std::cout << "[CPU] Sum = " << *hout_cpu << std::endl;
    checkResults(hout, gridSize, hout_cpu);

    // free resource
    cudaFree(din);
    cudaFree(dout);

    free(hin);
    free(hout);
    free(hout_cpu);

}
