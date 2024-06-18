#include "pch.cuh"
#define N 25600000

template<int blockSize>
__global__ void accumulator_v3(const float *input, float *output) {
    // Example for using shared memory, each block has individual shared memory space.

    /* In v2 each block processes 256 float data, each thread processes 2 float numbers, so in iter1, only 128 threads are working
     * and 128 threads are idle. In each iteration half of threads will not join next iteration, which means in iter2, 64 threads working,
     * 192 threads idle.
     * Optimization object: let all threads working all the time.
     * Let each thread add block1 data and block2 data. Half idle threads now work.
     * */
    __shared__ float shared_float_128[blockSize]; // init an array stores 128 float numbers on shared memory for blockSize(128) threads to use
    unsigned int tid = threadIdx.x;  // thread id in this block
    unsigned int global_tid = 4 * blockSize * blockIdx.x + threadIdx.x;  // thread id in global thread pool
    shared_float_128[tid] = input[global_tid] +
                            input[global_tid + blockSize] +
                            input[global_tid + 2 * blockSize] +
                            input[global_tid + 3 * blockSize]; // Changing loading data to an add operation for each thread, reduce 4x blockSize perform best
    __syncthreads();

    // By reduce blockSize to blockSize / 2, actually we just reassigned the first iteration in v2 method to each thread, so in this v3
    for (unsigned int index = blockDim.x / 2; index > 0; index >>= 1) {
        if (tid < index) {
            shared_float_128[tid] += shared_float_128[tid + index];
        }
        __syncthreads();
    }
    if (tid == 0) output[blockIdx.x] = shared_float_128[0];
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
    int gridSize = std::min(((N + blockSize - 1) / blockSize), deviceProp.maxGridSize[0]);  // keep gridSize because we are not reallocate data to block we are reallocate threads

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
    dim3 Block(blockSize / 4);  // number of threads
    float millisecond;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    accumulator_v3<blockSize / 4><<<Grid, Block>>>(din, dout);
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
