#include "pch.cuh"
#define N 25600000

template<int blockSize>
__global__ void accumulator_v1(const float *input, float *output) {
    // Example for using shared memory, each block has individual shared memory space.
    __shared__ float shared_float_256[blockSize]; // init an array stores 256 float numbers on shared memory for blockSize(256) threads to use
    unsigned int tid = threadIdx.x;  // thread id in this block
    unsigned int global_tid = blockDim.x * blockIdx.x + threadIdx.x;  // thread id in global thread pool
    shared_float_256[tid] = input[global_tid];  // load corresponding float value into block memory
    __syncthreads();

    // this method eliminates warp divergence overhead
    // First step = 1, means threads [(0,1), (2,3), ..., (254, 255)] will add together and store results at [0, 2, 4, ..., 254]
    // Second step = 2, first round of results have been stored at tid [0, 2, 4, ..., 254], then add [(0,2), (4,6), ..., (252, 254)], stored at [0, 4, 8, ..., 252]
    // keep iteration till all data added and stored
    for (unsigned int step = 1; step < blockDim.x; step *= 2) {
//        if (tid % (2 * step) == 0) {  // v0.1
//        if ((tid & (2 * step - 1)) == 0) {  // v0.2 should be a little bit faster
        unsigned int index = 2 * step * tid;  // v0.3 step=1, first 4 warp(128 threads) process; step=2 first 2 warps process;  step=4, first 1 warp process
        if (index < blockDim.x) {
//             shared_float_256[tid] += shared_float_256[tid + step];  // v0.1/v0.2
            shared_float_256[index] += shared_float_256[index + step];  // v0.3
            __syncthreads();
        }
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
    accumulator_v1<blockSize><<<Grid, Block>>>(din, dout);
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
