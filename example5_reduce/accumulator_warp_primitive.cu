#include "pch.cuh"
#define N 25600000
#define WarpSize 32

__device__ float warpReduce(float sum) {
    /* parsing the curr value store in curr thread register(which is the input param sum) to threads after fixed offset
     * this method don't need pre-allocated public space like global mem or shared mem, so we don't need to allocate space
     * and parse pointer to this function. Instead, we can use __shfl_down_sync to load data parallel and don't need to sync warp or threads
     * */
    sum += __shfl_down_sync(0xffffffff, sum, 16); // sum stored in tid0 shifted to tid16, then sum value stored in tid16 returned, then value(tid0) = value(tid0) + value(tid16)
    sum += __shfl_down_sync(0xffffffff, sum, 8);
    sum += __shfl_down_sync(0xffffffff, sum, 4);
    sum += __shfl_down_sync(0xffffffff, sum, 2);
    sum += __shfl_down_sync(0xffffffff, sum, 1);
    return sum;
}

template<int blockSize>
__global__ void accumulator_warp_primitive(const float *input, float *output, int size) {
    /* Using warp instead of block to handle data, now */
    unsigned int tid = threadIdx.x;  // thread id in this block
    unsigned int global_tid = blockDim.x * blockIdx.x + threadIdx.x;  // thread id in global thread pool
    unsigned int steps = gridDim.x * blockDim.x;  // let one block handle multiple blocks data

    // first assign data to this block by adding additional data by steps
    float sum = 0.0f;  // This sum contains sum of multiple values and is stored in the register of curr thread
    for (int i = global_tid; i < size; i += steps) {
        sum += input[i];
    }

    __shared__ float WarpSums[blockSize / WarpSize]; // each index stores sum of one warp(sum of 32 threads)
    const int laneID = tid % WarpSize;  // tid in each warp, ep: tid = 0 -> laneID = 0(warp0), tid=32 -> laneID = 0(warp1)
    const int warpID = tid / WarpSize;  // which warp curr thread in

    sum = warpReduce(sum);

    if (laneID == 0) WarpSums[warpID] = sum;
    __syncthreads();  // WarpSums is on shared mem, we need to sync threads after operations on it.

    /* All float data in this warp now is added together and stored in the first thread inside this warp
     * Now we have blockSize / WarpSize data stored in the array WarpSums, we still need to add these data together
     * We can use the first warp to do this, before we do this, we have to shift data in WarpSums to the first warp */
    sum = (tid < blockSize / warpSize) ? WarpSums[laneID] : 0.0f;  // ep: blockSize / WarpSize = 256 / 32 = 8, we need to shift 8 float data to first 8 threads in warp0 and set left 24 threads to 0

    if (warpID == 0) sum = warpReduce(sum);

    if (tid == 0) output[blockIdx.x] = sum;
}

void checkResults(const float *input, float *groundTruth, int size) {
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
    float *din, *dout, *dout_block;

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
    accumulator_warp_primitive<blockSize><<<Grid, Block>>>(din, dout_block, N);  // first get result of each block
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);

    // copy GPU result to CPU
    cudaMemcpy(hout, dout_block, gridSize * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;


    // call CPU function
    accumulator_baseline_cpu(hin, hout_cpu);
    std::cout << "[CPU] Sum = " << *hout_cpu << std::endl;
    checkResults(hout,hout_cpu, gridSize);

    // free resource
    cudaFree(din);
    cudaFree(dout);

    free(hin);
    free(hout);
    free(hout_cpu);

}
