#include "pch.cuh"
template<typename T>
__device__ __forceinline__ T WarpReduce(T val) {
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);  // accumulator reduce 
    }
    return val;
}

template<typename T>
__device__ T BlockReduce(T val) {
    static __shared__ T shared[32]; // shared space inside block
    int lane = threadIdx.x % warpSize;
    int wid = threadIdx.x / warpSize;

    val = WarpReduce(val); // warp reduce

    if (lane == 0) shared[wid] = val; // reduce result of this block
    __syncthreads();

    val = (threadIdx.x < blockDim.x / warpSize) ? shared[lane] : 0;

    if (wid == 0) val = WarpReduce(val); // block reduce

    return val;
}

__global__ void warpReduceKernel(float *data, float *result, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? data[tid] : 0;

    val = WarpReduce(val);

    if (threadIdx.x % warpSize == 0) {
        atomicAdd(result, val);
    }
}

__global__ void blockReduceKernel(float *data, float *result, int N) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (tid < N) ? data[tid] : 0;

    val = BlockReduce(val);

    if (threadIdx.x == 0) {
        atomicAdd(result, val);
    }
}

void benchmark(int N) {
    float *d_data, *d_result;
    float *h_result = new float;

    cudaMalloc(&d_data, N * sizeof(float));
    cudaMalloc(&d_result, sizeof(float));

    float *h_data = new float[N];
    for (int i = 0; i < N; i++) h_data[i] = 1.0f;

    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(256);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Warp Reduce
    cudaMemset(d_result, 0, sizeof(float));
    cudaEventRecord(start);
    warpReduceKernel<<<gridSize, blockSize>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float warpReduceTime;
    cudaEventElapsedTime(&warpReduceTime, start, stop);

    // Block Reduce
    cudaMemset(d_result, 0, sizeof(float));
    cudaEventRecord(start);
    blockReduceKernel<<<gridSize, blockSize>>>(d_data, d_result, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float blockReduceTime;
    cudaEventElapsedTime(&blockReduceTime, start, stop);

    cudaMemcpy(h_result, d_result, sizeof(float), cudaMemcpyDeviceToHost);

    if (warpReduceTime < blockReduceTime) {
        printf("N = %d, Warp Reduce Time = %f ms, Block Reduce Time = %f ms, Result = %f, Warp Faster\n",
               N, warpReduceTime, blockReduceTime, *h_result);
    }

    if (warpReduceTime > blockReduceTime) {
        printf("N = %d, Warp Reduce Time = %f ms, Block Reduce Time = %f ms, Result = %f, Block Faster\n",
               N, warpReduceTime, blockReduceTime, *h_result);
    }

    delete[] h_data;
    delete h_result;
    cudaFree(d_data);
    cudaFree(d_result);
}

int main() {
    for (int N = 32; N <= 1024 * 1024 * 128; N *= 2) {
        benchmark(N);
    }
    return 0;
}
