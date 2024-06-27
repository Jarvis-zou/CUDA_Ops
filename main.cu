#include "pch.cuh"
#include <algorithm>
#define N 1024000  // 1024 values/row, 1000 rows
#define warpSize 32

template<typename T, int size>
struct alignas(sizeof(T) * size) VectorType {
    // This struct has a T * Size alignment in space
    T data[size];

    // Overload[] for user-friendly call
    __host__ __device__ inline const T& operator[](int i) const { return data[i]; }  // This won't be used (No const members)
    __host__ __device__ inline T& operator[](int i) { return data[i]; }
};

// sum of two float
template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T a, const T b) { return __fadd_rn(a, b); }
};


// maximum of two float
template<typename T>
struct MaxOp {
    __device__ __forceinline__ T operator()(const T a, const T b) const { return fmaxf(a, b); }
};


// base e exponential of x (e ^ x)
template<typename T>
__device__ __forceinline__ T Exp(T x) { return exp(x); }
template<>
__device__ __forceinline__ float Exp<float>(float x) { return expf(x); }


// Inf
template<typename T>
__device__ __forceinline__ T Inf() { return static_cast<T>(10000000000); }
template<>
__device__ __forceinline__ float Inf<float>() { return 10000000000.0f; }


// Divide a with b, a / b
template<typename T>
__device__ __forceinline__ T Div(const T a, const T b) { return a / b; }
template<>
__device__ __forceinline__ float Div (const float a, const float b) { return fdividef(a, b); }


template<template<typename> class ReductionOp, typename T>
__device__ __forceinline__ T WarpReduce(T val) {
    // normal warp-level reduce operations
    # pragma unroll
    for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
        val = ReductionOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<typename T, int vectorSize>
__device__ void write(T *src, T *dst, const unsigned int rowIdx, const unsigned int rowSize, const unsigned int colIdx) {
    using VecType = VectorType<T, vectorSize>;  // demonstrate write size
    const unsigned int offset = (rowIdx * rowSize + colIdx) / vectorSize;  // before:0, 4, 8, after vectorized:0, 1, 2
    *(reinterpret_cast<VecType*>(dst) + offset) = *reinterpret_cast<VecType*>(src);
}

template<typename T, int vectorSize>
__device__ void load(T *src, T *dst, const unsigned int rowIdx, const unsigned int rowSize, const unsigned int colIdx) {
    using VecType = VectorType<T, vectorSize>;  // demonstrate load size
    const unsigned int offset = (rowIdx * rowSize + colIdx) / vectorSize;  // before:0, 4, 8, after vectorized:0, 1, 2
    *reinterpret_cast<VecType*>(dst) = *(reinterpret_cast<VecType*>(src) + offset);
}

template<typename T, int vectorSize, int bufferCols>
__global__ void Softmax(T *input, T *dst, const unsigned int rows, const unsigned int cols) {
    const unsigned int laneID = threadIdx.x; // threadID inside each warp, i.e 0-31
    const unsigned int global_warpID = blockIdx.y * blockDim.y + threadIdx.y;  // 0 * 8 + 0 = 0; 1 * 8 + 0 = 8...
    const unsigned int stride = gridDim.y * blockDim.y; // numBlocks * numWarpPerBlock = 125 * 8 == 1000

    // Init data buffer, bufferRows = 2 if dataSize % (warpSize * vectorSize)
    VectorType<T, vectorSize> buf[bufferCols];  // each buffer contains data handled by this thread

    // In case we don't have enough warps to handle all rows of data, need some warps to handle multiple rows of data
    for (unsigned int row = global_warpID; row < rows; row += stride) {
        /*
         * Step1: Get max value of this row
         */
        // Set registers to store maximum value of all data handled by this thread
        VectorType<T, vectorSize> threadMax{};  // register contains max value of all data in this row handled by this thread
        for (int i = 0; i < vectorSize; i++) {
            threadMax[i] = -Inf<T>(); // init maximum
        }

        // Using warp slide through this row, this mean one thread handle multiple data, i.e. thread0 get maximum value among index 0, 32, 64...
        int bufIdx = 0;
        for (unsigned int col = laneID * vectorSize; col < cols; col += warpSize * vectorSize) {  // warp sliding window process
            // Only load vector data once
            load<T, vectorSize>(input, buf[bufIdx].data, row, cols, col);  // load x into buffer

            // If we choose to load 4 values per thread, we still need to do 4 softmax ops separately, so we will do
            // warpMax on 4 values separately,
            #pragma unroll
            for (int i = 0; i < vectorSize; i++) {
                threadMax[i] = MaxOp<T>()(threadMax[i], buf[bufIdx][i]);  // update maximum value
            }
            bufIdx++;
        }

        // Get maximum value of this row using warp-level reduce operator
        T rowMax;
        #pragma unroll
        for (int i = 0; i < vectorSize; i++) {
            rowMax = WarpReduce<MaxOp, T>(threadMax[i]);
        }

        /*
         * Step2: Get sum of this row        *
         */
        // Using warp slide through this row, this mean one thread handle multiple data, i.e. thread0 get sum of index 0, 32, 64...
        bufIdx = 0;  // reset
        T threadSum = 0.0f;
        for (unsigned int col = laneID * vectorSize; col < cols; col += warpSize * vectorSize) {  // warp sliding window process
            #pragma unroll
            for (int i = 0; i < vectorSize; i++) {
                buf[bufIdx][i] = Exp<T>(buf[bufIdx][i] - rowMax);  // exp(x - MAX)
                threadSum = SumOp<T>()(threadSum, buf[bufIdx][i]);  // update sum
            }
            bufIdx++;
        }

        // Get maximum value of this row using warp-level reduce operator
        T rowSum;
        #pragma unroll
        for (int i = 0; i < vectorSize; i++) {
            rowSum = WarpReduce<SumOp, T>(threadSum);
        }

        /*
         * Step3: Divide each value by warpSum
         * res = exp(x-MAX) / warpSum
         */
        bufIdx = 0;  // reset
        for (unsigned int col = laneID * vectorSize; col < cols; col += warpSize * vectorSize) {  // warp sliding window process
            #pragma unroll
            for (int i = 0; i < vectorSize; i++) {
                buf[bufIdx][i] = Div<T>(buf[bufIdx][i], rowSum);
            }
            // write data into same address in global memory
            write<T, vectorSize>(buf[bufIdx].data, dst, row, cols, col);
            bufIdx++;
        }
    }
}

void SoftmaxCPU(const float *input, float *res_cpu, const uint32_t rows, const uint32_t cols) {

    for (auto row = 0; row < rows; row++) {
        float MAX = *std::max_element(input+ row * cols, input + (row + 1) * cols); // max value of this row
        float total = 0;  // sum of this row

        // calculate sum of this row
        for (auto col = 0; col < cols; col++) {
            total += std::exp(input[row * cols + col] - MAX);
        }

        // calculate softmax result of each element in this row
        for (auto col = 0; col < cols; col++) {
            res_cpu[row * cols + col] = std::exp(input[row * cols + col] - MAX) / total;
        }
    }
}

bool checkResults(const float *res_cpu, const float *res_gpu) {
    const float epsilon = 1e-7;
    for (auto i = 0 ; i < N; i++) {
        if (std::fabs(res_gpu[i] - res_cpu[i]) > epsilon) {
            std::cout << "Index:" << i << std::endl;
            printf("Check Failed: GPU Result=%.32f, CPU Result=%.32f\n", res_gpu[i], res_cpu[i]);
            return false;
        }
    }
    return true;
}

int main() {
    float *hx, *hy, *hy_cpu;
    float *dx, *dy;


    // Allocate CPU memory
    hx = (float*)malloc(N * sizeof(float));
    hy = (float*)malloc(N * sizeof(float));
    hy_cpu = (float*)malloc(N * sizeof(float));


    // Init data
    for (auto i = 0; i < N; i++) {
        hx[i] = static_cast<float>(i % 8);
    }


    // Allocate GPU memory
    cudaMalloc((void**)&dx, N * sizeof(float));
    cudaMalloc((void**)&dy, N * sizeof(float));

    // Copy data
    cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);

    // Check if the memory is aligned

    // Call kernel function
    float millisecond = 0.0f;

    dim3 Grid(1, 125);  // 1 block/row, 125 rows, each block handle 8 rows of data ,total 1000 rows, we need 1000 / 8 = 125 blocks.
    dim3 Block(32, 8);  // 32 threads(1 warp)/row, 8 rows, one row handle one row of data, so each block handle 8 rows of data
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    Softmax<float, 4, 8><<<Grid, Block>>>(dx, dy, 1000, 1024);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);


    // Copy GPU result to CPU
    cudaMemcpy(hy, dy, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;

    // Check results
    SoftmaxCPU(hx, hy_cpu, 1000, 1024);
    checkResults(hy_cpu, hy);


    // Free resources
    cudaFree(dx);
    cudaFree(dy);

    free(hx);
    free(hy);
    free(hy_cpu);


    return 0;
}
