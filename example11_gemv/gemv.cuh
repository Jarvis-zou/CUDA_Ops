#include "pch.cuh"
#define warpSize 32

static const char* _cudaGetErrorEnum(cudaError_t error) { return cudaGetErrorString(error); }


#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)

// vectorized load
template<typename T>
struct Vec {
    static constexpr int size = 4;
};

template<>
struct Vec<half> {
    static constexpr int size = 8;
};

template<typename T>
struct SumOp {
    __device__ __forceinline__ T operator()(const T &a, const T &b) const { return a + b; }
};

template<>
struct SumOp<half> {
    __device__ __forceinline__ half operator()(const half &a, const half &b) { return __hadd(a, b); }
};



template<template<typename> class ReduceOp, typename T>
__device__ __forceinline__ T warpReduce(T val) {
    for (int mask = warpSize / 2; mask > 0; mask >>= 1) {
        val = ReduceOp<T>()(val, __shfl_xor_sync(0xffffffff, val, mask));
    }
    return val;
}

template<template<typename> class ReduceOp, typename T>
__device__ __forceinline__ T blockReduce(T val) {
    uint32_t tid = threadIdx.x;
    uint32_t warpId = tid / 32;
    uint32_t laneId = tid % 32;
    uint32_t warpPerBlock = (blockDim.x + 31) / 32;
    __shared__ float warpRes[8];

    val = warpReduce<ReduceOp, T>(val);  // sum of all threads in this warp

    if (laneId == 0) { warpRes[warpId] = val; }
    __syncthreads();

    T warpSum = tid < warpPerBlock ? warpRes[tid] : 0;

    return warpReduce<ReduceOp, T>(warpSum);
}

template<int VEC_PER_THREAD, int VEC_SIZE>
__global__ void gemv(float *mat, float *vec, float *res, int cols) {
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    float threadSum = 0.0f;

    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        float4 mat4 = reinterpret_cast<float4*>(mat)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid];
        float4 vec4 = reinterpret_cast<float4*>(vec)[i * blockDim.x + tid];
        threadSum += mat4.x * vec4.x;
        threadSum += mat4.y * vec4.y;
        threadSum += mat4.z * vec4.z;
        threadSum += mat4.w * vec4.w;
    }

    float blockSum = blockReduce<SumOp, float>(threadSum);

    // write result
    if (tid == 0) { res[blockIdx.x] = blockSum; }
}

template<int VEC_PER_THREAD, int VEC_SIZE>
__global__ void gemv(half *mat, half *vec, half *res, int cols) {
    uint32_t tid = threadIdx.x;
    uint32_t bid = blockIdx.x;
    float threadSum = 0.0f;

    for (int i = 0; i < VEC_PER_THREAD; ++i) {
        float4 mat4 = reinterpret_cast<float4*>(mat)[bid * (cols / VEC_SIZE) + i * blockDim.x + tid];
        float4 vec4 = reinterpret_cast<float4*>(vec)[i * blockDim.x + tid];

        // convert float4 to half2
        auto *vec_h1 = (half2*)&vec4.x;
        auto *vec_h2 = (half2*)&vec4.y;
        auto *vec_h3 = (half2*)&vec4.z;
        auto *vec_h4 = (half2*)&vec4.w;
        auto *mat_h1 = (half2*)&mat4.x;
        auto *mat_h2 = (half2*)&mat4.y;
        auto *mat_h3 = (half2*)&mat4.z;
        auto *mat_h4 = (half2*)&mat4.w;
        half2 res1 = __hmul2(*mat_h1, *vec_h1);
        half2 res2 = __hmul2(*mat_h2, *vec_h2);
        half2 res3 = __hmul2(*mat_h3, *vec_h3);
        half2 res4 = __hmul2(*mat_h4, *vec_h4);
        half2 tempRes = __hadd2(__hadd2(__hadd2(res1, res2), res3), res4);
        threadSum = __hadd(tempRes.x, tempRes.y);
    }

    half blockSum = blockReduce<SumOp, half>(threadSum);

    // write result
    if (tid == 0) { res[blockIdx.x] = blockSum; }
}

template<int VEC_PER_THREAD, int VEC_SIZE, int NUM_THREADS>
struct DispatchLauncher {
    template<typename T>
    static void launcher(T *d_mat, T *d_vec, T *d_dst, int M, int N) {
        dim3 Grid(M);
        dim3 Block(NUM_THREADS);

        // launch
        float time = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        gemv<VEC_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N);
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr) {
            throw std::runtime_error(std::string("[CUDA] Kernel Error: ") +
                                     std::string(_cudaGetErrorEnum(kernelErr)) + " " +__FILE__ + ": " +
                                     std::to_string(__LINE__) + "\n");
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("[CUDA] GEMV Latency: %fms\n", time);
    }
};


namespace gemv2 {
    struct half8 {
        half2 h1;
        half2 h2;
        half2 h3;
        half2 h4;

        __device__ half8& operator=(const half8& h8) {
            h1 = h8.h1;
            h2 = h8.h2;
            h3 = h8.h3;
            h4 = h8.h4;
            return *this;
        }
    };

    template<int M, typename T>
    struct get_threads_per_mat_row { static const int value = M * sizeof(T) / 16; };

    inline __device__ float add(float a, float b) { return a + b; }
    inline __device__ half add(const half a, const half b) { return __hadd(a, b); }

    inline __device__ float4 add(float4 a, float4 b){
        float4 c;
        c.x = gemv2::add(a.x, b.x);
        c.y = gemv2::add(a.y, b.y);
        c.z = gemv2::add(a.z, b.z);
        c.w = gemv2::add(a.w, b.w);
        return c;
    }



    inline __device__ half2 add(const half2& a, const half2& b) {
        half2 res;
        res.x = gemv2::add(a.x, b.x);
        res.y = gemv2::add(a.y, b.y);
        return res;
    }

    inline __device__ half8 add(const half8& a, const half8& b) {
        half8 c{};
        c.h1 = gemv2::add(a.h1, b.h1);
        c.h2 = gemv2::add(a.h2, b.h2);
        c.h3 = gemv2::add(a.h3, b.h3);
        c.h4 = gemv2::add(a.h4, b.h4);
        return c;
    }

    inline __device__ half fma(half a, half b, half c) { return __float2half((float)a * (float)b + (float)c); }


    inline __device__ half2 fma(half a, const half2& b, const half2& c) {
        half2 res;
        res.x = gemv2::fma(a, b.x, c.x);
        res.y = gemv2::fma(a, b.y, c.y);
        return res;
    }

    inline __device__ half8 fma(half a, const half8& b, const half8& c) {
        half8 d{};
        d.h1 = gemv2::fma(a, b.h1, c.h1);
        d.h2 = gemv2::fma(a, b.h2, c.h2);
        d.h3 = gemv2::fma(a, b.h3, c.h3);
        d.h4 = gemv2::fma(a, b.h4, c.h4);
        return d;
    }

    inline __device__ float fma(float a, float b, float c) { return a * b + c; }

    inline __device__ float4 fma(float a, float4 b, float4 c) {
        float4 d;
        d.x = gemv2::fma(a, b.x, c.x);
        d.y = gemv2::fma(a, b.y, c.y);
        d.z = gemv2::fma(a, b.z, c.z);
        d.w = gemv2::fma(a, b.w, c.w);
        return d;
    }
}


template<int THREADS_PER_BLOCK, int VALUES_PER_THREAD, int VEC_SIZE>
__global__ void gemv2_kernel(float* matrix, float* vector, float* res, int N, int M) {
    uint32_t tid = threadIdx.x;
    int mat_o = tid / VALUES_PER_THREAD;
    int mat_i = tid % VALUES_PER_THREAD * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / VALUES_PER_THREAD;
    __shared__ float out_smem[512];
    float4 out;

    // Row accumulator
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);
        float logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }

    // block-level binary accumulator
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();
        if (mat_o < midpoint) {
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }

    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;
    }
}

template<int THREADS_PER_BLOCK, int VALUES_PER_THREAD, int VEC_SIZE>
__global__ void gemv2_kernel(half* matrix, half* vector, half* res, int N, int M) {
    uint32_t tid = threadIdx.x;
    int mat_o = tid / VALUES_PER_THREAD;
    int mat_i = tid % VALUES_PER_THREAD * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / VALUES_PER_THREAD;
    __shared__ half out_smem[2048];
    gemv2::half8 out{};

    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        gemv2::half8 mat = *reinterpret_cast<gemv2::half8*>(&matrix[ti * M + mat_i]);
        half logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }

    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<gemv2::half8*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();

        if (mat_o < midpoint) {
            out = gemv2::add(*reinterpret_cast<gemv2::half8*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<gemv2::half8*>(&res[mat_i]) = out;
    }
}

template<int THREADS_PER_BLOCK, int VALUES_PER_THREAD, int VEC_SIZE, typename T>
__global__ void gemv2_kernel_template(T* matrix, T* vector, T* res, int N, int M) {
    int tid = threadIdx.x;
    int mat_o = tid / VALUES_PER_THREAD;
    int mat_i = tid % VALUES_PER_THREAD * VEC_SIZE;
    constexpr int ROW_PER_ITER = THREADS_PER_BLOCK / VALUES_PER_THREAD;
    __shared__ T out_smem[512];
    float4 out;
    for (int ti = mat_o; ti < N; ti += ROW_PER_ITER) {
        float4 mat = *reinterpret_cast<float4*>(&matrix[ti * M + mat_i]);
        T logits = vector[ti];
        out = gemv2::fma(logits, mat, out);
    }
    for (int ROWS_PER_BLOCK = ROW_PER_ITER; ROWS_PER_BLOCK >= 2; ROWS_PER_BLOCK /= 2) {
        int midpoint = ROWS_PER_BLOCK / 2;
        if (mat_o >= midpoint && mat_o < ROWS_PER_BLOCK) {
            *reinterpret_cast<float4*>(&out_smem[(mat_o - midpoint) * M + mat_i]) = out;
        }
        __syncthreads();
        if (mat_o < midpoint) {
            out = gemv2::add(*reinterpret_cast<float4*>(&out_smem[mat_o * M + mat_i]), out);
        }
        __syncthreads();
    }
    if (mat_o == 0) {
        *reinterpret_cast<float4*>(&res[mat_i]) = out;
    }
}

template<int THREADS_PER_BLOCK, int VALUES_PER_THREAD, int VEC_SIZE>
struct DispatchLauncher2
{
    template<typename T>
    static void launcher(T* d_mat, T* d_vec, T* d_dst, int M, int N){
        dim3 Grid(1);
        dim3 Block(THREADS_PER_BLOCK);

        float time = 0;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        gemv2_kernel<THREADS_PER_BLOCK, VALUES_PER_THREAD, VEC_SIZE><<<Grid, Block>>>(d_mat, d_vec, d_dst, N, M);
        cudaError_t kernelErr = cudaGetLastError();
        if (kernelErr) {
            throw std::runtime_error(std::string("[CUDA] Kernel Error: ") +
                                     std::string(_cudaGetErrorEnum(kernelErr)) + " " +__FILE__ + ": " +
                                     std::to_string(__LINE__) + "\n");
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        printf("[CUDA] GEMV Latency: %fms\n", time);
    }
};






































