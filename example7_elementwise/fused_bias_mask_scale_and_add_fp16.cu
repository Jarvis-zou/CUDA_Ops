#include "pch.cuh"
#define N 25600000

template<typename T, int size>
struct alignas(sizeof(T) * size) AlignedVectorHalf {
// This struct has a T * Size alignment in space
float data[size];

// Overload[] for user-friendly call
__host__ __device__ inline const T& operator[](int i) const { return data[i]; }  // This won't be used (No const members)
__host__ __device__ inline T& operator[](int i) { return data[i]; }
};

template<typename T>
struct BiasMaskScaleAddFunctor {
    const T *bias;
    const int biasSize;
    const T *mask;
    const T scale;
    const T *add;

    // Init members
    BiasMaskScaleAddFunctor(T* bias, const int biasSize, T *mask, const T scale, T *add)
            : bias(bias), biasSize(biasSize), mask(mask), scale(scale), add(add) {}

    __device__ void compute(T *address, T *dst, int idx) {
        auto x = *reinterpret_cast<const half2*>(address); // x contains 2 fp16 values
        auto bias2 = *reinterpret_cast<const half2*>(bias+idx);
        auto mask2 = *reinterpret_cast<const half2*>(mask+idx);
        auto scale2 = __float2half2_rn(scale);
        auto add2 = *reinterpret_cast<const half2*>(add+idx);
//        printf("G_TID: %d\n", idx);
//        printf("x[idx].x=%f, x[idx].y=%f\n", __half2float(x.x), __half2float(x.y));
//        printf("bias2[idx].x=%f, bias2[idx].y=%f\n", __half2float(bias2.x), __half2float(bias2.y));
//        printf("mask2[idx].x=%f, mask2[idx].y=%f\n", __half2float(mask2.x), __half2float(mask2.y));
//        printf("add2[idx].x=%f, add2[idx].y=%f\n", __half2float(add2.x), __half2float(add2.y));

        const half2 res = __hadd2(__hmul2(__hmul2(__hadd2(x, bias2), mask2), scale2), add2);
//        printf("res[idx].x=%f, res[idx].y=%f\n", __half2float(res.x), __half2float(res.y));

        *reinterpret_cast<half2*>(dst) = res;
    }
};

template<typename Functor, typename T, int vectorSize>
__global__ void FusedBiasMaskScaleAddKernel(Functor functor, T *x, T *y) {
    unsigned int global_tid = (blockDim.x * blockIdx.x + threadIdx.x) * vectorSize;
    unsigned int stride = (gridDim.x * blockDim.x) * vectorSize;

    // In case number of threads < total size of data
    for (; global_tid < N; global_tid += stride) {
        for (int i = 0; i < vectorSize; i += 2) {
            functor.compute(x + global_tid + i, y + global_tid, global_tid + i);
        }
    }
}


int main() {
    half *hx, *hy;
    half *dx, *dy;

    float scale = 0.5f;
    int biasSize = 10; // recurrently apply 10 bias to all x
    half *h_mask, *d_mask;
    half *h_bias, *d_bias;
    half *h_add, *d_add;

    // Allocate CPU memory
    hx = (half*)malloc(N * sizeof(half));
    hy = (half*)malloc(N * sizeof(half));
    h_bias = (half*)malloc(N * sizeof(half));
    h_mask = (half*)malloc(N * sizeof(half));
    h_add = (half*)malloc(N * sizeof(half));

    // Init data
    for (auto i = 0; i < N; i++) {
        h_bias[i] = static_cast<half>(i % 10);
    }
    for (auto i = 0; i < N; i++) {
        hx[i] = static_cast<half>(i);
        h_mask[i] = static_cast<half>(i % 2);  // 010101...
        h_add[i] = static_cast<half>(i);
    }


    // Allocate GPU memory
    cudaMalloc((void**)&dx, N * sizeof(half));
    cudaMalloc((void**)&d_bias, N * sizeof(half));
    cudaMalloc((void**)&d_mask, N * sizeof(half));
    cudaMalloc((void**)&d_add, N * sizeof(half));
    cudaMalloc((void**)&dy, N * sizeof(half));

    // Convert float input to half input
    cudaMemcpy(dx, hx, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, N * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_add, h_add, N * sizeof(half), cudaMemcpyHostToDevice);

    // Get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 1024;  // max threads per block
    const int gridSize = (N / 8 + blockSize - 1) / blockSize;

    // Check if the memory is aligned
    auto is_aligned = [](const void *p, const int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    // Call kernel function
    float millisecond = 0.0f;
    constexpr auto vectorAlignmentHalf = alignof(AlignedVectorHalf<half, 8>);
    if (N % 8 == 0 && is_aligned(dx, vectorAlignmentHalf) && is_aligned(dy, vectorAlignmentHalf)) {
        dim3 Grid(gridSize);  // number of blocks
        dim3 Block(blockSize);  // number of threads
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        auto fused_func = BiasMaskScaleAddFunctor<half>(d_bias, biasSize, d_mask, scale, d_add);
        FusedBiasMaskScaleAddKernel<BiasMaskScaleAddFunctor<half>, half,8><<<Grid, Block>>>(fused_func, dx, dy);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&millisecond, start, stop);
    }


    // Copy GPU result to CPU
    cudaMemcpy(hy, dy, N * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;


    // Free resources
    cudaFree(dx);
    cudaFree(d_bias);
    cudaFree(d_mask);
    cudaFree(d_add);
    cudaFree(dy);

    free(hx);
    free(hy);
    free(h_bias);
    free(h_mask);
    free(h_add);

    return 0;
}
