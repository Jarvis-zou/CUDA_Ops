#include "pch.cuh"
#define N 4

template<int size>
struct alignas(sizeof(float) * size) AlignedVectorFloat {
// This struct has a T * Size alignment in space
float data[size];

// Overload[] for user-friendly call
__host__ __device__ inline const float& operator[](int i) const { return data[i]; }  // This won't be used (No const members)
__host__ __device__ inline float& operator[](int i) { return data[i]; }
};

template<typename T>
struct BiasMaskScaleAddFunctor {
    const T *bias;
    const int biasSize;
    uint32_t *mask;
    const T scale;
    T *add;

    // Init members
    BiasMaskScaleAddFunctor(const float* bias, const int biasSize, uint32_t *mask, const float scale, float *add)
            : bias(bias), biasSize(biasSize), mask(mask), scale(scale), add(add) {}

    __device__ void compute(T *address, T *dst, int idx) {
        float4 x = *reinterpret_cast<const float4*>(address); // x contains 4 float values
        int4 mask4 = *reinterpret_cast<const int4*>(mask);
        float4 add4 = *reinterpret_cast<const float4*>(add);

        x.x = __fadd_rn(__fmul_rn(__fmul_rn(__fadd_rn(x.x, bias[idx % biasSize]), mask4.x), scale), add4.x);
        x.y = __fadd_rn(__fmul_rn(__fmul_rn(__fadd_rn(x.y, bias[(idx + 1) % biasSize]), mask4.y), scale), add4.y);
        x.z = __fadd_rn(__fmul_rn(__fmul_rn(__fadd_rn(x.z, bias[(idx + 2) % biasSize]), mask4.z), scale), add4.z);
        x.w = __fadd_rn(__fmul_rn(__fmul_rn(__fadd_rn(x.w, bias[(idx + 3) % biasSize]), mask4.w), scale), add4.w);

        *reinterpret_cast<float4*>(dst) = x;
    }
};

template<typename Functor, typename T, int vectorSize>
__global__ void FusedBiasMaskScaleAddKernel(Functor functor, T *x, T *y) {
    unsigned int global_tid = (blockDim.x * blockIdx.x + threadIdx.x) * vectorSize;
    unsigned int stride = (gridDim.x * blockDim.x) * vectorSize;

    // In case number of threads < total size of data
    for (; global_tid < N; global_tid += stride) {
        functor.compute(x + global_tid, y + global_tid, global_tid);
    }
}

void FusedBiasMaskScaleAdd_CPU(const float *x, const float* bias, const int biasSize, const uint32_t *mask, const float scale, const float *add, float *y) {
    for (auto i = 0; i < N; i++) {
        y[i] = (x[i] + bias[i % biasSize]) * static_cast<float>(mask[i]) * scale + add[i];
    }
}

bool checkResults(const float *res_cpu, const float *res_gpu) {
    for (unsigned int i = 0 ; i < N; i++) {
        if (res_cpu[i] != res_gpu[i]) {
            std::cout << "Check Failed: res=" << res_gpu[i] << " Ground Truth=" << res_cpu[i] << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    float *hx, *hy, *hy_cpu;
    float *dx, *dy;

    float scale = 0.5f;
    float biasSize = 10; // recurrently apply 10 bias to all x
    uint32_t *h_mask, *d_mask;
    float *h_bias, *d_bias;
    float *h_add, *d_add;

    // Allocate CPU memory
    hx = (float*)malloc(N * sizeof(float));
    hy = (float*)malloc(N * sizeof(float));
    hy_cpu = (float*)malloc(N * sizeof(float));
    h_bias = (float*)malloc(biasSize * sizeof(float));
    h_mask = (uint32_t*)malloc(N * sizeof(uint32_t));
    h_add = (float*)malloc(N * sizeof(float));

    // Init data
    for (auto i = 0; i < biasSize; i++) {
        h_bias[i] = static_cast<float>(i);
    }
    for (auto i = 0; i < N; i++) {
        hx[i] = static_cast<float>(i);
        h_mask[i] = static_cast<uint32_t>(i % 2);  // 010101...
        h_add[i] = static_cast<float>(i);
    }


    // Allocate GPU memory
    cudaMalloc((void**)&dx, N * sizeof(float));
    cudaMalloc((void**)&d_bias, biasSize * sizeof(float));
    cudaMalloc((void**)&d_mask, N * sizeof(uint32_t));
    cudaMalloc((void**)&d_add, N * sizeof(float));
    cudaMalloc((void**)&dy, N * sizeof(float));

    // Convert float input to half input
    cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, biasSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, N * sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_add, h_add, N * sizeof(float), cudaMemcpyHostToDevice);

    // Get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 1024;  // max threads per block
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Check if the memory is aligned
    auto is_aligned = [](const void *p, const int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };

    // Call kernel function
    float millisecond = 0.0f;
    constexpr auto vectorAlignmentFloat = alignof(AlignedVectorFloat<4>);
    if (N % 4 == 0 && is_aligned(dx, vectorAlignmentFloat) && is_aligned(dy, vectorAlignmentFloat)) {
        dim3 Grid(gridSize / 4);  // number of blocks
        dim3 Block(blockSize);  // number of threads
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        auto fused_func = BiasMaskScaleAddFunctor<float>(d_bias, biasSize, d_mask, scale, d_add);
        FusedBiasMaskScaleAddKernel<BiasMaskScaleAddFunctor<float>, float, 4><<<1, 1>>>(fused_func, dx, dy);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&millisecond, start, stop);
    }


    // Copy GPU result to CPU
    cudaMemcpy(hy, dy, N * sizeof(float), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;

    // Check results
    FusedBiasMaskScaleAdd_CPU(hx, h_bias, biasSize, h_mask, scale, h_add, hy_cpu);
    checkResults(hy_cpu, hy);


    // Free resources
    cudaFree(dx);
    cudaFree(d_bias);
    cudaFree(d_mask);
    cudaFree(d_add);
    cudaFree(dy);

    free(hx);
    free(hy);
    free(hy_cpu);
    free(h_bias);
    free(h_mask);
    free(h_add);

    return 0;
}
