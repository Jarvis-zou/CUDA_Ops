#include "pch.cuh"
#define N 256000000

template<typename T>
struct BiasMaskScaleAddFunctor {
    const T *bias;
    const int biasSize;
    const uint8_t *mask;
    const T scale;
    const T *add;

    // Init members
    BiasMaskScaleAddFunctor(const float* bias, const int biasSize, const uint8_t *mask, const float scale, const float *add)
            : bias(bias), biasSize(biasSize), mask(mask), scale(scale), add(add) {}

    __device__ T compute(T *x, unsigned int idx) {
//        return __fadd_rn(__fmul_rn(__fmul_rn(__fadd_rn(x[idx], bias[idx % biasSize]), static_cast<float>(mask[idx])), scale), add[idx]);
        return (x[idx] + bias[idx % biasSize]) * static_cast<float>(mask[idx]) * scale + add[idx];
    }

};

template<typename Functor, typename T>
__global__ void FusedBiasMaskScaleAddKernel(Functor functor, T *x, T *y) {
    unsigned int global_tid = blockDim.x * blockIdx.x + threadIdx.x;

    // In case number of threads < total size of data
    for (auto i = global_tid; i < N; i += gridDim.x * blockDim.x) {
        y[i] = functor.compute(x, i);
    }
}

void FusedBiasMaskScaleAdd_CPU(const float *x, const float* bias, const int biasSize, const uint8_t *mask, const float scale, const float *add, float *y) {
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
    uint8_t *h_mask, *d_mask;
    float *h_bias, *d_bias;
    float *h_add, *d_add;

    // Allocate CPU memory
    hx = (float*)malloc(N * sizeof(float));
    hy = (float*)malloc(N * sizeof(float));
    hy_cpu = (float*)malloc(N * sizeof(float));
    h_bias = (float*)malloc(biasSize * sizeof(float));
    h_mask = (uint8_t*)malloc(N * sizeof(uint8_t));
    h_add = (float*)malloc(N * sizeof(float));

    // Init data
    for (auto i = 0; i < biasSize; i++) {
        h_bias[i] = static_cast<float>(i);
    }
    for (auto i = 0; i < N; i++) {
        hx[i] = static_cast<float>(i);
        h_mask[i] = static_cast<uint8_t>(i % 2);  // 010101...
        h_add[i] = static_cast<float>(i);
    }


    // Allocate GPU memory
    cudaMalloc((void**)&dx, N * sizeof(float));
    cudaMalloc((void**)&d_bias, biasSize * sizeof(float));
    cudaMalloc((void**)&d_mask, N * sizeof(uint8_t));
    cudaMalloc((void**)&d_add, N * sizeof(float));
    cudaMalloc((void**)&dy, N * sizeof(float));

    // Convert float input to half input
    cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias, h_bias, biasSize * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, N * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_add, h_add, N * sizeof(float), cudaMemcpyHostToDevice);

    // Get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 1024;  // max threads per block
    const int gridSize = (N + blockSize - 1) / blockSize;

    // Call kernel function
    float millisecond = 0.0f;
    dim3 Grid(gridSize);  // number of blocks
    dim3 Block(blockSize);  // number of threads
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    auto fused_func = BiasMaskScaleAddFunctor<float>(d_bias, biasSize, d_mask, scale, d_add);
    FusedBiasMaskScaleAddKernel<BiasMaskScaleAddFunctor<float>, float><<<Grid, Block>>>(fused_func, dx, dy);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);


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
