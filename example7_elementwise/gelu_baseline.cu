#include <iostream>
#include <cuda_fp16.h>
#define N 25600000
#define M_PI 3.14159265358979323846

__device__ half gelu(half x) {
    // first convert fp16 to fp32 to fit tanh and sqrt operations, then convert back to fp16.
    // This method is not 100% correct because when converting from fp32 to fp16, precision is different.
    float x_float = __half2float(x);
    float tanh_in = sqrtf(2.0 / M_PI) * (x_float + 0.044715 * x_float * x_float * x_float);
    float tanh_out = tanhf(tanh_in);
    float gelu_out = 0.5 * x_float * (1.0 + tanh_out);
    return __float2half(gelu_out);
}

__global__ void gelu_baseline(const half *input, half *output, const unsigned int data_size) {
    int global_tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (global_tid < data_size) {
        output[global_tid] = gelu(input[global_tid]);
    }
}

void gelu_cpu(const float *input, float *output, const unsigned int data_size) {
    for (unsigned int i = 0; i < data_size; ++i) {
        output[i] = 0.5 * input[i] * (1.0 + tanh(sqrt(2.0 / M_PI) * (input[i] + 0.044715 * input[i] * input[i] * input[i])));
    }
}

bool checkResults(const float *dst_cpu, const half *dst_gpu, const unsigned int data_size) {
    for (unsigned int i = 0; i < data_size; ++i) {
        if (fabs(dst_cpu[i] - __half2float(dst_gpu[i])) > 1e-5) {
            std::cout << "Check Failed at index " << i << ": CPU result = " << dst_cpu[i] << " GPU result = " << __half2float(dst_gpu[i]) << std::endl;
            return false;
        }
    }
    return true;
}

int main() {
    float *h_in = (float*)malloc(N * sizeof(float));
    float *h_out_cpu = (float*)malloc(N * sizeof(float));
    half *h_out = (half*)malloc(N * sizeof(half));
    half *d_in, *d_out;

    // Init data
    for (int i = 0; i < N; i++) {
        h_in[i] = static_cast<float>(i) / N;
    }

    // Allocate GPU memory
    cudaMalloc((void**)&d_in, N * sizeof(half));
    cudaMalloc((void**)&d_out, N * sizeof(half));

    // Convert float input to half input
    half *h_in_half = (half*)malloc(N * sizeof(half));
    for (int i = 0; i < N; i++) {
        h_in_half[i] = __float2half(h_in[i]);
    }
    cudaMemcpy(d_in, h_in_half, N * sizeof(half), cudaMemcpyHostToDevice);

    // Get device property and calculate block needed
    cudaDeviceProp deviceProp;
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 256;  // 256 threads/block
    int gridSize = (N + blockSize - 1) / blockSize;

    // Call kernel function
    dim3 Grid(gridSize);  // number of blocks
    dim3 Block(blockSize);  // number of threads
    float millisecond;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    gelu_baseline<<<Grid, Block>>>(d_in, d_out, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);

    // Copy GPU result to CPU
    cudaMemcpy(h_out, d_out, N * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;

    // Check Result
    gelu_cpu(h_in, h_out_cpu, N);
    if (checkResults(h_out_cpu, h_out, N)) {
        std::cout << "Results are correct." << std::endl;
    } else {
        std::cout << "Results are incorrect." << std::endl;
    }

    // Free resources
    cudaFree(d_in);
    cudaFree(d_out);
    free(h_in);
    free(h_out);
    free(h_out_cpu);
    free(h_in_half);

    return 0;
}
