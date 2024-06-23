#include "pch.cuh"
#define N 25600000

template <int Size>
// if we don't set alignas(sizeof(__half) * Size), alignof() will return 2 bytes
// we explicitly set alignof this struct as 8 half(16 bytes), so we must load 8 half values each time
struct alignas(sizeof(__half) * Size) AlignedVectorHalf {
    // This struct has a T * Size alignment in space
    __half data[Size];

    // Overload[] for user-friendly call
    __host__ __device__ inline const __half& operator[](int i) const { return data[i]; }  // This won't be used (No const members)
    __host__ __device__ inline __half& operator[](int i) { return data[i]; }
};

__device__ void gelu_forward(__half *address, __half *dst){
    const __half2 alpha = __float2half2_rn(0.7978845608028654); // sqrt(2.0 / M_PI)
    const __half2 beta = __float2half2_rn(0.044714998453855515);  // standard deviation


    half2 x = *reinterpret_cast<const half2*>(address); // x contains 2 half value
    x = __hmul2(alpha, __hadd2(x, __hmul2(x, __hmul2(x, __hmul2(x, beta))))); // alpha * (x + beta * x * x * x)
    const float2 tanh_in = __half22float2(x);  // because tanh cant process half2, we have to convert half2 to float2 first
    float2 tanh_out;
    tanh_out.x = tanhf(tanh_in.x);
    tanh_out.y = tanhf(tanh_in.y);

    const __half2 res = __hmul2(__hmul2(__float2half2_rn(0.5f), x),
                                __hadd2(__float2half2_rn(1.0f), __float22half2_rn(tanh_out))); // 0.5 * x * (1.0 + tanh_out);

    *reinterpret_cast<half2*>(dst) = res; // write 2 half values to dst address one time
}

template<int vectorSize>
__global__ void GeluFP16(__half *input, __half *output, const unsigned int data_size) {
    /* Each thread will handle 8 half values, but in forward function, we use half2 intrinsic to process 2 half values
     * at the same time, so we only need to do 4 times operations and N/8 read & write */
    //global_tid * vectorSize, result will write into this global address in output for this thread
    unsigned int global_offset = (threadIdx.x + blockIdx.x * blockDim.x) * vectorSize;
    unsigned int stride = (blockDim.x * gridDim.x) * vectorSize;

    // In case number of threads < data size, assign overflow data to some threads
    for (; global_offset < data_size; global_offset += stride) {
        for (int i = 0; i < vectorSize; i += 2){
            gelu_forward(input + global_offset + i, output + global_offset);  // do operations and update one global memory
        }
    }
}

int main() {
    auto *h_in = (__half*)malloc(N * sizeof(__half));
    auto *h_out = (__half*)malloc(N * sizeof(__half));
    __half *d_in, *d_out;

    // Init data
    for (auto  i = 0; i < N; i++) {
        h_in[i] = __float2half(static_cast<float>(i) / N) ;
    }

    // Allocate GPU memory
    cudaMalloc((void**)&d_in, N * sizeof(__half));
    cudaMalloc((void**)&d_out, N * sizeof(__half));

    // Convert float input to half input
    cudaMemcpy(d_in, h_in, N * sizeof(__half), cudaMemcpyHostToDevice);

    // Get device property and calculate block needed
    cudaDeviceProp deviceProp{};
    int device;
    cudaGetDevice(&device);
    cudaGetDeviceProperties(&deviceProp, device);
    const int blockSize = 1024;  // max threads per block
    const int gridSize = (N / 8 + blockSize - 1) / blockSize;  // 8 half/thread, so we only need 1/8 blocks to handle all data when blockSize is fixed

    // Check if the memory is aligned
    auto is_aligned = [](const void *p, const int alignment) {
        return reinterpret_cast<uintptr_t>(p) % alignment == 0;
    };


    // Get the alignment of a data struct, In this struct we have 8 half data, so takes 16 bytes space
    // We set sizeof(data)/read = 8 because vector load in CUDA has 64/128 bit constrains, so we can only read 4 half(4*16=64 bit) or 8 half(8*16=128 bits)
    constexpr auto vectorAlignmentHalf = alignof(AlignedVectorHalf<8>);

    // Data size must be multiples of 4/8, then we can use vector load in cuda
    float millisecond = 0.0f;
    if (N % 8 == 0 && is_aligned(d_in, vectorAlignmentHalf) && is_aligned(d_out, vectorAlignmentHalf)) {
        // Call kernel function
        dim3 Grid(gridSize);  // number of blocks
        dim3 Block(blockSize);  // number of threads
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        GeluFP16<8><<<Grid, Block>>>(d_in, d_out, N);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&millisecond, start, stop);
    }


    // Copy GPU result to CPU
    cudaMemcpy(h_out, d_out, N * sizeof(half), cudaMemcpyDeviceToHost);
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;

    // Free resources
    cudaFree(d_in);
    cudaFree(d_out);

    free(h_in);
    free(h_out);

    return 0;
}
