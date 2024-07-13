#include "pch.cuh"

// Template Exception check
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

template <typename T, int VecSize>
struct alignas(sizeof(T) * VecSize) VectorType {
    T val[VecSize];
};

template <typename T>
struct uniform_distribution {
    __device__ T operator()(curandStatePhilox4_32_10_t* state) { return static_cast<T>(curand_uniform(state)); }
    static constexpr int Count = 1;
};

template <>
struct uniform_distribution<float> {
    __device__ float4 operator()(curandStatePhilox4_32_10_t* state) { return curand_uniform4(state); }
    static constexpr int Count = 4;
};

template <typename T>
struct dropoutFunctor {
    const float m_ratio;
    const bool m_upscale;
    float inv_prob;
    __device__ dropoutFunctor(const float ratio, const bool upscale)
            : m_ratio(ratio), m_upscale(upscale) {
        inv_prob = 1.0f / (1 - m_ratio);
    }

    __device__ void operator()(T* dst, const T* src, const T* rand) {
        static constexpr int vecSize = uniform_distribution<T>::Count;
        for (int i = 0; i < vecSize; i++) {
            if (rand[i] < m_ratio) {
                dst[i] = static_cast<T>(0);
                dst[i + vecSize] = dst[i];
            } else {
                dst[i] = m_upscale ? static_cast<T>(src[i] * inv_prob) : static_cast<T>(src[i]);
                dst[i + vecSize] = static_cast<T>(1);
            }
        }
    }
};


template<typename T>
__global__ void dropout(const T* x, const uint8_t* mask, T* y, const bool upscale, const float ratio, const int seed, const int increment, const size_t maxVectorizedSize, const size_t size) {
    constexpr int VecSize = 4; // Assuming VecSize is 4, you can adjust this
    const unsigned int tid = threadIdx.x;
    const unsigned int global_tid = blockIdx.x * blockDim.x + tid;
    const unsigned int stride = gridDim.x * blockDim.x * VecSize;

    // Init random state
    curandStatePhilox4_32_10_t state;
    curand_init(seed, global_tid, increment, &state);

    // Init functor
    auto dst_functor = dropoutFunctor<T>(ratio, upscale);
    T dst_mask[VecSize * 2];  // 0 ~ VecSize - 1: dst; VecSize ~ 2 * VecSize - 1: mask
    float rands[VecSize];
    uint8_t mask_result[VecSize];

    using VecType = float4;
    using VecType_u8 = VectorType<uint8_t, VecSize>;

    // Vectorized part
    for (unsigned int start = global_tid * VecSize; start < maxVectorizedSize; start += stride) {
        const auto* vec_input = reinterpret_cast<const VecType*>(x + start);
        VecType vec_temp_input = vec_input[tid];
        auto random_tuple = curand_uniform4(&state);
        for (int i = 0; i < VecSize; ++i) {
            dst_mask[i] = *(reinterpret_cast<T*>(&vec_temp_input) + i);
            rands[i] = static_cast<float>((&random_tuple.x)[i]);
        }

        // mask
        dst_functor(&dst_mask[0], &dst_mask[0], &rands[0]);

        // write result
        T* res = y + start;
        auto* vec_dst_output = reinterpret_cast<VecType*>(res);
        vec_dst_output[tid] = *(reinterpret_cast<VecType*>(&dst_mask[0]));

        for (int i = 0; i < VecSize; ++i) {
            mask_result[i] = static_cast<uint8_t>(dst_mask[i + VecSize]);
        }

        auto* mask_res = const_cast<uint8_t*>(mask + start);
        auto* vec_mask_output = reinterpret_cast<VecType_u8*>(mask_res);
        vec_mask_output[tid] = *(reinterpret_cast<VecType_u8*>(mask_result));
    }

    // Can't be vectorized part
    unsigned int remain = size - global_tid * VecSize;
    if (remain > 0) {
        // load
        unsigned int thread_offset = tid * VecSize;
        const T* src_remain = x + global_tid * VecSize;
        auto random_tuple = curand_uniform4(&state);
        for (int i = 0; i < VecSize; ++i) {
            if (i + thread_offset < remain) {
                dst_mask[i] = src_remain[thread_offset + i];
            }
            rands[i] = static_cast<float>((&random_tuple.x)[i]);
        }

        // mask
        dst_functor(&dst_mask[0], &dst_mask[0], &rands[0]);

        // write
        T* res = y + global_tid * VecSize;
        auto* mask_res = const_cast<uint8_t*>(mask + global_tid * VecSize);
        for (int i = 0; i < VecSize; ++i) {
            if ((thread_offset + i) < remain) {
                res[thread_offset + i] = dst_mask[i];
                mask_result[i] = static_cast<uint8_t>(dst_mask[i + VecSize]);
                mask_res[thread_offset + i] = mask_result[i];
            }
        }
    }
}


template<typename T>
void launch(const T* x, const uint8_t* mask, T* y, const bool eval, const bool upscale, const float ratio, const int seed, const size_t size) {
    if (!eval) {
        if (ratio == 0) {
            cudaMemset(y, 0, size);
            return;
        }

        // launch dropout kernel
        size_t blockSize = 256;
        size_t gridSize = 2;
        dim3 grid(gridSize);
        dim3 block(blockSize);
        const int increment = 0;
        constexpr int randVecSize = uniform_distribution<float>::Count;
        const size_t maxVectorizedSize = size - size % (blockSize * gridSize * randVecSize);

        float time = 0.0f;
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        dropout<T><<<grid, block>>>(x, mask, y, upscale, ratio, seed, increment, maxVectorizedSize, size);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time, start, stop);
        std::cout << "Time Cost: " << time << " ms" << std::endl;
    } else {
//       y = x;
        cudaMemcpy(y, x, size * sizeof(T), cudaMemcpyDeviceToDevice);
    }
}

                                                                                               \

int main()
{
    constexpr size_t N = 2050;  // data size that cant be divided by 4 or 8
    float *hx, *dx, *hy, *dy;
    uint8_t *h_mask, * d_mask;

    hx = (float*)malloc(N * sizeof(float));
    h_mask = (uint8_t*)malloc(N * sizeof(uint8_t));
    hy = (float*)malloc(N * sizeof(float));

    // Init data
    for (int i = 0; i < N; i++) {
        hx[i] = 1.0f;
    }


    // Init device memory
    CHECK(cudaMalloc((void**)&dx, N * sizeof(float)));
    CHECK(cudaMalloc((void**)&d_mask, N * sizeof(uint8_t)));
    CHECK(cudaMalloc((void**)&dy, N * sizeof(float)));

    // Copy data
    CHECK(cudaMemcpy(dx, hx, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mask, h_mask, N * sizeof(uint8_t), cudaMemcpyHostToDevice));

    // Init config
    constexpr bool eval = false;  // drop out only works under training process
    constexpr bool upscale = false;  // x = x / (1 - p)
    constexpr float ratio = 0.5;  // dropout ration
    constexpr int seed = 10000;  // random seed

    // Start kernel
    launch<float>(dx, d_mask, dy, eval, upscale, ratio, seed, N);

    // Check Result
    CHECK(cudaMemcpy(hy, dy, N * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(h_mask, d_mask, N * sizeof(uint8_t), cudaMemcpyDeviceToHost));
    for (int i = N - 3; i < N; i++){
        printf("[%d] y is %f\n",i, hy[i]);
        printf("[%d] mask is %d\n",i, h_mask[i]);
    }

    // Free
    free(hx);
    free(h_mask);
    free(hy);

    cudaFree(dx);
    cudaFree(d_mask);
    cudaFree(dy);


}
