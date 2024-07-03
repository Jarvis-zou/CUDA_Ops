#include "pch.cuh"
// Kernel performance on RTX 4090:
// PerTensor + Sym: 0.58ms
// PerChannel + Sym: 0.46ms

// CPU version equation
// PerTensor + Sym: scale = max(abs(weight)) / 127 , zeropoint = 0, input_int8 = clamp(input_fp32/scale ,-128, 127)
// PerTensor + Asym: scale = (max(weight) - min(weight)) / 255, zeropoint = -round(min(weight))/scale
// PerChannel + Sym: scale[channel_id] = max(abs(weight[channel_id])) / 127 , zeropoint = 0, input_int8[channel_id * HW + (channel_id + 1) * HW] = clamp(input_fp32[channel_id * HW + (channel_id + 1) * HW]/scale[channel_id] ,-128, 127)
// PerChannel + Asym: scale[channel_id] = (max(weight[channel_id]) - min(weight[channel_id])) / 255, zeropoint[channel_id] = -round(min(weight[channel_id]))/scale[channel_id]

template<typename T>
void getScalePerTensorSymmetricCPU(const T* in_ptr, const int quantization_bit,
                                   const int num_elements, T* scale, T* zero_point) {
    T in_max = *std::max_element(in_ptr, in_ptr + num_elements);  // Max(weight)
    T in_min = *std::min_element(in_ptr, in_ptr + num_elements);  // Min(weight)
    T out_max = std::max(std::abs(in_max), std::abs(in_min));  // AbsMax(weight)
    T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;  // 127
    *scale = out_max / denominator;  // Max(weight) / 127
    *zero_point = 0;  // zp = 0 in symmetric quantization
}

template<typename T>
void symmetricQuantizePerTensorCPU(const T* in_ptr, const T scale, const int quantization_bit,
                                   const int num_elements, T* out_ptr) {
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    T lower_bound = -upper_bound - 1;
    for(int j = 0; j < num_elements; j++) {
        T out = std::nearbyint(in_ptr[j] / scale);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[j] = out;
    }
}

template<typename T>
void getScalePerChannelSymmetricCPU(const T* in_ptr, const int quantization_bit, const int HW, const int channel,
                                    T* scale, T* zero_point) {
    for (int cid = 0; cid < channel; cid++){
        int start = cid * HW;
        int end = (cid + 1) * HW;
        T channel_max = *std::max_element(in_ptr + start, in_ptr + end);
        T channel_min = *std::min_element(in_ptr + start, in_ptr + end);
        T out_max = std::max(std::abs(channel_max), std::abs(channel_min));
        T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
        scale[cid] = out_max / denominator;
        zero_point[cid] = 0;
    }
}
template<typename T>
void symmetricQuantizePerChannelCPU(const T* in_ptr, const T* scale, const int quantization_bit, const int HW,
                                    const int num_elements, T* out_ptr) {
    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    T lower_bound = -upper_bound - 1;
    for(int j = 0; j < num_elements; j++) {
        T out = std::nearbyint(in_ptr[j] / scale[j / HW]);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[j] = out;
    }
}

bool checkResult(const float *gpuRes, const float* cpuRes, int nums){
    if (!gpuRes || !cpuRes) { return false; }

    for (int i = 0; i < nums; i++){
        if (cpuRes[i] != gpuRes[i]) {
            printf("Wrong Result at index %d, the CPU Result is %f, the GPU Result is %f\n", i, cpuRes[i], gpuRes[i]);
            return false;
        }
    }
    return true;
}

// CUDA atomicMAX/MIN doesn't support FP32, so here is the overloaded version, refer to https://forums.developer.nvidia.com/t/cuda-atomicmax-for-float/194207
inline __device__ float atomicMax(float *address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,  __float_as_int(fmaxf(val, __int_as_float(assumed))));

    } while (old != assumed);

    return __int_as_float(old);
}

inline __device__ float atomicMin(float *address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i;
    int assumed = 0;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,  __float_as_int(fminf(val, __int_as_float(assumed))));

    } while (old != assumed);

    return __int_as_float(old);
}

// Block Reduce
template<typename T>
__global__ void reduceMaxMinPerTensor(const T* input_ptr, const int nums, T* max_ptr, T* min_ptr, const int channel, const int HW) {
    // block stage result(dynamic space)
    extern __shared__ unsigned char shared_max_min_memory[];
    T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
    T* shared_min = shared_max + blockDim.x;
    auto stride = blockDim.x * gridDim.x;

    auto tid = threadIdx.x;
    auto gid = blockDim.x * blockIdx.x + tid;
    shared_max[tid] = FLT_MIN;
    shared_min[tid] = FLT_MAX;

    // In case num of threads < data size
    for (auto i = gid; i < nums; i += stride) {
        // Load data
        shared_max[tid] = max(shared_max[tid], input_ptr[i]);
        shared_min[tid] = min(shared_min[tid], input_ptr[i]);
    }
    __syncthreads();

    // Block Reduce
    for (auto size = blockDim.x / 2; size > 0; size >>= 1) {
        if (tid < size && gid < nums) {
            shared_max[tid] = max(shared_max[tid], shared_max[tid + size]);
            shared_min[tid] = min(shared_min[tid], shared_min[tid + size]);
        }
        __syncthreads();
    }

    // Final Result stored in tid=0
    if (tid == 0) {
        atomicMax(max_ptr, shared_max[0]);
        atomicMin(min_ptr, shared_min[0]);
    }
}

// Per Channel
template<typename T>
__global__ void reduceMaxMinPerChannel(const T* input_ptr,
                                       const int nums,
                                       T* max_ptr,
                                       T* min_ptr,
                                       const int num_channels,
                                       const uint32_t HW) {

    extern __shared__ unsigned char shared_max_min_memory[];
    T* shared_max = reinterpret_cast<T*>(shared_max_min_memory);
    T* shared_min = shared_max + blockDim.x;
    auto cur_channel = blockIdx.x;
    auto tid = threadIdx.x;
    auto gid = blockIdx.x * blockDim.x + tid;
    // get min/max of each channel
    while (cur_channel < num_channels) {
        shared_max[tid] = FLT_MIN;
        shared_min[tid] = FLT_MAX;
        auto index = (HW * cur_channel) + tid;
        auto end = HW * (cur_channel + 1);

        while (index < end && index < nums) {
            shared_max[tid] = max(shared_max[tid], input_ptr[index]);
            shared_min[tid] = min(shared_min[tid], input_ptr[index]);
            index += blockDim.x;
        }
        __syncthreads();

        for (auto s = blockDim.x / 2; s > 0; s >>= 1) {
            if (tid < s) {
                shared_max[tid] = max(shared_max[tid], shared_max[tid + s]);
                shared_min[tid] = min(shared_min[tid], shared_min[tid + s]);
            }
            __syncthreads();
        }

        if (tid == 0) {
            atomicMax(&max_ptr[cur_channel], shared_max[0]);
            atomicMin(&min_ptr[cur_channel], shared_min[0]);
//            // debug info
//            if(blockIdx.x==0){
//                printf("max = %f\n", max_ptr[0]);
//                printf("min = %f\n", min_ptr[0]);
//            }
        }
        cur_channel += gridDim.x;
    }
}


template<typename T>
__global__ void getScaleAndZPSymmetric(const T* max_ptr,
                                       const T* min_ptr,
                                       const int nums,
                                       const uint32_t quantization_bit,
                                       T* scale,
                                       T* zero_point) {
    auto tid = threadIdx.x;
    auto gid = blockDim.x * blockIdx.x + tid;
    while (gid < nums) {
        T weight_max = max(fabs(max_ptr[gid]), fabs(min_ptr[gid]));
        T denominator = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
        scale[gid] = weight_max / denominator;
        zero_point[gid] = 0;
        gid += gridDim.x * blockDim.x;
    }
}

template<typename T>
__global__ void getScaleAndZPAsymmetric(const T* max_ptr,
                                        const T* min_ptr,
                                        const int nums,
                                        const uint32_t quantization_bit,
                                        T* scale,
                                        T* zero_point) {
    auto tid = threadIdx.x;
    auto gid = (blockDim.x * blockIdx.x) + tid;
    while (gid < nums) {
        T denominator = static_cast<T>(pow(2.0, quantization_bit)) - 1;
        T min = -min_ptr[gid];
        T s = (max_ptr[gid] - min) / denominator;
        scale[gid] = s;
        zero_point[gid] = -1 * std::nearbyint(min / s);
        gid += gridDim.x * blockDim.x;
    }
}

template<typename T>
__global__ void quantizePerChannelSymmetric(const T* in_ptr,
                                            const T* scale_ptr,
                                            const int nums,
                                            const uint32_t quantization_bit,
                                            T* out_ptr,
                                            const int scale_size,
                                            const int HW) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    T lower_bound = -upper_bound - 1;

    while (gid < nums) {
        auto channel_index = gid / HW;
        auto scale_idx = min(scale_size - 1, channel_index);
        T scale = scale_ptr[scale_idx];

        T out = std::nearbyint(in_ptr[gid] / scale);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;

        gid += step;
    }
}

template<typename T>
__global__ void quantizePerChannelAsymmetric(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                             const int scale_size, const int nums,
                                             const int HW, const uint32_t quantization_bit,
                                             T* out_ptr) {
    auto gid = (blockDim.x * blockIdx.x) + threadIdx.x;
    auto step = gridDim.x * blockDim.x;

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
    T lower_bound = 0;

    while (gid < nums) {
        auto channel_index = gid / HW;
        auto scale_idx = min(scale_size - 1, channel_index);

        T scale = scale_ptr[scale_idx];
        T zero_point = zero_point_ptr[scale_idx];

        T out = std::nearbyint(in_ptr[gid] / scale + zero_point);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;

        gid += step;
    }
}

// element wise operation
template<typename T>
__global__ void quantizePerTensorSymmetric(const T* in_ptr, const T* scale_ptr,
                                           const int nums, const double quantization_bit, T* out_ptr, const int channel, const int HW) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit - 1)) - 1;
    T lower_bound = -upper_bound - 1;
    T scale = *scale_ptr;
//    // debug info
//    if (gid==0) printf("scaleGPU is %f\n", scale);
    while (gid < nums) {
        T out = rintf(in_ptr[gid] / scale);
//        // debug info
//        if (gid==328) printf("328 in_ptr is %f, out is %f\n", in_ptr[gid], out);
//        if (gid==1587) printf("1587 in_ptr is %f, out is %f\n", in_ptr[gid], out);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;
        gid += step;
    }
}

template<typename T>
__global__ void quantizePerTensorAsymmetric(const T* in_ptr, const T* scale_ptr, const T* zero_point_ptr,
                                            const int nums, const double quantization_bit, T* out_ptr,
                                            const int channel, const int HW) {
    auto gid = blockDim.x * blockIdx.x + threadIdx.x;
    auto step = gridDim.x * blockDim.x;

    T upper_bound = static_cast<T>(pow(2.0, quantization_bit)) - 1;
    T lower_bound = 0;
    T scale = *scale_ptr;
    T zero_point = *zero_point_ptr;
    while (gid < nums) {
        T out = nearbyint(in_ptr[gid] / scale + zero_point);
        out = out > upper_bound ? upper_bound : out;
        out = out < lower_bound ? lower_bound : out;
        out_ptr[gid] = out;
        gid += step;
    }
}

// Macro replace redundant code
#define LAUNCH_GPU_KERNEL(getMinMaxFunc, quantFunc, scale_size, channel, HW) \
    cudaMalloc((void **)&d_scale, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_zeropoint, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_max, scale_size * sizeof(float)); \
    cudaMalloc((void **)&d_min, scale_size * sizeof(float)); \
    cudaEvent_t start, stop; \
    cudaEventCreate(&start); \
    cudaEventCreate(&stop); \
    cudaEventRecord(start); \
    getMinMaxFunc<float><<<gridSize, blockSize, blockSize * 2 * sizeof(float), 0>>>(d_input, nums, d_max, d_min, channel, HW);  \
    getScaleAndZPSymmetric<float><<<1, blockSize>>>(d_max, d_min, channel, quantization_bit, d_scale, d_zeropoint); \
    quantFunc<float><<<gridSize, blockSize>>>(d_input, d_scale, nums, quantization_bit, d_output, channel, HW); \
    cudaEventRecord(stop); \
    cudaEventSynchronize(stop); \
    cudaEventElapsedTime(&milliseconds, start, stop);

int main() {

    // Init test data
    float milliseconds = 0;
    constexpr int HW = 20 * 10;  // H * W
    constexpr int channel = 400;
    constexpr int nums = channel * HW;
    constexpr int quantization_bit = 8;  // quantize to 8-bit

    // Allocate host memory
    auto* input = (float*)malloc(sizeof(float)* nums);
    float cpu_min = FLT_MAX;
    float cpu_max = FLT_MIN;
    for(int i = 0; i < nums; i++) {
        // generate float input in range [-1, 1], [-3,3]
        input[i] = -3 + static_cast <float>(rand()) / (static_cast <float> (RAND_MAX / 6));
        cpu_min = std::min(input[i], cpu_min);
        cpu_max = std::max(input[i], cpu_max);
    }


    auto* output = (float*)malloc(sizeof(float) * nums);
    float *d_input, *d_output;
    cudaMalloc((void **)&d_input, nums * sizeof(float));
    cudaMalloc((void **)&d_output, nums * sizeof(float));
    cudaMemcpy(d_input, input, sizeof(float) * nums, cudaMemcpyHostToDevice);

    // Kernel config
    cudaDeviceProp deviceProp{};
    cudaGetDeviceProperties(&deviceProp, 0);
    int maxBlocks = deviceProp.maxGridSize[0];
    int blockSize = 256;
    int gridSize = std::min<int>((nums + blockSize - 1) / blockSize,  std::min<int>(maxBlocks, channel));
    printf("GridSize=%d, BlockSize=%d\n", gridSize, blockSize);
    float *d_scale, *d_zeropoint, *d_max, *d_min;

    // Switch
    bool per_tensor_quantize = false;
    if(per_tensor_quantize) {
        LAUNCH_GPU_KERNEL(reduceMaxMinPerTensor, quantizePerTensorSymmetric, 1, nums, HW);  // per tensor
    } else {
        LAUNCH_GPU_KERNEL(reduceMaxMinPerChannel, quantizePerChannelSymmetric, channel, channel, HW);  // per channel
    }
    cudaMemcpy(output, d_output, sizeof(float) * nums, cudaMemcpyDeviceToHost);

    // CPU Result
    float* CPUOutput= (float*) malloc(sizeof(float) * nums);
    if(per_tensor_quantize) {
        float* scale = (float*) malloc(sizeof(float) * 1);
        float* zeropoint = (float*) malloc(sizeof(float) * 1);
        getScalePerTensorSymmetricCPU<float>(input, quantization_bit, nums, scale, zeropoint);
        symmetricQuantizePerTensorCPU<float>(input, *scale, quantization_bit, nums, CPUOutput);
        free(scale);
        free(zeropoint);
    } else {
        float* scale = (float*) malloc(sizeof(float) * channel);
        float* zeropoint = (float*) malloc(sizeof(float) * channel);
        getScalePerChannelSymmetricCPU<float>(input, quantization_bit, HW, channel, scale, zeropoint);
        symmetricQuantizePerChannelCPU<float>(input, scale, quantization_bit, HW, nums, CPUOutput);
        free(scale);
        free(zeropoint);
    }
    if (checkResult(output, CPUOutput, nums)) {
        printf("Result correct.");
    } else {
        printf("Result wrong\n");
    }
    printf("Quantize kernel latency = %f ms\n", milliseconds);

    free(input);
    free(output);
    free(CPUOutput);
    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_scale);
    cudaFree(d_zeropoint);
    cudaFree(d_max);
    cudaFree(d_min);
}
