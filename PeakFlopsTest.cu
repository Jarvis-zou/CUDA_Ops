#include "pch.cuh"
#define LOOP_TIMES 5000
#define N 1536  // maximum number of threads / SM
//RTX4090 fp32: 83.8 TFLOPS
__global__ void FP32FLOPS(int* start, int* stop, float* x, float* y, float* result) {
    int gtid = blockDim.x * blockIdx.x + threadIdx.x;
    float d1 = x[gtid];
    float d2 = y[gtid];
    float res = 0;
    int start_time = 0;
    // only measure the computation time, eliminate the memory access time
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(start_time) :: "memory");
#pragma unroll
    for (int i = 0; i < LOOP_TIMES; i++) {
//        asm volatile ("{\n\t""fma.rn.f32 %0, %1, %2 , %0; \n\t"
//                             "fma.rn.f32 %0, %1, %2 , %0; \n\t"
//                             "fma.rn.f32 %0, %1, %2 , %0; \n\t"
//                             "fma.rn.f32 %0, %1, %2 , %0; \n\t"
//                             "}" : "+f"(res), "+f"(d1),"+f"(d2)); // res + d1 * d2 = res
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
        res = d1 * d2 + res;
    }
    asm volatile("bar.sync 0;");   //sync 1536 threads

    int stop_time = 0;
    asm volatile ("mov.u32 %0, %%clock;" : "=r"(stop_time) :: "memory");
    start[gtid] = start_time;
    stop[gtid] = stop_time;
    result[gtid] = res;
}

int main() {
    float *x = (float*)malloc(N * sizeof(float));
    float *y = (float*)malloc(N * sizeof(float));
    float *d_x;
    float *d_y;
    cudaMalloc((void **)&d_x, N * sizeof(float));
    cudaMalloc((void **)&d_y, N * sizeof(float));
    for(int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i);
    }
    cudaMemcpy(d_x, x, N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_y, y, N * sizeof(float), cudaMemcpyHostToDevice);

    float *d_result;
    int *startClock = (int*)malloc(N * sizeof(int));  // count time cost of each thread
    int *stopClock = (int*)malloc(N * sizeof(int));
    int *d_startClock;
    int *d_stopClock;
    cudaMalloc((void **)&d_result, N * sizeof(float));
    cudaMalloc((void **)&d_startClock, N * sizeof(int));
    cudaMalloc((void **)&d_stopClock, N * sizeof(int));
    // confirm launch max threads of SM = 1536 to do FMA to saturate SM resource
    FP32FLOPS<<<2, 768>>>(d_startClock, d_stopClock, d_x, d_y, d_result);
    cudaMemcpy(startClock, d_startClock, N * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(stopClock, d_stopClock, N * sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);

    // avg time cost of each thread
    float total_time = 0.0f;
    for (int i = 0; i < N; i++) {
        float cost = stopClock[i] - startClock[i];

        total_time += cost;
    }
    float avg_time = total_time / N;
    printf( "Avg Time: %0.2f ms\n" , avg_time);


    float FLOPS = (LOOP_TIMES * 4 * 2 * N) /  avg_time;  // FLOPS/s
    printf( "GPU Max Clock rate: %0.2f GHz\n" , props.clockRate * 1e-6f);
    printf("SM counts is %d\n", props.multiProcessorCount);
    printf("actual NVIDIA RTX4090 GPU peak FLOPS is %f (TFLOPS) \n", FLOPS * props.clockRate * 1e-9 * props.multiProcessorCount);
    free(x);
    free(y);
    free(startClock);
    free(stopClock);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_result);
    cudaFree(d_startClock);
    cudaFree(d_stopClock);
}
