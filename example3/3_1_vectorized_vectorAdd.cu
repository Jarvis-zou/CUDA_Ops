#include "../pch.cuh"

#define ARRAY_SIZE 100000000   //Array size has to exceed L2 size to avoid L2 cache residence
#define MEMORY_OFFSET 10000000  // numbers of float data, each time kernel will read this much data to process
#define BENCH_ITER 10  // BENCH_ITER = ARRAY_SIZE / MEMORY_OFFSET we can use 1 kernel to process all ARRAY_SIZE data or we can set 10 kernels(each process MEMORY_OFFSET data)
#define THREADS_NUM 256



__global__ void vectorized_add(float *x, float *y, float *z) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto* x_float4 = reinterpret_cast<float4*>(x);  // store x[i] to a float4 data type,x_float4 contains 4 float data start from x[i]
    auto* y_float4 = reinterpret_cast<float4*>(y);
    auto* z_float4 = reinterpret_cast<float4*>(z);

    for (uint32_t i = idx; i < MEMORY_OFFSET / 4; i += gridDim.x * blockDim.x) {
        z_float4[i].x = x_float4[i].x + y_float4[i].x;
        z_float4[i].y = x_float4[i].y + y_float4[i].y;
        z_float4[i].z = x_float4[i].z + y_float4[i].z;
        z_float4[i].w = x_float4[i].w + y_float4[i].w;
    }
}

void vec_add_cpu(float *x, float *y, float *z, int N)
{
    for (int i = 0; i < 20; i++)
        z[i] = y[i] + x[i];
}

int main() {
    float *hx, *hy, *hz;  // host pointers
    float *dx, *dy, *dz;  // device pointers

    // allocate CPU Memory
    hx = (float*)malloc(ARRAY_SIZE * sizeof(float));
    hy = (float*)malloc(ARRAY_SIZE * sizeof(float));
    hz = (float*)malloc(ARRAY_SIZE * sizeof(float));

    // init data on CPU
    for (auto i = 0; i < ARRAY_SIZE; i++) {
        hx[i] = static_cast<float>(i % 100);
        hy[i] = static_cast<float>(i % 200);
    }

    // allocate CUDA Memory
    cudaMalloc((void**)&dx,ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&dy,ARRAY_SIZE * sizeof(float));
    cudaMalloc((void**)&dz,ARRAY_SIZE * sizeof(float));

    // Copy data from CPU to GPU
    cudaMemcpy(dx, hx, ARRAY_SIZE* sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dy, hy, ARRAY_SIZE* sizeof(float), cudaMemcpyHostToDevice);

    // Call kernel function
    int BlockNums = MEMORY_OFFSET / THREADS_NUM;
    float milliseconds = 0;

    std::cout << "Warm up start" << std::endl;
    vectorized_add<<<BlockNums / 4, THREADS_NUM>>>(dx, dy, dz);  // warm up to occupy L2 cache
    std::cout << "Warm up stop" << std::endl;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    for (int i = BENCH_ITER - 1; i >= 0; --i) {
        vectorized_add<<<BlockNums / 4, THREADS_NUM>>>(dx + i * MEMORY_OFFSET, dy + i * MEMORY_OFFSET, dz + i * MEMORY_OFFSET);
    }
//    vectorized_add<<<BlockNums, THREADS_NUM>>>(dx, dy, dz);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time OverHead: " << milliseconds << std::endl;

    // Copy GPU result data to CPU
//    cudaMemcpy(hx, dx, ARRAY_SIZE* sizeof(float), cudaMemcpyDeviceToHost);  // don't really need to copy back x and y
//    cudaMemcpy(hy, dy, ARRAY_SIZE* sizeof(float), cudaMemcpyDeviceToHost);  // z is the only result we want
    cudaMemcpy(hz, dz, ARRAY_SIZE* sizeof(float), cudaMemcpyDeviceToHost);

    // CPU compute
    auto* hz_cpu = (float*)malloc(20 * sizeof(float));
    vec_add_cpu(hx, hy, hz_cpu, ARRAY_SIZE);

    // check GPU result with CPU
    for (int i = 0; i < 20; ++i) {
        if (fabs(hz_cpu[i] - hz[i]) > 1e-6) {
            printf("Result verification failed at element index %d!\n", i);
        }
    }
    std::cout << "Result right" << std::endl;
    unsigned N = ARRAY_SIZE * 4;
    printf("Mem BW= %f (GB/sec)\n", 3 * (float)N / milliseconds / 1e6);

    // free resource
    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu);

}
