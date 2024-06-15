#include "../pch.cuh"


void vector_add_cpu(const float *x, const float *y, float *z, int size) {
    for(int i = 0; i < size; i++){
        z[i] = x[i] + y[i];
    }
}

__global__ void vector_add(float *x, float *y, float *z, int *count, int size) {
    int idx = blockDim.x * (blockIdx.x + blockIdx.y * gridDim.x) + threadIdx.x; // 2d grid

//    int idx = blockDim.x * blockIdx.x + threadIdx.x; // 1d grid
    if (idx < size) {
        z[idx] = x[idx] + y[idx];
        atomicAdd(count, 1);
    }
}

int main() {
    int vectorSize = 10000;
    int vectorBytes = vectorSize * sizeof(float);

    int bd = 32;

    int s = ceil(sqrt((vectorSize + bd - 1.) / bd));
    dim3 grid(s, s);

    // init pointers
    float *hx, *dx;  // first vector
    float *hy, *dy;  // second vector
    float *hz, *hz_cpu, *dz;  // result vector
    int *hcount, *dcount; // count operations

    // allocate GPU memory
    cudaMalloc((void**)&dx, vectorBytes);
    cudaMalloc((void**)&dy, vectorBytes);
    cudaMalloc((void**)&dz, vectorBytes);
    cudaMalloc((void**)&dcount, sizeof(int));

    // allocate CPU data memory
    hx = (float*)malloc(vectorBytes);
    hy = (float*)malloc(vectorBytes);
    hz = (float*)malloc(vectorBytes);
    hz_cpu = (float*)malloc(vectorBytes);
    hcount = (int*)malloc(sizeof(int));

    // init vector data
    *hcount = 0;
    for (int i = 0; i < vectorSize; i++) {
        hx[i] = 1;
        hy[i] = 1;
    }

    // copy data to GPU
//    cudaMemcpy(dx, hx, vectorBytes, cudaMemcpyHostToDevice);
//    cudaMemcpy(dy, hy, vectorBytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dcount, hcount, sizeof(int), cudaMemcpyHostToDevice);

    // call kernel function, change the total number of threads, you can see difference in hcount
    float milliseconds = 0;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    vector_add<<<grid, bd>>>(dx, dy, dz, dcount, vectorSize);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);

    // call cpu function
    vector_add_cpu(hx, hy, hz_cpu, vectorSize);
    std::cout << hz_cpu[0] << std::endl;



    // copy data from GPU back to CPU
    cudaMemcpy(hz, dz, vectorBytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hcount, dcount, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "[CUDA] Time Spend: " << milliseconds << "ms" << std::endl;
    std::cout << *hcount << std::endl;



    // check GPU result
    for (int i = 0; i < vectorSize; i++) {
        if (fabs(hz_cpu[i] - hz[i]) > 1e-6) {
            std::cout << "Result verification failed at element index " << i << std::endl;
        }
    }

    cudaFree(dx);
    cudaFree(dy);
    cudaFree(dz);
    cudaFree(dcount);

    free(hx);
    free(hy);
    free(hz);
    free(hz_cpu);
    free(hcount);
}
