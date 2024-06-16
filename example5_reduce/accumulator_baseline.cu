#include "pch.cuh"
#define N 25600000

__global__ void accumulator_baseline(const int *input, int *output) {
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    *output = sum;
}

void accumulator_baseline_cpu(const int *input, int *output) {
    int sum = 0;
    for (int i = 0; i < N; i++) {
        sum += input[i];
    }
    *output = sum;
}

int main() {
    int *hin, *hout, *hout_cpu;
    int *din, *dout;

    // allocate CPU data memory
    hin = (int*)malloc(N * sizeof(int));
    hout = (int*)malloc(sizeof(int));
    hout_cpu = (int*)malloc(sizeof(int));

    // init data
    for (int i = 0; i < N; i++) {
        hin[i] = 1;
    }

    //allocate GPU memory
    cudaMalloc((void**)&din, N * sizeof(int));
    cudaMalloc((void**)&dout, sizeof(int));

    // copy data to GPU
    cudaMemcpy(din, hin, N * sizeof(int), cudaMemcpyHostToDevice);

    // call kernel function
    float millisecond;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    accumulator_baseline<<<1, 1>>>(din, dout);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&millisecond, start, stop);

    // copy GPU result to CPU
    cudaMemcpy(hout, dout, sizeof(int), cudaMemcpyDeviceToHost);
    std::cout << "[GPU] Sum = " << *hout << std::endl;
    std::cout << "Time Spent: " << millisecond << "ms." << std::endl;


    // call CPU function
    accumulator_baseline_cpu(hin, hout_cpu);
    std::cout << "[CPU] Sum = " << *hout_cpu << std::endl;
    if (*hout_cpu != *hout) std::cout << "Result is Wrong!" << std::endl;

    // free resource
    cudaFree(din);
    cudaFree(dout);

    free(hin);
    free(hout);
    free(hout_cpu);










}
