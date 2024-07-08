#include "gemv.cuh"


/*
 * Matrix(row/col=M/N) * Vector(row/col=M/1) = NewMatrix(row/col=M/1)
 */
template<typename T>
void gemvCPU(const T *mat, const T *vec, float *res, int M, int N) {
    for (int i = 0; i < M; ++i) {
        res[i] = 0.0f;
        for (int j = 0; j < N; ++j) {
            res[i] += (float)(mat[i * N + j] * vec[j]);
        }
    }
}


template<typename T>
bool checkResult(const float *cpuRes, const T* gpuRes, int M) {
    for (int i = 0; i < M; ++i) {
        if (cpuRes[i] != (float)gpuRes[i]) {
            std::cout << "Wrong Result: Index=" << i << ", CPU=" << cpuRes[i] << ", GPU=" << (float)gpuRes[i] << std::endl;
            return false;
        }
    }
    return true;
}

template<typename T>
void gemv_kernel(T *h_vec, T *d_vec, T *h_mat, T *d_mat, T *h_res, T *d_res){
    constexpr int M = 256;
    constexpr int N = 2048;


    // allocate memory
    h_vec = (T*)malloc(N * sizeof(T));
    h_mat = (T*)malloc(M * N * sizeof(T));
    h_res = (T*)malloc(M * sizeof(T));
    cudaMalloc((void**)&d_vec, N * sizeof(T));
    cudaMalloc((void**)&d_mat, M * N * sizeof(T));
    cudaMalloc((void**)&d_res, M * sizeof(T));

    // init dummy data
    for(int i = 0; i < N; i++) { h_vec[i] = (T)1; }
    for(int i = 0; i < N * M; i++) { h_mat[i] = (T)1; }

    // init device memory
    CHECK(cudaMemcpy(d_vec, h_vec, N * sizeof(T), cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(d_mat, h_mat, M * N * sizeof(T), cudaMemcpyHostToDevice));

    // cuda config
    constexpr int NUM_THREADS = 256;
    constexpr int VEC_SIZE = Vec<T>::size;
    constexpr int VEC_PER_THREAD = (N / NUM_THREADS) / VEC_SIZE;  // 2048 / 256 = 8 values/thread 8/4 = 2 vec/thread 8/8 = 1 vec/thread
    DispatchLauncher<VEC_PER_THREAD, VEC_SIZE, NUM_THREADS>::template launcher<T>(d_mat, d_vec, d_res, M, N);
    CHECK(cudaMemcpy(h_res, d_res, M * sizeof(T), cudaMemcpyDeviceToHost));


    auto *cpu_res = (float*)malloc(M * sizeof(float));
    gemvCPU<T>(h_mat, h_vec, cpu_res, M, N);
    bool resRight = checkResult<T>(cpu_res, h_res, M);
    if (resRight) {
        printf("Result Right.\n");
    } else {
        printf("Result Wrong.\n");
    }

    cudaFree(d_vec);
    cudaFree(d_mat);
    cudaFree(d_res);
    free(h_vec);
    free(h_mat);
    free(h_res);
    free(cpu_res);
}

int main() {
    if(true) {
        float *h_vec = nullptr;
        float *d_vec = nullptr;
        float *h_mat = nullptr;
        float *d_mat = nullptr;
        float *h_dst = nullptr;
        float *d_dst = nullptr;
        gemv_kernel<float>(h_vec, d_vec, h_mat, d_mat, h_dst, d_dst);
    } else {
        half *h_vec = nullptr;
        half *d_vec = nullptr;
        half *h_mat = nullptr;
        half *d_mat = nullptr;
        half *h_dst = nullptr;
        half *d_dst = nullptr;
        gemv_kernel<half>(h_vec, d_vec, h_mat, d_mat, h_dst, d_dst);
    }
}
