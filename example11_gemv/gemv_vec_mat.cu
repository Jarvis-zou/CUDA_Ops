#include "gemv.cuh"


/*
 *  Vector(row/col=1/M) * Matrix(row/col=N/M)= NewMatrix(row/col=1/M)
 */
template<typename T>
void gemvCPU(const T *mat, const T *vec, float *res, int M, int N) {
    for (int i = 0; i < M; ++i) {
        res[i] = 0.0f;
        for (int j = 0; j < N; ++j) {
            res[i] += (float)(vec[j] * mat[i * N + j]);
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

// vec.shape = [1, N]
// mat.shape = [N, M] and matrix is in row major order
#define GEMV_KERNEL(dtype)                                                                                                  \
    dtype *d_vec;                                                                                                           \
    dtype *d_mat;                                                                                                           \
    dtype *d_dst;                                                                                                           \
    constexpr int N = 256;                                                                                                  \
    constexpr int M = 256;                                                                                                  \
    dtype *vec = (dtype *)malloc(N * sizeof(dtype));                                                                        \
    cudaMalloc((void **)&d_vec, N * sizeof(dtype));                                                                         \
    dtype *mat = (dtype *)malloc(M * N * sizeof(dtype));                                                                    \
    cudaMalloc((void **)&d_mat, M *N * sizeof(dtype));                                                                      \
    dtype *dst = (dtype *)malloc(M * sizeof(dtype));                                                                        \
    cudaMalloc((void **)&d_dst, M * sizeof(dtype));                                                                         \
    for (int i = 0; i < N; i++) {                                                                                           \
        vec[i] = (dtype)1;                                                                                                  \
    }                                                                                                                       \
    for (int i = 0; i < N * M; i++) {                                                                                       \
        mat[i] = (dtype)1;                                                                                                  \
    }                                                                                                                       \
    cudaMemcpy(d_vec, vec, N * sizeof(dtype), cudaMemcpyHostToDevice);                                                      \
    cudaMemcpy(d_mat, mat, M *N * sizeof(dtype), cudaMemcpyHostToDevice);                                                   \
    constexpr int THREADS_PER_BLOCK = 256;                                                                                  \
    constexpr int VEC_SIZE = Vec<dtype>::size;                                                                              \
    constexpr int VALUES_PER_THREAD = gemv2::get_threads_per_mat_row<M, dtype>::value;                                      \
    DispatchLauncher2<THREADS_PER_BLOCK, VALUES_PER_THREAD, VEC_SIZE>::template launcher<dtype>(d_mat, d_vec, d_dst, M, N); \
    cudaMemcpy(dst, d_dst, M * sizeof(dtype), cudaMemcpyDeviceToHost);                                                      \
    float *groudtruth = (float *)malloc(sizeof(float) * M);                                                                 \
    gemvCPU(mat, vec, groudtruth, M, N);                                                                                    \
    bool is_right = checkResult(groudtruth, dst, M);                                                                        \
    if (is_right)                                                                                                           \
    {                                                                                                                       \
        printf("the ans is right\n");                                                                                       \
    }                                                                                                                       \
    else                                                                                                                    \
    {                                                                                                                       \
        printf("the ans is wrong\n");                                                                                       \
    }                                                                                                                       \
    cudaFree(d_vec);                                                                                                        \
    cudaFree(d_mat);                                                                                                        \
    cudaFree(d_dst);                                                                                                        \
    free(vec);                                                                                                              \
    free(mat);                                                                                                              \
    free(dst);                                                                                                              \
    free(groudtruth);                                                                                                       \

int main()
{
    if (true)
    {
        GEMV_KERNEL(float);
    }
    else
    {
        GEMV_KERNEL(half);
    }
}
