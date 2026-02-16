#include <cuda_runtime.h>
#include <stdio.h>
template <typename T, int BM, int BN, int TM, int BK>
__global__ void gemm_1D_BlockTiling(int M, int N, int K, float alpha, const T *A, const T *B, float beta, T *C) {
    // BLOCK LEVEL location
    int row = blockIdx.y * BM;
    int col = blockIdx.x * BN;
    T threadResult[TM] = {0};
    assert(BM % TM == 0 && blockDim.x == BM * BN / TM);
    //data load location 
    int loadRowA = threadIdx.x / BK;
    int loadColA = threadIdx.x % BK;
    int loadRowB = threadIdx.x / BN;
    int loadColB = threadIdx.x % BN;
    A += row * K;
    B += col;
    C += row * N + col;
    //thread location
    int threadRow = threadIdx.x / BN;
    int threadCol = threadIdx.x % BN;
    // shared memory
    __shared__ T AS[BM * BK];
    __shared__ T BS[BK * BN];
    for (int i  = 0; i < K; i += BK) {
        AS[loadRowA * BK + loadColA] = A[loadRowA * K + i + loadColA];
        BS[loadRowB * BN + loadColB] = B[(i + loadRowB) * N + loadColB];
        __syncthreads();
        for (int k = 0; k < BK; k++) {
            T bVal = BS[k * BN + threadCol];
            for (int t = 0; t < TM; t++) {
                threadResult[t] += AS[((threadRow * TM + t) * BK + k)] * bVal;
            }
        }
        __syncthreads();
    }
    for (int t = 0; t < TM; t++) {
        int c_row = threadRow * TM + t;
        int c_col = threadCol;
        if (row + c_row < M && col + c_col < N) {
            C[c_row * N + c_col] = alpha * threadResult[t] + beta * C[c_row * N + c_col];
        }
    }
}

int main() {
    constexpr int BM = 32;
    constexpr int BN = 32;
    constexpr int TM = 2;
    constexpr int BK = 16;
    int M = 1024;
    int N = 1024;
    int K = 1024;

    float *A, *B, *C;
    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, K * N * sizeof(float));
    cudaMalloc(&C, M * N * sizeof(float));
    cudaMemset(C, 0, M * N * sizeof(float));

    dim3 block(BM * BN / TM);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    gemm_1D_BlockTiling<float, BM, BN, TM, BK><<<grid, block>>>(M, N, K, 1.0f, A, B, 0.0f, C);
    cudaEventRecord(end);
    cudaEventSynchronize(end);
    float time;
    cudaEventElapsedTime(&time, start, end);
    printf("Time: %f ms\n", time);
    cudaFree(A);
    cudaFree(B);
    cudaFree(C);
    cudaEventDestroy(start);
    cudaEventDestroy(end);
}