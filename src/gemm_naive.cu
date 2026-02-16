#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 32

template <typename T>
__global__ void gemm_naive(int M, int N, int K, float alpha, const T *A, const T *B, float beta, T *C) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; ++k) {
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    }
}

int main() {
    int M = 1024;
    int N = 1024;
    int K = 1024;
    float alpha = 1.0f;
    float beta = 0.0f;

    float *A, *B, *C;
    cudaMalloc(&A, M * K * sizeof(float));
    cudaMalloc(&B, K * N * sizeof(float));
    cudaMalloc(&C, M * N * sizeof(float));
    cudaMemset(C, 0, M * N * sizeof(float));

    dim3 block(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE, (M + BLOCK_SIZE - 1) / BLOCK_SIZE);

    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventRecord(start);
    gemm_naive<float><<<grid, block>>>(M, N, K, alpha, A, B, beta, C);
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
