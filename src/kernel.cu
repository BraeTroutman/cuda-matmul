#include <vector>
#include <cuda.h>
#include <iostream>

#include "kernel.h"

__global__ void kernel(int* A, int* B, int* C, int M, int N, int K) {
  	unsigned int row, col;
  	row = blockIdx.y * blockDim.y + threadIdx.y;
  	col = blockIdx.x * blockDim.x + threadIdx.x;

	if (row < M && col < K) {
	  	unsigned int idx = col + row * K;

		// C[i][j] = SUM[k](A[i][k] * B[k][j])
		C[idx] = 0;
	  	for (int i = 0; i < N; i++) {
			C[idx] += A[row*N + i] * B[i*K + col];
		}
	}
}

std::vector<int> cudaMatmul(std::vector<int> &A, std::vector<int>& B, int M, int N, int K, char timed) {
	std::vector<int> C(M*K);

	int *d_A, *d_B, *d_C;
	cudaEvent_t start, stop;	
	if (timed) {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
		cudaEventRecord(start, 0);
	}

	cudaMalloc((void**) &d_A, sizeof(int)*M*N);
	cudaMalloc((void**) &d_B, sizeof(int)*N*K);
	cudaMalloc((void**) &d_C, sizeof(int)*M*K);

	cudaMemcpy(d_A, A.data(), sizeof(int)*M*N, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, B.data(), sizeof(int)*N*K, cudaMemcpyHostToDevice);
	
	dim3 gridSize(1,1), blockSize(K,M);

	if (M > 32 || K > 32) {
		gridSize.x = ceil((double)K/32);
		gridSize.y = ceil((double)M/32);
		blockSize.x = blockSize.y = 32;
	}

	kernel<<<gridSize, blockSize>>>(d_A, d_B, d_C, M, N, K);
	cudaDeviceSynchronize();

	if (timed) {
		cudaEventRecord(stop,0);
		cudaEventSynchronize(stop);
		float diff;
		cudaEventElapsedTime(&diff, start, stop);
		printf("time: %f ms\n", diff);
	}
	
	cudaMemcpy(C.data(), d_C, sizeof(int)*M*K, cudaMemcpyDeviceToHost);
	
	return C;
}


