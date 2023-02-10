#include <iostream>
#include <vector>
#include <cuda.h>
#include <assert.h>

#include "matrix.h"

using namespace std;

__global__ void matmul(int* A, int* B, int* C, int M, int N, int K) {
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

int main(int argc, char **argv)
{	
	if (argc != 4) {
		printf("Multiply MxN matrix by NxK matrix: %s M N K\n", argv[0]);
		return 0;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	vector<int> A_data = matrixAlloc(M,N);
	vector<int> B_data = matrixAlloc(N,K);	

	unsigned int Asize = A_data.size() * sizeof(int);
	unsigned int Bsize = B_data.size() * sizeof(int);
	unsigned int Csize = M * K * sizeof(int);

	vector<int> C_data(Csize);

	int *dev_A, *dev_B, *dev_C;

	cudaMalloc((void**) &dev_A, Asize);
	cudaMalloc((void**) &dev_B, Bsize);
	cudaMalloc((void**) &dev_C, Csize);

	cudaMemcpy(dev_A, A_data.data(), Asize, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B_data.data(), Bsize, cudaMemcpyHostToDevice);

	dim3 gridSize(1,1), blockSize(K,M);

	if (M > 32 || K > 32) {
		gridSize.x = ceil((double)K/32);
		gridSize.y = ceil((double)M/32);
		blockSize.x = blockSize.y = 32;
	}
	
	matmul<<<gridSize,blockSize>>>(dev_A, dev_B, dev_C, M, N, K);
	cudaDeviceSynchronize();

	cudaMemcpy(C_data.data(), dev_C, Csize, cudaMemcpyDeviceToHost);

	vector<int> checkC = seqMatmul(A_data, B_data, M, N, K);

	if (C_data != checkC)
		printMat(C_data, checkC, M, K);

	assert(C_data == checkC);
}

