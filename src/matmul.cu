#include <iostream>
#include <vector>
#include <cuda.h>
#include <assert.h>

using namespace std;

vector<int> seqMatmul(vector<int> A, vector<int> B, int M) {
	vector<int> C(A.size());

	int i,j,k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < M; j++) {
			for (k = 0; k < M; k++) {
				C[i*M + j] += A[i*M + k] * B[k*M + j];
			}
		}
	}
	return C;
}

void printMat(vector<int> A, vector<int> B, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			if (A[i*N+j] == B[i*N+j]) {
				printf("%-6d", A[i*N+j]);
			} else {
				printf("%2d/%-2d ", A[i*N+j], B[i*N+j]);
			}
		}
		puts("");
	}
}

void printMat(vector<int> M, int N) {
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < N; j++) {
			printf("%-5d", M[i*N + j]);
		}
		puts("");
	}
}

__global__ void matmul(int* A, int* B, int* C, int M) {
  	unsigned int row, col;
  	row = blockIdx.y * blockDim.y + threadIdx.y;
  	col = blockIdx.x * blockDim.x + threadIdx.x;
 	
	if (row < M && col < M) {
	  	unsigned int idx = col + row * M;
	       	
		// C[i][j] = SUM[k](A[i][k] * B[k][j])	
	  	int k;
	  	for (k = 0; k < M; k++) {
			C[idx] += A[row*M + k] * B[k*M + col];
		}
	}
}

int main(int argc, char **argv)
{	
	if (argc != 2) {
		printf("Command style: %s <matrix_size>\n", argv[0]);
		return 0;
	}

	int N = atoi(argv[1]);
	int A[N][N], B[N][N], i, j;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			A[i][j] = rand() % 10;
			B[i][j] = (i == j) ? 1 : 0;
		}
	}

	vector<int> A_data, B_data;
	for (i = 0; i < N; i++) {
		for (j = 0; j < N; j++) {
			A_data.push_back(A[i][j]);
			B_data.push_back(B[i][j]);
		}
	}
	vector<int> C_data(A_data.size());

	unsigned int size = A_data.size() * sizeof(int);

	int *dev_A, *dev_B, *dev_C;

	cudaMalloc((void**) &dev_A, size);
	cudaMalloc((void**) &dev_B, size);
	cudaMalloc((void**) &dev_C, size);

	cudaMemcpy(dev_A, A_data.data(), size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_B, B_data.data(), size, cudaMemcpyHostToDevice);

	dim3 gridSize(1,1), blockSize(N,N);

	if (N > 32) {
		gridSize.x = gridSize.y = ceil((double)N/32);
		blockSize.x = blockSize.y = 32;
	}

	matmul<<<gridSize,blockSize>>>(dev_A, dev_B, dev_C, N);
	cudaDeviceSynchronize();

	cudaMemcpy(C_data.data(), dev_C, size, cudaMemcpyDeviceToHost);
	
	if (C_data != A_data) {
		printMat(A_data, C_data, N);
	}

	assert(C_data == A_data);
}

