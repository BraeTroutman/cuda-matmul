#include <iostream>
#include <vector>
#include <cuda.h>
#include <assert.h>

using namespace std;

vector<int> seqMatmul(vector<int> A, vector<int> B, int M, int N, int K) {
	vector<int> C(M*K*sizeof(int));

	int i,j,k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < K; j++) {
			C[i*K + j] = 0;
			for (k = 0; k < N; k++) {
//				printf("C[%i][%i] += A[%i][%i] * B[%i][%i]\n",
//						i,j,
//						i,k,
//						k,j);
//				printf("%d = %d * %d\n",
//						A[i*N+k] * B[k*K+j],
//						A[i*N+k],
//						B[k*K+j]
//				);

				C[i*K + j] += A[i*N + k] * B[k*K + j];
			}
		}
	}
	return C;
}

void printMat(vector<int> A, vector<int> B, int H, int W) {
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			if (A[i*W+j] == B[i*W+j]) {
				printf("%2d/%-2d ", A[i*W+j], B[i*W+j]);
			} else {
				printf("%2d/%-2d ", A[i*W+j], B[i*W+j]);
			}
		}
		puts("");
	}
}

void printMat(vector<int> A, int H, int W) {
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			printf("%-5d", A[i*W + j]);
		}
		puts("");
	}
}

__global__ void matmul(int* A, int* B, int* C, int M, int N, int K) {
  	unsigned int row, col;
  	row = blockIdx.y * blockDim.y + threadIdx.y;
  	col = blockIdx.x * blockDim.x + threadIdx.x;
 	printf("(%i,%i)\n", row, col);	
	if (row < M && col < K) {
	  	unsigned int idx = col + row * K;

		// C[i][j] = SUM[k](A[i][k] * B[k][j])
		C[idx] = 0;
	  	for (int i = 0; i < N; i++) {
			printf("C[%i][%i] += A[%i][%i] * B[%i][%i]\n",
					row, col,
					row, i,
					i, col);
			printf("%d = %d * %d\n",
					A[row*N+i] * B[i*K+col],
					A[row*N+i],
					B[i*K+col]
			);

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

	int A[M][N], B[N][K];
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			A[i][j] = rand() % 10;
		}
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < K; j++) {
			B[i][j] = rand() % 10;
		}
	}	

	vector<int> A_data, B_data;
	for (int i = 0; i < M; i++) {
		for (int j = 0; j < N; j++) {
			A_data.push_back(A[i][j]);
		}
	}
	for (int i = 0; i < N; i++) {
		for (int j = 0; j < K; j++) {
			B_data.push_back(B[i][j]);
		}
	}

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

	cout << "A" << endl;
	printMat(A_data, M, N);
	cout << "B" << endl;
	printMat(B_data, N, K);
	
	if (C_data != checkC)
		printMat(C_data, checkC, M, K);

	assert(C_data == checkC);
}

