#include <iostream>
#include <vector>
#include <cuda.h>
#include <assert.h>

#include "matrix.h"

std::vector<int> seqMatmul(std::vector<int> A, std::vector<int> B, int M, int N, int K) {
	std::vector<int> C(M*K*sizeof(int));

	int i,j,k;
	for (i = 0; i < M; i++) {
		for (j = 0; j < K; j++) {
			C[i*K + j] = 0;
			for (k = 0; k < N; k++) {
				C[i*K + j] += A[i*N + k] * B[k*K + j];
			}
		}
	}
	return C;
}

void printMat(std::vector<int> A, std::vector<int> B, int H, int W) {
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

void printMat(std::vector<int> A, int H, int W) {
	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			printf("%-5d", A[i*W + j]);
		}
		puts("");
	}
}

std::vector<int> matrixAlloc(int H, int W) {
	std::vector<int> M(H*W);

	for (int i = 0; i < H; i++) {
		for (int j = 0; j < W; j++) {
			M[i*W+j] = rand() % 10;
		}
	}

	return M;
}

