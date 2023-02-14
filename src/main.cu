#include <iostream>
#include <vector>
#include <assert.h>

#include "matrix.h"
#include "kernel.h"

using namespace std;

int main(int argc, char **argv)
{	
	if (argc != 4) {
		printf("Multiply MxN matrix by NxK matrix: %s M N K\n", argv[0]);
		return 0;
	}

	int M = atoi(argv[1]);
	int N = atoi(argv[2]);
	int K = atoi(argv[3]);

	vector<int> A = matrixAlloc(M,N);
	vector<int> B = matrixAlloc(N,K);	

	vector<int> C = cudaMatmul(A, B, M, N, K);

	vector<int> checkC = seqMatmul(A, B, M, N, K);

	if (C != checkC)
		printMat(C, checkC, M, K);

	assert(C == checkC);
}

