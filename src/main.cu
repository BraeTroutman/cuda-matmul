#include <iostream>
#include <vector>
#include <assert.h>

#include "matrix.h"
#include "kernel.h"
#include "arguments.h"

using namespace std;

int main(int argc, char **argv)
{	
	options_t opts = parse_args(argc, argv);
	
	int optslen;
	for (optslen = 0; opts.remaining[optslen] != NULL; ++optslen);

	if (optslen < 3) {
		printf("Usage: %s [options] M N K\n", argv[0]);	
		return 1;
	}
	
	int M = atoi(opts.remaining[0]);
	int N = atoi(opts.remaining[1]);
	int K = atoi(opts.remaining[2]);

	vector<int> A = matrixAlloc(M,N);
	vector<int> B = matrixAlloc(N,K);	
	vector<int> C = cudaMatmul(A, B, M, N, K);
	
	if (opts.verbose) {
		cout << "A" << endl;
		printMat(A, M, N);
		cout << "B" << endl;
		printMat(B, N, K);
		cout << "C" << endl;
		printMat(C, M, K);
	}

	if (opts.check) {
		vector<int> checkC = seqMatmul(A, B, M, N, K);
		if (C != checkC) printMat(C, checkC, M, K);
	}
}

