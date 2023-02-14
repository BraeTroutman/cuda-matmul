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
	
	vector<int> A = matrixAlloc(opts.M,opts.N);
	vector<int> B = matrixAlloc(opts.N,opts.K);	
	vector<int> C = cudaMatmul(A, B, opts.M, opts.N, opts.K, opts.timed);
	
	if (opts.verbose) {
		cout << "A" << endl;
		printMat(A, opts.M, opts.N);
		cout << "B" << endl;
		printMat(B, opts.N, opts.K);
		cout << "C" << endl;
		printMat(C, opts.M, opts.K);
	}

	if (opts.check) {
		vector<int> checkC = seqMatmul(A, B, opts.M, opts.N, opts.K);
		if (C != checkC) {
			if (opts.verbose) printMat(C, checkC, opts.M, opts.K);
			puts("failure.");
		} else {
			puts("success!");
		}
	}
}

