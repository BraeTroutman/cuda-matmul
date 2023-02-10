#include <vector>
#include <cuda.h>

#include "kernel.h"

std::vector<int> cudaMatmul(std::vector<int> A, std::vector<int> B, int M, int N, int K) {
	std::vector<int> C(M*K);
	return C;
}

__global__ void kernel(int* A, int* B, int M, int N, int K) {
	
}

