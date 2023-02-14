#ifndef KERNEL_CUH
#define KERNEL_CUH

std::vector<int> cudaMatmul(std::vector<int>&, std::vector<int>&, int, int, int, char);
__global__ void kernel(int*, int*, int, int, int);

#endif
