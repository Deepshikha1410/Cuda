#include<iostream>
#include<cuda_runtime.h>

int main() {
    int maxThreadsPerBlock, maxBlocks;

    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&maxBlocks, cudaDevAttrMaxGridDimX, 0);

    std::cout << "Maximum Threads Per Block: " << maxThreadsPerBlock << std::endl;
    std::cout << "Maximum Blocks: " << maxBlocks << std::endl;

    return 0;
}