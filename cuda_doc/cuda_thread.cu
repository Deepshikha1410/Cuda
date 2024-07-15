#include<cuda_runtime.h>
#include<stdio.h>

__global__ void printThreadInfo(){
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;
    printf("ThreadIdx: %d, BlockIdx: %d, BlockDim: %d, Effective Thread ID: %d\n",threadIdx.x, blockIdx.x, blockDim.x, threadID);
}
    int main() {
        int numBlocks = 3;
        int threadsPerBlock = 4;

        printThreadInfo<<<numBlocks, threadsPerBlock>>>();
        cudaDeviceSynchronize();
    return 0;
}