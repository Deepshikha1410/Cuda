#include <stdio.h>
#include <cuda_runtime.h>

#define NUM 90000

// CUDA kernel function for printing prime numbers
__global__ void printPrimeNumbers(int start, int end) {
    int threadID = blockIdx.x * blockDim.x + threadIdx.x;

    if (threadID < end - start + 1) {
        int num = start + threadID;
        bool isPrime = true;

        for (int i = 2; i * i <= num; i++) {
            if (num % i == 0) {
                isPrime = false;
                break;
            }
        }

        if (isPrime && num > 1) {
            // printf("%d ", num);
        }
    }
}

int main() {
    int host_fromNum = 2;
    int host_toNum = NUM;

    // Declare device variables
    int *device_fromNum;
    int *device_toNum;

    // Allocate memory on the device
    cudaMalloc((void**)&device_fromNum, sizeof(int));
    cudaMalloc((void**)&device_toNum, sizeof(int));

    // Copy data from host to device
    cudaMemcpy(device_fromNum, &host_fromNum, sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_toNum, &host_toNum, sizeof(int), cudaMemcpyHostToDevice);

    // Calculate the number of threads per block and blocks per grid
    int threadPerBlock = 256;
    int blockPerGrid = (host_toNum - host_fromNum + threadPerBlock - 1) / threadPerBlock;

    // Start timing GPU execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Launch the CUDA kernel
    printPrimeNumbers<<<blockPerGrid, threadPerBlock>>>(host_fromNum, host_toNum);
    cudaDeviceSynchronize();

    // Stop timing GPU execution
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Print time taken by GPU
    printf("\nTime taken by GPU : %f milliseconds\n", milliseconds);

    // Free allocated memory
    cudaFree(device_fromNum);
    cudaFree(device_toNum);

    return 0;
}