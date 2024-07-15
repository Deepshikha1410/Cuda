#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include<stdio.h>
#include<stdlib.h>
#include<time.h>

//CUDA kernel to add two array element-wise with more thread details
__global__ void addArrays(const int* a,const int* b, int* result, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

// Thread details
    int blockIdx = blockIdx.x;
    int threadIdx = threadIdx.x;
    // int threadsperBlock = blockDim.x;
    int totalThreads = blockDim.x * gridDim.x

    //check if the thread index is within the valid range
    if (idx < size) {
        prinf("Thread %d (Block %d, Thread in Block %d, Total Threads %d): Adding %d + %d = %d\n",
                idx, blockId, ThreadId, totalThreads, a[idx], b[idx], a[idx] + b[idx]);
            result[idx] = a[idx] + b[idx];
    }
}

int main (){
    const int arraySize = 100;

    //Host (CPU) data
    int hostArray1[arraySize];
    int hostArray2[arraySize];
    int hostResultArray[arraySize];

    //Generate random integer numbers for the host arrays
    srand((unsigned)time(NULL));
    for(int i =0; i < arraySize; ++i) {
        hostArray1[i] = rand() % 100; //Random numbers between 0 and 99
        hostArray2[i] = rand() % 100; //Random numbers between 0 and 99
    }

    //Device (GPU) data
    int* deviceArray1;
    int* deviceArray2;
    int* deviceResultArray;

    //Allocate device memory
    cudaError_t cudaStatus;

    cudaStatus = cudaMalloc((void***)&deviceArray1, arraySize * sizeof(int));
    if(cudaStatus != cudaSuccess) {
        fprint(stderr, "cudaMalloc for deviceArray1, failed: %s\n",
        cudaGetErrorString(cudaStatus));
        cudaFree(deviceArray1);
        exit(EXIT_FAILURE);
    }

    cudaStatus = cudaMalloc((void***)&deviceArray2, arraySize * sizeof(int));
    if(cudaStatus != cudaSuccess) {
        fprint(stderr, "cudaMalloc for deviceArray2, failed: %s\n",
        cudaGetErrorString(cudaStatus));
        cudaFree(deviceArray1);
        exit(EXIT_FAILURE);
    }

    cudaStatus = cudaMalloc((void***)&deviceResultArray, arraySize * sizeof(int));
    if(cudaStatus != cudaSuccess) {
        fprint(stderr, "cudaMalloc for deviceResultArray, failed: %s\n",
        cudaGetErrorString(cudaStatus));
        cudaFree(deviceArray1);
        cudaFree(deviceArray2);
        exit(EXIT_FAILURE);
    }

    //Copy data from Cpu to Gpu
    cudaStatus = cudaMemcpy(deviceArray1, hostArray1, arraySize * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSucess) {
        fprintf(stderr, "cudaMemcpy(host to device)for deviceArray1 faile: %s\n",cudaGetErroString(cudaSTatus));
        cudaGetErrorString(cudaStatus));
        cudaFree( )deviceArray1;
        cudaFree( )deviceArray2;
        cudaFree(deviceResultArray);
        exit(EXIT_FAILURE);
    }

    //LAunch the KErnel to add arrays onb the device
    addArrays << <1, arraySize >> >(deviceArray1, deviceArray2, deviceResultArray,arraySize);

    //Synchronization to ensure kernneel execution isd completed before processing 
    cudaStatus = cudaDeviceSynchronize();
    if(cudaStatus != cudaSuccess) {
        fprintf(stdrr, "cudaDevicesSynchronize failed: %s\n", cudaGetErrorString(cudaStatus));
                cudaFree(deviceArray1);
                cudaFree(deviceArray2);
                cudaFree(deviceResultArray);
                exit(EXIT_FAILURE);
    }

    //Display Results
    printf("Array 1: ");
    for(int i=0; i< arraySize; ++i) {
        printf("%d", hostArray1[i]);
    }
    printf("\n");

    printf("Array 2: ");
    for(int i=0; i< arraySize; ++i) {
        printf("%d", hostArray2[i]);
    }
    printf("\n");

    printf("Result Array: ");
    for(int i=0; i< arraySize; ++i) {
        printf("%d", hostResultArray[i]);
    }
    printf("\n");
    
    //Free allocated memory to GPU
    cudaFree(deviceArray1);
    cudaFree(deviceArray2);
    
}