#include<cuda_runtime.h>
#include<device_launch_parameters.h>
#include<stdio.h>
#define arraySize 5
int main(){
    float hostArray[arraySize] = {1.0, 2.0, 3.0, 4.0, 5.0 };
    float resultArray[arraySize];

    float* deviceArray;
    cudaMalloc((void **)&deviceArray, arraySize * sizeof(float));

    cudaMemcpy(deviceArray, hostArray, arraySize * sizeof(float) ,cudaMemcpyHostToDevice);

    cudaMemcpy(resultArray, deviceArray, arraySize * sizeof(float), cudaMemcpyDeviceToHost);

    //Display results using printf
    printf("Original Array: ");
    for(int i =0; i < arraySize; ++i){
        printf("%f", hostArray[i]);
    }
    printf("\n");

    printf("Copied Array: ");
    for(int i = 0; i < arraySize; ++i){
        printf("%f", resultArray[i]);
    }
    printf("\n");

    //Free allocated memory on GPU
    cudaFree(deviceArray);

    return 0;
}