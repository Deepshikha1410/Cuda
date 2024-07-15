#include<stdio.h>
#include<cuda_runtime.h>
#define size 15000

__global__ void add_array(int *c, const int *a, const int *b, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < n) {
      c[i] = a[i] + b[i];
    }
}

int main(){

    int a[size] ;
    int b[size] ;
    int *d_c;

 // Initialize arrays a and b on the host
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i;
    }

    //Allocate memory on the device for array c
    cudaMalloc((void**)&d_c, size * sizeof(int));

    //copy array a and b to the device
    int *d_a, *d_b;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));

    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    //calculate the nubmber of blocks and thread per block
    int threads_per_blocks = 256;
    int number_of_blocks = (size + threads_per_blocks-1)/threads_per_blocks;

    //Start timing GPU execution
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    add_array<<<number_of_blocks, threads_per_blocks>>>(d_c, d_a, d_b, size);
    cudaDeviceSynchronize();

    //Start timing GPU execution
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    //copy the result back from the device
    int *c = (int*)malloc(size * sizeof(int));
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    // //Print the result
    // for(int i = 0; i < size; i++) {
    //     printf("%d: ", c[i]);
    // }

    printf("\n");

printf("\nTime taken by gpu %f : \n",milliseconds);
    //free memory on the device
    cudaFree(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}