#include<stdio.h>
#include<cuda_runtime>

__global__ void add_array(int*c, const int *a, const int *b, int size) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < size) {
    c[i] = a[i] + b[i];
    }
}

int main (){
    const int size = 5;
    int a[size] = {1,2,3,4,5};
    int b[size] = {1,2,3,4,5};
    int *d_c;

    //Allocate memory on the device for array c
    cudaMalloc((void**)&d_c, size * sizeof(int));

    //copy array a and b to the device
    int *d_a, *d_b;
    cudaMalloc((void**)&d_a, size * sizeof(int));
    cudaMalloc((void**)&d_b, size * sizeof(int));
    cudaMemcpy(d_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, b, size * sizeof(int), cudaMemcpyHostToDevice);

    add_array<<<5, 5>>>(d_c, d_a, d_b, size);
    cudaDeviceSynchronize();

    //copy the result back from the device
    int *c = (int*)malloc(5 * sizeof(int));
    cudaMemcpy(c, d_c, size * sizeof(int), cudaMemcpyDeviceToHost);

    //Print the result
    for(int i = 0; i < size; i++) {
        printf("%d:", c[i]);
    }

    printf("\n");

    //free memory on the device
    cudaFree(c);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}