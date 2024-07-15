#include<stdio.h>
#include<cuda_runtime.h>

__global__ void kernel(void){
  printf("Hello from GPU\n");
}


void cpu_print(void){
    printf("Hello from CPU\n");
}
int main(){
    kernel<<<1,1>>>();
    kernel<<<1,1>>>();
    kernel<<<1,1>>>();

   cudaDeviceSynchronize();
    cpu_print();
    cpu_print();
    cpu_print();
    
    return 0;
}