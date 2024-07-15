#include<stdio.h>
#include<cuda_runtime.h>

__global__ void kernel(void){
    // empty kernel, does nothing
}

int main(){
    kernel<<<1,1>>>();
    printf("hello world\n");
    return 0;
}