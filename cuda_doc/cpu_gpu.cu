#include<stdio.h>
#include<cuda_runtime.h>

__global__ void kernel(void){
    printf("Hello from GPU\n");
}

void cpu_print(void) {
    printf("Hello from CPU\n");
}

int main() {
        kernel <<<1, 1>>> ();
        kernel <<<1, 1>>> ();
        kernel <<<1, 1>>> ();
//this is the kind of gate first it will run the above command then it will 
//bind the next gpu
        cudaDeviceSynchronize(); 

        cpu_print();
        cpu_print();
        cpu_print();

        return 0;
}
