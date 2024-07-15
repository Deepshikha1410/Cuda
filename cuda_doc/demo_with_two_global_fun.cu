#include<stdio.h>
#include<cuda.h>


    __global__ void grandChildKernel(){
        printf("from CDAC\n");
    }

    __global__ void childKernel(){
            grandChildKernel<<<1,1>>>();

            cudaDeviceSynchronize();//wait for grandchild to compplete

            printf("hello\n");
        }

    __global__ void parentKernel(){
        childKernel<<<1,1>>>();

        //cudaDeviceSynchronize();//wait for child to compplete

        printf("world\n");
    }

    int main(){
        parentKernel<<<1,1>>>();
        cudaDeviceSynchronize();//wait for parent to complete
        return 0;
    }