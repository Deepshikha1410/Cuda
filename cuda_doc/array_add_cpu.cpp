#include<chrono>
#include<iostream>
#include<stdio.h>
#include<cstdint>
#define size 90000

// void add_array(int *c, const int *a, const int *b, int n) {
//     int i = blockIdx.x * blockDim.x + threadIdx.x;
//     if(i < n) {
//       c[i] = a[i] + b[i];
//     }
// }

int main(){

    int a[size] ;
    int b[size] ;
    int c[size] ;



 // Initialize arrays a and b on the host
    for (int i = 0; i < size; i++) {
        a[i] = i;
        b[i] = i;
    }
//Record start time
    auto start_time = std::chrono::high_resolution_clock::now();
// Perform computation task on cpu
    for (int i =0; i< size; i++){
        c[i] = a[i] + b[i];
    }
// //Record end time
  auto end_time = std::chrono::high_resolution_clock::now();
//     //Print the result
//     for(int i = 0; i < size; i++) {
//         printf("%d: ", c[i]);
//     }

//calculate duration
auto duration_ns =  std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
double milliseconds = duration_ns / 1000000.0; // convert nanoseconds to milliseconds
std::cout<< "Time taken by CPU: "<< milliseconds << " milliseconds" << std::endl;
    printf("\n");

    return 0;
}