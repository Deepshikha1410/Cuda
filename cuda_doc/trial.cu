#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define M 2  // Array rows
#define N 3 // Array columns

__global__ void addArrays2D(int* d_result, int* d_array1, int* d_array2) {
  int blockIdxX = blockIdx.x;
  int blockIdxY = blockIdx.y;
  int threadIdxX = threadIdx.x;
  int threadIdxY = threadIdx.y;

  // Calculate a unique thread ID within a block
  int uniqueThreadId = blockIdxX * blockDim.x * blockDim.y + 
                       blockIdxY * blockDim.x + 
                       threadIdxX + threadIdxY;

  int row = blockIdxY * blockDim.y + threadIdxY;
  int col = blockIdxX * blockDim.x + threadIdxX;

  // Check for valid element within array bounds
  if (row < M && col < N) {
    int result = d_array1[row * N + col] + d_array2[row * N + col];
    d_result[row * N + col] = result;
    // Print the thread ID along with operation details 
    printf("Glocally thread usnique id is : Thread %d: Adding A[%d][%d] + B[%d][%d] = Result[%d][%d]\n", 
           uniqueThreadId, row, col, row, col, row, col);
  }
}

int main() {
  // Host memory for the arrays
  int host_array1[M][N] = {
    {1, 2, 3},
    {6, 7, 8},
 
  };
  int host_array2[M][N] = {
    {10, 20, 30},
    {60, 70, 80},

  };

  // Allocate memory on device for the arrays
  int* d_array1, *d_array2, *d_result;
  cudaMalloc(&d_array1, M * N * sizeof(int));
  cudaMalloc(&d_array2, M * N * sizeof(int));
  cudaMalloc(&d_result, M * N * sizeof(int));

  // Copy arrays from host to device
  cudaMemcpy(d_array1, host_array1, M * N * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_array2, host_array2, M * N * sizeof(int), cudaMemcpyHostToDevice);

  // Define grid and block sizes
  int threadsPerBlockX = 6;
  int threadsPerBlockY = 3;
  dim3 threadsPerBlock(threadsPerBlockX, threadsPerBlockY, 1);
  int numBlocksX = (N + threadsPerBlockX - 1) / threadsPerBlockX;
  int numBlocksY = (M + threadsPerBlockY - 1) / threadsPerBlockY;
  dim3 blocksPerGrid(numBlocksX, numBlocksY, 1);

  // Launch the kernel
  addArrays2D<<<blocksPerGrid, threadsPerBlock>>>(d_result, d_array1, d_array2);

  // Synchronize threads

  cudaDeviceSynchronize();

  // Allocate memory on host to store results
  int host_result[M][N];

  // Copy results back from device to host
  cudaMemcpy(host_result, d_result, M * N * sizeof(int), cudaMemcpyDeviceToHost);

  // Print the result array
  printf("Resulting Array:\n");
  for (int i = 0; i < M; ++i) {
    for (int j = 0; j < N;++j) {
      printf("%d ", host_result[i][j]);
    }
     printf("\n");
  }

   // Free memory on device
   cudaFree(d_array1);
   cudaFree(d_array2);
   cudaFree(d_result);

  return 0;
}


