#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define N 10000  // Adjust array size as needed

__global__ void elementWiseSum(float *A, float *B) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    A[i] = A[i] + B[i];
  }
}

__global__ void squareElements(float *A) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < N) {
    A[i] = A[i] * A[i];
  }
}

int main() {
  // Allocate memory on the host for arrays A and B
  float *A_h, *B_h;
  cudaMalloc(&A_h, N * sizeof(float));
  cudaMalloc(&B_h, N * sizeof(float));

  // Initialize arrays A and B on the host (assuming initialization is not performance-critical)
  for (int i = 0; i < N; ++i) {
    A_h[i] = i * 0.1f;
    B_h[i] = i * 0.1f;
  }

  // Allocate memory on the device (GPU) for arrays A and B
  float *A_d, *B_d;
  cudaMalloc(&A_d, N * sizeof(float));
  cudaMalloc(&B_d, N * sizeof(float));

  // Start timing
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Transfer data from host to device
  cudaMemcpy(A_d, A_h, N * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(B_d, B_h, N * sizeof(float), cudaMemcpyHostToDevice);

  // Launch element-wise sum kernel
  int threadsPerBlock = 256;  // Adjust as needed based on GPU architecture
  int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
  elementWiseSum<<<blocksPerGrid, threadsPerBlock>>>(A_d, B_d);
  cudaDeviceSynchronize();  // Wait for kernel to finish

  // Launch square elements kernel
  squareElements<<<blocksPerGrid, threadsPerBlock>>>(A_d);
  cudaDeviceSynchronize();  // Wait for kernel to finish

  // Transfer data back from device to host
  cudaMemcpy(A_h, A_d, N * sizeof(float), cudaMemcpyDeviceToHost);

  // Stop timing
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float elapsed_time_ms;
  cudaEventElapsedTime(&elapsed_time_ms, start, stop);

  cudaFree(start);
  cudaFree(stop);

  // Print results
  printf("Required elements of (A+B):\n");
  // ... (print results as before)

  // Free memory on the host and device
  cudaFree(A_h);
  cudaFree(B_h);
  cudaFree(A_d);
  cudaFree(B_d);

  return 0;
}
