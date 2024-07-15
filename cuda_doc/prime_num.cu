#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>

__global__ void find_primes(int* numbers, int num_elements, int* prime_count) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int i = idx; i < num_elements; i += stride) {
    if (numbers[i] <= 1) continue;

    int is_prime = 1;
    for (int j = 2; j * j <= numbers[i]; j++) {
      if (numbers[i] % j == 0) {
        is_prime = 0;
        break;
      }
    }

    if (is_prime) {
      atomicAdd(prime_count, 1);
    }
  }
}

int main() {
  const int num_elements = 100001;

  // Allocate memory on host for numbers and prime count
  int* numbers = (int*)malloc(num_elements * sizeof(int));
  for (int i = 0; i < num_elements; i++) {
    numbers[i] = i;
  }
  int* prime_count = (int*)malloc(sizeof(int));
  *prime_count = 0;

  // Allocate memory on device for numbers and prime count
  int* d_numbers;
  int* d_prime_count;
  cudaMalloc(&d_numbers, num_elements * sizeof(int));
  cudaMalloc(&d_prime_count, sizeof(int));

  // Copy data from host to device
  cudaMemcpy(d_numbers, numbers, num_elements * sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(d_prime_count, prime_count, sizeof(int), cudaMemcpyHostToDevice);

  // Define grid and block size for kernel execution
  int threadsPerBlock = 256;
  int numBlocks = (num_elements + threadsPerBlock - 1) / threadsPerBlock;

  // Declare elapsed_time
  float elapsed_time;

  // Start timer
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, 0);

  // Launch kernel
  find_primes<<<numBlocks, threadsPerBlock>>>(d_numbers, num_elements, d_prime_count);

  // Wait for GPU to finish
  cudaDeviceSynchronize();

  // Stop timer
  cudaEventRecord(stop, 0);
  cudaEventElapsedTime(&elapsed_time, start, stop);
  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  // Copy prime count back from device to host
  cudaMemcpy(prime_count, d_prime_count, sizeof(int), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_numbers);
  cudaFree(d_prime_count);

  // Print results
  printf("Found %d prime numbers between 1 and 100001.\n", *prime_count);
  printf("Execution time on GPU: %.3f ms\n", elapsed_time);

  // Free host memory
  free(numbers);
  free(prime_count);

  return 0;
}