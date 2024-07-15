#include <iostream>
#include <chrono>
#include <cmath>

#define size 100

__global__ void findPrimes(int *primes, int a, int b) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= a && idx <= b) {
        bool isPrime = true;
        if (idx < 2) {
            isPrime = false;
        } else {
            for (int j = 2; j <= sqrt(idx); j++) {
                if (idx % j == 0) {
                    isPrime = false;
                    break;
                }
            }
        }

        if (isPrime) {
            primes[idx - a] = idx;
        }
    }
}

int main() {
    const int a = 2;
    const int b = 100;

    int *primes;
    cudaMalloc((void **)&primes, (b - a + 1) * sizeof(int));

    int blockSize = 256;
    int numBlocks = (b - a + 1 + blockSize - 1) / blockSize;

    auto start_time = std::chrono::high_resolution_clock::now();

    findPrimes<<<numBlocks, blockSize>>>(primes, a, b);

    cudaDeviceSynchronize();

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    double milliseconds = duration_ns / 1000000.0; // convert nanoseconds to milliseconds
    std::cout << "Time taken by GPU: " << milliseconds << " milliseconds" << std::endl;

    int *hostPrimes = new int[b - a + 1];
    cudaMemcpy(hostPrimes, primes, (b - a + 1) * sizeof(int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < b - a + 1; i++) {
        if (hostPrimes[i] != 0) {
            std::cout << hostPrimes[i] << " is a prime number" << std::endl;
        }
    }

    cudaFree(primes);
    delete[] hostPrimes;

    return 0;
}