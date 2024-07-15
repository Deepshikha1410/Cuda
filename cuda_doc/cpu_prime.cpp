#include <iostream>
#include <chrono>
#include <cmath>
#define size 100

int main() {
    const int a = 2;
    const int b = 100;

    auto start_time = std::chrono::high_resolution_clock::now();

    for (int i = a; i <= b; i++) {
        bool isPrime = true;
        if (i < 2) {
            isPrime = false;
        } else {
            for (int j = 2; j <= sqrt(i); j++) {
                if (i % j == 0) {
                    isPrime = false;
                    break;
                }
            }
        }

        if (isPrime) {
            std::cout << i << " is a prime number" << std::endl;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    double milliseconds = duration_ns / 1000000.0; // convert nanoseconds to milliseconds
    std::cout << "Time taken by CPU: " << milliseconds << " milliseconds" << std::endl;
    printf("\n");

    return 0;
}