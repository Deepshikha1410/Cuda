#include <chrono>
#include <iostream>

#define ARRAY_SIZE 90000

int main() {
    int a[ARRAY_SIZE];
    int b[ARRAY_SIZE];
    int c[ARRAY_SIZE];

    // Initialize arrays (more concise approach)
    std::fill(a, a + ARRAY_SIZE, 0);  // Fills with 0 (or any desired value)
    std::copy(a, a + ARRAY_SIZE, b);   // Efficiently copy a to b

    // Record start time (fix typo: now() instead of now)
    auto start_time = std::chrono::high_resolution_clock::now();

    // Perform computation
    for (int i = 0; i < ARRAY_SIZE; ++i) {
        c[i] = a[i] + b[i];
    }

    // Record end time (fix typo: now() instead of now)
    auto end_time = std::chrono::high_resolution_clock::now();

    // Optional: Calculate and print array elements (using std::cout)
    // ... (same as before)

    // Calculate duration
    auto duration_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end_time - start_time).count();
    double milliseconds = static_cast<double>(duration_ns) / 1000000.0;

    // Print execution time (fix typo: endl instead of end1)
    std::cout << "Time taken by CPU: " << milliseconds << " milliseconds" << std::endl;

    return 0;
}
