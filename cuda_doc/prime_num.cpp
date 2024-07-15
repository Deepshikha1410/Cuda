#include <stdio.h>
#include <time.h>

int is_prime(int num) {
  if (num <= 1) {
    return 0;
  }
  for (int i = 2; i * i <= num; i++) {
    if (num % i == 0) {
      return 0;
    }
  }
  return 1;
}

int main() {
  clock_t start_time = clock();

  int count = 0;
  for (int num = 1; num <= 100001; num++) {
    if (is_prime(num)) {
      count++;
    }
  }

  clock_t end_time = clock();

  double elapsed_time = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("Found %d prime numbers between 1 and 100001.\n", count);
  printf("Execution time: %.2f seconds\n", elapsed_time);

  return 0;
}
