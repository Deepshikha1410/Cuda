#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>

#define NUM_CLASSES 3
#define NUM_HOUSE_TYPES 5 

__global__ void countHouseTypes(int *houseTypeArray, int *houseTypeCounts, int totalRecords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalRecords) {
        atomicAdd(&houseTypeCounts[houseTypeArray[idx]], 1);
    }
}

__global__ void countGenders(int *genderArray, int *maleCount, int *femaleCount, int totalRecords) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < totalRecords) {
        if (genderArray[idx] == 0) {
            atomicAdd(maleCount, 1);
        } else {
            atomicAdd(femaleCount, 1);
        }
    }
}

int main() {
    // Read the CSV files
    FILE *file1 = fopen("/home/user3/CUDA/ASSIGNMENT/pune.csv", "r");
    FILE *file2 = fopen("/home/user3/CUDA/ASSIGNMENT/mumbai.csv", "r");
    FILE *file3 = fopen("/home/user3/CUDA/ASSIGNMENT/mumbai-suburuban.csv", "r");

    if (file1 == NULL || file2 == NULL || file3 == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }

    printf("Files opened successfully.\n");

    int numRecords1 = 0, numRecords2 = 0, numRecords3 = 0;

    // Count the number of records in each file
    while (!feof(file1)) {
        char buffer[265];
        if (fgets(buffer, sizeof(buffer), file1) == NULL) break;
        numRecords1++;
    }

    while (!feof(file2)) {
        char buffer[265];
        if (fgets(buffer, sizeof(buffer), file2) == NULL) break;
        numRecords2++;
    }

    while (!feof(file3)) {
        char buffer[265];
        if (fgets(buffer, sizeof(buffer), file3) == NULL) break;
        numRecords3++;
    }

    int totalRecords = numRecords1 + numRecords2 + numRecords3;

    // Allocate memory on CPU
    int *houseTypeArray = (int *) malloc(totalRecords * sizeof(int));
    int *genderArray = (int *) malloc(totalRecords * sizeof(int));

    // Reset the file pointers to the beginning
    rewind(file1);
    rewind(file2);
    rewind(file3);

    int i = 0;
    char line[256];
    while (fgets(line, sizeof(line), file1)!= NULL) {
        char *token = strtok(line, ",");
        // assume house type is the second column
        if (strcmp(token, "Apartment") == 0) {
            houseTypeArray[i] = 0; // apartment
        } else if (strcmp(token, "House") == 0) {
            houseTypeArray[i] = 1; // house
        } else if (strcmp(token, "Flat") == 0) {
            houseTypeArray[i] = 2; // flat
        } else if (strcmp(token, "Bungalow") == 0) {
            houseTypeArray[i] = 3; // bungalow
        } else if (strcmp(token, "Other") == 0) {
            houseTypeArray[i] = 4; // other
        }

        token = strtok(NULL, ",");
        // assume gender is the third column
        if (strcmp(token, "Male") == 0) {
            genderArray[i] = 0; // male
        } else if (strcmp(token, "Female") == 0) {
            genderArray[i] = 1; // female
        }
        i++;
    }

    i = numRecords1;
    while (fgets(line, sizeof(line), file2)!= NULL) {
        char *token = strtok(line, ",");
        // assume house type is the second column
        if (strcmp(token, "Apartment") == 0) {
            houseTypeArray[i] = 0; // apartment
        } else if (strcmp(token, "House") == 0) {
            houseTypeArray[i] = 1; // house
        } else if (strcmp(token, "Flat") == 0) {
            houseTypeArray[i] = 2; // flat
        } else if (strcmp(token, "Bungalow") == 0) {
            houseTypeArray[i] = 3; // bungalow
        } else if (strcmp(token, "Other") == 0) {
           houseTypeArray[i] = 4; // other
        }

        token = strtok(NULL, ",");
        // assume gender is the third column
        if (strcmp(token, "Male") == 0) {
            genderArray[i] = 0; // male
        } else if (strcmp(token, "Female") == 0) {
            genderArray[i] = 1; // female
        }
        i++;
    }

    i = numRecords1 + numRecords2;
    while (fgets(line, sizeof(line), file3)!= NULL) {
        char *token = strtok(line, ",");
        // assume house type is the second column
        if (strcmp(token, "Apartment") == 0) {
            houseTypeArray[i] = 0; // apartment
        } else if (strcmp(token, "House") == 0) {
            houseTypeArray[i] = 1; // house
        } else if (strcmp(token, "Flat") == 0) {
            houseTypeArray[i] = 2; // flat
        } else if (strcmp(token, "Bungalow") == 0) {
            houseTypeArray[i] = 3; // bungalow
        } else if (strcmp(token, "Other") == 0) {
            houseTypeArray[i] = 4; // other
        }

        token = strtok(NULL, ",");
        // assume gender is the third column
        if (strcmp(token, "Male") == 0) {
            genderArray[i] = 0; // male
        } else if (strcmp(token, "Female") == 0) {
            genderArray[i] = 1; // female
        }
        i++;
    }

    fclose(file1);
    fclose(file2);
    fclose(file3);

    // Allocate memory on GPU
    int *d_houseTypeArray;
    cudaMalloc((void **)&d_houseTypeArray, totalRecords * sizeof(int));
    cudaMemcpy(d_houseTypeArray, houseTypeArray, totalRecords * sizeof(int), cudaMemcpyHostToDevice);

    int *d_houseTypeCounts;
    cudaMalloc((void **)&d_houseTypeCounts, NUM_HOUSE_TYPES * sizeof(int));
    cudaMemset(d_houseTypeCounts, 0, NUM_HOUSE_TYPES * sizeof(int));

    int *d_genderArray;
    cudaMalloc((void **)&d_genderArray, totalRecords * sizeof(int));
    cudaMemcpy(d_genderArray, genderArray, totalRecords * sizeof(int), cudaMemcpyHostToDevice);

    int *d_maleCount;
    cudaMalloc((void **)&d_maleCount, sizeof(int));
    cudaMemset(d_maleCount, 0, sizeof(int));

    int *d_femaleCount;
    cudaMalloc((void **)&d_femaleCount, sizeof(int));
    cudaMemset(d_femaleCount, 0, sizeof(int));

    // auto cpuStart = std::chrono::high_resolution_clock::now();

    // Calculate the percentage of various house types
    int threadsPerBlock = 256;
    int blocksPerGrid = (totalRecords + threadsPerBlock - 1) / threadsPerBlock;
    countHouseTypes<<<blocksPerGrid, threadsPerBlock>>>(d_houseTypeArray, d_houseTypeCounts, totalRecords);

    // Calculate the ratio of males:females
    countGenders<<<blocksPerGrid, threadsPerBlock>>>(d_genderArray, d_maleCount, d_femaleCount, totalRecords);

    cudaDeviceSynchronize();

    // auto cpuEnd = std::chrono::high_resolution_clock::now();
    // std::chrono::duration<double> cpuElapsed = cpuEnd - cpuStart;
   

    auto gpuStart = std::chrono::high_resolution_clock::now();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    countHouseTypes<<<blocksPerGrid, threadsPerBlock>>>(d_houseTypeArray, d_houseTypeCounts, totalRecords);
    countGenders<<<blocksPerGrid, threadsPerBlock>>>(d_genderArray, d_maleCount, d_femaleCount, totalRecords);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float gpuElapsedTime;
    cudaEventElapsedTime(&gpuElapsedTime, start, stop);
   

    int houseTypeCounts[NUM_HOUSE_TYPES] = {0, 0, 0, 0, 0};
    cudaMemcpy(houseTypeCounts, d_houseTypeCounts, NUM_HOUSE_TYPES * sizeof(int), cudaMemcpyDeviceToHost);

    int maleCount, femaleCount;
    cudaMemcpy(&maleCount, d_maleCount, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&femaleCount, d_femaleCount, sizeof(int), cudaMemcpyDeviceToHost);

    // Free the allocated memory
    free(houseTypeArray);
    free(genderArray);
   cudaFree(d_houseTypeArray);
    cudaFree(d_houseTypeCounts);
    cudaFree(d_genderArray);
    cudaFree(d_maleCount);
    cudaFree(d_femaleCount);

    // Print the results
    printf("House type percentages:\n");
    for (int i = 0; i < NUM_HOUSE_TYPES; i++) {
        printf("%s: %.2f%%\n", i == 0? "Apartment" :i == 1? "House" : i == 2? "Flat" : i == 3? "Bungalow" : "Other",
               (float) houseTypeCounts[i] / totalRecords * 100);
    }

    printf("Ratio of males:females: %.2f:%.2f\n", (float) maleCount / totalRecords * 100,
           (float) femaleCount / totalRecords * 100);

        printf("GPU time: %.2f seconds\n", gpuElapsedTime);

    return 0;
}
