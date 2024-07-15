#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>
#include<cuda.h>

#define NUM_CLASSES 3
#define BLOCK_SIZE 256

__global__ void calculateSeatComfort(int *classArray, float *seatComfortArray, int numRecords, float *totalSeatComfort, int *classCounts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < numRecords) {
        int ClassIndex = classArray[idx];
        float seatComfort = seatComfortArray[idx];

        if (ClassIndex < NUM_CLASSES) {
            atomicAdd(&totalSeatComfort[ClassIndex], seatComfort);
            atomicAdd(&classCounts[ClassIndex], 1);
        }
    }
}

int main() {
    // Read the CSV file
    FILE *file = fopen("extracted.csv", "r");

    if (file == NULL) {
        fprintf(stderr, "Error opening file.\n");
        return 1;
    }
    printf("File opened successfully.\n");

    int numRecords = 0;

    // Count the number of records
    while (!feof(file)) {
        char buffer[265];
        if (fgets(buffer, sizeof(buffer), file) == NULL) break;
        numRecords++;
    }

    printf("Number of Records :%d\n", numRecords);

    // Allocate memory on CPU
    int *classArray = (int *) malloc(numRecords * sizeof(int));
    float *seatComfortArray = (float *) malloc(numRecords * sizeof(float));

    // Reset the file pointer to the beginning
    rewind(file);

    int i = 0;
    char line[256];
    while (fgets(line, sizeof(line), file)!= NULL) {
        char *token = strtok(line, ",");
        if (strcmp(token, "Business") == 0) {
            classArray[i] = 0; // business class
        } else if (strcmp(token, "Economy") == 0) {
            classArray[i] = 1; // economy class
        } else if (strcmp(token, "Economy Plus") == 0) {
            classArray[i] = 2; // economy plus class
        }

        token = strtok(NULL, ",");
        seatComfortArray[i] = atof(token);
        i++;
    }

    fclose(file);

    // Allocate memory on GPU
    int *d_classArray;
    float *d_seatComfortArray;
    float *d_totalSeatComfort;
    int *d_classCounts;

    cudaMalloc((void **)&d_classArray, numRecords * sizeof(int));
    cudaMalloc((void **)&d_seatComfortArray, numRecords * sizeof(float));
    cudaMalloc((void **)&d_totalSeatComfort, NUM_CLASSES * sizeof(float));
    cudaMalloc((void **)&d_classCounts, NUM_CLASSES * sizeof(int));

    // Copy data from CPU to GPU
    cudaMemcpy(d_classArray, classArray, numRecords * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_seatComfortArray, seatComfortArray, numRecords * sizeof(float), cudaMemcpyHostToDevice);

    // Initialize totalSeatComfort and classCounts arrays on GPU
    cudaMemset(d_totalSeatComfort, 0, NUM_CLASSES * sizeof(float));
    cudaMemset(d_classCounts, 0, NUM_CLASSES * sizeof(int));
    
     // Start timing GPU execution
     cudaEvent_t start, stop;
     cudaEventCreate(&start);
     cudaEventCreate(&stop);
     cudaEventRecord(start);

    // Calculate total seat comfort and counts for each class on GPU
    int numBlocks = (numRecords + BLOCK_SIZE - 1) / BLOCK_SIZE;
    calculateSeatComfort<<<numBlocks, BLOCK_SIZE>>>(d_classArray, d_seatComfortArray, numRecords, d_totalSeatComfort, d_classCounts);

    // Synchronize threads
    cudaDeviceSynchronize();
    
    // Stop timing GPU execution
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);


    // Copy results from GPU to CPU
    float totalSeatComfort[NUM_CLASSES];
    int classCounts[NUM_CLASSES];
    cudaMemcpy(totalSeatComfort, d_totalSeatComfort, NUM_CLASSES * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(classCounts, d_classCounts, NUM_CLASSES * sizeof(int), cudaMemcpyDeviceToHost);

    // Calculate and display average seat comfort for each class
    printf("Total seat comfort:\n");
    printf("Business class:%.2f\n", totalSeatComfort[0]);
    printf("Economy class:%.2f\n", totalSeatComfort[1]);
    printf("Economy Plus class:%.2f\n", totalSeatComfort[2]);

    printf("Average seat comfort:\n");
    if (classCounts[0] > 0) {
        printf("Business class: %.2f\n", totalSeatComfort[0] / classCounts[0]);
    } else {
        printf("Business class: N/A\n");
    }

    if (classCounts[1] > 0) {
        printf("Economy class: %.2f\n", totalSeatComfort[1] / classCounts[1]);
    } else {
        printf("Economy class: N/A\n");
    }

    if (classCounts[2] > 0) {
        printf("Economy Plus class: %.2f\n", totalSeatComfort[2] / classCounts[2]);
    } else {
        printf("Economy Plus class: N/A\n");
    }
    // Print time taken by GPU
    printf("\nTime taken by GPU : %f milliseconds\n", milliseconds);
    printf("\n");
    // Free the allocated memory
    free(classArray);
    free(seatComfortArray);
    cudaFree(d_classArray);
    cudaFree(d_seatComfortArray);
    cudaFree(d_totalSeatComfort);
    cudaFree(d_classCounts);

    return 0;
}