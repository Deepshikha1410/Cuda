#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <chrono>
#include <iostream>

#define NUM_CLASSES 3
#define NUM_HOUSE_TYPES 5 // assume 5 house types

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

    auto start = std::chrono::high_resolution_clock::now();

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

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    

    // Calculate the percentage of various house types
    int houseTypeCounts[NUM_HOUSE_TYPES] = {0, 0, 0, 0, 0};
    for (int i = 0; i < totalRecords; i++) {
        houseTypeCounts[houseTypeArray[i]]++;
    }

    printf("House type percentages:\n");
    for (int i = 0; i < NUM_HOUSE_TYPES; i++) {
        printf("%s: %.2f%%\n", i == 0? "Apartment" : i == 1? "House" : i == 2? "Flat" : i == 3? "Bungalow" : "Other",
               (float) houseTypeCounts[i] / totalRecords * 100);
    }

    // Calculate the ratio of males:females
    int maleCount = 0, femaleCount = 0;
    for (int i = 0; i < totalRecords; i++) {
        if (genderArray[i] == 0) {
            maleCount++;
        } else {
            femaleCount++;
        }
    }

    printf("Ratio of males:females: %.2f:%.2f\n", (float) maleCount / totalRecords * 100,
           (float) femaleCount / totalRecords * 100);

    printf("CPU time: %.2f miliseconds\n", elapsed.count());

    // Free the allocated memory
    free(houseTypeArray);
    free(genderArray);

    return 0;
}