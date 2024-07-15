#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_LINE_LENGTH 1024
#define MAX_CITY_LENGTH 50
#define MAX_HOUSE_TYPE_LENGTH 20

typedef struct {
    char city[MAX_CITY_LENGTH];
    char house_type[MAX_HOUSE_TYPE_LENGTH];
    int male;
    int female;
} Data;

int main() {
    FILE *file = fopen("Economic_Census_Data.csv", "r");
    if (file == NULL) {
        printf("Error opening file\n");
        return 1;
    }

    char line[MAX_LINE_LENGTH];
    fgets(line, MAX_LINE_LENGTH, file); // Skip the header

    int pune_count = 0, mumbai_count = 0, mumbai_subarban_count = 0;
    int pucca_count = 0, semi_pucca_count = 0, kaccha_count = 0;
    int male_count = 0, female_count = 0;

    while (fgets(line, MAX_LINE_LENGTH, file)) {
        char *token = strtok(line, ",");
        Data data;

        strcpy(data.city, token);
        token = strtok(NULL, ",");
        strcpy(data.house_type, token);
        token = strtok(NULL, ",");
        data.male = atoi(token);
        token = strtok(NULL, ",");
        data.female = atoi(token);

        if (strcmp(data.city, "Pune") == 0) {
            pune_count++;
        } else if (strcmp(data.city, "Mumbai") == 0) {
            mumbai_count++;
        } else if (strcmp(data.city, "Mumbai Subarban") == 0) {
            mumbai_subarban_count++;
        }

        if (strcmp(data.house_type, "Pucca") == 0) {
            pucca_count++;
        } else if (strcmp(data.house_type, "Semi Pucca") == 0) {
            semi_pucca_count++;
        } else if (strcmp(data.house_type, "Kaccha") == 0) {
            kaccha_count++;
        }

        male_count += data.male;
        female_count += data.female;
    }

    fclose(file);

    int total_count = pune_count + mumbai_count + mumbai_subarban_count;
    float pucca_percentage = (float)pucca_count / total_count * 100;
    float semi_pucca_percentage = (float)semi_pucca_count / total_count * 100;
    float kaccha_percentage = (float)kaccha_count / total_count * 100;

    printf("Percentage of Pucca houses: %.2f%%\n", pucca_percentage);
    printf("Percentage of Semi Pucca houses: %.2f%%\n", semi_pucca_percentage);
    printf("Percentage of Kaccha houses: %.2f%%\n", kaccha_percentage);

    float male_female_ratio = (float)male_count / female_count;
    printf("Male to Female Ratio: %.2f\n", male_female_ratio);

    return 0;
}