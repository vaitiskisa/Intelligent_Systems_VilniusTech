#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <math.h>

#ifndef M_PI
    #define M_PI 3.14159265
#endif

#define USE_GAUSSIAN_CLASSIFER 1

#define TRAINING_SAMPLE_SIZE   6
#define VALIDATION_SAMPLE_SIZE 7
#define TRAINING_EPOCH_LIMIT   1000

typedef struct {
    double x1;
    double x2;
    int label; // 1 or -1
} SampleData;

static double gaussianFunction(double x, double mean, double var)
{
    double coeff = 1.0 / (sqrt(2.0 * M_PI * var));
    double exponent = -((x - mean) * (x - mean)) / (2.0 * var);
    return coeff * exp(exponent);
}

static int readDataset(const char *path, SampleData *out_items)
{
    FILE *f = fopen(path, "r");
    if(!f) {
        perror("Couldn't open file\n");
        return -1;
    }

    size_t len = 0;

    char line[256];
    while(fgets(line, sizeof(line), f)) {
        double x1, x2;
        int y;

        int n = sscanf(line, " %lf,%lf,%d", &x1, &x2, &y);
        if(n < 3) {
            fprintf(stderr, "Parse error: '%s'\n", line);
            fclose(f);
            return -1;
        }

        out_items[len++] = (SampleData){
            .x1 = x1,
            .x2 = x2,
            .label = y,
        };
    }

    fclose(f);

    return 0;
}

static void perceptronClassifier(SampleData *train, SampleData *valid)
{
    srand(time(NULL));
    double w1 = (double)rand() / RAND_MAX * 2.0 - 1.0;
    double w2 = (double)rand() / RAND_MAX * 2.0 - 1.0;
    double b = (double)rand() / RAND_MAX * 2.0 - 1.0;

    printf("Initial weights: w1=%.5f, w2=%.5f, b=%.5f\n", w1, w2, b);

    double learning_rate = 0.15;
    int epochs = 0;

    while(1) {
        int err_accum = 0;
        for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
            double v = w1 * train[i].x1 + w2 * train[i].x2 + b;
            int y = (v > 0) ? 1 : -1;

            w1 += learning_rate * (train[i].label - y) * train[i].x1;
            w2 += learning_rate * (train[i].label - y) * train[i].x2;
            b += learning_rate * (train[i].label - y);

            err_accum += abs(train[i].label - y);
        }

        if(err_accum == 0 || epochs++ >= TRAINING_EPOCH_LIMIT) {
            break;
        }
    }
    printf("Training completed in %d epochs, weights: w1=%.5f, w2=%.5f, b=%.5f\n", epochs, w1, w2, b);

    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        double v = w1 * valid[i].x1 + w2 * valid[i].x2 + b;
        int y = (v > 0) ? 1 : -1;

        if(y != valid[i].label) {
            printf("Validation sample %d misclassified: expected %d, got %d\n", i, valid[i].label, y);
        } else {
            printf("Validation sample %d classified correctly: expected %d, got %d\n", i, valid[i].label, y);
        }
    }
}

static void gaussianNaiveBayesClassifier(SampleData *train, SampleData *valid)
{
    // Calculate prior probabilities
    int pear_count = 0;
    int apple_count = 0;

    for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        if(train[i].label == -1) {
            pear_count += 1;
        } else {
            apple_count += 1;
        }
    }

    double total = pear_count + apple_count;
    double pear_probability = pear_count / total;
    double apple_probability = apple_count / total;

    printf("Prior probabilities: P(pear) = %.5f, P(apple) = %.5f\n", pear_probability, apple_probability);

    // Calculate means
    double pear_mean_x1 = 0.0, pear_mean_x2 = 0.0;
    double apple_mean_x1 = 0.0, apple_mean_x2 = 0.0;

    for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        if(train[i].label == -1) {
            pear_mean_x1 += train[i].x1;
            pear_mean_x2 += train[i].x2;
        } else {
            apple_mean_x1 += train[i].x1;
            apple_mean_x2 += train[i].x2;
        }
    }

    pear_mean_x1 /= pear_count;
    pear_mean_x2 /= pear_count;
    apple_mean_x1 /= apple_count;
    apple_mean_x2 /= apple_count;

    printf("Pear means: x1=%.5f, x2=%.5f\n", pear_mean_x1, pear_mean_x2);
    printf("Apple means: x1=%.5f, x2=%.5f\n", apple_mean_x1, apple_mean_x2);

    // Calculate variances
    double pear_var_x1 = 0.0, pear_var_x2 = 0.0;
    double apple_var_x1 = 0.0, apple_var_x2 = 0.0;

    for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        if(train[i].label == -1) {
            pear_var_x1 += (train[i].x1 - pear_mean_x1) * (train[i].x1 - pear_mean_x1);
            pear_var_x2 += (train[i].x2 - pear_mean_x2) * (train[i].x2 - pear_mean_x2);
        } else {
            apple_var_x1 += (train[i].x1 - apple_mean_x1) * (train[i].x1 - apple_mean_x1);
            apple_var_x2 += (train[i].x2 - apple_mean_x2) * (train[i].x2 - apple_mean_x2);
        }
    }

    double var_pooled_x1 = 0.0, var_pooled_x2 = 0.0;
    // Pooled variance calculation
    var_pooled_x1 = (pear_var_x1 + apple_var_x1) / total;
    var_pooled_x2 = (pear_var_x2 + apple_var_x2) / total;

    printf("Pear variances: x1=%.5f, x2=%.5f\n", pear_var_x1, pear_var_x2);
    printf("Apple variances: x1=%.5f, x2=%.5f\n", apple_var_x1, apple_var_x2);

    // Validation
    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        double p_pear = gaussianFunction(valid[i].x1, pear_mean_x1, var_pooled_x1) *
                        gaussianFunction(valid[i].x2, pear_mean_x2, var_pooled_x2) * pear_probability;
        double p_apple = gaussianFunction(valid[i].x1, apple_mean_x1, var_pooled_x1) *
                         gaussianFunction(valid[i].x2, apple_mean_x2, var_pooled_x2) * apple_probability;
        int predicted_label = (p_apple > p_pear) ? 1 : -1;

        if(predicted_label != valid[i].label) {
            printf("Validation sample %d misclassified: expected %d, got %d\n", i, valid[i].label, predicted_label);
        } else {
            printf("Validation sample %d classified correctly: expected %d, got %d\n", i, valid[i].label,
                   predicted_label);
        }
    }
}

int main(void)
{
    const char *train_path = "datasets/training_data.txt";
    const char *valid_path = "datasets/validation_data.txt";

    SampleData train[TRAINING_SAMPLE_SIZE] = { 0 };
    SampleData valid[VALIDATION_SAMPLE_SIZE] = { 0 };

    if(readDataset(train_path, train) != 0) {
        fprintf(stderr, "Failed to read training data from %s\n", train_path);
        return 1;
    }
    if(readDataset(valid_path, valid) != 0) {
        fprintf(stderr, "Failed to read validation data from %s\n", valid_path);
        return 1;
    }

    if(USE_GAUSSIAN_CLASSIFER) {
        gaussianNaiveBayesClassifier(train, valid);
    } else {
        perceptronClassifier(train, valid);
    }

    return 0;
}
