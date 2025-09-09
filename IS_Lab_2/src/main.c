#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
    #define M_PI 3.14159265
#endif

#define TRAINING_SAMPLE_SIZE   20
#define VALIDATION_SAMPLE_SIZE 10
#define TRAINING_EPOCH_LIMIT   10000

typedef struct {
    struct {
        /* Weights */
        double w11_1;
        double w21_1;
        double w31_1;
        double w41_1;
        double w51_1;
        /* Biases */
        double b1_1;
        double b2_1;
        double b3_1;
        double b4_1;
        double b5_1;
    } FirstLayer;
    struct {
        /* Weights */
        double w11_2;
        double w12_2;
        double w13_2;
        double w14_2;
        double w15_2;
        /* Biases */
        double b1_2;
    } SecondLayer;
} NetworkLayers;

static NetworkLayers bestNetwork = { { 0 } };

static inline double sigmoid(double x)
{
    return 1.0 / (1.0 + exp(-x));
}

/* y = (1 + 0.6*sin(2*pi*x/0.7)) + 0.3*sin(2*pi*x))/2; */
static double targetFunction(double x)
{
    return (1.0 + 0.6 * sin(2.0 * M_PI * x / 0.7) + 0.3 * sin(2.0 * M_PI * x)) / 2.0;
}

static void saveCheckpoint(NetworkLayers currNetwork)
{
    bestNetwork.FirstLayer.w11_1 = currNetwork.FirstLayer.w11_1;
    bestNetwork.FirstLayer.w21_1 = currNetwork.FirstLayer.w21_1;
    bestNetwork.FirstLayer.w31_1 = currNetwork.FirstLayer.w31_1;
    bestNetwork.FirstLayer.w41_1 = currNetwork.FirstLayer.w41_1;
    bestNetwork.FirstLayer.w51_1 = currNetwork.FirstLayer.w51_1;

    bestNetwork.FirstLayer.b1_1 = currNetwork.FirstLayer.b1_1;
    bestNetwork.FirstLayer.b2_1 = currNetwork.FirstLayer.b2_1;
    bestNetwork.FirstLayer.b3_1 = currNetwork.FirstLayer.b3_1;
    bestNetwork.FirstLayer.b4_1 = currNetwork.FirstLayer.b4_1;
    bestNetwork.FirstLayer.b5_1 = currNetwork.FirstLayer.b5_1;

    bestNetwork.SecondLayer.w11_2 = currNetwork.SecondLayer.w11_2;
    bestNetwork.SecondLayer.w12_2 = currNetwork.SecondLayer.w12_2;
    bestNetwork.SecondLayer.w13_2 = currNetwork.SecondLayer.w13_2;
    bestNetwork.SecondLayer.w14_2 = currNetwork.SecondLayer.w14_2;
    bestNetwork.SecondLayer.w15_2 = currNetwork.SecondLayer.w15_2;

    bestNetwork.SecondLayer.b1_2 = currNetwork.SecondLayer.b1_2;
}

static void applyCheckpoint(NetworkLayers *currNetwork)
{
    currNetwork->FirstLayer.w11_1 = bestNetwork.FirstLayer.w11_1;
    currNetwork->FirstLayer.w21_1 = bestNetwork.FirstLayer.w21_1;
    currNetwork->FirstLayer.w31_1 = bestNetwork.FirstLayer.w31_1;
    currNetwork->FirstLayer.w41_1 = bestNetwork.FirstLayer.w41_1;
    currNetwork->FirstLayer.w51_1 = bestNetwork.FirstLayer.w51_1;

    currNetwork->FirstLayer.b1_1 = bestNetwork.FirstLayer.b1_1;
    currNetwork->FirstLayer.b2_1 = bestNetwork.FirstLayer.b2_1;
    currNetwork->FirstLayer.b3_1 = bestNetwork.FirstLayer.b3_1;
    currNetwork->FirstLayer.b4_1 = bestNetwork.FirstLayer.b4_1;
    currNetwork->FirstLayer.b5_1 = bestNetwork.FirstLayer.b5_1;

    currNetwork->SecondLayer.w11_2 = bestNetwork.SecondLayer.w11_2;
    currNetwork->SecondLayer.w12_2 = bestNetwork.SecondLayer.w12_2;
    currNetwork->SecondLayer.w13_2 = bestNetwork.SecondLayer.w13_2;
    currNetwork->SecondLayer.w14_2 = bestNetwork.SecondLayer.w14_2;
    currNetwork->SecondLayer.w15_2 = bestNetwork.SecondLayer.w15_2;

    currNetwork->SecondLayer.b1_2 = bestNetwork.SecondLayer.b1_2;
}

static void saveDataToCSV(double *data, size_t data_size, char *filename)
{
    FILE *f = fopen(filename, "w");
    if(f == NULL) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    for(uint8_t i = 0; i < data_size; i++) {
        fprintf(f, "%.5f\n", data[i]);
    }

    fclose(f);
}

int main(void)
{
    srand(time(NULL));

    double train_data[TRAINING_SAMPLE_SIZE] = { 0.0 };
    double valid_data[VALIDATION_SAMPLE_SIZE] = { 0.0 };
    for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        train_data[i] = (1.0 / TRAINING_SAMPLE_SIZE) * i;
    }
    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        valid_data[i] = (double)rand() / (double)RAND_MAX;
    }
    saveDataToCSV(train_data, TRAINING_SAMPLE_SIZE, "train_data.csv");
    saveDataToCSV(valid_data, VALIDATION_SAMPLE_SIZE, "valid_data.csv");

    double train_data_y[TRAINING_SAMPLE_SIZE] = { 0.0 };
    double valid_data_y[VALIDATION_SAMPLE_SIZE] = { 0.0 };
    for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        train_data_y[i] = targetFunction(train_data[i]);
    }
    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        valid_data_y[i] = targetFunction(valid_data[i]);
    }
    saveDataToCSV(train_data_y, TRAINING_SAMPLE_SIZE, "train_expected.csv");
    saveDataToCSV(valid_data_y, VALIDATION_SAMPLE_SIZE, "expected.csv");

    NetworkLayers network = {
        .FirstLayer.w11_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w21_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w31_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w41_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w51_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.b1_1 = 0.0,
        .FirstLayer.b2_1 = 0.0,
        .FirstLayer.b3_1 = 0.0,
        .FirstLayer.b4_1 = 0.0,
        .FirstLayer.b5_1 = 0.0,
        .SecondLayer.w11_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.w12_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.w13_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.w14_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.w15_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.b1_2 = 0.5,
    };

    double learning_rate = 0.03;
    double best_cost = INT32_MAX;
    int epochs = 0;

    while(epochs++ < TRAINING_EPOCH_LIMIT) {
        double cost_accum = 0;
        for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
            double v1_1 = network.FirstLayer.w11_1 * train_data[i] + network.FirstLayer.b1_1;
            double v2_1 = network.FirstLayer.w21_1 * train_data[i] + network.FirstLayer.b2_1;
            double v3_1 = network.FirstLayer.w31_1 * train_data[i] + network.FirstLayer.b3_1;
            double v4_1 = network.FirstLayer.w41_1 * train_data[i] + network.FirstLayer.b4_1;
            double v5_1 = network.FirstLayer.w51_1 * train_data[i] + network.FirstLayer.b5_1;

            double y1_1 = sigmoid(v1_1);
            double y2_1 = sigmoid(v2_1);
            double y3_1 = sigmoid(v3_1);
            double y4_1 = sigmoid(v4_1);
            double y5_1 = sigmoid(v5_1);

            double v1_2 = network.SecondLayer.w11_2 * y1_1 + network.SecondLayer.w12_2 * y2_1 +
                          network.SecondLayer.w13_2 * y3_1 + network.SecondLayer.w14_2 * y4_1 +
                          network.SecondLayer.w15_2 * y5_1 + network.SecondLayer.b1_2;

            double error = train_data_y[i] - v1_2;

            double grad1_2 = error;
            double grad11_1 = grad1_2 * network.SecondLayer.w11_2 * y1_1 * (1 - y1_1);
            double grad21_1 = grad1_2 * network.SecondLayer.w12_2 * y2_1 * (1 - y2_1);
            double grad31_1 = grad1_2 * network.SecondLayer.w13_2 * y3_1 * (1 - y3_1);
            double grad41_1 = grad1_2 * network.SecondLayer.w14_2 * y4_1 * (1 - y4_1);
            double grad51_1 = grad1_2 * network.SecondLayer.w15_2 * y5_1 * (1 - y5_1);

            network.FirstLayer.w11_1 += learning_rate * grad11_1 * train_data[i];
            network.FirstLayer.w21_1 += learning_rate * grad21_1 * train_data[i];
            network.FirstLayer.w31_1 += learning_rate * grad31_1 * train_data[i];
            network.FirstLayer.w41_1 += learning_rate * grad41_1 * train_data[i];
            network.FirstLayer.w51_1 += learning_rate * grad51_1 * train_data[i];
            network.FirstLayer.b1_1 += learning_rate * grad11_1;
            network.FirstLayer.b2_1 += learning_rate * grad21_1;
            network.FirstLayer.b3_1 += learning_rate * grad31_1;
            network.FirstLayer.b4_1 += learning_rate * grad41_1;
            network.FirstLayer.b5_1 += learning_rate * grad51_1;

            network.SecondLayer.w11_2 += learning_rate * grad1_2 * y1_1;
            network.SecondLayer.w12_2 += learning_rate * grad1_2 * y2_1;
            network.SecondLayer.w13_2 += learning_rate * grad1_2 * y3_1;
            network.SecondLayer.w14_2 += learning_rate * grad1_2 * y4_1;
            network.SecondLayer.w15_2 += learning_rate * grad1_2 * y5_1;
            network.SecondLayer.b1_2 += learning_rate * grad1_2;
        }

        for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
            double v1_1 = network.FirstLayer.w11_1 * valid_data[i] + network.FirstLayer.b1_1;
            double v2_1 = network.FirstLayer.w21_1 * valid_data[i] + network.FirstLayer.b2_1;
            double v3_1 = network.FirstLayer.w31_1 * valid_data[i] + network.FirstLayer.b3_1;
            double v4_1 = network.FirstLayer.w41_1 * valid_data[i] + network.FirstLayer.b4_1;
            double v5_1 = network.FirstLayer.w51_1 * valid_data[i] + network.FirstLayer.b5_1;

            double y1_1 = sigmoid(v1_1);
            double y2_1 = sigmoid(v2_1);
            double y3_1 = sigmoid(v3_1);
            double y4_1 = sigmoid(v4_1);
            double y5_1 = sigmoid(v5_1);

            double v1_2 = network.SecondLayer.w11_2 * y1_1 + network.SecondLayer.w12_2 * y2_1 +
                          network.SecondLayer.w13_2 * y3_1 + network.SecondLayer.w14_2 * y4_1 +
                          network.SecondLayer.w15_2 * y5_1 + network.SecondLayer.b1_2;

            double error = valid_data_y[i] - v1_2;
            double cost = (error * error) / 2.0;

            cost_accum += cost;
        }
        double mean_cost = cost_accum / VALIDATION_SAMPLE_SIZE;

        if(mean_cost < best_cost) {
            best_cost = mean_cost;
            saveCheckpoint(network);
        }
    }
    printf("Training completed in %d epochs with best cost: %.5f\n", epochs, best_cost);

    applyCheckpoint(&network);

    double predictions[VALIDATION_SAMPLE_SIZE] = { 0.0 };

    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        double v1_1 = network.FirstLayer.w11_1 * valid_data[i] + network.FirstLayer.b1_1;
        double v2_1 = network.FirstLayer.w21_1 * valid_data[i] + network.FirstLayer.b2_1;
        double v3_1 = network.FirstLayer.w31_1 * valid_data[i] + network.FirstLayer.b3_1;
        double v4_1 = network.FirstLayer.w41_1 * valid_data[i] + network.FirstLayer.b4_1;
        double v5_1 = network.FirstLayer.w51_1 * valid_data[i] + network.FirstLayer.b5_1;

        double y1_1 = sigmoid(v1_1);
        double y2_1 = sigmoid(v2_1);
        double y3_1 = sigmoid(v3_1);
        double y4_1 = sigmoid(v4_1);
        double y5_1 = sigmoid(v5_1);

        double v1_2 = network.SecondLayer.w11_2 * y1_1 + network.SecondLayer.w12_2 * y2_1 +
                      network.SecondLayer.w13_2 * y3_1 + network.SecondLayer.w14_2 * y4_1 +
                      network.SecondLayer.w15_2 * y5_1 + network.SecondLayer.b1_2;

        predictions[i] = v1_2;
    }

    saveDataToCSV(predictions, VALIDATION_SAMPLE_SIZE, "predictions.csv");

    return 0;
}