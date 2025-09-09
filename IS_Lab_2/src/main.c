#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
    #define M_PI 3.14159265
#endif

#define USE_2D_MODEL 1

#ifdef USE_2D_MODEL
    #define TRAINING_SAMPLE_SIZE   20
    #define VALIDATION_SAMPLE_SIZE 200
#else
    #define TRAINING_SAMPLE_SIZE   20
    #define VALIDATION_SAMPLE_SIZE 10
#endif

#define TRAINING_EPOCH_LIMIT 10000

typedef struct {
    double x1;
    double x2;
} Sample2D;

typedef struct {
    struct {
        /* Weights */
        double w11_1;
        double w21_1;
        double w31_1;
        double w41_1;
        double w51_1;
        /* Weights for 2D approx. model */
        double w12_1;
        double w22_1;
        double w32_1;
        double w42_1;
        double w52_1;
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
static double targetFunction1D(double x)
{
    return (1.0 + 0.6 * sin(2.0 * M_PI * x / 0.7) + 0.3 * sin(2.0 * M_PI * x)) / 2.0;
}

/* y = (1 + 0.6*sin(2*pi*x1/0.7 + x2)) + 0.3*sin(2*pi*x2)/2; */
static double targetFunction2D(double x1, double x2)
{
    return (1.0 + 0.6 * sin(2.0 * M_PI * x1 / 0.7 + x2) + 0.3 * sin(2.0 * M_PI * x2)) / 2.0;
}

static void saveCheckpoint(NetworkLayers currNetwork)
{
    bestNetwork.FirstLayer.w11_1 = currNetwork.FirstLayer.w11_1;
    bestNetwork.FirstLayer.w21_1 = currNetwork.FirstLayer.w21_1;
    bestNetwork.FirstLayer.w31_1 = currNetwork.FirstLayer.w31_1;
    bestNetwork.FirstLayer.w41_1 = currNetwork.FirstLayer.w41_1;
    bestNetwork.FirstLayer.w51_1 = currNetwork.FirstLayer.w51_1;

    bestNetwork.FirstLayer.w12_1 = currNetwork.FirstLayer.w12_1;
    bestNetwork.FirstLayer.w22_1 = currNetwork.FirstLayer.w22_1;
    bestNetwork.FirstLayer.w32_1 = currNetwork.FirstLayer.w32_1;
    bestNetwork.FirstLayer.w42_1 = currNetwork.FirstLayer.w42_1;
    bestNetwork.FirstLayer.w52_1 = currNetwork.FirstLayer.w52_1;

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

    currNetwork->FirstLayer.w12_1 = bestNetwork.FirstLayer.w12_1;
    currNetwork->FirstLayer.w22_1 = bestNetwork.FirstLayer.w22_1;
    currNetwork->FirstLayer.w32_1 = bestNetwork.FirstLayer.w32_1;
    currNetwork->FirstLayer.w42_1 = bestNetwork.FirstLayer.w42_1;
    currNetwork->FirstLayer.w52_1 = bestNetwork.FirstLayer.w52_1;

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

    for(uint32_t i = 0; i < data_size; i++) {
        fprintf(f, "%.5f\n", data[i]);
    }

    fclose(f);
}

static void save2DDataToCSV(Sample2D *data, size_t data_size, char *filename)
{
    FILE *f = fopen(filename, "w");
    if(f == NULL) {
        fprintf(stderr, "Failed to open %s for writing\n", filename);
        return;
    }

    for(uint32_t i = 0; i < data_size; i++) {
        fprintf(f, "%.5f,%.5f\n", data[i].x1, data[i].x2);
    }

    fclose(f);
}

static void approximationModel1D(void)
{
    double train_data[TRAINING_SAMPLE_SIZE] = { 0.0 };
    double valid_data[VALIDATION_SAMPLE_SIZE] = { 0.0 };
    for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        train_data[i] = (1.0 / TRAINING_SAMPLE_SIZE) * i;
    }
    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        valid_data[i] = (double)rand() / (double)RAND_MAX;
    }
    saveDataToCSV(train_data, TRAINING_SAMPLE_SIZE, "train_data_1D.csv");
    saveDataToCSV(valid_data, VALIDATION_SAMPLE_SIZE, "valid_data_1D.csv");

    double train_data_y[TRAINING_SAMPLE_SIZE] = { 0.0 };
    double valid_data_y[VALIDATION_SAMPLE_SIZE] = { 0.0 };
    for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
        train_data_y[i] = targetFunction1D(train_data[i]);
    }
    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        valid_data_y[i] = targetFunction1D(valid_data[i]);
    }
    saveDataToCSV(train_data_y, TRAINING_SAMPLE_SIZE, "train_expected_1D.csv");
    saveDataToCSV(valid_data_y, VALIDATION_SAMPLE_SIZE, "expected_1D.csv");

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

    saveDataToCSV(predictions, VALIDATION_SAMPLE_SIZE, "predictions_1D.csv");
}

static void approximationModel2D(void)
{
    Sample2D train_data[TRAINING_SAMPLE_SIZE * TRAINING_SAMPLE_SIZE] = { { 0.0 } };
    Sample2D valid_data[VALIDATION_SAMPLE_SIZE] = { { 0.0 } };
    double train_data_y[TRAINING_SAMPLE_SIZE * TRAINING_SAMPLE_SIZE] = { 0.0 };
    double valid_data_y[VALIDATION_SAMPLE_SIZE] = { 0.0 };

    for(uint32_t i = 0; i < TRAINING_SAMPLE_SIZE; ++i) {
        double x1 = (double)i / (double)(TRAINING_SAMPLE_SIZE - 1);
        for(uint32_t j = 0; j < TRAINING_SAMPLE_SIZE; ++j) {
            double x2 = (double)j / (double)(TRAINING_SAMPLE_SIZE - 1);
            size_t k = (size_t)i * (size_t)TRAINING_SAMPLE_SIZE + (size_t)j;
            train_data[k].x1 = x1;
            train_data[k].x2 = x2;
            train_data_y[k] = targetFunction2D(x1, x2);
        }
    }

    for(uint32_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        valid_data[i].x1 = (double)rand() / (double)RAND_MAX;
        valid_data[i].x2 = (double)rand() / (double)RAND_MAX;
        valid_data_y[i] = targetFunction2D(valid_data[i].x1, valid_data[i].x2);
    }

    save2DDataToCSV(train_data, TRAINING_SAMPLE_SIZE * TRAINING_SAMPLE_SIZE, "train_data_2D.csv");
    save2DDataToCSV(valid_data, VALIDATION_SAMPLE_SIZE, "valid_data_2D.csv");
    saveDataToCSV(train_data_y, TRAINING_SAMPLE_SIZE * TRAINING_SAMPLE_SIZE, "train_expected_2D.csv");
    saveDataToCSV(valid_data_y, VALIDATION_SAMPLE_SIZE, "expected_2D.csv");

    NetworkLayers network = {
        .FirstLayer.w11_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w21_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w31_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w41_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w51_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w12_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w22_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w32_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w42_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .FirstLayer.w52_1 = (double)rand() / RAND_MAX * 2.0 - 1.0,
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
        for(uint32_t i = 0; i < TRAINING_SAMPLE_SIZE * TRAINING_SAMPLE_SIZE; i++) {
            double v1_1 = network.FirstLayer.w11_1 * train_data[i].x1 + network.FirstLayer.w12_1 * train_data[i].x2 +
                          network.FirstLayer.b1_1;
            double v2_1 = network.FirstLayer.w21_1 * train_data[i].x1 + network.FirstLayer.w22_1 * train_data[i].x2 +
                          network.FirstLayer.b2_1;
            double v3_1 = network.FirstLayer.w31_1 * train_data[i].x1 + network.FirstLayer.w32_1 * train_data[i].x2 +
                          network.FirstLayer.b3_1;
            double v4_1 = network.FirstLayer.w41_1 * train_data[i].x1 + network.FirstLayer.w42_1 * train_data[i].x2 +
                          network.FirstLayer.b4_1;
            double v5_1 = network.FirstLayer.w51_1 * train_data[i].x1 + network.FirstLayer.w52_1 * train_data[i].x2 +
                          network.FirstLayer.b5_1;

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

            network.FirstLayer.w11_1 += learning_rate * grad11_1 * train_data[i].x1;
            network.FirstLayer.w21_1 += learning_rate * grad21_1 * train_data[i].x1;
            network.FirstLayer.w31_1 += learning_rate * grad31_1 * train_data[i].x1;
            network.FirstLayer.w41_1 += learning_rate * grad41_1 * train_data[i].x1;
            network.FirstLayer.w51_1 += learning_rate * grad51_1 * train_data[i].x1;

            network.FirstLayer.w12_1 += learning_rate * grad11_1 * train_data[i].x2;
            network.FirstLayer.w22_1 += learning_rate * grad21_1 * train_data[i].x2;
            network.FirstLayer.w32_1 += learning_rate * grad31_1 * train_data[i].x2;
            network.FirstLayer.w42_1 += learning_rate * grad41_1 * train_data[i].x2;
            network.FirstLayer.w52_1 += learning_rate * grad51_1 * train_data[i].x2;

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

        for(uint32_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
            double v1_1 = network.FirstLayer.w11_1 * valid_data[i].x1 + network.FirstLayer.w12_1 * valid_data[i].x2 +
                          network.FirstLayer.b1_1;
            double v2_1 = network.FirstLayer.w21_1 * valid_data[i].x1 + network.FirstLayer.w22_1 * valid_data[i].x2 +
                          network.FirstLayer.b2_1;
            double v3_1 = network.FirstLayer.w31_1 * valid_data[i].x1 + network.FirstLayer.w32_1 * valid_data[i].x2 +
                          network.FirstLayer.b3_1;
            double v4_1 = network.FirstLayer.w41_1 * valid_data[i].x1 + network.FirstLayer.w42_1 * valid_data[i].x2 +
                          network.FirstLayer.b4_1;
            double v5_1 = network.FirstLayer.w51_1 * valid_data[i].x1 + network.FirstLayer.w52_1 * valid_data[i].x2 +
                          network.FirstLayer.b5_1;

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

    for(uint32_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        double v1_1 = network.FirstLayer.w11_1 * valid_data[i].x1 + network.FirstLayer.w12_1 * valid_data[i].x2 +
                      network.FirstLayer.b1_1;
        double v2_1 = network.FirstLayer.w21_1 * valid_data[i].x1 + network.FirstLayer.w22_1 * valid_data[i].x2 +
                      network.FirstLayer.b2_1;
        double v3_1 = network.FirstLayer.w31_1 * valid_data[i].x1 + network.FirstLayer.w32_1 * valid_data[i].x2 +
                      network.FirstLayer.b3_1;
        double v4_1 = network.FirstLayer.w41_1 * valid_data[i].x1 + network.FirstLayer.w42_1 * valid_data[i].x2 +
                      network.FirstLayer.b4_1;
        double v5_1 = network.FirstLayer.w51_1 * valid_data[i].x1 + network.FirstLayer.w52_1 * valid_data[i].x2 +
                      network.FirstLayer.b5_1;

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

    saveDataToCSV(predictions, VALIDATION_SAMPLE_SIZE, "predictions_2D.csv");
}

int main(void)
{
    srand(time(NULL));

    if(USE_2D_MODEL) {
        /* Additional task - approximation model (2D) */
        approximationModel2D();
    } else {
        /* Main task - approximation model (1D) */
        approximationModel1D();
    }

    return 0;
}
