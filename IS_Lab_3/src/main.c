#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifndef M_PI
    #define M_PI 3.14159265
#endif

#define LEARN_GAUSSIAN_CONSTANTS 1

#define TRAINING_SAMPLE_SIZE   22
#define VALIDATION_SAMPLE_SIZE 50

#define TRAINING_EPOCH_LIMIT 10000

typedef struct {
    double c; /* Center */
    double r; /* Radius */
} GaussianRadialBasisConstants;

typedef struct {
    struct {
        /* Weights */
        double w11_2;
        double w12_2;
        /* Biases */
        double b1_2;
    } SecondLayer;
} NetworkLayers;

static NetworkLayers bestNetwork = { { 0 } };

/* F = exp(-(x-c)^2/(2*r^2)). */
static inline double gaussianRadialBasis(double x, GaussianRadialBasisConstants rbfConstants)
{
    double exponent = -(x - rbfConstants.c) * (x - rbfConstants.c) / (2 * rbfConstants.r * rbfConstants.r);
    return exp(exponent);
}

/* Linear interpolation of x where segment (x1,y1)-(x2,y2) crosses level h. */
static inline double interp_cross(double x1, double y1, double x2, double y2, double h)
{
    /* Assume y1 != y2 and h is between y1 and y2. */
    return x1 + (h - y1) * (x2 - x1) / (y2 - y1);
}

/* Estimate Gaussian radius r around a peak at index ip using FWHM. */
static int estimate_radius_fwhm(const double *x, const double *y, size_t n, size_t ip, double *out_r)
{
    if(ip == 0 || ip + 1 >= n)
        return -1; /* need neighbors */

    const double A = y[ip];
    const double h = A * 0.5; /* half-maximum */

    /* Find left crossing of level h */
    int found_left = 0, found_right = 0;
    double xL = x[ip], xR = x[ip];

    for(size_t i = ip; i-- > 0;) {
        double y1 = y[i], y2 = y[i + 1];
        if((y1 - h) * (y2 - h) <= 0.0 && y1 != y2) {
            xL = interp_cross(x[i], y1, x[i + 1], y2, h);
            found_left = 1;
            break;
        }
        if(i == 0)
            break;
    }

    /* Find right crossing of level h */
    for(size_t i = ip; i + 1 < n; ++i) {
        double y1 = y[i], y2 = y[i + 1];
        if((y1 - h) * (y2 - h) <= 0.0 && y1 != y2) {
            xR = interp_cross(x[i], y1, x[i + 1], y2, h);
            found_right = 1;
            break;
        }
    }

    if(found_left && found_right) {
        double fwhm = xR - xL;
        if(fwhm <= 0.0)
            return -1;
        *out_r = fwhm / (2.0 * sqrt(2.0 * log(2.0)));
        return 0;
    }

    /* Fallback: curvature-based estimate around the peak (requires roughly uniform spacing). */
    double dx1 = x[ip] - x[ip - 1];
    double dx2 = x[ip + 1] - x[ip];
    if(dx1 <= 0 || dx2 <= 0)
        return -1;
    double dx = 0.5 * (dx1 + dx2);
    double ypp = (y[ip + 1] - 2.0 * y[ip] + y[ip - 1]) / (dx * dx);
    if(ypp >= 0.0)
        return -1; /* not concave */
    *out_r = sqrt(-A / ypp);
    return 0;
}

/* y = (1 + 0.6 * sin (2 * pi * x / 0.7)) + 0.3 * sin (2 * pi * x)) / 2; */
static double targetFunction1D(double x)
{
    return (1.0 + 0.6 * sin(2.0 * M_PI * x / 0.7) + 0.3 * sin(2.0 * M_PI * x)) / 2.0;
}

static void saveCheckpoint(NetworkLayers currNetwork)
{
    bestNetwork.SecondLayer.w11_2 = currNetwork.SecondLayer.w11_2;
    bestNetwork.SecondLayer.w12_2 = currNetwork.SecondLayer.w12_2;

    bestNetwork.SecondLayer.b1_2 = currNetwork.SecondLayer.b1_2;
}

static void applyCheckpoint(NetworkLayers *currNetwork)
{
    currNetwork->SecondLayer.w11_2 = bestNetwork.SecondLayer.w11_2;
    currNetwork->SecondLayer.w12_2 = bestNetwork.SecondLayer.w12_2;

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

static void gaussianRadialBasisNetwork(void)
{
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
        train_data_y[i] = targetFunction1D(train_data[i]);
    }
    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        valid_data_y[i] = targetFunction1D(valid_data[i]);
    }
    saveDataToCSV(train_data_y, TRAINING_SAMPLE_SIZE, "train_expected.csv");
    saveDataToCSV(valid_data_y, VALIDATION_SAMPLE_SIZE, "expected.csv");

    NetworkLayers network = {
        .SecondLayer.w11_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.w12_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.b1_2 = 0.5,
    };

    GaussianRadialBasisConstants rbfConstants1 = {
        .c = 0.19,
        .r = 0.18,
    };
    GaussianRadialBasisConstants rbfConstants2 = {
        .c = 0.91,
        .r = 0.18,
    };

    double learning_rate = 0.03;
    double best_cost = INT32_MAX;
    int epochs = 0;

    while(epochs++ < TRAINING_EPOCH_LIMIT) {
        double cost_accum = 0;
        for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
            double v1_1 = gaussianRadialBasis(train_data[i], rbfConstants1);
            double v2_1 = gaussianRadialBasis(train_data[i], rbfConstants2);

            double v1_2 =
                    network.SecondLayer.w11_2 * v1_1 + network.SecondLayer.w12_2 * v2_1 + network.SecondLayer.b1_2;

            double error = train_data_y[i] - v1_2;

            double grad1_2 = error;

            network.SecondLayer.w11_2 += learning_rate * grad1_2 * v1_1;
            network.SecondLayer.w12_2 += learning_rate * grad1_2 * v2_1;

            network.SecondLayer.b1_2 += learning_rate * grad1_2;
        }

        for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
            double v1_1 = gaussianRadialBasis(valid_data[i], rbfConstants1);
            double v2_1 = gaussianRadialBasis(valid_data[i], rbfConstants2);

            double v1_2 =
                    network.SecondLayer.w11_2 * v1_1 + network.SecondLayer.w12_2 * v2_1 + network.SecondLayer.b1_2;

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
        double v1_1 = gaussianRadialBasis(valid_data[i], rbfConstants1);
        double v2_1 = gaussianRadialBasis(valid_data[i], rbfConstants2);

        double v1_2 = network.SecondLayer.w11_2 * v1_1 + network.SecondLayer.w12_2 * v2_1 + network.SecondLayer.b1_2;

        predictions[i] = v1_2;
    }

    saveDataToCSV(predictions, VALIDATION_SAMPLE_SIZE, "predictions.csv");
}

static void gaussianRadialBasisNetworkDynamic(void)
{
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
        train_data_y[i] = targetFunction1D(train_data[i]);
    }
    for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
        valid_data_y[i] = targetFunction1D(valid_data[i]);
    }
    saveDataToCSV(train_data_y, TRAINING_SAMPLE_SIZE, "train_expected.csv");
    saveDataToCSV(valid_data_y, VALIDATION_SAMPLE_SIZE, "expected.csv");

    NetworkLayers network = {
        .SecondLayer.w11_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.w12_2 = (double)rand() / RAND_MAX * 2.0 - 1.0,
        .SecondLayer.b1_2 = 0.5,
    };

    GaussianRadialBasisConstants rbfConstants1 = {
        .c = 0.0, // 0.19,
        .r = 0.0, // 0.18,
    };
    GaussianRadialBasisConstants rbfConstants2 = {
        .c = 0.0, // 0.91,
        .r = 0.0, // 0.18,
    };

    for(uint8_t i = 1; i < TRAINING_SAMPLE_SIZE - 1; i++) {
        if(train_data_y[i] > train_data_y[i + 1] && train_data_y[i] > train_data_y[i - 1]) {
            if(rbfConstants1.c == 0.0) {
                rbfConstants1.c = train_data[i];
                double r_est = 0.0;
                if(estimate_radius_fwhm(train_data, train_data_y, TRAINING_SAMPLE_SIZE, i, &r_est) == 0 &&
                   r_est > 0.0) {
                    rbfConstants1.r = r_est;
                } else {
                    rbfConstants1.r = 0.1; /* conservative default */
                }

            } else if(rbfConstants2.c == 0.0) {
                rbfConstants2.c = train_data[i];
                double r_est = 0.0;
                if(estimate_radius_fwhm(train_data, train_data_y, TRAINING_SAMPLE_SIZE, i, &r_est) == 0 &&
                   r_est > 0.0) {
                    rbfConstants2.r = r_est;
                } else {
                    rbfConstants2.r = 0.1;
                }
            }
        }
    }

    printf("RBF1: c=%.5f r=%.5f\n", rbfConstants1.c, rbfConstants1.r);
    printf("RBF2: c=%.5f r=%.5f\n", rbfConstants2.c, rbfConstants2.r);

    double learning_rate = 0.03;
    double best_cost = INT32_MAX;
    int epochs = 0;

    while(epochs++ < TRAINING_EPOCH_LIMIT) {
        double cost_accum = 0;
        for(uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
            double v1_1 = gaussianRadialBasis(train_data[i], rbfConstants1);
            double v2_1 = gaussianRadialBasis(train_data[i], rbfConstants2);

            double v1_2 =
                    network.SecondLayer.w11_2 * v1_1 + network.SecondLayer.w12_2 * v2_1 + network.SecondLayer.b1_2;

            double error = train_data_y[i] - v1_2;

            double grad1_2 = error;

            network.SecondLayer.w11_2 += learning_rate * grad1_2 * v1_1;
            network.SecondLayer.w12_2 += learning_rate * grad1_2 * v2_1;

            network.SecondLayer.b1_2 += learning_rate * grad1_2;
        }

        for(uint8_t i = 0; i < VALIDATION_SAMPLE_SIZE; i++) {
            double v1_1 = gaussianRadialBasis(valid_data[i], rbfConstants1);
            double v2_1 = gaussianRadialBasis(valid_data[i], rbfConstants2);

            double v1_2 =
                    network.SecondLayer.w11_2 * v1_1 + network.SecondLayer.w12_2 * v2_1 + network.SecondLayer.b1_2;

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
        double v1_1 = gaussianRadialBasis(valid_data[i], rbfConstants1);
        double v2_1 = gaussianRadialBasis(valid_data[i], rbfConstants2);

        double v1_2 = network.SecondLayer.w11_2 * v1_1 + network.SecondLayer.w12_2 * v2_1 + network.SecondLayer.b1_2;

        predictions[i] = v1_2;
    }

    saveDataToCSV(predictions, VALIDATION_SAMPLE_SIZE, "predictions.csv");
}

int main(void)
{
    srand(time(NULL));

    if(LEARN_GAUSSIAN_CONSTANTS) {
        /* Additional task */
        gaussianRadialBasisNetworkDynamic();
    } else {
        /* Main task */
        gaussianRadialBasisNetwork();
    }

    return 0;
}
