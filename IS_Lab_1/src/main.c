#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define TRAINING_SAMPLE_SIZE 6
#define VALIDATION_SAMPLE_SIZE 7
#define TRAINING_EPOCH_LIMIT 1000

typedef struct {
  double x1;
  double x2;
  int label; // 1 or -1
} SampleData;

static int read_dataset(const char *path, SampleData *out_items) {

  FILE *f = fopen(path, "r");
  if (!f) {
    perror("Couldn't open file\n");
    return -1;
  }

  size_t len = 0;

  char line[256];
  while (fgets(line, sizeof(line), f)) {
    double x1, x2;
    int y;

    int n = sscanf(line, " %lf,%lf,%d", &x1, &x2, &y);
    if (n < 3) {
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

int main(void) {
  const char *train_path = "datasets/training_data.txt";
  const char *valid_path = "datasets/validation_data.txt";

  SampleData train[TRAINING_SAMPLE_SIZE] = {0};
  SampleData valid[VALIDATION_SAMPLE_SIZE] = {0};

  if (read_dataset(train_path, train) != 0) {
    fprintf(stderr, "Failed to read training data from %s\n", train_path);
    return 1;
  }
  if (read_dataset(valid_path, valid) != 0) {
    fprintf(stderr, "Failed to read validation data from %s\n", valid_path);
    return 1;
  }

  srand(time(NULL));
  double w1 = (double)rand() / RAND_MAX * 2.0 - 1.0;
  double w2 = (double)rand() / RAND_MAX * 2.0 - 1.0;
  double b = (double)rand() / RAND_MAX * 2.0 - 1.0;

  printf("Initial weights: w1=%.5f, w2=%.5f, b=%.5f\n", w1, w2, b);

  double learning_rate = 0.15;
  int epochs = 0;

  while (1) {
    int err_accum = 0;
    for (uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
      double v = w1 * train[i].x1 + w2 * train[i].x2 + b;
      int y = (v > 0) ? 1 : -1;

      w1 += learning_rate * (train[i].label - y) * train[i].x1;
      w2 += learning_rate * (train[i].label - y) * train[i].x2;
      b += learning_rate * (train[i].label - y);

      err_accum += abs(train[i].label - y);
    }

    if (err_accum == 0 || epochs++ >= TRAINING_EPOCH_LIMIT) {
      break;
    }
  }
  printf("Training completed in %d epochs, weights: w1=%.5f, w2=%.5f, b=%.5f\n",
         epochs, w1, w2, b);

  for (uint8_t i = 0; i < TRAINING_SAMPLE_SIZE; i++) {
    double v = w1 * train[i].x1 + w2 * train[i].x2 + b;
    int y = (v > 0) ? 1 : -1;

    if (y != train[i].label) {
      printf("Training sample %d misclassified: expected %d, got %d\n", i,
             train[i].label, y);
    } else {
      printf("Training sample %d classified correctly: expected %d, got %d\n",
             i, train[i].label, y);
    }
  }

  return 0;
}
