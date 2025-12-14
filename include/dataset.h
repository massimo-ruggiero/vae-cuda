#pragma once
#include <cstdint>

struct Dataset {
    int N;
    int D;
    float *data;
};

Dataset load_dataset(const char *filename);
void free_dataset(Dataset &ds);