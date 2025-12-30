#include "dataset.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>

Dataset load_dataset(const char* filename) {
    FILE* fp = fopen(filename, "rb");
    if (!fp) {
        perror("Errore apertura file dataset");
        exit(EXIT_FAILURE);
    }

    int32_t N, D;
    size_t r;

    r = fread(&N, sizeof(int32_t), 1, fp);
    if (r != 1) {
        fprintf(stderr, "Errore lettura N\n");
        exit(EXIT_FAILURE);
    }

    r = fread(&D, sizeof(int32_t), 1, fp);
    if (r != 1) {
        fprintf(stderr, "Errore lettura D\n");
        exit(EXIT_FAILURE);
    }

    float* data = (float*)malloc((size_t)N * D * sizeof(float));
    if (!data) {
        fprintf(stderr, "Errore malloc dataset\n");
        exit(EXIT_FAILURE);
    }

    size_t to_read = (size_t)N * D;
    r = fread(data, sizeof(float), to_read, fp);
    if (r != to_read) {
        fprintf(stderr, "Errore lettura dati: letti %zu invece di %zu\n", r, to_read);
        exit(EXIT_FAILURE);
    }

    fclose(fp);

    Dataset ds;
    ds.N = (int)N;
    ds.D = (int)D;
    ds.data = data;
    return ds;
}

void free_dataset(Dataset& ds) {
    if (ds.data) {
        free(ds.data);
        ds.data = nullptr;
    }
}