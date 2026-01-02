#pragma once

#include <cstddef>

class MNISTLoader {
private:
    float* images_;
    unsigned char* labels_;
    int* indices_;

    int num_samples_;
    int feature_dim_;
    int current_idx_;

public:
    explicit MNISTLoader(const char* filename);
    ~MNISTLoader();

    void shuffle();
    bool next_batch(float* batch_dst, int batch_size);

    int num_samples() const { return num_samples_; }
    int feature_dim() const { return feature_dim_; }
};
