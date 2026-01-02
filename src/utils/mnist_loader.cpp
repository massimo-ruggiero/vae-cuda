#include "mnist_loader.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <random>

MNISTLoader::MNISTLoader(const char* filename)
    : images_(nullptr),
      labels_(nullptr),
      indices_(nullptr),
      num_samples_(0),
      feature_dim_(784),
      current_idx_(0) {

    FILE* f = std::fopen(filename, "rb");
    if (!f) {
        std::cerr << "[Loader] ERROR: cannot open " << filename << "\n";
        std::exit(EXIT_FAILURE);
    }

    if (std::fread(&num_samples_, sizeof(int), 1, f) != 1) {
        std::cerr << "[Loader] ERROR: cannot read n_samples from " << filename << "\n";
        std::fclose(f);
        std::exit(EXIT_FAILURE);
    }

    std::cout << "[Loader] ⚙️ Loading " << num_samples_
              << " images (float, dim = " << feature_dim_ << ") from "
              << filename << "...\n";

    images_ = new float[num_samples_ * feature_dim_];
    labels_ = new unsigned char[num_samples_];
    indices_ = new int[num_samples_];

    unsigned char* temp_buffer = new unsigned char[num_samples_ * feature_dim_];

    std::fread(temp_buffer, sizeof(unsigned char), num_samples_ * feature_dim_, f);

    for (int i = 0; i < num_samples_ * feature_dim_; ++i) {
        images_[i] = static_cast<float>(temp_buffer[i]) / 255.0f;
    }

    delete[] temp_buffer;

    std::fread(labels_, sizeof(unsigned char), num_samples_, f);

    std::fclose(f);

    for (int i = 0; i < num_samples_; ++i) indices_[i] = i;

    std::cout << "[Loader] ✅ Loaded " << num_samples_
              << " samples (feature_dim = " << feature_dim_ << ")\n";
}

MNISTLoader::~MNISTLoader() {
    if (images_) delete[] images_;
    if (labels_) delete[] labels_;
    if (indices_) delete[] indices_;
}

void MNISTLoader::shuffle() {
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices_, indices_ + num_samples_, g);
    current_idx_ = 0;
}

bool MNISTLoader::next_batch(float* batch_dst, int batch_size) {
    if (current_idx_ + batch_size > num_samples_) return false;

    for (int i = 0; i < batch_size; ++i) {
        int real_idx = indices_[current_idx_ + i];

        std::memcpy(batch_dst + (i * feature_dim_),
                    images_ + (real_idx * feature_dim_),
                    feature_dim_ * sizeof(float));
    }

    current_idx_ += batch_size;
    return true;
}
