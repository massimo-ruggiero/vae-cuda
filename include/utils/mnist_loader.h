#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstring> // per memcpy
#include <algorithm>
#include <random>
#include <vector>

class MNISTLoader {
private:
    float* images_;       
    unsigned char* labels_; 
    
    int* indices_; 
    int num_samples_;
    int feature_dim_; 
    int current_idx_;

public:
    MNISTLoader(const char* filename) 
        : images_(nullptr), labels_(nullptr), indices_(nullptr), 
          num_samples_(0), feature_dim_(784), current_idx_(0) 
    {
        FILE* f = fopen(filename, "rb");
        if (!f) {
            fprintf(stderr, "Error: cannot open %s\n", filename);
            exit(EXIT_FAILURE);
        }

        if (fread(&num_samples_, sizeof(int), 1, f) != 1) {
            fprintf(stderr, "Error reading n_samples\n");
            exit(EXIT_FAILURE);
        }

        printf("[Loader] Caricamento di %d immagini in formato FLOAT...\n", num_samples_);

        images_ = new float[num_samples_ * feature_dim_];
        labels_ = new unsigned char[num_samples_];
        indices_ = new int[num_samples_];

        unsigned char* temp_buffer = new unsigned char[num_samples_ * feature_dim_];
        
        fread(temp_buffer, sizeof(unsigned char), num_samples_ * feature_dim_, f);
        
        for (int i = 0; i < num_samples_ * feature_dim_; ++i) {
            images_[i] = (float)temp_buffer[i] / 255.0f;
        }

        delete[] temp_buffer;

        fread(labels_, sizeof(unsigned char), num_samples_, f);

        fclose(f);

        for(int i = 0; i < num_samples_; ++i) indices_[i] = i;
    }

    ~MNISTLoader() {
        if (images_) delete[] images_;
        if (labels_) delete[] labels_;
        if (indices_) delete[] indices_;
    }

    void shuffle() {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(indices_, indices_ + num_samples_, g);
        current_idx_ = 0;
    }

    bool next_batch(float* batch_dst, int batch_size) {
        if (current_idx_ + batch_size > num_samples_) return false;

        for (int i = 0; i < batch_size; ++i) {
            int real_idx = indices_[current_idx_ + i];
            
            memcpy(batch_dst + (i * feature_dim_), 
                   images_ + (real_idx * feature_dim_), 
                   feature_dim_ * sizeof(float));
        }

        current_idx_ += batch_size;
        return true;
    }
    
    int num_samples() const { return num_samples_; }
};