#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "dataset.h"
#include "trainer.cuh"
#include "utils.cuh"
#include "vae_buffers.cuh"

namespace {

void run_checkpoint_inference(const TrainingConfig& config,
                              const Dataset& dataset,
                              const std::string& checkpoint_path) {
    const int infer_batch = std::min(config.batch_size, dataset.N);
    if (infer_batch <= 0) {
        std::printf("No samples available for inference.\n");
        return;
    }

    VaeConfig vae_cfg = make_vae_config(config, dataset.D, false);
    vae_cfg.max_batch_size = config.batch_size;
    VaeModel model(vae_cfg);
    model.load(checkpoint_path);

    DeviceTensor d_input(static_cast<size_t>(infer_batch) * dataset.D);
    DeviceTensor d_output(static_cast<size_t>(infer_batch) * dataset.D);

    CUDA_CHECK(cudaMemcpy(d_input.data(),
                          dataset.data,
                          d_input.bytes(),
                          cudaMemcpyHostToDevice));

    model.reconstruct(d_input.data(), d_output.data(), infer_batch);

    std::vector<float> recon(static_cast<size_t>(infer_batch) * dataset.D);
    CUDA_CHECK(cudaMemcpy(recon.data(),
                          d_output.data(),
                          recon.size() * sizeof(float),
                          cudaMemcpyDeviceToHost));
    std::printf("Checkpoint inference completed for %d samples\n", infer_batch);
}

} // namespace

int main(int argc, char** argv) {
    const std::string dataset_path = (argc > 1) ? argv[1] : "data/train.bin";
    const std::string checkpoint_path = (argc > 2) ? argv[2] : "vae_checkpoint.bin";
    const std::string mode = (argc > 3) ? argv[3] : "train";

    Dataset dataset = load_dataset(dataset_path.c_str());

    TrainingConfig config;
    config.epochs = 10;
    config.batch_size = 128;
    config.latent_dim = 64;
    config.encoder_hidden_dims = {512, 256};
    config.decoder_hidden_dims = {256, 512};
    config.learning_rate = 1e-3f;
    config.beta = 1.0f;
    config.leaky_relu_alpha = 0.1f;
    config.log_every = 50;
    config.seed = 42ull;
    config.checkpoint_path = checkpoint_path;

    if (mode == "train") {
        Trainer trainer(config, dataset.D);
        trainer.fit(dataset);
        trainer.save_weights(checkpoint_path);

        const int preview_batch = std::min(config.batch_size, dataset.N);
        if (preview_batch > 0) {
            std::vector<float> recon(static_cast<size_t>(preview_batch) * dataset.D);
            trainer.infer(dataset.data, preview_batch, recon.data());
            std::printf("Preview inference completed for %d samples\n", preview_batch);
        }

        run_checkpoint_inference(config, dataset, checkpoint_path);
    } else if (mode == "infer") {
        run_checkpoint_inference(config, dataset, checkpoint_path);
    } else {
        std::fprintf(stderr, "Unknown mode '%s'. Use 'train' or 'infer'.\n", mode.c_str());
        free_dataset(dataset);
        return 1;
    }

    free_dataset(dataset);
    return 0;
}
