#include "vae_forward.cuh"
#include "vae_config.cuh"
#include "linear.cuh"
#include "activations.cuh"
#include "reparametrization.cuh"
#include "fused.cuh"


namespace vae {

    void encoder_pass(VAEBuffers& buf){
        int batch_size = buf.config.batch_size;
        int input_dim = buf.config.input_dim;
        int hidden_dim = buf.config.hidden_dim;
        int latent_dim = buf.config.latent_dim;
        VAEStrategy strategy = buf.config.strategy;

        if (strategy == VAEStrategy::KERNEL_FUSION) {
            fused::forward::linear_lrelu_tc(buf.d_X.ptr,
                                            buf.enc1.W.ptr,
                                            buf.enc1.b.ptr,
                                            buf.enc1.A.ptr,
                                            batch_size,
                                            input_dim,
                                            hidden_dim,
                                            0.2f);

            fused::forward::linear(buf.enc1.A.ptr,
                                   buf.enc2_mu.W.ptr,
                                   buf.enc2_mu.b.ptr,
                                   buf.enc2_mu.Z.ptr, // mu
                                   batch_size,
                                   hidden_dim,
                                   latent_dim);

            fused::forward::linear(buf.enc1.A.ptr,
                                   buf.enc2_logvar.W.ptr,
                                   buf.enc2_logvar.b.ptr,
                                   buf.enc2_logvar.Z.ptr, // logvar
                                   batch_size,
                                   hidden_dim,
                                   latent_dim);
        } else {
            linear::forward(buf.d_X.ptr,
                            buf.enc1.W.ptr,
                            buf.enc1.b.ptr,
                            buf.enc1.Z.ptr,
                            batch_size,
                            input_dim,
                            hidden_dim,
                            strategy);
            
            activations::leaky_relu::forward(buf.enc1.Z.ptr,
                                                buf.enc1.A.ptr,
                                                0.2f,
                                                batch_size * hidden_dim,
                                                strategy);

            linear::forward(buf.enc1.A.ptr,
                        buf.enc2_mu.W.ptr,
                        buf.enc2_mu.b.ptr,
                        buf.enc2_mu.Z.ptr, // mu
                        batch_size,
                        hidden_dim,
                        latent_dim,
                        strategy);

            linear::forward(buf.enc1.A.ptr,
                            buf.enc2_logvar.W.ptr,
                            buf.enc2_logvar.b.ptr,
                            buf.enc2_logvar.Z.ptr, // logvar
                            batch_size,
                            hidden_dim,
                            latent_dim,
                            strategy);
        }
    }

    void decoder_pass(VAEBuffers& buf){
        int batch_size = buf.config.batch_size;
        int input_dim = buf.config.input_dim;
        int hidden_dim = buf.config.hidden_dim;
        int latent_dim = buf.config.latent_dim;
        VAEStrategy strategy = buf.config.strategy;
        
        if (strategy == VAEStrategy::KERNEL_FUSION) {
            fused::forward::linear_lrelu_tc(buf.d_z.ptr,
                                            buf.dec1.W.ptr,
                                            buf.dec1.b.ptr,
                                            buf.dec1.A.ptr,
                                            batch_size,
                                            latent_dim,
                                            hidden_dim,
                                            0.2f);

            fused::forward::linear_sigmoid_tc(buf.dec1.A.ptr,
                                              buf.dec2.W.ptr,
                                              buf.dec2.b.ptr,
                                              buf.d_X_hat.ptr,
                                              batch_size,
                                              hidden_dim,
                                              input_dim);
        } else {
            linear::forward(buf.d_z.ptr,
                            buf.dec1.W.ptr,
                            buf.dec1.b.ptr,
                            buf.dec1.Z.ptr,
                            batch_size,
                            latent_dim,
                            hidden_dim,
                            strategy);
            
            activations::leaky_relu::forward(buf.dec1.Z.ptr,
                                                buf.dec1.A.ptr,
                                                0.2f,
                                                batch_size * hidden_dim,
                                                strategy);

            linear::forward(buf.dec1.A.ptr,
                            buf.dec2.W.ptr,
                            buf.dec2.b.ptr,
                            buf.dec2.Z.ptr,
                            batch_size,
                            hidden_dim,
                            input_dim,
                            strategy);

            activations::sigmoid::forward(buf.dec2.Z.ptr,
                                            buf.d_X_hat.ptr,
                                            batch_size * input_dim,
                                            strategy);
        }
    }

    void forward(VAEBuffers& buf) {
        int batch_size = buf.config.batch_size;
        int latent_dim = buf.config.latent_dim;
        VAEStrategy strategy = buf.config.strategy;

        // Encoder
        encoder_pass(buf);
        
        // Reparametrization trick
        reparametrization::forward(buf.enc2_mu.Z.ptr,         // mu
                                    buf.enc2_logvar.Z.ptr,    // logvar
                                    buf.d_z.ptr,
                                    buf.d_epsilon.ptr,
                                    buf.d_states,
                                    batch_size * latent_dim,
                                    strategy);

        // Decoder
        decoder_pass(buf);
    }

} // namespace vae 
