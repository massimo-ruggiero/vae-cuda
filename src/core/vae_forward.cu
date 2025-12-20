#include "vae_forward.cuh"
#include "linear.cuh"
#include "activations.cuh"
#include "reparametrization.cuh"


namespace vae {

    namespace naive {

        void encoder_pass(VAEBuffers& buf){
            int batch_size = buf.config.batch_size;
            int input_dim = buf.config.input_dim;
            int hidden_dim = buf.config.hidden_dim;
            int latent_dim = buf.config.latent_dim;

            linear::naive::forward(buf.d_X.ptr,
                                   buf.enc1.W.ptr,
                                   buf.enc1.b.ptr,
                                   buf.enc1.Z.ptr,
                                   batch_size,
                                   input_dim,
                                   hidden_dim);
            
            activations::leaky_relu::forward(buf.enc1.Z.ptr,
                                             buf.enc1.A.ptr,
                                             batch_size * hidden_dim);

            linear::naive::forward(buf.enc1.A.ptr,
                                   buf.enc2_mu.W.ptr,
                                   buf.enc2_mu.b.ptr,
                                   buf.enc2_mu.Z.ptr, // mu
                                   batch_size,
                                   hidden_dim,
                                   latent_dim);

            linear::naive::forward(buf.enc1.A.ptr,
                                   buf.enc2_logvar.W.ptr,
                                   buf.enc2_logvar.b.ptr,
                                   buf.enc2_logvar.Z.ptr, // logvar
                                   batch_size,
                                   hidden_dim,
                                   latent_dim);
        }

        void decoder_pass(VAEBuffers& buf){
            int batch_size = buf.config.batch_size;
            int input_dim = buf.config.input_dim;
            int hidden_dim = buf.config.hidden_dim;
            int latent_dim = buf.config.latent_dim;
            
            linear::naive::forward(buf.d_z.ptr,
                                   buf.dec1.W.ptr,
                                   buf.dec1.b.ptr,
                                   buf.dec1.Z.ptr,
                                   batch_size,
                                   latent_dim,
                                   hidden_dim);
            
            activations::leaky_relu::forward(buf.dec1.Z.ptr,
                                             buf.dec1.A.ptr,
                                             batch_size * hidden_dim);

            linear::naive::forward(buf.dec1.A.ptr,
                                   buf.dec2.W.ptr,
                                   buf.dec2.b.ptr,
                                   buf.dec2.Z.ptr,
                                   batch_size,
                                   hidden_dim,
                                   input_dim);

            activations::sigmoid::forward(buf.dec2.Z.ptr,
                                          buf.d_X_hat.ptr,
                                          batch_size * input_dim);
        }

        void forward(VAEBuffers& buf) {
            int batch_size = buf.config.batch_size;
            int latent_dim = buf.config.latent_dim;

            // Encoder
            encoder_pass(buf);
            
            // Reparametrization trick
            reparametrization::forward(buf.enc2_mu.Z.ptr,         // mu
                                       buf.enc2_logvar.Z.ptr,     // logvar
                                       buf.d_z.ptr,
                                       buf.d_epsilon.ptr,
                                       buf.d_states,
                                       batch_size * latent_dim);

            // Decoder
            decoder_pass(buf);
        }

    }

}