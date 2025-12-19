#include "vae_backward.cuh"
#include "linear.cuh"
#include "activations.cuh"
#include "reparametrization.cuh"
#include "loss.cuh"


namespace vae {

    namespace naive {

        void backward(VAEBuffers& buf, VAEGradients& grads, float beta = 1.0f){
            int batch_size = buf.config.batch_size;
            int input_dim = buf.config.input_dim;
            int hidden_dim = buf.config.hidden_dim;
            int latent_dim = buf.config.latent_dim;

            // loss
            loss::backward::bce(buf.d_X.ptr,
                                buf.d_X_hat.ptr,
                                grads.dec2.dZ.ptr,
                                batch_size * input_dim);

            // decoder        
            linear::naive::backward(buf.dec1.A.ptr,
                                    buf.dec2.W.ptr,
                                    grads.dec2.dZ.ptr,
                                    grads.dec1.dA.ptr,
                                    grads.dec2.dW.ptr,
                                    grads.dec2.db.ptr,
                                    batch_size,
                                    hidden_dim,
                                    input_dim);
            
            activations::leaky_relu::backward(buf.dec1.Z.ptr,
                                            grads.dec1.dA.ptr,
                                            grads.dec1.dZ.ptr,
                                            batch_size * hidden_dim);

            linear::naive::backward(buf.d_z.ptr,
                                    buf.dec1.W.ptr,
                                    grads.dec1.dZ.ptr,
                                    grads.dec1.dA.ptr,
                                    grads.dec1.dW.ptr,
                                    grads.dec1.db.ptr,
                                    batch_size,
                                    latent_dim,
                                    hidden_dim);

            // reparametrization trick
            reparametrization::backward(grads.d_dz.ptr,
                                        buf.d_logvar,
                                        buf.d_epsilon.ptr,
                                        grads.d_dmu,
                                        grads.d_dlogvar,
                                        batch_size * latent_dim);
            
            loss::backward::kl(buf.d_mu,
                            buf.d_logvar,
                            grads.d_dmu,
                            grads.d_dlogvar,
                            batch_size * latent_dim,
                            beta);

            // encoder
            linear::naive::backward(buf.enc1.A.ptr,
                                    buf.enc2.W.ptr,
                                    grads.enc2.dZ.ptr,
                                    grads.enc1.dA.ptr,
                                    grads.enc2.dW.ptr,
                                    grads.enc2.db.ptr,
                                    batch_size,
                                    hidden_dim,
                                    latent_dim * 2);

            activations::leaky_relu::backward(buf.enc1.Z.ptr,
                                            grads.enc1.dA.ptr,
                                            grads.enc1.dZ.ptr,
                                            batch_size * hidden_dim);

            linear::naive::backward(buf.d_X.ptr,
                                    buf.enc1.W.ptr,
                                    grads.enc1.dZ.ptr,
                                    nullptr,
                                    grads.enc1.dW.ptr,
                                    grads.enc1.db.ptr,
                                    batch_size,
                                    input_dim,
                                    hidden_dim);

        }

    }

}