#include "optimizers/adam.cuh"
#include "optimizers.cuh"
#include <cstdio>


Adam::Adam(const VAEConfig& config, float lr, float beta1, float beta2, float epsilon) 
    : lr_(lr), 
      beta1_(beta1), 
      beta2_(beta2), 
      epsilon_(epsilon), 
      timestep_(0) {
      
    state_ = new VAEAdamState(config);
}

Adam::~Adam() {
    if (state_) {
        delete state_;
        state_ = nullptr;
    }
}

void Adam::apply_layer(LinearLayer& l, LinearGrads& g, AdamState& s) {
    // update W
    optimizers::adam::step(
        g.dW.ptr, 
        l.W.ptr, 
        s.m_W.ptr, 
        s.v_W.ptr, 
        timestep_, 
        l.input_dim * l.output_dim, 
        state_.config.strategy,
        lr_, 
        beta1_, 
        beta2_, 
        epsilon_
    );

    // update b
    optimizers::adam::step(
        g.db.ptr, 
        l.b.ptr, 
        s.m_b.ptr, 
        s.v_b.ptr, 
        timestep_, 
        l.output_dim, 
        state_.config.strategy,
        lr_, 
        beta1_, 
        beta2_, 
        epsilon_
    );
}

void Adam::step(VAEBuffers& params, VAEGradients& grads) {
    timestep_++;
    
    apply_layer(params.enc1, grads.enc1, state_->enc1);
    apply_layer(params.enc2_mu, grads.enc2_mu, state_->enc2_mu);
    apply_layer(params.enc2_logvar, grads.enc2_logvar, state_->enc2_logvar);
    apply_layer(params.dec1, grads.dec1, state_->dec1);
    apply_layer(params.dec2, grads.dec2, state_->dec2);
}
