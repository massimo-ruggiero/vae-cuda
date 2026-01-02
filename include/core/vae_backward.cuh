#pragma once
#include "vae_buffers.cuh"


namespace vae {

    void backward(VAEBuffers& buf, VAEGradients& grads);
    
}
