#pragma once
#include "vae_buffers.cuh"


namespace vae {

    namespace naive {

        void backward(VAEBuffers& buf, VAEGradients& grads);

    }
    
}
