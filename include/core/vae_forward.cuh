#pragma once
#include "vae_buffers.cuh"


namespace vae {

    void encoder_pass(VAEBuffers& buf);

    void decoder_pass(VAEBuffers& buf);
    
    void forward(VAEBuffers& buf);

}
