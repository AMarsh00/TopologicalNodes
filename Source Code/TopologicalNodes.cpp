/*
* TopologicalNodes.cpp
* Alexander Marsh
* Last Edit 10 September 2025
*
* GNU Affero General Public License
*
* Method definitions for the LooseTopologicalNodeImpl pointer class.
* Implemented in c++ for speed.
*/

#include "TopologicalNodes.h"

// Constructor: Default leak factor, no hidden layers
LooseTopologicalNodeImpl::LooseTopologicalNodeImpl(int input_dim, int output_dim) {
    build_layers(0.1, input_dim, output_dim, 0, 0);
}

// Constructor: With configurable leak factor and hidden layers
LooseTopologicalNodeImpl::LooseTopologicalNodeImpl(float leak_factor, int input_dim, int output_dim, int num_hidden_layers, int hidden_layer_size) {
    build_layers(leak_factor, input_dim, output_dim, num_hidden_layers, hidden_layer_size);
}

// Build encoder-decoder architecture
void LooseTopologicalNodeImpl::build_layers(float leak_factor, int input_dim, int output_dim, int num_hidden_layers, int hidden_layer_size) {
    encoder_layers = torch::nn::Sequential();
    decoder_layers = torch::nn::Sequential();

    if (num_hidden_layers == 0) {
        // Simple 2-layer autoencoder with bottleneck of 4
        encoder_layers->push_back(torch::nn::Linear(input_dim, 4));
        decoder_layers->push_back(torch::nn::Linear(4, output_dim));
    } else {
        // Encoder
        encoder_layers->push_back(torch::nn::Linear(input_dim, hidden_layer_size));
        encoder_layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(leak_factor)));

        for (int i = 1; i < num_hidden_layers; i++) {
            encoder_layers->push_back(torch::nn::Linear(hidden_layer_size, hidden_layer_size));
            encoder_layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(leak_factor)));
        }

        // Latent bottleneck: fixed at size 4
        encoder_layers->push_back(torch::nn::Linear(hidden_layer_size, 4));

        // Decoder
        decoder_layers->push_back(torch::nn::Linear(4, hidden_layer_size));
        decoder_layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(leak_factor)));

        for (int i = 1; i < num_hidden_layers; i++) {
            decoder_layers->push_back(torch::nn::Linear(hidden_layer_size, hidden_layer_size));
            decoder_layers->push_back(torch::nn::LeakyReLU(torch::nn::LeakyReLUOptions().negative_slope(leak_factor)));
        }

        decoder_layers->push_back(torch::nn::Linear(hidden_layer_size, output_dim));
    }

    // Register modules so they’re tracked and saved properly
    register_module("encoder_layers", encoder_layers);
    register_module("decoder_layers", decoder_layers);
    skip_weight = register_module("skip_weight", torch::nn::Linear(4, 4));
}

// Normalize 4D vector to torus: first 2 dims to S¹, next 2 dims to S¹
torch::Tensor LooseTopologicalNodeImpl::normalize_to_torus(torch::Tensor z) {
    // Normalize first 2 dimensions (xy)
    auto xy = z.narrow(1, 0, 2);
    auto xy_norm = xy.norm(2, 1, true).clamp_min(1e-8);
    auto xy_normalized = xy / xy_norm;  // out-of-place division

    // Normalize next 2 dimensions (uv)
    auto uv = z.narrow(1, 2, 2);
    auto uv_norm = uv.norm(2, 1, true).clamp_min(1e-8);
    auto uv_normalized = uv / uv_norm;  // out-of-place division

    // Now create a new tensor that replaces those slices in z
    // Because xy and uv are slices of z, we can't just assign new tensors to them.
    // So let's clone z and overwrite those slices with normalized values:

    auto z_normalized = z.clone();
    z_normalized.narrow(1, 0, 2).copy_(xy_normalized);
    z_normalized.narrow(1, 2, 2).copy_(uv_normalized);

    return z_normalized;
}

torch::Tensor LooseTopologicalNodeImpl::angle_layer(const torch::Tensor& z2) {
    // Assumes z2 is of shape [batch_size, 4]
    auto angle_2d = torch::atan2(z2.select(1, 1), z2.select(1, 0));
    auto angle_3d = torch::atan2(z2.select(1, 3), z2.select(1, 2));

    return torch::stack({angle_2d, angle_3d}, 1); // Shape: [batch_size, 2]
}

// Forward pass: encode >> skip >> normalize >> decode
std::tuple<torch::Tensor, torch::Tensor> LooseTopologicalNodeImpl::forward(torch::Tensor x) {
    auto z = encoder_layers->forward(x);                                       // Encode to latent space (4D)
    auto post_skip_z = z + skip_weight->forward(torch::abs(z));                // Apply learnable skip connection
    auto normalized_z = normalize_to_torus(post_skip_z);                       // Push latent to 2-torus (S¹ × S¹)
    return std::make_tuple(decoder_layers->forward(normalized_z), angle_layer(normalized_z));   // Decode back to output
}
