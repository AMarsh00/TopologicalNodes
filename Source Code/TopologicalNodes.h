/*
* TopologicalNodes.h
* Alexander Marsh
* 10 September 2025
*
* GNU Affero General Public License
*
* LibTorch implementation of our loose topological node.
*/

#pragma once

#include <torch/torch.h>

// Implementation of the LooseTopologicalNode module.
// The `Impl` suffix is automatically handled by the TORCH_MODULE macro below.
struct LooseTopologicalNodeImpl : torch::nn::Module {
private:
    // Network components
    torch::nn::Sequential encoder_layers{nullptr};
    torch::nn::Sequential decoder_layers{nullptr};
    torch::nn::Linear skip_weight{nullptr};

public:
    // Constructors
    LooseTopologicalNodeImpl(int input_dim, int output_dim);
    LooseTopologicalNodeImpl(float leak_factor, int input_dim, int output_dim,
                             int num_hidden_layers, int hidden_layer_size);

    // Forward pass
    torch::Tensor forward(torch::Tensor x);

private:
    // Utility to build encoder/decoder
    void build_layers(float leak_factor, int input_dim, int output_dim,
                      int num_hidden_layers, int hidden_layer_size);

    // Normalize latent vector to live on S¹ × S¹ (torus)
    torch::Tensor normalize_to_torus(torch::Tensor z);
    // Get the angle representation of data for return
    torch::Tensor angle_layer(const torch::Tensor& z2);
};

// Creates a shared_ptr wrapper around the implementation class
// This is the modern idiomatic way to define custom torch modules in C++
TORCH_MODULE(LooseTopologicalNode);
