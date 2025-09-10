/*
* Bindings.cpp
* Alexander Marsh
* Last Edit 10 September 2025
*
* GNU Affero General Public License
*
* Bind the TopologicalNode LibTorch module to Python via PyBind11.
* This works by allowing us to compile the module as a python .so file, which we can import in PyTorch.
*/

#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include "TopologicalNodes.h"

namespace py = pybind11;

PYBIND11_MODULE(LooseTopologicalNode, m) {
    m.doc() = "LooseTopologicalNode PyTorch C++ extension module";

    // Bind the module holder (this is what Python will use)
    py::class_<LooseTopologicalNode, std::shared_ptr<LooseTopologicalNode>>(m, "LooseTopologicalNode")
        .def(py::init<int, int>(), "Simplified constructor (input_dim, output_dim)")
        .def(py::init<float, int, int, int, int>(), "Full constructor (leak_factor, input_dim, output_dim, num_hidden_layers, hidden_layer_size)")
        .def("forward", [](LooseTopologicalNode& self, torch::Tensor x) {
            return self->forward(x);
        }, "Forward pass through the module")
        .def("parameters", [](LooseTopologicalNodeImpl &self) {
            return self.parameters();
        });
}
