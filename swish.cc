#include <torch/extension.h>

at::Tensor swish_forward(torch::Tensor input) {
    return input * torch::sigmoid(input);
}

at::Tensor swish_backward(torch::Tensor grad_input, torch::Tensor input) {
    // sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
    auto sig = torch::sigmoid(input);
    return grad_input * (sig + input * sig * (1 - sig));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &swish_forward, "swish forward");
    m.def("backward", &swish_backward, "swish backward");    
}
