#include <torch/extension.h>
using namespace pybind11::literals;

// Forward declaration of kernels
void swish_forward_cuda(torch::Tensor &output, const torch::Tensor &input);
void swish_backward_cuda(torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &grad_out);

// Forward declaration of hosts
// also can run on device, but not fully optimized
at::Tensor swish_forward_cpu(torch::Tensor input) {
    return input * torch::sigmoid(input);
}

at::Tensor swish_backward_cpu(torch::Tensor grad_input, torch::Tensor input) {
    auto sig = torch::sigmoid(input);
    return grad_input * (sig * (1 + input * (1 - sig)));
}

torch::Tensor
swish_forward(const torch::Tensor &input, const at::optional<torch::Tensor> out) {
  auto input_arg = torch::TensorArg(input, "input", 0);
  if (out) {
    auto out_arg = torch::TensorArg(*out, "out", 1);
    torch::checkSameType("swish_forward", input_arg, out_arg);
    torch::checkSameSize("swish_forward", input_arg, out_arg);
  }
  auto o = out.value_or(torch::empty_like(input));
  switch (input.device().type()) {
    case c10::kCUDA:
      swish_forward_cuda(o, input);
      break;
    default:
      TORCH_CHECK(false, "Unsupported device type, should be CUDA but got ", input.device().type());
  }
  return o;
}

torch::Tensor
swish_backward(const torch::Tensor &grad_out, const torch::Tensor &input) {
  auto grad_out_arg = torch::TensorArg(grad_out, "grad_out", 0);
  auto input_arg = torch::TensorArg(input, "input", 1);
  torch::checkSameType("swish_backward", grad_out_arg, input_arg);

  auto grad_inp = torch::empty_like(input);
  switch (input.device().type()) {
    case c10::kCUDA:
      swish_backward_cuda(grad_inp, input, grad_out);
      break;
    default:
      TORCH_CHECK(false, "Unsupported device type, should be CUDA but got ", input.device().type());
  }
  return grad_inp;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // TODO note here when there is a optional parameter
    // pybind11 must specify the args name and default value
    m.def("forward", &swish_forward, "swish forward func", "input"_a, "out"_a = nullptr);
    m.def("backward", &swish_backward, "swish backward func", "grad_out"_a, "input"_a);    
}
