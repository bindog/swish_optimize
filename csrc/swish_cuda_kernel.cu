#include <torch/types.h>
#include <torch/extension.h>
// #include <ATen/cuda/CUDAApplyUtils.cuh>

#include "CUDAApplyUtils.cuh"

#include <cuda.h>
#include <cuda_runtime.h>

// TORCH_CHECK replaces AT_CHECK in PyTorch 1,2, support 1.1 as well.
#ifndef TORCH_CHECK
#define TORCH_CHECK AT_CHECK
#endif

#ifndef __CUDACC_EXTENDED_LAMBDA__
#error "please compile with --expt-extended-lambda"
#endif


namespace kernel {
#include "swish.h"

using at::cuda::CUDA_tensor_apply2;
using at::cuda::CUDA_tensor_apply3;
using at::cuda::TensorArgType;

template <typename scalar_t>
void
swish_forward(
  torch::Tensor &output,
  const torch::Tensor &input
) {
  CUDA_tensor_apply2<scalar_t,scalar_t>(
    output, input,
    [=] __host__ __device__ (scalar_t &out, const scalar_t &inp) {
      swish_fwd_func(out, inp);
    },
    TensorArgType::ReadWrite, TensorArgType::ReadOnly
  );
}

template <typename scalar_t>
void
swish_backward(
  torch::Tensor &grad_inp,
  const torch::Tensor &input,
  const torch::Tensor &grad_out
) {
  CUDA_tensor_apply3<scalar_t,scalar_t,scalar_t>(
    grad_inp, input, grad_out,
    [=] __host__ __device__ (scalar_t &grad_inp, const scalar_t &inp, const scalar_t &grad_out) {
      swish_bwd_func(grad_inp, inp, grad_out);
    },
    TensorArgType::ReadWrite, TensorArgType::ReadOnly, TensorArgType::ReadOnly
  );
}

} // namespace kernel

template <typename scalar_t>
__device__ __forceinline__ scalar_t sigmoid(scalar_t z) {
  return 1.0 / (1.0 + exp(-z));
}

void
swish_forward_cuda(
    torch::Tensor &output, const torch::Tensor &input
) {
  auto in_arg  = torch::TensorArg(input,  "input",  0),
       out_arg = torch::TensorArg(output, "output", 1);
  torch::checkAllDefined("swish_forward_cuda", {in_arg, out_arg});
  torch::checkAllSameGPU("swish_forward_cuda", {in_arg, out_arg});
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(input.scalar_type(), "swish_forward_cuda", [&] {
      kernel::swish_forward<scalar_t>(output, input);
  });
}

void
swish_backward_cuda(
  torch::Tensor &grad_inp, const torch::Tensor &input, const torch::Tensor &grad_out
) {
  auto gi_arg = torch::TensorArg(grad_inp, "grad_inp", 0),
       in_arg = torch::TensorArg(input,    "input",    1),
       go_arg = torch::TensorArg(grad_out, "grad_out", 2);
  torch::checkAllDefined("swish_backward_cuda", {gi_arg, in_arg, go_arg});
  torch::checkAllSameGPU("swish_backward_cuda", {gi_arg, in_arg, go_arg});
  AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_inp.scalar_type(), "swish_backward_cuda", [&] {
      kernel::swish_backward<scalar_t>(grad_inp, input, grad_out);
  });
}
