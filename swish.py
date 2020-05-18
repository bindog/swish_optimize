import math
import torch
import torch.nn as nn

import swish_cpp


def swish_naive(x):
    return x * torch.sigmoid(x)

######################################
sigmoid = torch.nn.Sigmoid()
class Swish(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))

swish = Swish.apply

class Swish_module(nn.Module):
    def forward(self, x):
        return swish(x)

swish_layer = Swish_module()

######################################
class SwishFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        ctx.input = input
        outputs = swish_cpp.forward(input)
        return outputs

    @staticmethod
    def backward(ctx, grad_input):
        outputs = swish_cpp.backward(grad_input, ctx.input)
        return outputs

swish_c = SwishFunction.apply

if __name__ == "__main__":
     x = torch.rand((4, 5))
     print(x)
     print("="*50)
     print("check forward...")
     print(swish_c(x))
     print(swish_naive(x))
     print("="*50)
     print("check backward...")
     a = torch.rand((4, 5), requires_grad=True)
     b = torch.zeros_like(a, requires_grad=True)
     b.data = a.data
     al = swish_c(a).sum()
     bl = swish_naive(b).sum()
     al.backward()
     bl.backward()
     print(a.grad)
     print(b.grad)
