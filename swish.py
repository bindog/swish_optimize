import math
import torch

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


