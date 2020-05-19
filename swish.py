import math
import torch
import torch.nn as nn

# import swish_cpp


def swish_naive(x):
    return x * torch.sigmoid(x)


class SwishFuncV1(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x * torch.sigmoid(x)

    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_variables
        sigmoid_x = torch.sigmoid(x)
        return grad_output * (sigmoid_x * (1 + x * (1 - sigmoid_x)))

swish_v1 = SwishFuncV1.apply


# class SwishFuncV2(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, x):
#         ctx.save_for_backward(x)
#         return swish_cpp.forward(x)

#     @staticmethod
#     def backward(ctx, grad_output):
#         x, = ctx.saved_variables
#         return swish_cpp.backward(grad_output, x)

# swish_v2 = SwishFuncV2.apply


class SwishActivation(nn.Module):
    def __init__(self, swish_func_impl):
        super(SwishActivation, self).__init__()
        self.swish_func = swish_func_impl

    def forward(self, x):
        return self.swish_func(x)
