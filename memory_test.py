import os
import time

import torch
import torch.nn as nn
import torch.optim as optim

from resnet import *
from swish import *


def replace_relu_with_swish(model, swish_func_str=None):
    if swish_func_str == "v1":
        swish_act = SwishActivation(swish_v1)
    elif swish_func_str == "v2":
        swish_act = SwishActivation(swish_v2)
    else:
        swish_act = SwishActivation(swish_naive)

    return convert_recursive(model, swish_act)


# def convert_recursive(module, swish_act):
#     mod = module
#     if isinstance(module, torch.nn.modules.activation.ReLU):
#         mod = swish_act
#     else:
#         for name, child in module.named_children():
#             mod.add_module(name, convert_recursive(child, swish_act))
#     del module
#     return mod


def convert_recursive(module, swish_act):
    if isinstance(module, torch.nn.modules.activation.ReLU):
        module = swish_act
    else:
        for name, child in module.named_children():
            module.add_module(name, convert_recursive(child, swish_act))
    return module


def train_loop(model, optimizer, criterion):
    random_data = torch.rand((32, 3, 224, 224)).cuda()
    random_label = torch.randint(1000, (32, ), dtype=torch.int64).cuda()
    print("start training now...please check GPU memory with nvidia-smi")
    print("Stop the program by Ctrl-C")
    for step in range(1000000):
        torch.cuda.synchronize()
        start = time.time()
        optimizer.zero_grad()
        output = model(random_data)
        loss = criterion(output, random_label)
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        end = time.time()
        if step % 20 == 0 and step > 0:
            print("forward and backward time: ", end - start)


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    model = resnet50()
    model = replace_relu_with_swish(model, "v2").cuda()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss().cuda()

    train_loop(model, optimizer, criterion)
