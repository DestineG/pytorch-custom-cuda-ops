import torch
import online_softmax_cuda

def fast_softmax(input):
    # 这里可以做一些检查
    return online_softmax_cuda.forward(input)