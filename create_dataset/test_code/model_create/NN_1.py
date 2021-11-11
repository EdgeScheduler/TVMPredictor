import torch
import torch.nn as nn
import numpy as np
import torchvision
import tvm
import tvm.relay
import torch.nn.functional as F
import os

def calc_size(shape):
    size = 1
    for i in shape:
        size*=i
    return size

class Net(torch.nn.Module):
    # def __init__(self):
    def __init__(self,shape,middle_size_1=256,middle_size_2=64,out_size=10):
        super().__init__()
        self.fc1 = nn.Linear(in_features=calc_size(shape[1:]), out_features=middle_size_1)
        self.fc2 = nn.Linear(in_features=middle_size_1, out_features=middle_size_2)
        self.output = nn.Linear(in_features=middle_size_2, out_features=out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x

def create_onnx(dshape,onnx_name="onnx_tmp.onnx",**args):
    input_shape = dshape[0]
    onnx_name = os.path.join(os.path.dirname(os.path.abspath(__file__)),"../ONNX_models/"+onnx_name)

    net = Net(input_shape,**args)
    input = torch.randn(input_shape[0],calc_size(input_shape[1:]))
    torch.onnx.export(net, input, onnx_name,input_names=['input'],output_names=['output'])
    return onnx_name