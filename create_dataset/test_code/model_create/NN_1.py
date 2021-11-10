# from create_dataset.test_code.model_create import *
import torch
import torch.nn as nn
import numpy as np
import torchvision
import tvm
import tvm.relay

def calc_size(shape):
    size = 1
    for i in shape:
        size*=i
    return size

class Net(torch.nn.Module):
    # def __init__(self):
    def __init__(self,shape,middle_size=128,out_size=10):
        super().__init__()
        print(middle_size)
        self.layer = nn.Sequential(
            torch.nn.Linear(64, middle_size),
            torch.nn.Linear(middle_size,out_size))
            
    def forward(self,x):
        return self.layer(x)


def create_onnx(input_shape,onnx_name="onnx_tmp.onnx",**args):
    net = Net(input_shape,**args)
    input = torch.randn(*input_shape)
    torch.onnx.export(net, input, onnx_name,input_names=['input'],output_names=['output'])

create_onnx((3,64,64),middle_size=64,out_size=10)