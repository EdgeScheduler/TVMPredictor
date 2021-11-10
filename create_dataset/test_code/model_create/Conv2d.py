import torch
import torch.nn as nn
import numpy as np
import torchvision
from tvm.contrib.download import download_testdata
import tvm
import tvm.relay

# class Net(nn.Module):
#     def __init__(self, in_channels=3, out_channels=64, stride=1):
#         super().__init__()
#         self.layer = nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
#             nn.BatchNorm2d(out_channels),
#             torch.nn.Linear(out_channels, 128),
#             torch.nn.Linear(128, 10))
            
#     def forward(self,x):
#         return self.layer(x)

# onnx_name = "net" + ".onnx"
# net = Net()
# input = torch.randn(1,3,224,224)
# output = net(input)
# torch.onnx.export(net, input, onnx_name,input_names=['input'],output_names=['output'])

# def pp():
#     print("hello")

class Net(nn.Module):
    # def __init__(self, in_channels=3, out_channels=64, stride=1):
    def __init__(self, batch_size,shape,out_channels=64,stride=1):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(shape[0], out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            torch.nn.Linear(out_channels, 128),
            torch.nn.Linear(128, 10))
            
    def forward(self,x):
        return self.layer(x)

onnx_name = "net" + ".onnx"
net = Net()
input = torch.randn(1,3,224,224)
output = net(input)
torch.onnx.export(net, input, onnx_name,input_names=['input'],output_names=['output'])

def pp():
    print("hello")