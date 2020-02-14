import math

import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim


def round_value(value, scale):
    return int(math.ceil(value * scale))

# ================================================
class ConvBnRelu(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1):
        super(ConvBnRelu, self).__init__()
        pad_size = kernel_size // 2
        
        self.conv = nn.Conv1d(in_channel,out_channel,kernel_size,padding=pad_size,stride=stride)
        self.bn = nn.BatchNorm1d(out_channel)
        self.act = nn.ReLU()

    def forward(self,x):
        out = self.act(self.bn(self.conv(x)))
        return out


# ================================================
class CNNmodule(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, depth_scale, width_scale, initial=False):
        super(CNNmodule, self).__init__()
        
        depth = round_value(2, depth_scale)
        width = round_value(out_channel, width_scale)

        if initial: layers = [ConvBnRelu(in_channel, width, kernel_size, stride=2)]
        else: layers = [ConvBnRelu(round_value(in_channel, width_scale), width, kernel_size, stride=2)]

        for i in range(depth-1):
            layers += [ConvBnRelu(width, width, kernel_size)]
        self.cnn_module = nn.Sequential(*layers)

    def forward(self, x):
        return self.cnn_module(x)


# ================================================
class CNN1d_adaptive(nn.Module):
    def __init__(self, kernel_size, num_classes, alpha, beta, phi):
        super(CNN1d_adaptive, self).__init__()

        depth_scale = alpha ** phi
        width_scale = beta ** phi
        self.last_channel = round_value(128, width_scale)

        self.feature = nn.Sequential(
            CNNmodule(1, 16, kernel_size, depth_scale, width_scale, initial=True),
            CNNmodule(16, 32, kernel_size, depth_scale, width_scale),
            CNNmodule(32, 64, kernel_size, depth_scale, width_scale),
            CNNmodule(64, 128, kernel_size, depth_scale, width_scale)
        )
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(self.last_channel,num_classes)

    def forward(self, x):
        x = self.feature(x)
        x = self.pool(x)
        x = x.view(-1, self.last_channel)
        out = self.fc(x)
        return out


# ================================================
if __name__ == "__main__":
    x = torch.rand(16,1,5000)
    # m = CNNmodule(1, 16, 3, 1, 1)
    m = CNN1d_adaptive(3, 5, 1, 1, 1)
    out = m(x)
    

