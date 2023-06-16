# 开发时间: 2023/6/14 15:22
import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size):
        super(MLP, self).__init__()

        # 定义输入层和隐藏层
        self.layers = nn.ModuleList()
        sizes = [input_size] + hidden_sizes
        for i in range(len(hidden_sizes)):
            self.layers.append(nn.Linear(sizes[i], sizes[i + 1]))
            self.layers.append(nn.ReLU())
            
        # 定义输出层
        self.output_layer = nn.Linear(hidden_sizes[-1], output_size)

    def forward(self, x):
        # 前向传播
        for layer in self.layers:
            x = layer(x)
        x = self.output_layer(x)
        return x