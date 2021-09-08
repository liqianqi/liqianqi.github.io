---
title: Pytorch Nueral Network
author: Harry-hhj
date: 2021-09-04 08:00:00 +0800
categories: [Tutorial, Pytorch]
tags: [getting started, computer, pytorch]
math: true
mermaid: false
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-04-Pytorch-Building-Neural-Network.assets/neural-network.jpeg?raw=true
  width: 940
  height: 530
---



# Pytorch 搭建神经网络

## 一、热身

`torch.nn` 包依赖 `autograd` 包来定义模型并求导。 一个 `nn.Module` 包含：

-   各个层
-   一个 `forward(input)` 方法，该方法返回网络的 `output` 

在模型中**必须要定义 `forward()` 函数**， `backward()` 函数（用来计算梯度）会被 `autograd` **自动创建**。 可以在 `forward()` 函数中使用**任何**针对 Tensor 的操作。

神经网络的典型训练过程如下：

1. 定义包含一些可学习的参数（权重）神经网络模型； 
2. 在数据集上迭代； 
3. 通过神经网络处理输入； 
4. 计算损失（输出结果和正确值的差值大小）；
5. 将梯度反向传播回网络的参数； 
6. 更新网络的参数，主要使用如下简单的更新原则： 
`weight = weight - learning_rate * gradient`

本篇将着重于模型定义，而[下一篇]()将着重于网络训练。在讲解 pytorch 的网络包之前，我们先尝试使用 `torch.nn` 包自己搭建一个简单的网络。我们先来看这样一个网络：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)  # 3 通道输入，6 通道输出，5x5 卷积核
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 全连接层 y = Wx + b
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))  # (2, 2) 最大池化
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)  # 方形卷积核可以用一个数字代替大小
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # 第 0 维是 batch 大小
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)
```

为了更好地观察网络结构，我们可以借助可视化工具 Netron 来输出网络的结构图。使用如下的代码，有关 Netron 的更多内容，可以查看[这篇教程]()。

```python
x = torch.randn(1, 3, 32, 32)
torch.onnx.export(net, x, "example.onnx")

import netron
import os
model_path = os.path.join(os.getcwd(),"example.onnx")
netron.start(model_path)
```

运行这段代码会自动跳出一个网页，如下图：

<img src="2021-09-04-Pytorch-Neural-Network.assets/example.onnx.svg" alt="example.onnx" style="zoom:50%;" />







## 二、torch.nn

### 1. 





## 三、各层权重命名规则

### 1. init 中的变量定义



### 2. nn.Sequential



### 3. DataParallel/DistributedDataParallel



## 、参考文献

1.   

