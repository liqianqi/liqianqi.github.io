---
title: Pytorch Building Nueral Network
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

<img src="2021-09-04-Pytorch-Building-Neural-Network.assets/example.onnx.svg" alt="example.onnx" style="zoom:50%;" />

先请读者思考一个问题，这个网络的输入大小是多少，为什么？

想要了解网络的结构，我们需要看 `forward()` 的函数，注意不是看 `init()` 。 `__init__()` 中可以定义网络的比较关键的部件，这样的做的好处是：便于变量名称管理（见 [【三】](#命名规则) ）和规定层的参数，增加代码的可读性。

解析：对于大小为 $(w, h, c)$ 的输入 x ，首先经过 conv1 ，可以看出 conv1 的输入通道是 $3$ ，因此 $c=3$ ，输出 $6$ 通道。经过 $(5 \times 5)$ 的卷积，大小变为 $(w-4, h-4, 6)$ 。经过 $(2,2)$ 的最大池化层，大小缩减一半，通道数不变，为 $(\cfrac{w-4}{2}, \cfrac{h-4}{2}, 6)$ 。再经过一层 $(5 \times 5)$ 卷积，卷积核数量为 $16$ ，此时特征图尺寸 $(\cfrac{w-4}{2}-4, \cfrac{h-4}{2}-4, 16)$ 。再经过 $(2,2)$ 的最大池化层，变为 $(\cfrac{\cfrac{w-4}{2}-4}{2}, \cfrac{\cfrac{h-4}{2}-4}{2}, 16)$ 。全连接层的输入大小是固定的，因此我们得到以下的等式：
$$
\left \{
\begin{array}{c}
\cfrac{\cfrac{w-4}{2}-4}{2} = 5\\
\cfrac{\cfrac{h-4}{2}-4}{2} = 5\\
\end{array}
\right .
$$
得到 $w=h=32$ ，因此答案是 $(32, 32, 3)$ 。

如果对于上面的推导没有看懂的读者，建议先学习神经网络原理，再来看本篇教程。



## 二、torch.nn

### 1. 卷积层

### 1) nn.Conv2d

二维输入卷积层

```python
class torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros', device=None, dtype=None)
```

参数：

-   **in_channels** (`int`) – 输入的通道数量
-   **out_channels** (`int`) – 输出的通道数量
-   **kernel_size** (`int` or `tuple`) – 卷积核的大小
-   **stride** (`int` or `tuple`，可选) – 卷积步长，默认为 1 
-   **padding** (`int` or `tuple` or `str`，可选) – Padding added to all four sides of the input. Default: 0
-   **padding_mode** (`string**`，可选) – `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`，默认`'zeros'`
-   **dilation** (`int` or `tuple`，可选) – 内核元素之间的距离，默认 1 
-   **groups** (`int`，可选) – 从输入通道到输出通道到阻塞连接数，默认为 1 
-   **bias** (`bool`，可选) – 如果 `bias=True` ，添加科学系的偏置到输出中 

参数说明：





## <span id="命名规则">三、各层变量命名规则</span>

PyTorch 在对于网络中的参数，采用以下的规则命名变量。了解一下规则，能够帮助我们在需要时知道该如何调用某个变量，比如某层的权重。

### 1) init

对于 `__init__()` 中使用 `self` 定义的变量会使用这个**变量的名字**作为存储时的名字。

举例：

```python
self.conv1 = torch.nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)  # 卷积层有 2 个参数，对应 conv1.weight 和 conv.bias 
self.bn1 = torch.nn.BatchNorm2d(12)  # 标准化层油 5 个参数，对应 bn1.weight, bn1.bias, bn1.running_mean, bn1.running_var, bn1.num_batches_tracked 
```

### 2) nn.Sequential

使用 `nn.Sequential` 时会根据**传入 list 的顺序**对其进行标号。

举例：

```python
conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
bn1 = nn.BatchNorm2d(12)
s1 = [conv1, bn1]
self.stage1 = nn.Squential(*s1)
# stage1.0.weight, stage1.0.bias
# stage1.1.weight, stage1.1.bias, stage1.1.running_mean, stage1.1.running_var, stage1.1.num_batches_tracked
```

注意此时的 `conv1` 和 `bn1` 都没有 `self` ， `stage1` 有 `self` ，而 `s1` 是 python 基本数据类型。

### 3) DataParallel/DistributedDataParallel

当一个 `module` 被 `from torch.nn import DataParallel` 或者 `from torch.nn.parallel import DistributedDataParallel` 包围住后，会在这个变量名后面加上 `module.` 。

举例：

```python
conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
bn1 = nn.BatchNorm(12)
s1 = [conv1, bn1]
stage1 = nn.Sequential(*s1)
self.stage2 = DataParallel(stage1)
# stage2.module.0.weight, stage2.module.0.bias
# stage2.module.1.weight, stage2.module.1.bias, stage2.module.1.running_mean, stage2.module.1.running_var, stage2.module.1.num_batches_tracked
```

注意只有 `stage2` 前面有 `self` 。

### 4) 综合

下面举两个综合起来的例子供读者练习。

Example 1：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
 
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.s1 = [self.conv1, self.bn1]
        self.stage1 = nn.Sequential(*self.s1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 24 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
 
if __name__ == '__main__':
    model = CNN()
    for name in model.state_dict():
        print(name)
```

解析： `self.conv1` 和 `self.bn1` 通过 `self.s1` 传入 `Sequential`	 ，所以 `self.stage` 会根据出现顺序进行编号，但原本的 `self.conv1` 和 `self.bn1` 仍然存在，同时 `self.s1` 并没有，虽然它有 `self` ，但是它不是 `pytorch` 自带的层，是 `python` 的基本数据结构。
```text
conv1.weight、conv1.bias
bn1.weight、bn1.bias、bn1.running_mean、bn1.running_var、bn1.num_batches_tracked
stage1.0.weight、stage1.0.bias
stage1.1.weight、stage1.1.bias、stage1.1.running_mean、stage1.1.running_var、stage1.1.num_batches_tracked
conv2.weight、conv2.bias
bn2.weight、bn2.bias、bn2.running_mean、bn2.running_var、bn2.num_batches_tracked
fc1.weight、fc1.bias
fc2.weight、fc2.bias
```

Example2：

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
 
 
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        conv1 = nn.Conv2d(3, 12, kernel_size=3, stride=1, padding=1)
        bn1 = nn.BatchNorm2d(12)
        s1 = [conv1, bn1]
        self.stage1 = nn.Sequential(*s1)
        self.stage2 = DataParallel(self.stage1)
        self.conv2 = nn.Conv2d(12, 24, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 10)
 
    def forward(self, x):
        x = self.stage2(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = x.view(-1, 24 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
 
 
if __name__ == '__main__':
    model = CNN()
    model = DataParallel(model)
    for name in model.state_dict():
        print(name)
```

解析：`self.stage1` 按照 `Sequential` 进行编号， `self.stage` 通过 `DataParallel` 进行包裹，因此会在 `stage2` 后面多出 `module.` ，由于最后的 `model` 也被 `DataParallel` 包裹，所以 CNN 里面所有变量前面都多了 `module.` 。

```text
module.stage1.0.weight、module.stage1.0.bias
module.stage1.1.weight、module.stage1.1.bias、module.stage1.1.running_mean、module.stage1.1.running_var、module.stage1.1.num_batches_tracked
module.stage2.module.0.weight、module.stage2.module.0.bias
module.stage2.module.1.weight、module.stage2.module.1.bias、module.stage2.module.1.running_mean、module.stage2.module.1.running_var、module.stage2.module.1.num_batches_tracked
module.conv2.weight、module.conv2.bias
module.bn2.weight、module.bn2.bias、module.bn2.running_mean、module.bn2.running_var、module.bn2.num_batches_tracked
module.fc1.weight、module.fc1.bias
module.fc2.weight、module.fc2.bias
```





## 、参考文献

1.   [TORCH.NN](https://pytorch.org/docs/stable/nn.html)
2.   [pytorch中存储各层权重参数时的命名规则，为什么有些层的名字中带module.](https://blog.csdn.net/u014734886/article/details/106230535)
3.   [pytorch中文文档-torch.nn常用函数-待添加](https://www.cnblogs.com/wanghui-garcia/p/10775859.html)



-----

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

