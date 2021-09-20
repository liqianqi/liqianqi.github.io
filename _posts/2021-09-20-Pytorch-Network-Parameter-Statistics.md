---
title: Pytorch Network Parameter Statistics
author: Harry-hhj
date: 2021-09-20 18:40:00 +0800
categories: [Tutorial, Pytorch]
tags: [computer, pytorch, tools]
math: false
mermaid: false
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-20-Pytorch-Network-Parameter-Statistics.assets/statics.jpg?raw=true
  width: 1024
  height: 520
---



# PyTorch 统计网络参数量

神经网络的参数统计是很重要的，它反映了一个网络的硬件需求与性能。PyTorch 可以使用第三方库 torchsummary 来统计参数并打印层结构。但是想要正确统计出参数量，需要对如何统计参数有一定的了解。

## Case1 无参数共享（最常见）

```python
import torch
import torch.nn as nn
import torchsummary
from torch.nn import init


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv2=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self,x):
        x=self.conv1(x)
        out_map=self.conv2(x)
        return out_map
    

def count_parameters(model):
    '''
    model.parameters() 取得模型的参数，在参数可求导 p.requires_grad 的情况下，使用 numel()统计 numpy 数组里面的元素的个数。
    '''
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = BaseNet()
torchsummary.summary(model, (1, 512, 512))
print('parameters_count:',count_parameters(model))
```

结果：

![image-20210920180042475](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-20-Pytorch-Network-Parameter-Statistics.assets/image-20210920180042475.png){: .shadow width="1386" height="690" style="max-width: 90%" }

在这个案例中，使用 torchsummary 和自己统计得到相同的结果。



## Case2 参数共享

```python
import torch
import torch.nn as nn
import torchsummary
from torch.nn import init


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self,x):
        x=self.conv1(x)
        out_map=self.conv1(x)
        return out_map
    

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = BaseNet()
torchsummary.summary(model, (1, 512, 512))
print('parameters_count:',count_parameters(model))
```

结果：

![image-20210920181012205](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-20-Pytorch-Network-Parameter-Statistics.assets/image-20210920181012205.png){: .shadow width="1390" height="696" style="max-width: 90%" }

在这里例子中， `parameter_count` 统计的是 9 个参数，而 `torchsummary` 统计的是 18 个参数，为什么会出现这种问题？在这个网络中，我们只初始化了一个卷积层对象 `conv1` ，然后在网络构建时（ `forward` 函数中），重复调用了`conv1` ，以实现参数共享，即 `Conv2d-1` 和 `Conv2d-2` 层共享了 `conv1` 的参数。因此本例中 `parameter_count` 的计算是对的，而 `torchsummary` 计算时是先把层结构打印下来，然后统计各个层的参数并求和，不区分 `Conv2d-1` 和 `Conv2d-2` 层的参数是否相同。

结论：**在遇到参数共享的时候， `torchsummary` 统计的是不正确的！**



## Case3 初始化无用变量

```python
import torch
import torch.nn as nn
import torchsummary
from torch.nn import init


class BaseNet(nn.Module):
    def __init__(self):
        super(BaseNet,self).__init__()
        self.conv1=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)
        self.conv2=nn.Conv2d(in_channels=1,out_channels=1,kernel_size=3,stride=1,padding=1,bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self,x):
        x=self.conv1(x)
        out_map=self.conv1(x)
        return out_map


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = BaseNet()
torchsummary.summary(model, (1, 512, 512))
print('parameters_count:',count_parameters(model))
```

结果：

![image-20210920181904154](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-20-Pytorch-Network-Parameter-Statistics.assets/image-20210920181904154.png){: .shadow width="1386" height="684" style="max-width: 90%" }

这个例子中我们在初始化时多初始化了一个 `conv2` 卷积层对象，但是没有在 `forward` 中使用。此时 `parameter_count` 出现了错误，即使没有在 `forward` 中调用，但是也会被算在 `model.parameters()` 中。但是要注意，尽管 `torchsummary` 和 `parameter_count` 都出现了同样结果的错误，两者出现错误的原因是不同的。





## 参考文献

1.   [PyTorch几种情况下的参数数量统计](https://zhuanlan.zhihu.com/p/64425750)



---

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

