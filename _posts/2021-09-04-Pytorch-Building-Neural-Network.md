---
title: Pytorch Building Nueral Network
author: Harry-hhj
date: 2021-09-04 08:00:00 +0800
categories: [Tutorial, Pytorch]
tags: [getting started, computer science, pytorch]
math: true
mermaid: false
image:
  src: https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2021-09-04-Pytorch-Building-Neural-Network.assets/neural-network.jpeg
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

解析：对于大小为 $(w, h, c)$ 的输入 x ，首先经过 conv1 ，可以看出 conv1 的输入通道是 $3$ ，因此 $c=3$ ，输出 $6$ 通道。经过 $(5 \times 5)$ 的卷积，大小变为 $(w-4, h-4, 6)$ 。经过 $(2,2)$ 的最大池化层，默认 `stride=kernel_size=2`，大小缩减一半，通道数不变，为 $(\cfrac{w-4}{2}, \cfrac{h-4}{2}, 6)$ 。再经过一层 $(5 \times 5)$ 卷积，卷积核数量为 $16$ ，此时特征图尺寸 $(\cfrac{w-4}{2}-4, \cfrac{h-4}{2}-4, 16)$ 。再经过 $(2,2)$ 的最大池化层，变为 $(\cfrac{\cfrac{w-4}{2}-4}{2}, \cfrac{\cfrac{h-4}{2}-4}{2}, 16)$ 。全连接层的输入大小是固定的，因此我们得到以下的等式：
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



在定义一个自己的网络时，我们



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
-   **padding** (`int` or `tuple` or `str`，可选) – 向输入四边添加的填充宽度，默认为 0 
-   **padding_mode** (`string**`，可选) – `'zeros'`, `'reflect'`, `'replicate'` or `'circular'`，默认`'zeros'`
-   **dilation** (`int` or `tuple`，可选) – 内核元素之间的距离，默认 1 
-   **groups** (`int`，可选) – 从输入通道到输出通道到阻塞连接数，默认为 1 
-   **bias** (`bool`，可选) – 如果 `bias=True` ，添加科学系的偏置到输出中，默认为 `true`

参数说明：

-   `dilation` ：**空洞卷积**。下图蓝色为输入，绿色为输出（注意参考文献 4 中的阐述有误）
    
    -   `dilation=1` 
    
        <img src="2021-09-04-Pytorch-Building-Neural-Network.assets/full_padding_no_strides_transposed.gif" alt="full_padding_no_strides_transposed" style="zoom:50%;" />
    
    -   `dilation=2` 
    
        <img src="2021-09-04-Pytorch-Building-Neural-Network.assets/dilation.gif" alt="dilation" style="zoom:50%;" />
    
    -   以此类推
    
    -   其实这里 `dilation` 的定义和 `stride` 是一致的， `dilation` 并不代表跳过多少个元素，而代表两个内核元素之间的距离，因此默认值是 1 。
    
    -   好处：使用 `dilation` 的好处是增大单次计算时的感受域（即覆盖的面积），在**增大感受域**的同时却**没有增加计算量**，保留了更多的细节信息。例如在上面的例子中， `dilation=1` 时感受域为 $3*3=9$ ， `dilation=2` 时感受域为 $5*5=25$ 。
    
-   `groups` ：**深度可分离卷积**。

    -   `groups=1` ：输出是所有的输入的卷积

        ```python
        conv = nn.Conv2d(in_channels=6, out_channels=3, kernel_size=1, groups=1)
        conv.weight.data.size()  # torch.Size([3, 6, 1, 1])
        ```

    -   `groups=2` ：此时相当于有并排的两个卷积层，每个卷积层计算输入通道的一半，并且产生的输出是输出通道的一半，随后将这两个输出连接起来得到结果

        ```python
        conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=2)
        conv.weight.data.size()  # torch.Size([6, 3, 1, 1])
        ```

    -   `groups=in_channels` ：每一个输入通道和它对应的卷积核进行卷积，该对应的卷积核大小为 $$\cfrac{C_{out}}{C_{in}}$$ 

        ```python
        conv = nn.Conv2d(in_channels=6, out_channels=6, kernel_size=1, groups=6)
        conv.weight.data.size()  # torch.Size([6, 1, 1, 1])
        ```

        提示一下，**一般卷积网络中出现的大小都是四维数组，他们分别表示 $(数量N, 通道C, 高度H, 宽度W)$ **。注意顺序。

    -   `groups` 的值必须能**同时整除 `in_channels` 和 `out_channels`** 。
    
    -   **【理解】**其**规律**是共有 `out_channels` 个卷积核，被分为 `groups` 组，即每组有 `out_channels/groups` 个卷积核，输入的通道也被分为 `groups` 组，每组的通道数为`in_channels/groups` ，因此每个卷积核的通道数为 `in_channels/groups` 组，每一组卷积核负责输入的一组，最后将 `groups` 组的卷积结果拼接起来，就得到了最终的输出。具体的实验结果可以查看[这篇教程](https://blog.csdn.net/cxx654/article/details/109681004)。
    
    -   好处：深度可分离卷积的目的是**减少卷积操作的参数量和计算量**，从而提升运算速度。在实际实验中，同样的网络结构下，这种分组的卷积效果是**好于**未分组的卷积的效果的。
    
-   参数 `kernel_size` ， `stride` ， `padding` ， `dilation` 

    -   可以是一个 `int` 的数据，此时卷积 `height` 和 `width` 值相同
    -   也可以是一个 `tuple` 数组， `tuple` 的第一维度表示 `height` 的数值， `tuple` 的第二维度表示 `width` 的数值

大小推导：

假设输入大小为 $(N, C_{in}, H_{in}, W_{in})$ ，输出为 $(N, C_{out}, H_{out}, W_{out})$ ，满足以下关系：



$$
H_{out} = \lfloor \cfrac{H_{in}+2 \times \text{padding}[0]-\text{dilation}[0] \times (\text{kernel\_size}[0]-1)-1}{\text{stride}[0]}+1 \rfloor
$$


$$
W_{out} = \lfloor \cfrac{W_{in}+2 \times \text{padding}[1]-\text{dilation}[1] \times (\text{kernel\_size}[1]-1)-1}{\text{stride}[1]}+1 \rfloor
$$



卷积层含有两个变量：

-   **~Conv2d.weight** (Tensor)：权重，可学习参数，大小为 $(\text{out\_channels}, \frac{\text{in\_channels}}{\text{groups}}, \text{kernel\_size}[0], \text{kernel\_size}[1])$ 
-   **~Conv2d.bias** (Tensor)：偏置，可学习参数

用法：

```python
m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
```



### 2) nn.ConvTranspose2d

微步卷积（fractionally-strided convolutions）或解卷积（deconvolutions），也可以看作是输入卷积的梯度。注意，解卷积并**不是卷积的逆过程**， 不能还原卷积前的数据。

```python
class torch.nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=True, dilation=1, padding_mode='zeros', device=None, dtype=None)
```

参数：

-   **in_channels** (`int`) – 输入的通道数量
-   **out_channels** (`int`) – 输出的通道数量
-   **kernel_size** (`int` or `tuple`) – 卷积核的大小
-   **stride** (`int` or `tuple`，可选) – 卷积步长，默认为 1 
-   **padding** (`int` or `tuple`，可选) – 向输入四边添加 `dilation * (kernel_size - 1) - padding` 零填充，默认为 0 
-   **output_padding** (`int` or `tuple`，可选) – 向输出图片一边添加的填充，默认为 0 
-   **groups** (`int`，可选) – 从输入通道到输出通道到阻塞连接数，默认为 1 
-   **bias** (`bool`，可选) – 如果 `bias=True` ，添加科学系的偏置到输出中，默认为 `true` 
-   **dilation** (`int` or `tuple`，可选) – 内核元素之间的距离，默认 1 

参数说明：

-   本质同样是卷积，很多参数的意义都和 conv 一样

-   `out_padding` ：这种填充只会填充一边，目的是规定输出的大小。

    -   在进行卷积时，对于不同大小的输入，规定不同的 `padding` 、 `stride` 和卷积类型，可能得到相同大小的输出。举个例子，对于 $6\times6$  的输入特征图，定义 `stride` 为 $2$ ，`kernel_size` 为 $3$ ，`padding` 为 $2$，输出的特征图大小为 $3\times3$ 。对于 $5\times5$  的输入特征图，定义 `stride` 为 $2$ ，`kernel_size` 为 $3$ ，`padding` 为 $1$，输出的特征图大小也为 $3\times3$ 。既然不同大小的图片经过卷积运算能够得到相同尺寸的输出，那么作为解卷积，同样的一样图片可以得到不同尺寸的合法输出，这就引发了歧义。当我们后续的操作涉及到尺寸有关的行为时，就无法保证网络按照预期进行计算。为了解决这种模糊性，pytorch 巧妙地引入了参数 `out_padding` 来获得固定大小的输出。

    -   这里有一个默认前提，一般情况下我们希望经过卷积/解卷积处理后的图像尺寸比例与步长相等，即 $输入特征图大小/输出特征图大小=\text{stride}$ 。

        我们先来算为了满足这个前提 `padding` 应该设置为多少。根据公式
        $$
        H_{out} = \lfloor \cfrac{H_{in}+2 \times \text{padding}[0]-\text{dilation}[0] \times (\text{kernel\_size}[0]-1)-1}{\text{stride}[0]}+1 \rfloor = \cfrac{H_{in}}{\text{stride}[0]}
        $$
        得到：
        $$
        \text{padding}[0] = \cfrac{\text{dilation}[0] \times (\text{kernel\_size[0] - 1}) + 1 - \text{stride}[0]}{2}
        $$
        根据这个 padding 求 outpadding ，代入公式
        $$
        H_{out} = (H_{in}-1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0] \times (\text{kernel\_size}[0]-1) + \text{out\_padding}[0] + 1 = H_{in} \times \text{stride}[0]
        $$
        得到：
        $$
        \text{out\_padding}[0] = 0
        $$
        这说明，当按照完全相同的尺寸逆卷积，且规定网络参数一致，输入输出特征图的比例互为倒数（即整个大小运算中应没有使用到下取整），此时无需 `out_padding` 。而当运算使用到了下取整，就意味着我们需要通过 `out_padding` 补足这部分被舍弃的大小。
        
        而这一点，参考文献[6](https://www.cnblogs.com/wanghui-garcia/p/10791778.html)的表述就略有问题了，请读者注意。

大小推导：

假设输入大小为 $(N, C_{in}, H_{in}, W_{in})$ ，输出为 $(N, C_{out}, H_{out}, W_{out})$ ，满足以下关系：


$$
H_{out} = (H_{in}-1) \times \text{stride}[0] - 2 \times \text{padding}[0] + \text{dilation}[0] \times (\text{kernel\_size}[0]-1) + \text{out\_padding}[0] + 1
$$

$$
W_{out} = (W_{in}-1) \times \text{stride}[1] - 2 \times \text{padding}[1] + \text{dilation}[1] \times (\text{kernel\_size}[1]-1) + \text{out\_padding}[1] + 1
$$


逆卷积层含有两个变量：

-   **~ConvTranspose2d.weight** (Tensor)：权重，可学习参数，大小为 $(\text{in\_channels}, \frac{\text{out\_channels}}{\text{groups}}, \text{kernel\_size}[0], \text{kernel\_size}[1])$ 
-   **~ConvTranspose2d.bias** (Tensor)：权重，可学习参数，大小为 $(\text{out\_channels})$ 

更详细的说明可以查看这篇 [Deconvolution教程](https://harry-hhj.github.io/posts/Deconvolution/) 。

用法：

```python
upsample = nn.ConvTranspose2d(16, 16, 3, stride=2, padding=1)
```



### 2. 池化层

#### 1) nn.MaxPool2d

下采样的一种。

```python
torch.nn.MaxPool2d(kernel_size, stride=None, padding=0, dilation=1, return_indices=False, ceil_mode=False)
```

参数：

-   **kernel_size** – 取最大值的窗口大小
-   **stride** – 窗口滑动的步幅。**默认值为 `kernel_size`** （注意不是 $1$ ）
-   **padding** – 要在两侧添加隐式零填充
-   **dilation** – 控制窗口中元素步幅的参数
-   **return_indices** – 如果`True`，将返回最大索引和输出。对之后的 `torch.nn.MaxUnpool2d` 有用。
-   **ceil_mode** – 当为 True 时，将使用上取整 ceil （即填充数据）而不是下取整 floor （即舍弃数据）来计算**输出形状**

参数说明：

-   `stride` ：默认值等于卷积核的大小

-   `ceil_mode` ：当 `ceil_mode = true` 时，将保存剩余的不足为 `kernel_size` 大小的数据保存，自动补足 $\text{NAN}$ 至 `kernel_size` 大小；当 `ceil_mode = False` 时，剩余数据不足 `kernel_size` 大小时，直接舍弃。

    举例：对于如下原始数据，进行 $(2\times2)$ MaxPool2d：

    ```text
            0		0		|		0		0		|		0
            1		1		|		1		1		|		1
            ————————————----————————————
            2		2		|		2		2		|		2
            3		3		|		3		3		|		3
            —————————————---————————————
            4		4		|		4		4		|		4
    ```

    

    -   ```text
        当ceil_mode = True时：
                0		0		|		0		0		|		0		×
                1		1		|		1		1		|		1		×
                ————————————----————————————----
                2		2		|		2		2		|		2		×
                3		3		|		3		3		|		3		×
                —————————————---————————————----
                4		4		|		4		4		|		4		×
                ×		×		|		×		×		|		×		×
                ————————————————————————————----
                
        输出：
                [	1 	1 	1
                 	3 	3 	3
                 	4 	4 	4	]
        
        即：当数据不足以构成 2*2 时，仍然对剩余数据进行计算
        ```

    -   ```text
        当ceil_mode = False时：
                0		0		|		0		0
                1		1		|		1		1
                ————————————----——-
                2		2		|		2		2
                3		3		|		3		3
        
        输出：
        				[	1		1
        					3  	3	]
        
        即：当数据不足以构成 2*2 时，舍弃
        ```

大小推导：

假设输入大小为 $(N, C_{in}, H_{in}, W_{in})$ ，输出为 $(N, C_{out}, H_{out}, W_{out})$ ，满足以下关系：
$$
H_{out} = \lfloor \cfrac{H_{in}+2 \times \text{padding}[0]-\text{dilation}[0] \times (\text{kernel\_size}[0]-1)-1}{\text{stride}[0]}+1 \rfloor
$$

$$
W_{out} = \lfloor \cfrac{W_{in}+2 \times \text{padding}[1]-\text{dilation}[1] \times (\text{kernel\_size}[1]-1)-1}{\text{stride}[1]}+1 \rfloor
$$

用法：

```python
m = nn.MaxPool2d((3, 2), stride=(2, 1))
```



#### 2) nn.MaxUnpool2d

一种上采样方法， `MaxPool2d` 的部分逆运算，因为非最大值已经丢失了。

`MaxUnpool2d` 将`MaxPool2d` 包含最大值索引的输出作为输入，并计算部分逆，其中所有非最大值都设置为零。

```python
```

参数：

-   **kernel_size** ( `int` or `tuple` ) – 最大池化窗口的大小
-   **stride** ( `int` *or* `tuple` ) – 最大池化窗口的步幅。默认设置为 `kernel_size` 。
-   **padding** ( `int` *or* `tuple` ) – 添加到输入的填充，可以通过输入 `output_size` 来确定

输入：

-   `input` ：输入张量
-   `indices` ：`MaxPool2d` 返回
-   `output_size` (optional)：目标输出大小

用法：

```python
pool = nn.MaxPool2d(2, stride=2, return_indices=True)
unpool = nn.MaxUnpool2d(2, stride=2)
input = torch.tensor([[[[ 1.,  2,  3,  4],
                        [ 5,  6,  7,  8],
                        [ 9, 10, 11, 12],
                        [13, 14, 15, 16]]]])
output, indices = pool(input)
unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
# tensor([[[[  0.,   0.,   0.,   0.,   0.],
#           [  6.,   0.,   8.,   0.,   0.],
#           [  0.,   0.,   0.,  14.,   0.],
#           [ 16.,   0.,   0.,   0.,   0.],
#           [  0.,   0.,   0.,   0.,   0.]]]])
```



#### 3) nn.AvgPool2d

平均池化。
$$
\text{out}(N_i, C_j, h, w) = \frac{1}{kH*kW}\sum_{m=0}^{kH-1}\sum_{n=0}^{kW-1} \text{input}(N_i, C_j, \text{stride}[0] \times h+m, \text{stride}[1] \times w+n)
$$

````python
class torch.nn.AvgPool2d(kernel_size, stride=None, padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None)
````

参数：

-   **kernel_size** – 窗口的大小
-   **stride** – 窗口的步幅，默认值为 `kernel_size`
-   **padding** – 要在两侧添加的隐式零填充
-   **ceil_mode** – 当为 True 时，将使用 ceil 而不是 floor 来计算输出形状
-   **count_include_pad** – 当为 True 时，将在平均计算中包括零填充
-   **divisor_override** – 除法因子，如果指定，它将被用作平均时的除数，否则 `kernel_size` 将被使用

用法：

```python
m = nn.AvgPool2d(3, stride=2)
```



#### 4) nn.AdaptiveMaxPool2d

二维自适应最大池化。对于任何大小的输入，可以产生指定输出大小。

```python
class torch.nn.AdaptiveMaxPool2d(output_size, return_indices=False)
```

参数：

-   **output_size** – H 和 W 可以是 `int`，或者 `None` 表示大小将与输入相同。
-   **return_indices** – 如果`True`，将返回索引和输出。传递给 nn.MaxUnpool2d 很有用。默认：`False`

AdaptiveMaxPool2d 自动计算了 `kernel_size` 、 `stride` 、 `padding` ，它可以用以下公式与 MaxPool2d 转换：
$$
\begin{equation}\begin{split}
& \text{kernel\_size} = \text{input\_size} - (\text{output\_size}-1) * \text{stride}\\
& \text{stride} = \lfloor \frac{\text{input\_size}}{\text{output\_size}} \rfloor\\
& padding = 0
\end{split}\end{equation}
$$
用法：

```python
m = nn.AdaptiveMaxPool2d((None, 7))
```



#### 5) nn.AdaptiveAvgPool2d





### 3. 非线性激活函数

#### 1) nn.ELU



#### 2) nn.Hardshrink



#### 3) nn.Hardsigmoid



#### 4) nn.Hardtanh



#### 5) nn.Hardwish



#### 6) nn.LeakyReLu



#### 7) nn.LogSigmoid



#### 8) nn.MultiheadAttention



#### 9) nn.PReLu



####  10) nn.ReLu



#### 11) nn.ReLu6



#### 12) RReLu



#### 13) nn.SELU



#### 14) nn.CELU



#### 15) nn.GELU



#### 16) nn.Sigmoid



#### 17) nn.SiLU



#### 18) nn.Mish



#### 19) nn.Softplus



####  20) nn.Softsign



#### 21) nn.Tanh



#### 22) nn.Tanhshrink



#### 23) nn.Threshold



### 4. 其他非线性操作

#### 1) nn.Softmax



#### 2) nn.LogSoftmax



#### 3) nn.AdaptiveLogSoftmaxWithLoss





### 5. 标准化层

#### 1) nn.BatchNorm2d

```python
class torch.nn.BatchNorm2d(num_features, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True, device=None, dtype=None)
```

论文来源： [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) 。

对于一个 4D 的输入（小批量的带通道的二维输入）：
$$
y = \cfrac{x-\mathbf{E}[x]}{\sqrt{\mathbf{Var}[x]+\epsilon}}*\gamma+\beta
$$
平均值和标准差是在小批量上的每个维度计算的，并且 $\gamma$ 和 $\beta$ 是大小为 $C$ 的可学习参数向量（其中 $C$ 是输入大小）。默认情况下，元素 $\gamma$ 被设置为 1 和元素 $\beta$ 被设置为 $0$ 。标准偏差是通过偏置估计器计算的，相当于 `torch.var(input, unbiased=False)` 。

此外，默认情况下，在训练期间，该层**不断运行其计算出的**均值和方差的估计值，然后在评估期间将其用于归一化。运行估计保持默认值 `momentum` 0.1。更新规则：$\hat x_{new} = 1 - \text{momentum} \times \hat x + \text{momentum} \times x_t$ ，其中 $\hat x$ 是估计值， $x_t$ 是新的观测值。

参数：

-   **num_features** – 输入大小 $(N, C, H, W)$ 中的 $C$ 
-   **eps** – 为数值稳定性添加到分母的值（当分母趋近于 $0$ 时会出现数值爆炸）。默认值：`1e-5`
-   **momentum** – 用于 running_mean 和 running_var 计算的值。可以设置 `None` 为累积移动平均（即简单平均）。默认值：`0.1`
-   **affine** – bool，当设置为 `True` 时，该模块具有可学习的仿射参数。默认：`True`
-   **track_running_stats** – bool，当设置为 `True` ，该模块跟踪运行的均值和方差，当设定为 `False` ，该模块不跟踪这些统计数据，并初始化统计缓冲区 `running_mean` 和 `running_var` 作为 `None` 。当这些缓冲区为 `None` 时，此模块始终使用 batch 统计信息。在训练和评估模式中。默认：`True`

参数说明：

-   `afine` ：

    -   当 `afine=true` 时， weight($\gamma$) 和 bias($\beta$) 将被使用，即进行缩放和平移

        举例：

        ````python
        import torch
        from torch import nn
        
        m = nn.BatchNorm2d(2,affine=True)
        # 通过梯度下降更新，初始化为不缩放、不平移
        print(m.weight)  	# tensor([1., 1.], requires_grad=True)
        print(m.bias)  		# tensor([0., 0.], requires_grad=True)
        
        # 操作 m.weight 和 m.bias ，你将看到网络产生不同的输出
        input = torch.arange(1*2*3*4, dtype=torch.float).view(1,2,3,4)
        output = m(input)
        print(output)
        ````

    -   当 `afine=true` 时， weight($\gamma$) 和 bias($\beta$) 都为 `None`

        举例：

        ```python
        m = torch.nn.BatchNorm2d(2,affine=False)
        print(m.weight) 		# None
        print(m.bias)				# None
        ```

-   `track_running_stats` ：如果为 `True` ，则针对每次 mini-batch 结合上次的历史信息动态统计，如果为 `false` ，则使用该次 mini-batch 的静态统计信息。

计算步骤：

1.   先对输入按照通道进行归一化， $\mathbf E(x)$ 为计算的均值， $\mathbf{Var}(x)$ 为计算的方差
2.   然后对归一化的结果按照通道进行缩放和平移，设置 `affine=True` ，即意味着 weight($\gamma$) 和 bias($\beta$) 将被使用

大小推导：

输入：$(N, C, H, W)$ ，输出：$(N, C, H, W)$

用法：

```python
m = nn.BatchNorm2d(100)  # 输入通道 100 
```



##### batch & mini-batch

这里顺便聊聊 batch 和 mini-batch 的区别：

-   **batch** ：整个数据集，遍历一次完整的数据集称为一个 **epoch** 。遍历完所有输入数据后才进行参数更新。
-   **mini-batch** ：将整个数据集分成几组，每次同时送入的一批数据称为一个 mini-batch 。每次迭代进行参数更新。

其中 mini-batch 的方案在现在应用更为广泛，它是以下两种方案的折中：

-   批梯度下降 Batch gradient descent ：

    -   遍历全部数据集算一次损失函数，然后计算梯度更新参数。这种方法每更新一次参数需要遍历整个数据集，计算量开销大，计算速度慢，不支持在线学习。但是由于梯度下降方向稳定，不容易震荡。

        <img src="2021-09-04-Pytorch-Building-Neural-Network.assets/image-20210920115442951.png" alt="image-20210920115442951" style="zoom:30%;" />

-   随机梯度下降 Stochastic gradient descent ：

    -   每输入一个数据就更新参数。这种方法速度比较快，但**不容易收敛**，会在最有点附近剧烈晃动，且梯度下降过程**震荡剧烈**。

        <img src="2021-09-04-Pytorch-Building-Neural-Network.assets/image-20210920115506580.png" alt="image-20210920115506580" style="zoom:30%;" />

为了结合上述两种方法的优势，克服各自的缺点，我们采用小批量梯度下降。一个 mini-batch 中的数据共同决定了本次梯度的方向，下降起来就不容易跑偏，减少了随机性。另一方面因为 mini-batch 的样本数与整个数据集相比小了很多，每次更新所需的计算量也不是很大。

<img src="2021-09-04-Pytorch-Building-Neural-Network.assets/image-20210920115526710.png" alt="image-20210920115526710" style="zoom:30%;" />



#### 2) nn.GroupNorm

```python
class torch.nn.GroupNorm(num_groups, num_channels, eps=1e-05, affine=True, device=None, dtype=None)
```

论文来源： [Group Normalization](https://arxiv.org/abs/1803.08494) 。
$$
y = \cfrac{x-\mathbf E[x]}{\sqrt{\mathbf{Var}[x]+\epsilon}} * \gamma + \beta
$$
输入通道被分成 `num_groups` 个组，每组有 `num_channels / num_groups` 个通道，有各自独立计算的均值和标准差。 $\gamma$ 和 $\beta$ 是针对每个通道可学习的

参数：

-   **num_groups** ( [*int*](https://docs.python.org/3/library/functions.html#int) ) – 将通道分成的组数
-   **num_channels** ( [*int*](https://docs.python.org/3/library/functions.html#int) ) – 输入中预期的通道数
-   **eps** – 为数值稳定性添加到分母的值。默认值：1e-5
-   **affine** – bool，当设置为 时 `True` ，该模块具有可学习的每通道仿射参数，初始化为 `1`（对于权重）和 `0`（对于偏差）。默认值：`True`。

大小推导：

输入：$(N, C, *)$ ，输出：$(N, C, *)$

用法：

```python
m = nn.GroupNorm(3, 6)  # Separate 6 channels into 3 groups
```





### 6. 循环层

#### 1) 





### Vision Layers（上采样）

#### 1) nn.PixelShuffle



#### 2) nn.Upsample



https://samuel92.blog.csdn.net/article/details/82855946

https://zhuanlan.zhihu.com/p/98081181



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





<br/>

**如果觉得本教程不错或对您有用，请前往项目地址 [https://github.com/Harry-hhj/Harry-hhj.github.io](https://github.com/Harry-hhj/Harry-hhj.github.io) 点击 Star :) ，这将是对我的肯定和鼓励，谢谢！**

<br/>



## 、参考文献

1.   [TORCH.NN](https://pytorch.org/docs/stable/nn.html)
2.   [pytorch中存储各层权重参数时的命名规则，为什么有些层的名字中带module.](https://blog.csdn.net/u014734886/article/details/106230535)
3.   [pytorch中文文档-torch.nn常用函数-待添加](https://www.cnblogs.com/wanghui-garcia/p/10775859.html)
4.   [pytorch的函数中的dilation参数的作用](https://www.cnblogs.com/wanghui-garcia/p/10775367.html)
5.   [pytorch卷积操作nn.Conv中的groups参数用法解释](https://blog.csdn.net/cxx654/article/details/109681004)
6.   [nn.ConvTranspose2d的参数output_padding的作用](https://www.cnblogs.com/wanghui-garcia/p/10791778.html)
7.   [Batch、Mini-batch和随机梯度下降的区别和Python示例](https://baijiahao.baidu.com/s?id=1665861710638558010&wfr=spider&for=pc)
8.   https://github.com/vdumoulin/conv_arithmetic
9.   [Pytorch池化层Maxpool2d中ceil_mode参数](https://blog.csdn.net/seungribariumgd/article/details/107066502)
10.   



-----

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

