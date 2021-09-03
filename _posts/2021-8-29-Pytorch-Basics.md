---
title: PyTorch-Basics
author: Harry-hhj
date: 2021-08-29 11:12:00 +0800
categories: [Tutorial, PyTorch]
tags: [getting started]
math: true
mermaid: true
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-29-Pytorch-Basics.assets/pytorch.jpeg?raw=true
  width: 480
  height: 230
---



# PyTorch - Basics

## PyTorch 是什么

**Torch** 是一个有大量机器学习算法支持的科学计算框架，是一个与 Numpy 类似的张量 Tensor 操作库。

**PyTorch** 是一个基于 Torch 的 Python 开源机器学习库，用于自然语言处理等应用程序。

优点：

-   作为 Numpy 的替代品，可以使用 **GPU** 的强大计算能力
-   提供最大的**灵活性**和高速的深度学习研究平台

缺点：

-   全面性：目前 PyTorch 还不支持快速傅里叶、沿维翻转张量和检查无穷与非数值张量
-   性能：针对移动端、嵌入式部署以及高性能服务器端的部署其性能表现有待提升
-   文档：社区还没有那么强大，其 C 库大多数没有文档

## 环境搭建

Miniconda3 + PyTorch

首先去 Miniconda [官网](https://docs.conda.io/en/latest/miniconda.html)下载对应系统和 Python 版本的安装包，打开终端运行脚本，按照指令完成 conda 环境搭建。然后前往 PyTorch [官网](https://pytorch.org)，按照需求选择，其中 Language 选择 `Python` ，Compute Platform 根据自己的硬件选择，有 Nvidia GPU 的选择 CUDA 版，有 AMD GPU 的选择 ROC 版（还需另外安装 ROC 环境），不需要或者没有 GPU 的选择 CPU 版。注意 MacOS 系统只能安装 CPU 版。复制命令并在命令行执行即可安装。



## 预备知识

### Tensor

Tensors （张量），与 Numpy 中的 ndarrays 类似，但是在 PyTorch 中 Tensors 可以使用 GPU 进行计算。

在讨论其语法之前，先来说说什么是张量。

1.   标量：一个单独的数
2.   向量：一列有序排列的数，通过次序中的索引可以确定一个数
3.   矩阵：二维数组，每个元素被两个索引唯一确定
4.   张量：几何代数中定义的张量是基于向量和矩阵的推广，通俗一点理解的话，标量是零阶张量，矢量是一阶张量，矩阵是二阶张量

举个例子，对于任意一张彩色照片，可以表示成一个三阶张量，三个维度分别是图片的高度、宽度和 RGB 通道。下图是一个白色图片的示例：

![img](2021-8-29-Pytorch-Basics.assets/640.jpeg)

我们继续将这一例子拓展：即：我们可以用四阶张量表示一个包含多张图片的数据集，这四个维度分别是：图片在数据集中的编号，图片高度、宽度，以及 RGB 通道。这种数据表示形式在计算机视觉中非常常见，你可以在这里先有个印象。

**张量**在深度学习中是一个很重要的概念，因为它是一个深度学习框架中的一个核心组件，后续的所有运算和优化算法几乎都是基于张量进行的。



常用操作：

```python
import torch
# 创建一个未初始化的 Tensor
x = torch.empty(5, 3)

# 创建一个随机初始化的 Tensor
x = torch.rand(5, 3)  # torch.rand(*sizes, out=None)->Tensor: [0, 1) 均匀分布
x = torch.randn(5, 3)  # torch.randn(*sizes, out=None)->Tensor: 标准正态分布（均值为 0 ，方差为 1 ，即高斯白噪声）
x = torch.randint(1, 4, (2, 3, 2))  # torch.randint(low = 0, high, size, out=None, dtype=None)->Tensor: 整数范围 [low, high)
x = torch.randperm(3)  # torch.randperm(n, out=None, dtpe=torch.int64)->LongTensor: 1 到 n 这些数的一个随机序列

# 创建 Tensor 并使用现有数据初始化
x = torch([5.5, 3])

# 其他特殊的创建 Tensor 的方法
x = torch.zeros(5, 3, dtype=torch.long)  # 全 0 
x = torch.ones(5, 3, dtype=torch.double)  # 全 1 
x = torch.eye(5, 3)  # 对角线为 1 
x = torch.arange(2, 10, 2)  # torch.arange(s, e, step)->Tensor: 从 s 到 e ，步长为 step 
x = torch.linspace(2, 10, 3)  # torch.linspace(s, e, step)->Tensor: 从 s 到 e ，均匀切分成 steps 份
x = torch.normal(0, 3, (5, 3))  # torch.normal(mean:float, std:float, size:tuple)->Tensor: 均值为 mean ，方差为 std ，大小为 size 
x = torch.Tensor(5, 3).uniform_(-1, 1)  # 均匀分布 [from, to)

# new_* 方法来创建对象
x = torch.new_ones(5, 3, dtype=torch.double)

# 根据现有的张量创建张量，重用输入张量的属性，例如 dtype ，除非设置新的值进行覆盖
x = torch.randn_like(x, dtype=torch.float)    # 覆盖 dtype ，但 size 相同

# 获取 size
x.size()  # torch.Size 返回值是 tuple 类型, 所以它支持 tuple 类型的所有操作
```



### 运算

```python
# 加法
x = torch.rand(5, 3)
y = torch.rand(5, 3)
# Method 1
z = x + y
# Method 2
z = torch.add(x, y)
# Method 3
z = torch.empty(5, 3)
torch.add(x, y, out=z)  # 提供输出tensor作为参数
# Method 4
y.add_(x)  # 会改变原变量的值

# 索引
x[:, 1]  # 与 Numpy 索引方式相同
x = torch.randn(5, 3).index_select(0, torch.linspace(0, 4, 2, dtype=torch.int32))  # .index_select(dim:int, index:Tensor(int32/64))->Tensor: 从 dim 维选取 index 的数据
x = x.masked_select(x>0)  # .masked_select(mask)->Tensor: 选取掩膜为 1 处的元素，不保留原始位置信息
x = x.nonzero()  # 返回非零元素的下标

# torch.gather(input, dim, index:torch.long, out=None)->Tensor：根据index，在dim维度上选取数据
# out[i][j][k]...[i+dim]...[z] = input[i][j][k]...[index[i][j][k]...[z]]_{i+dim}...[z]
t = torch.Tensor([[1,2],[3,4]])
torch.gather(t, 1, torch.LongTensor([[0,0],[1,0]]))	# [[1 1] [4 3]]
																							# index 中元素范围 [0, n_dim-1]，用来指定第 dim 维的选取的位置
# 关于 torch.gather 的更多用法请参考教程：https://zhuanlan.zhihu.com/p/352877584

# 改变张量的维度和大小
x = torch.randn(4, 4)
y = x.view(16)  # x 和 y 共享数据
z = x.view(-1, 8)  #  size：-1 从其他维度推断

# 
x = x.squeeze()
unsqueeze()
```

注意：

-   任何以 `_` 结尾的操作都会用结果**替换原变量**。例如：`x.copy_(y)` ， `x.t_()` ，都会改变 `x` 。
-   view() 返回的新 Tensor 与原 Tensor 虽然可能有不同的 size ，但是是**共享 data 的**（ view 仅仅是改变了对这个张量的观察角度，内部数据并未改变）。如果需要副本先使用 `.clone()` 。



### Python 数据类型转换

如果你有只有一个元素的张量，使用 `.item()` 来得到 Python 数据类型的数值。

```python
x = torch.randn(1)
print(x.item())
```



### Numpy 转换

将一个 Torch Tensor 转换为 NumPy 数组是一件轻松的事，反之亦然。

Torch Tensor 与 NumPy 数组**共享底层内存地址**，修改一个会导致另一个的变化。

```python
# Torch Tensor -> NumPy数组
a = torch.ones(5)
b = a.numpy()
a.add_(1)  # 此时 b 发生变化
# NumPy Array -> Torch Tensor
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)  # 此时 a 发生变化
```



### Broadcasting

当对两个形状不同的 Tensor 按元素运算时，可能会触发广播（broadcasting）机制：先适当复制元素使这两个 Tensor 形状相同后再按元素运算。

```python
x = torch.arange(1, 3).view(1, 2)
y = torch.arange(1, 4).view(3, 1)
z = x + y  # torch.Size([3, 2])
```



### CUDA

使用 `.to` 方法 可以将 Tensor 移动到任何设备中。

```python
# is_available 函数判断是否有 cuda 可以使用
# `torch.device` 将张量移动到指定的设备中
if torch.cuda.is_available():
    device = torch.device("cuda")          # 一个 CUDA 设备对象
    y = torch.ones_like(x, device=device)  # 直接从 GPU 创建张量
    x = x.to(device)                       # 或者直接使用 `.to("cuda")` 将张量移动到 cuda 中
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # `.to` 也会对变量的类型做更改
```



<br/>

更多内容，请查看[官网教程](https://pytorch.org/docs/torch)。



## Autograd：自动求导机制

PyTorch 中所有神经网络的核心是 autograd 包，它为张量上的所有操作提供了**自动求导**。 它是一个在**运行时定义**的框架，这意味着反向传播是根据你的代码来确定如何运行，并且每次迭代可以是不同的。

在自动求导计算中有两个重要的类：

-   Tensor
    -   如果设置 `.requires_grad` 为 `True`，那么将会追踪所有对于该张量的操作。当完成计算后通过调用 `.backward()`，自动计算所有的梯度，这个张量的所有梯度将会自动积累到 `.grad` 属性。
    -   为了防止跟踪历史记录（和使用内存），可以将代码块包装在 `with torch.no_grad():` 中。 这在评估模型时特别有用，因为模型可能具有 `requires_grad = True` 的可训练参数，但是我们不需要梯度计算。
-   Function
    -   Tensor 和 Function 互相连接并生成一个**非循环图**，它表示和存储了完整的计算历史。 每个张量都有一个 `.grad_fn` 属性，这个属性引用了一个创建了 Tensor 的 Function ，即该 Tensor 是不是通过某些运算得到的，若是，则 grad_fn 返回一个与这些运算相关的对象（除非这个张量是用户手动创建的，即，这个张量的 `grad_fn` 是 `None` ）。
    -   如果需要计算导数，你可以在 Tensor 上调用 `.backward()` 。 如果 Tensor 是一个标量（即它包含一个元素数据）则不需要为 `backward()` 指定任何参数， 但是如果它有更多的元素，你需要指定一个 `gradient`  参数来匹配张量的形状。

注意：在其他的文章中你可能会看到说将 Tensor 包裹到 Variable 中提供自动梯度计算， Variable 这个在 0.41 版中已经被标注为过期了，现在可以直接使用 Tensor ，官方文档在[这里](https://pytorch.org/docs/stable/autograd.html#variable-deprecated)。



### 打开关闭微分

```python
x = torch.ones(2, 2, requires_grad=True)  # x.grad_fn = None
y = x + 2  # y.grad_fn = <AddBackward0 object at 0x...>
z = y * y * 3  # z.grad_fn = <MulBackward0 object at 0x...>
out = z.mean()  # out.grad_fn = <MeanBackward0 object at 0x...>
```

`x` 是直接创建的，所以么有 `grad_fn` ，` y` 作为操作的结果被创建，因此具有 `grad_fn` 。像 `x` 这样的节点被称为**叶子节点**，叶子节点对应的 `grad_fn` 是 `None` 。

输入的 `requires_grad` 在**没有给定参数的情况下**默认是 `False` ，可以通过 `requires_grad_()` 来改变张量的 `requires_grad` 属性。如果输入的  `requires_grad` 是 `False` ，那么之后所有计算结果的变量的  `requires_grad` 属性都将是 `False` ，且 `grad_fn` 为 None。



### 求梯度

在调用 `y.backward()` 时，如果 `y` 是标量，则不需要为 `backward()` 传入任何参数；否则，需要传入一个与 `y` 同形的 `Tensor` 。因为不允许张量对张量求导，只允许标量对张量求导，求导结果是和自变量同形的张量。

在数学上，如果我们有向量值函数  $\overrightarrow y = f(\overrightarrow x)$，且 $\overrightarrow y$ 关于 $\overrightarrow x$ 的梯度是一个雅可比矩阵（Jacobian matrix）：
$$
J = 
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1}&\cdots&\frac{\partial y_1}{\partial x_n}\\
\cdots&\cdots&\cdots\\
\frac{\partial y_m}{\partial x_1}&\cdots&\frac{\partial y_m}{\partial x_n}\\
\end{pmatrix}
$$
一般来说，`torch.autograd` 就是用来计算 vector-Jacobian product 的工具。也就是说，给定任一向量 $\overrightarrow v = (v_1\ v_2\ \cdots\ v_m)^T$ ，计算 $v^T \cdot J$ 。如果 $v$ 恰好是标量函数 $l = g(\overrightarrow y)$ 的梯度，也就是说 $v = (\frac{\partial l}{y_1}\ \cdots \frac{\partial l}{y_m})^T$ ，那么根据链式法则，vector-Jacobian product 是  $l$ 关于 $\overrightarrow x$ 的梯度：
$$
J^T \cdot v = 
\begin{pmatrix}
\frac{\partial y_1}{\partial x_1}&\cdots&\frac{\partial y_1}{\partial x_n}\\
\cdots&\cdots&\cdots\\
\frac{\partial y_m}{\partial x_1}&\cdots&\frac{\partial y_m}{\partial x_n}\\
\end{pmatrix}
\begin{pmatrix}
\frac{\partial l}{\partial x_1}\\
\cdots\\
\frac{\partial l}{\partial x_n}\\
\end{pmatrix}
$$
（注意，$v^T \cdot J$ 给出了一个行向量，可以通过 $J^T \cdot v$ 将其视为列向量）

vector-Jacobian product 这种特性使得**将外部梯度返回到具有非标量输出的模型**变得非常方便。

以下是两个例子：

-   标量求导

    ```python
    x = torch.ones(2, 2, requires_grad=True)
    y = x + 2
    z = y * y * 3
    out = z.mean()
    out.backward()  #  因为 out 是一个纯量（scalar），out.backward() 等于 out.backward(torch.tensor(1))
    print(x.grad)  # tensor([[4.5000, 4.5000], [4.5000, 4.5000]])
    ```

-   非标量求导

    ```python
    x = torch.randn(3, requires_grad=True)
    y = x * 2
    while y.data.norm() < 1000:
        y = y * 2
    gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
    y.backward(gradients)
    print(x.grad)
    ```

    在这个情形中，`y` 不再是个标量， `torch.autograd` 无法直接计算出完整的雅可比矩阵，但是如果我们只想要 vector-Jacobian product ，只需将向量作为参数传入 `backward` 。



<br/>

更多内容，请查看[官网教程](https://pytorch.org/docs/autograd)。









## 参考文档

1.   [标量，向量，矩阵与张量](https://www.jianshu.com/p/abe7515c6c7f)
2.   [torch.rand()、torch.randn()、torch.randint()、torch.randperm()用法](https://blog.csdn.net/leilei7407/article/details/107710852)
3.   [我对torch中的gather函数的一点理解](https://zhuanlan.zhihu.com/p/110289027)
4.   [pytorch简介和准备知识](https://zhuanlan.zhihu.com/p/97234180)



-----

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

