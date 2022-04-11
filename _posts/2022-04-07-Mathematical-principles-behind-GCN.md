---
title: Mathematical principles behind GCN
author: Harry-hhj
date: 2022-04-06 20:12:00 +0800
categories: [Course, Machine Learning]
tags: [maths, ml, gcn]
math: true
mermaid: false
pin: false
image:
  src: https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-04-07-Mathematical-principles-behind-GCN.assets/cover.jpg
  width: 550
  height: 295
---



# GCN 背后的数学原理

## 一、定义

GNN 的更新公式为
$$
H^{(l+1)} = f(A, H^{(l)})
$$
即根据当前的隐藏状态，根据与其他节点的连接关系，进行某种映射，得到下一步隐藏状态。所有的 GNN 网络其实都是在设计 $f$ 。而 GCN 的 $f$ 定义如下
$$
H^{(l+1)} = \sigma(\hat D^{-\frac12} \hat A \hat D^{-\frac12} H^{(l)} \theta)
$$
这里的 GCN 考虑的是无向简单图（无向、无自环、无重边）。

$A$ 代表邻接矩阵， $D$ 代表度矩阵， $\hat A$ 代表添加了自环后的邻接矩阵，即 $\hat A = A + I$ ， $\hat D$ 代表添加了自环后的度矩阵， $\hat D = D + I$ 。 $\hat D^{-\frac12} \hat A \hat D^{-\frac12}$ 是 $\hat A$ 的对称归一化矩阵。

## 二、基础知识

### 1. 图理论

依据线性代数研究与图相关的矩阵的性质。

首先复习线性代数的一些知识：

-   特征值和特征向量：若 $A \vec x = \lambda \vec x$ ，且 $\vec x \ne \vec 0$ ，那么 $\vec x$ 称为 $A$ 的一个特征向量， $\lambda$ 是 $A$ 的一个特征值。

-   定理：如果一个矩阵是实对称阵，那么它一定有 $n$ 个特征值，并且这 $n$ 个特征值对应着 $n$ 个互相正交的特征向量，其中 $n$ 代表矩阵维度。数学表达如下：
    
    
    $$
    A = U \Lambda U^T\\
    UU^T = I\\
    \Lambda = diag(\lambda_1, \lambda_2, \dots, \lambda_n)\\
    $$
    
    
-   半正定矩阵：所有特征值都大于等于 $0$ 

-   二次型：给定一个矩阵 $A$ ，向量 $x$ 对于矩阵 $A$ 的二次型为 $\vec x^T A \vec x$ 

-   Rayleigh 熵：一个向量关于 $A$  的二次型和它关于 $I$ 的二次型的比值，即 $\frac{\vec x^T A \vec x}{\vec x^T \vec x}$ ，它与矩阵的特征值有着密切的联系。假设 $\vec x$ 是 $A$ 的特征向量，则可以证明 Rayleigh 熵等于矩阵的特征值。证明如下：
    
    
    $$
    \frac{\vec x^T A \vec x}{\vec x^T \vec x} = \frac{\vec x^T \lambda \vec x}{\vec x^T \vec x} = \frac{\lambda (\vec x^T \vec x)}{\vec x^T \vec x} = \lambda
    $$



然后我们研究和 GCN 相关的最重要的两个矩阵 $L$ 和 $L_{sym}$ 。

图的拉普拉斯矩阵定义为 $L = D - A$ ，其对称归一化矩阵为 $L_{sym} = D^{-\frac12} L D^{-\frac12}$ 。这两个矩阵都是实对称矩阵，因此它们都有 $n$ 个特征值和 $n$ 个正交的特征向量，且都为半正定矩阵。证明如下：
$$
\begin{aligned}
&若要证明 L 为半正定阵，则只要证明 \forall \vec x, \frac{\vec x^T L \vec x}{\vec x^T \vec x} \ge 0 ，则当 \vec x 为 L 的特征向量时，有 \lambda \ge 0\\
&又 \vec x^T \vec x \gt 0 ，故只需证明 \vec x^T L \vec x \ge 0\\
&构造 G_{(i,j)} = 
\begin{bmatrix}
\ddots &&&&\\
&1&\cdots&-1&\\
&\vdots&\ddots&\vdots&\\
&-1&\cdots&1&\\
&&&&\ddots\\
\end{bmatrix}
，则 L = \sum_{(i,j) \in E} G_{(i,j)}\\
&研究 \vec x^T G_{(i,j)} \vec x = \vec x^T 
\begin{bmatrix}
\vdots \\ x_i-x_j \\ \vdots \\ x_j-x_i \\ \vdots \\
\end{bmatrix}
= (x_i-x_j)^2\\
&故 \vec x^T L \vec x = \vec x^T (\sum_{(i,j) \in E} G_{(i,j)}) \vec x = \sum_{(i,j) \in E} (\vec x^T G_{(i,j)} \vec x) = \sum_{(i,j) \in E}(x_i-x_j)^2 \ge 0\\
&结论成立\\
\end{aligned}
$$


对于 $L_{sym}$ ，证明如下：
$$
\begin{aligned}
&\vec x^T L_{sym} \vec x = \vec x^T D^{-\frac12} L D^{-\frac12} \vec x = (\vec x^T D^{-\frac12}) L (D^{-\frac12} \vec x)\\
&依据之前的结论\\
&\vec x^T L_{sym} \vec x = \sum_{(i,j) \in E}(\frac{x_i}{\sqrt d_i}-\frac{x_j}{\sqrt d_j})^2 \ge 0\\
\end{aligned}
$$


这仅仅说明了 $L_{sym}$ 特征值是非负数，但其实我们可以证明更准确的一个范围 $[0,2]$ 。证明如下：
$$
\begin{aligned}
&构造 G_{(i,j)}^\prime = 
\begin{bmatrix}
\ddots &&&&\\
&1&\cdots&1&\\
&\vdots&\ddots&\vdots&\\
&1&\cdots&1&\\
&&&&\ddots\\
\end{bmatrix}
，则 \vec x^T G_{(i,j)}^\prime \vec x  = (x_i+x_j)^2\\
&定义 L^\prime = D + A = \sum_{(i,j) \in E} G_{(i,j)}^\prime，\\
&有 \vec x^T L^\prime \vec x = \sum_{(i,j) \in E} (x_i+x_j)^2\\
&又 L_{sym}^\prime = D^{-\frac12} L^\prime D^{-\frac12}，有 \vec x^T L_{sym}^\prime \vec x = \sum_{(i,j) \in E}(\frac{x_i}{\sqrt d_i}+\frac{x_j}{\sqrt d_j})^2 \ge 0\\
&由于 L_{sym}^\prime = I + D^{-\frac12} A D^{-\frac12}，代入上面的不等式得到\\
&\vec x^T L_{sym}^\prime \vec x = \vec x^T (I + D^{-\frac12} A D^{-\frac12}) \vec x = \vec x^T \vec x + \vec x^T D^{-\frac12} A D^{-\frac12} \vec x  \ge 0\\
&进一步变换可得 2 \vec x^T \vec x \ge \vec x^T (I - D^{-\frac12} A D^{-\frac12}) \vec x ，有\\
&2 \vec x^T \vec x \ge \vec x^T L_{sym} \vec x ，即 \frac{\vec x^T L_{sym} \vec x}{\vec x^T \vec x} \le 2\\
\end{aligned}
$$



### 2. 傅立叶变换

什么是傅立叶变换？傅立叶变换就是从另一个域研究问题。例如，声波在时域是一个复杂的波形，根据傅立叶变换，任何函数都可以表达为一系列的正弦波，因此将声波转换到频域，表现为在不同频率下不同振幅的正弦波。此时我们就很容易区分男声和女声（假设女声频率普遍高于男声），而这在时域上是不容易操作的。这里推荐一篇讲解傅立叶变换的[教程](https://zhuanlan.zhihu.com/p/19763358)。

举个例子，假设我们有两个多项式需要相乘 $f(x) = a_0 + a_1x + \cdots + a_nx^n$ ， $g(x) = b_0 + b_1x + \cdots + b_nx^n$ 。简单的暴力方法嵌套遍历两组系数，复杂度是 $O(n^2)$ ，而通过 FFT ，一个 $n$ 次多项式可以由 $n+1$ 个点确定： $f(x) \Leftrightarrow (x_1, f(x_1)), \dots, (x_n, f(x_n))$ ， $g(x) \Leftrightarrow (x_1, g(x_1)), \dots, (x_n, g(x_n))$ 。此时，计算两个多项式就是 $n$ 个点相乘，复杂度 $O(n)$ ，而这种变换的复杂度为 $O(n \log n)$ ，所以最终算法的复杂度为 $O(n \log n)$ ，效率提升。

什么是图上的傅立叶变换，又为什么需要？因为不同于图像（image），图（graph）是一个非欧氏空间，节点的邻居数量不固定，我们无法确定 kernel 的形状，因此在空间域做图的卷积是非常困的。在图上进行傅立叶变换，就是将图变换到一个更容易进行卷积的域中。

我们看看 $Lx$ 做了什么。 $x$ 的每一行都可以看作是一个节点的 feature ，可以看出这种乘法实现了和邻居特征的聚合。由于 $L$ 是一个实对称、半正定阵，因此 $Lx = U \Lambda U^T x$ ，其中 $U$ 和 $U^T$ 都是正交阵。而一个向量乘以一个正交阵，就是一种基底变换。因此 $U \Lambda U^T x$ 表示先将 $x$ 变换基底表示，然后在不同维度上进行放缩，然后再变换回原来的坐标系。

但实际上对拉普拉斯矩阵进行特征分解需要 $O(n^2)$ 的复杂度。GCN 所做的就是对带特征分解的傅立叶变换进行限制，寻找一种不需要特征分解、复杂度与边的数量成线性关系的方法。



## 三、图卷积公式推导

首先需要定义图上的卷积操作。假设我们有一个关于图的邻接矩阵的函数 $F(A) \mapsto L~or~L_{sym}$ ，输入是图的邻接矩阵，输出是一种关于图的性质比较好的矩阵，例如 $L~or~L_{sym}$ ，只要它满足实对称阵都还算不错。我们定义 $F(A) = U \Lambda U^T$ 。图上的卷积操作 $g_\theta * x$ 就可以定义为 $U g_{\theta}(\Lambda) U^T x$ ，其中限定 $g_{\theta}(\lambda)$ 是一个多项式函数 $g_{\theta}(\Lambda) = \theta_0 \Lambda^0 + \theta_1 \Lambda^1 + \cdots + \theta_n \Lambda^n+ \cdots$ ，这样的好处是 $U g_{\theta}(\Lambda) U^T = g_{\theta}(U \Lambda U^T) = g_\theta(F(A))$ ，就不需要再对 $A$ 做特征分解了。证明如下：
$$
\begin{aligned}
&(U \Lambda U^T)^k = U \Lambda U^T \cdots U \Lambda U^T = U \Lambda^k U^T\\
&U g_{\theta}(\Lambda) U^T = U \sum_k(\theta_k \Lambda^k) U^T = \sum_k(\theta_k (U \Lambda^k U^T)) = g_{\theta}(U \Lambda U^T)\\
\end{aligned}
$$
但在实际操作中，我们并不是用系数的形式拟合多项式的，因为随着 $n$ 的变大这种方法存在梯度消失和梯度爆炸的问题。事实上使用切比雪夫多项式 $\Gamma_n(x) = 2\Gamma_{n-1}(x) - \Gamma_{n-2}(x),~\Gamma_0(x) = 1, \Gamma_1(x) = x$ ，它存在一个非常好的性质 $\Gamma_n(\cos \theta) = \cos(n\theta)$ ，即不论 $n$ 多大，其数值上都有一个稳定的摆动趋势。但它的缺点是要求自变量的取值范围为 $[-1,1]$ ，即 $\lambda \in [-1,1]$ 。为了满足这个性质，只需要将 $L_{sym}$ 减去 $I$ 即可，所以最终选择 $F(A) = L_{sym}-I$ 。因此


$$
\begin{aligned}
g_\theta * x
&= U (\sum_{k=0}^K \theta_k \Gamma_k(\Lambda)) U^T x\\
&= \sum_{k=0}^K \theta_k U \Gamma_k(\Lambda) U^T x\\
&= \sum_{k=0}^K \theta_k \Gamma_k(U \Lambda U^T) x\\
&= \sum_{k=0}^K \theta_k \Gamma_k(L_{sym}-I) x\\
\end{aligned}
$$


但实际上这个运算的复杂度还是很高，需要计算矩阵的 $k$ 次方。因此 GCN 的实际做法是使用一阶近似


$$
\begin{aligned}
g_\theta * x
&\approx \theta_0 \Gamma_0(L_{sym}-I)x + \theta_1 \Gamma_1(L_{sym}-I)x\\
&= \theta_0 x + \theta_1 (L_{sym}-I)x\\
&= \theta_0 x + \theta_1 (I-D^{-\frac12} A D^{-\frac12}-I)x\\
&= \theta_0 x + \theta_1 D^{-\frac12} A D^{-\frac12}x\\
\end{aligned}
$$


在 GCN 中还应用了一些正则化，令 $\theta_1 = -\theta_0$ ，有 $g_\theta * x = \theta_0 (I + D^{-\frac12} A D^{-\frac12})x$ ，则 $\theta_0$ 也可以省略。其次使用 renormalize ，将 $I$ 移入，得到 $g_\theta * x = D^{-\frac12} \hat A D^{-\frac12}x$ 。至于原因，只是因为它的效果更好。



## 四、参考资料

1.   https://www.bilibili.com/video/BV1Vw411R7Fj





<br/>

**如果觉得本教程不错或对您有用，请前往项目地址 [https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io](https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io) 点击 Star :) ，这将是对我的肯定和鼓励，谢谢！**

<br/>



---

作者：Harry-hhj，Github主页：[传送门](https://raw.githubusercontent.com/Harry-hhj)

