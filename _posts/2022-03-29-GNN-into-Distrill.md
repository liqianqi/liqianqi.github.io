---
title: GNN
author: Harry-hhj
date: 2022-03-29 10:20:00 +0800
categories: [Tutorial, GNN]
tags: [gnn, graph]
math: true
mermaid: false
pin: false
image:
  src: https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/cover.png
  width: 640
  height: 320
---



# GNN

>   https://distill.pub/2021/gnn-intro/

## 一、什么是图

图 (graph) 是表示实体 (entities) 间的一些关系。实体就是顶点 (nodes) ，关系就是边 (edges) 。

现在定义如下标记：顶点 V 、边 E 、全局 U ，这些标记所蕴含的信息就叫 attribute 。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325112206754.png" alt="image-20220325112206754" style="zoom:50%;" />

图一般分为两类：有向图和无向图。例如，微信上两个人互为朋友，这种关系是没有方向的，但在 B 站上你关注了主播，主播却没有关注你，这种关系就是有方向的。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325112610688.png" alt="image-20220325112610688" style="zoom:50%;" />

那么数据是如何表示成图的？

-   图片

    首先观察我们熟悉的图像。假定有一张 $244\times244\times3$ 的图像，将其看成一个图（非图像），那么每个像素就是一个顶点，像素间的邻接关系就是边。

    <img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325114104486.png" alt="image-20220325114104486" style="zoom:50%;" />

    中间的矩阵叫做邻接矩阵 $\mathbf A$ (adjacency matrix) ，它的行和列都是顶点。
    $$
    \begin{cases}
    a_{ij} = 1 & if ~ v_i\text{ is connected with } v_j\\
    a_{ij} = 0 & otherwise
    \end{cases}
    $$
    邻接矩阵通常是很大且稀疏的矩阵。

-   文本

    文本可以看作是一个序列，可以将其中的每一个词作为顶点，一个词和下一个词时间有一个有向边。

    <img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325114623963.png" alt="image-20220325114623963" style="zoom:50%;" />

-   分子图

    每一个原子表示成图里的一个点，原子间的化学键表示成一条边。例如咖啡因：

    <img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325155408826.png" alt="image-20220325155408826" style="zoom:50%;" />

-   社交网络

    以下是《奥塞德》所有任务的交互图。任何人物如果同时出现在一个场景中，就会在他们之间连一条边。

    <img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325160306565.png" alt="image-20220325160306565" style="zoom:50%;" />

-   引用图

    文章之间的引用关系被表示为有向边。

实际中我们碰到的图的平均大小如下：

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325160527491.png" alt="image-20220325160527491" style="zoom:50%;" />

在我们把数据表示为图之后，我们看看可以在这些图上定义那些问题。主要有三大类问题：图级别 (graph-level) 的、顶点级别 (node-level) 的和边级别 (edge-level) 的。

-   图级别的任务：判断整个图的性质。

    比如下图中哪个图含有两个环。

    <img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325161000563.png" alt="image-20220325161000563" style="zoom:50%;" />

-   顶点级别的任务：判断节点在图中的角色。

    比如通过两个跆拳道老师和学生打过比赛的图，判断某个学生属于哪个老师管辖。

    <img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325161354733.png" alt="image-20220325161354733" style="zoom:50%;" />

-   边级别的任务

    比如对于一张图片，先进行语义分割，将人物和背景提取出后，判断任务之间是什么关系，即学习顶点之间边的属性。

    <img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325161646052.png" alt="image-20220325161646052" style="zoom:50%;" />

    <img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325161714275.png" alt="image-20220325161714275" style="zoom:50%;" />

那么将图用到机器学习上有哪些挑战呢？神经网络使用图的一个最核心的问题是如何表示图，使得它与神经网络是兼容的。一个图上有四种信息：

-   顶点的属性
-   边的属性
-   全局信息
-   连接性

前面三个都可以用一个向量来表示，但表示连接性更为复杂。使用邻接矩阵虽然可以达到效果，但是它会非常大，而其中存在的边却不多。或许我们可以使用稀疏矩阵解决存放的问题，但是稀疏矩阵的高效运算尤其是如何在 GPU 上高效运算是一个很难的问题。另一个问题是对于邻接矩阵，交换任意的行和列都不会产生影响，因此神经网络需要能够处理这些看上去完全不同却代表同一张图的邻接矩阵。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325162657500.png" alt="image-20220325162657500" style="zoom:50%;" />

假设有四个顶点和四条边，以下是所有可能的邻接矩阵。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325162853661.png" alt="image-20220325162853661" style="zoom:50%;" />

如果我们既想高效存储，又想不受存储顺序的影响，可以采用顶点属性、边属性、全局属性都采用标量/向量来表示，边的关系采用邻接列表的形式，元素 $i$ 表示第 $i$ 条边 $e_i = (v_s, v_e)$ ，表示连接的两个节点。

![image-20220325163131107](https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325163131107.png)

那么给定这种输入的形式，神经网络应该如何处理？

## 二、图神经网络 (Graph Neural Networks)

GNN 是一个对图上的所有属性（包括顶点、边和全局的上下文 (global-context) ）进行的可以优化的变换，这个变换能够保持图的对称信息，即改变节点的排序后，结果不变。接下来我们使用信息传递网络框架 (“message passing neural network” framework) 搭建 GNN 。GNN 的输入是图，输出也是图，它会变换节点、边、全局的属性嵌入 (embeddings) ，但不会改变图的连接性。

### 1. 最简单的 GNN

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325164402314.png" alt="image-20220325164402314" style="zoom:50%;" />

对于顶点、边、全局向量，我们分别构造一个多层感知机 (MLP) ，其输入大小和输出大小一致。这三个 MLP 就组成了 GNN 的一个层。经过这一层，图的属性将被更新，但是图的结果不发生变化。由于 MLP 是每个向量独自作用的，它不会考虑顺序，因此顶点排序的改变不会改变结果。

接下来考虑最后一层的输出如何得到我们要的预测值。假设我们想对顶点做预测，这其实和一般的 NN 没有区别。比如之前提到的学生归属哪个老师的问题，我们在之后加入一个输出维度为 2 的全连接层，再加上一个 softmax ，就可以完成任务。这里所有顶点嵌入共享一个全连接层的参数。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325165317817.png" alt="image-20220325165317817" style="zoom:50%;" />

我们将问题稍微复杂化一些。假设我们还是想要分类顶点，但是每个顶点没有嵌入向量。此时我们会用到一个技术 —— 汇聚 (pooling) 。我们会利用与它有关的边和全局信息来生成它的嵌入向量，这里假设所有属性向量的长度相同，如果不同我们还需要先进行投影变换。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325165843837.png" alt="image-20220325165843837" style="zoom:50%;" />

然后我们就可以用新得到的向量预测。以下分别是没有顶点属性和没有边属性进行预测的例子。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325170707800.png" alt="image-20220325170707800" style="zoom:50%;" />

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220325170750305.png" alt="image-20220325170750305" style="zoom:50%;" />

因此，不管缺少哪一类属性，都可以通过汇聚操作的到想要的属性值。

于是，我们得到了一个最简单的 GNN 模型。给定一个输入的图，首先进入一系列的 GNN 层，每个层里有 3 个 MLP ，对应不同的属性，输出会得到结构不变、属性已经发生变化的一个图。然后根据需要对哪一个属性做预测，添加对应的输出层完成预测，其中如果缺失信息就加入合适的汇聚层。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326113534840.png" alt="image-20220326113534840" style="zoom:50%;" />

这个模型会有很大的局限性，主要在于 MLP 的位置没有使用图结构信息，每个顶点进入 MLP 时仅考虑自身信息。

接下来我们考虑如何改进，放入图的信息，要用到的技术是之前提到过的信息传递。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326135539990.png" alt="image-20220326135539990" style="zoom:50%;" />

我们将某个顶点的向量和邻居节点的向量相加，得到汇聚的向量，然后对这个向量进行更新 $f$ 。这一步骤类似于卷积操作，唯一不同的是所有邻居顶点的权重都为 $1$ 。当这种层叠加时，顶点的感受野增大，融合的不仅是邻居的特征。我们规定符号 $\rho_{_{V_n \to V_n}}$ 表示顶点和 1 近邻的信息传递过程。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326141251502.png" alt="image-20220326141251502" style="zoom:50%;" />

当某种属性缺失时，我们之前在最后从别的属性汇聚过来弥补这个属性。在中间层我们也可以通过信息传递完成更新。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326141518065.png" alt="image-20220326141518065" style="zoom:50%;" />

注意这种信息传递的先后顺序会导致结果的不同，目前没有研究证明优劣。

-   先顶点后边学习
-   先边后顶点学习
-   交替更新 (Weave Layer) ：node to node (linear), edge to edge (linear), node to edge (edge layer), edge to node (node layer) 。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326144129489.png" alt="image-20220326144129489" style="zoom:50%;" />

上面我们讲述了顶点和边之间的信息交换，接下来我们看一下全局信息是怎么做的。首先为什么要有全局信息？在没有全局信息的情况下，顶点或边的视图只有邻居，当图很大且稀疏时，两个很远的点需要通过很多次更新才能交互。解决方法就是使用图的全局表示 $U$ ，也称为 master node 或 context vector 。这个虚拟的顶点与所有的顶点和边相连，这在图上想象会非常抽象。因为它与所有的顶点和边相连，因此在做顶点或边信息传递时，这个顶点的信息也是会考虑的。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326145111832.png" alt="image-20220326145111832" style="zoom:50%;" />

此时，所有的三类属性都学到了对应的向量，并进行过了消息传递。因此我们可以在池化期间通过调节我们感兴趣的属性相对于其余属性的信息来利用它们。一种可能的方式是将向量 concat 起来，另一种可能的方式是通过线性映射将不同属性向量映射到相同的空间或应用特征调制层 (feature-wise modulation layer) 。

## 三、实验

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326150948422.png" alt="image-20220326150948422" style="zoom:50%;" />

这一部分需要在浏览器上自行实验，我们看一下作者关于模型超参数选择的解释。

先来看看模型大小（参数数量）和表现（AUC）之间的关系。当模型参数变多时， AUC 的上限是增加的，但是如果模型超参数设置得不好，参数增多也不会有更好的效果。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326152215414.png" alt="image-20220326152215414" style="zoom:50%;" />

固定某种向量的长度，变换其他的参数，观察模型的表现。这里使用了箱线图 (box plot) ，其中横线表示中值， bar 是 25% - 75% 的数据，最高/低的点是最大/小值。我们会希望中值越高越好， bar 也不要太长（太敏感）。从图中看来，维度越大，效果稍微好一点，但不是很明显。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326153125723.png" alt="image-20220326153125723" style="zoom:50%;" />

接下来观察不同的层数对精度的影响。 $x$ 轴是学习参数的个数， $y$ 轴是测试精度。可以看出，层数越少，学习参数越少，从左图看不出精度的差别。但在箱线图中，随着层数的增加，精度也在增加。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326153557786.png" alt="image-20220326153557786" style="zoom:50%;" />

接下来观察不同聚合操作的影响。三者几乎没有什么区别。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326154122691.png" alt="image-20220326154122691" style="zoom:50%;" />

最后观察需要在哪些属性上传递信息。绿色表示不聚合任何信息，也就是我们之前讨论的最简单的 GNN ，效果是最差的。随着不断加入各种信息传递，模型效果越来越好。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326154229031.png" alt="image-20220326154229031" style="zoom:50%;" />

可以看出 GNN 对超参数还是比较敏感的，可以调节的参数有：层数、嵌入向量大小、汇聚操作类型、如何进行信息传递。

## 四、GNN 相关

### 1. 其他类型的图

-   mutigraphs

    定点之间有多种类型的边

-   hierarchical graphs

    图是分层的，一些顶点是一个子图。不同的图结构在神经网络如何进行信息汇聚产生一定的影响。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326155654540.png" alt="image-20220326155654540" style="zoom:50%;" />

### 2. 如何对图进行采样 (Sampling) 和 Batching

之前我们讲过，假设有很多层 GNN ，最后一层的顶点可以看到很广的范围，在连通性足够的情况下，可能会看到整个图的信息。在计算梯度时，我们需要把所有的中间变量存储下来，这可能导致最后的计算代价时无法承受的。因此，我们需要对图进行采样，即我们把图拆封成子图，在子图上进行信息汇聚。有几种采样的方法：

-   Random node sampling ：随机采样一些点，找出这些点的邻居，在这些子图上进行计算。通过控制每次采样多少个点，避免图过大，能够存入内存。
-   Random walk sampling ：从某一个顶点，随机找到它的一条边，找到下一个顶点，通过规定最多随机走多少步，获得一个子图。
-   Random walk with nerghborhood ：随机游走后找出所有顶点的邻居。
-   Diffusion Sampling ：选择一个节点，找出它的 k 阶邻，即进行宽度遍历，取出子图。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326160641089.png" alt="image-20220326160641089" style="zoom:50%;" />

具体哪种方法有效还是取决于数据集。

跟采样相关的问题是进行 batching 。我们不希望对每一个顶点逐步更新，其计算量太小，不利于并行。但是每个顶点的邻居数量是不一样的，如何合并成一个规整的张量是一个有挑战性的问题。

### 3. Inductive biases

任何神经网络都基于一些假设。比如，CNN 假设空间平移不变性， RNN 假设时间平移不变性。 GNN 的假设是保持了图的对称性。

### 4. 不同的汇聚操作

其实 `sum` 、 `mean` 、 `max` 没有一种是非常理想的。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326161847191.png" alt="image-20220326161847191" style="zoom:50%;" />

### 5. GCN 作为子图的函数近似

GCN 如果有 k 个层，每个层看 1 阶邻，最后每一个顶点看到的是一个 k-step 的子图，即这个顶点的表示就是这个子图的 embedding 。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326162249214.png" alt="image-20220326162249214" style="zoom:50%;" />

### 6. 可以将点和边对偶 (Dual)

根据图论，讲一个图的点变成边、边变成点，图的邻接关系不变。

### 7. 图卷积作为矩阵乘法、矩阵乘法作为图上的游走

在图上进行卷积或者随机游走等价于在邻接矩阵上进行矩阵乘法。图卷积作为矩阵乘法也是如何高效实现 GCN 的关键。

### 8. 图注意力网络 (Graph Attention Networks)

卷积对每个邻居有个权重，但卷积的权重是和位置相关的，但这对于图是无用的，因为图的邻居个数和顺序不固定。一种做法是用注意力机制的方法，权重取决于两个顶点向量之间的关系。具体看 Transformer 中的原理。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-03-29-GNN-into-Distrill.assets/image-20220326163120622.png" alt="image-20220326163120622" style="zoom:50%;" />

### 9. 图的可解释性

### 10. 生成图

如何对图的拓扑结构进行有效的建模

## 五、结论

图表是一种强大而丰富的结构化数据类型，其优势和挑战与图像和文本截然不同。 在本文中，我们概述了研究人员在构建基于神经网络的图形处理模型方面提出的一些里程碑。 我们已经介绍了使用这些架构时必须做出的一些重要设计选择，希望 GNN 游乐场可以直观地了解这些设计选择的经验结果是什么。 近年来 GNN 的成功为解决各种新问题创造了绝佳机会，我们很高兴看到该领域将带来什么。

## 六、回顾

作者先介绍了什么是图，用向量表示图的属性（顶点、边、全局），然后介绍了现实生活中的数据如何表示成图，如何对图做预测，以及机器学习算法用到图上的时候有什么挑战。接下来作者开始介绍 GNN 。 GNN 就是对属性做变化，而不改变图的结构，并设计了一个简单的例子。当属性有缺失时，可以使用聚合 (pooling) 的操作。然后作者介绍了我们日常使用中真正意义上的 GNN ，通过每一层将信息不断传递，每个顶点可以看到邻接顶点、相连边、全局的信息。 GNN 能够对图的结构进行有效的发掘。然后就是一个实际的 GNN 模型实验。接下来他讨论了一些有关的拓展问题。尤其关注作者的精美的图。



<br/>

**如果觉得本教程不错或对您有用，请前往项目地址 [https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io](https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io) 点击 Star :) ，这将是对我的肯定和鼓励，谢谢！**

<br/>



---

作者：Harry-hhj，Github主页：[传送门](https://raw.githubusercontent.com/Harry-hhj)

