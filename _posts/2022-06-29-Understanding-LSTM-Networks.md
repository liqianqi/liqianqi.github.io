---
title: Transformers are Graph Neural Networks
author: Harry-hhj
date: 2022-06-29 12:00:00 +0800
categories: [Tutorial, RNN]
tags: [lstm, ml, rnn]
math: true
mermaid: false
pin: false
image:
  src: https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/cover.png
  width: 942
  height: 592
---

# Understanding LSTM Networks

>   https://colah.github.io/posts/2015-08-Understanding-LSTMs/

## Recurrent Neural Networks

人的思维具有持续性，我们依据对之前的词的理解来理解下一个词。传统的神经网络不具备这样的能力，这似乎是一个主要缺点。

RNNs 就是为了解决这一问题而诞生的，它们是内部带有循环的网络，允许信息持续存在。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/RNN-rolled.png" alt="img" style="zoom:20%;" />

在上图中，一块神经网络 $A$ 查看某个输入 $x_t$ 并输出一个值 $h_t$ 。 循环允许信息从网络的一个步骤传递到下一个步骤。

这么看着可能很神秘，但我们展开后， RNN 其实和普通的神经网络相同。循环神经网络可以被认为是同一网络的多个副本，每个副本都将消息传递给后继者。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/RNN-unrolled.png" alt="An unrolled recurrent neural network." style="zoom:36%;" />

这种链状性质表明循环神经网络与序列和列表密切相关。 它们是用于此类数据的神经网络的自然架构。

事实上确实是这样。 RNNs 被广泛用于 speech recognition, language modeling, translation, image captioning ……感兴趣的可以观看 Andrej Karpathy 的优秀博文[《循环神经网络的不合理有效性》](http://karpathy.github.io/2015/05/21/rnn-effectiveness/)。

LSTMs 就是一种特殊的 RNN ，在多种任务上表现良好。

## The Problem of Long-Term Dependencies

RNNs 的诱人之处在于它们也许能够将以前的信息与当前的任务联系起来，但事实是这样吗？不一定。

Case 1 ：预测句子“the clouds are in the *sky*.”中的最后一个单词。在这种情况下，相关信息与所需位置之间的差距很小，RNN 可以学习使用过去的信息。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/RNN-shorttermdepdencies.png" alt="img" style="zoom:36%;" />

Case 2 ：预测句子“I grew up in France… I speak fluent *French*.”中的最后一个词。 RNNs 能够根据上下文推测这里应该是一种语言，但要确定是哪种语言，需要在非常靠前的 France 。不幸的是，随着 gap 的增大， RNNs 没有办法再连接起信息。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/RNN-longtermdependencies.png" alt="Neural networks struggle with long term dependencies." style="zoom:36%;" />

理论上， RNNs 应该具有处理 long-term dependency 的能力。遗憾的是，在实际中， RNNs 似乎不具备这种能力。 [Hochreiter (1991) [German]](http://people.idsia.ch/~juergen/SeppHochreiter1991ThesisAdvisorSchmidhuber.pdf) 和 [Bengio 等人 (1994)](http://www-dsi.ing.unifi.it/~paolo/ps/tnn-94-gradient.pdf) 深入探讨了这个问题。

幸运的是， LSTMs 没有这个问题！

## LSTM

Long Short Term Memory networks (LSTMs) ，一种特殊的 RNN ，能够学习 long-term dependencies ，由 [Hochreiter & Schmidhuber (1997)](http://www.bioinf.jku.at/publications/older/2604.pdf) 提出。

所有循环神经网络都具有神经网络的重复模块链的形式。在标准 RNNs 中，重复模块结构简单，如 tanh 层。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-SimpleRNN.png" alt="img" style="zoom:36%;" />

LSTMs 拥有四层神经网络，以特殊的方式交互。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-chain.png" alt="A LSTM neural network." style="zoom:36%;" />

## LSTMs 核心思想

LSTM 的关键是单元状态，即贯穿图表顶部的水平线。它直接沿着整个链条运行，只有一些较小的线性相互作用。 信息很容易沿着它不变地流动。

![img](https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-C-line.png)

LSTM 确实有能力将信息删除或添加到细胞状态，由称为门的结构小心调节。

门，顾名思义，放行或阻止信息通过。它由一个 sigmoid 和元素乘法构成。 sigmoid 的输出范围 0-1 ，意味着对每个组件应该放行多少。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-gate.png" alt="img" style="zoom:36%;" />

## Step-by-Step LSTM Walk Through

想要了解 LTSM 的机制，需要了解三扇门。

### Forget gate

Forget gate 决定单元状态中要丢掉的部分。输入 $h_{t-1}$ 和 $x_t$ ，通过 sigmoid 生成每一位的分数 (0-1) 。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-focus-f.png" alt="img" style="zoom:36%;" />

### Input gate

Input gate 决定什么新的信息需要被存储起来。这分成两步： sigmoid 对输入生成分数 (0-1) ，然后 tanh 生成候选的状态值。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-focus-i.png" alt="img" style="zoom:36%;" />

此时，我们就可以更新单元状态了。把每个分数乘上对应的数值，然后相加作为新的单元状态。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-focus-C.png" alt="LSTM3-focus-C" style="zoom:36%;" />

### Output gate

输出基于单元状态，但需要经过修改。对输入 $h_{t-1}$ 和 $x_t$ 应用 sigmoid 生成分数，然后对单元状态应用 tanh 规范到 $[-1,1]$ ，两者相乘。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-focus-o.png" alt="img" style="zoom:36%;" />

## Variants on Long Short Term Memory

### Peephole connections

>   ftp://ftp.idsia.ch/pub/juergen/TimeCount-IJCNN2000.pdf

Peephole connections 让所有门能够看见单元状态。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-var-peepholes.png" alt="img" style="zoom:36%;" />

当然，可以选择一部分 peephole connections 。

### Coupled forget and input gates

Coupled forget and input gates 同时决定遗忘和学习的内容。我们只会忘记何时要输入一些东西。 只有当我们忘记旧的东西时，我们才会向状态输入新的值。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-var-tied.png" alt="img" style="zoom:36%;" />

### GRU

>   http://arxiv.org/pdf/1406.1078v3.pdf

它将 forget gate 和 input gate 组合成一个 update gate 。 它还合并了单元格状态和隐藏状态，并进行了一些其他更改。最终的模型更加简单，也更流行。

<img src="https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2022-06-29-Understanding-LSTM-Networks.assets/LSTM3-var-GRU.png" alt="A gated recurrent unit neural network." style="zoom:36%;" />

这些变体中哪个最好？ 差异重要吗？ [Greff 等人 (2015)](http://arxiv.org/pdf/1503.04069.pdf) 对流行的变体进行了很好的比较，发现它们都差不多。 [Jozefowicz 等人 (2015)](http://jmlr.org/proceedings/papers/v37/jozefowicz15.pdf) 测试了超过一万个 RNN 架构，发现其中一些在某些任务上比 LSTM 效果更好。

## Conclusion

LSTM 是我们可以使用 RNN 完成的一大步。很自然地想知道：还有一大步吗？研究人员的普遍看法是：“是的！还有下一步，要注意了！”这个想法是让 RNN 的每一步都从更大的信息集合中挑选要查看的信息。例如，如果您使用 RNN 创建描述图像的标题，它可能会选择图像的一部分来查看它输出的每个单词。事实上，Xu 等人 (2015) 正是这样做的——如果你想探索注意力，这可能是一个有趣的起点！使用注意力已经产生了许多非常令人兴奋的结果，而且似乎还有更多的结果即将到来……

注意力并不是 RNN 研究中唯一令人兴奋的主题。例如，Kalchbrenner 等人 (2015) 的 Grid LSTM 似乎非常有前途。在生成模型中使用 RNN 的工作——例如 Gregor 等人 (2015)、Chung 等人 (2015) 或 Bayer & Osendorfer (2015)——似乎也很有趣。过去几年对于循环神经网络来说是一个激动人心的时期，而未来的神经网络承诺只会更是如此！





<br/>

**如果觉得本教程不错或对您有用，请前往项目地址 [https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io](https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io) 点击 Star :) ，这将是对我的肯定和鼓励，谢谢！**

<br/>



---

作者：Harry-hhj，Github主页：[传送门](https://raw.githubusercontent.com/Harry-hhj)

