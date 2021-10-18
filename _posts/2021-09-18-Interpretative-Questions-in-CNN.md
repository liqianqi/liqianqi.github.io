---
title: CNN 中的解释性问题
author: Harry-hhj
date: 2021-09-18 16:20:00 +0800
categories: [Question, CNN]
tags: [computer science, questions]
math: true
mermaid: false
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-18-Interpretative-Questions-in-CNN.assets/sea.jpg?raw=true
  width: 1366
  height: 768
---



# 目录

1.   [为什么 CNN 中卷积核的大小一般为奇数？](#为什么 CNN 中卷积核的大小一般为奇数？)



# Q&A

## <span id="为什么 CNN 中卷积核的大小一般为奇数？">Q1：为什么 CNN 中卷积核的大小一般为奇数？</span>

卷积核大小一般是奇数，原因是：

-   容易找到卷积锚点（主要）：

    使用奇数尺寸的滤波器可简化索引，并更为直观，因为滤波器的中心落在整数值上。奇数相对于偶数，有中心点，对边沿、对线条更加敏感，可以更有效的提取边沿信息。因此，奇数大小的卷积核效率更高。

-   便于进行 padding （次要）：

    padding 的做法是双边填充，这就导致不管我们怎么填充最终图像增长的长度是偶数。然而我们知道在卷积时如果最后的剩余部分比卷积核小，那么就会损失部分边缘信息，这是我们不希望看到的。我们假设卷积大小 $(k \times k)$ ，令 $\text{stride}=1$ ， $\text{dilation}=1$ ，则 $\text{padding} = \frac{\text{kernel\_size}-1}{2}$ 。





<br/>

**如果觉得本教程不错或对您有用，请前往项目地址 [https://github.com/Harry-hhj/Harry-hhj.github.io](https://github.com/Harry-hhj/Harry-hhj.github.io) 点击 Star :) ，这将是对我的肯定和鼓励，谢谢！**

<br/>

# 参考文献

1.   deeplearning.ai 《卷积神经网络》



-----

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

