---
title: RM 教程 2 —— 安装 OpenCV
author: Harry-hhj
date: 2021-10-02 13:30:00 +0800
categories: [Course, RM]
tags: [getting started, robomaster, opencv]
math: false
mermaid: false
pin: false
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-10-02-RM-Tutorial-2-Install-OpenCV.assets/IMG_4633.JPG?raw=true
  width: 639
  height: 639
---



# RM 教程 2 —— 安装 OpenCV

> 机械是血肉，电控是大脑，视觉是灵魂。

---

## 一、简介

OpenCV 是计算机视觉中经典的专用库，其支持多语言，跨平台，功能强大。 opencv-python 为OpenCV 提供了 Python 接口，使得使用者在 Python 中能够调用 C/C++ ，在保证易读性和运行效率的前提下，实现所需的功能。

OpenCV 现在支持与计算机视觉和机器学习有关的多种算法，并且正在日益扩展。OpenCV 支持多种编程语言，例如 C++、 Python 、 Java 等，并且可以在 Windows 、 Linux 、 OS X 、 Android 和 IOS 等不同平台上使用。基于 CUDA 和 OpenCL的高速GPU操作的接口也在积极开发中。



## 二、快速安装

注意：仅适合新手，队员参与实际项目时还请按照【三】完成安装。

### C++

打开终端，输入以下命令：

```cpp
sudo apt-get install libopencv-dev python-opencv libopencv-contrib-dev
```

以 Clion IDE 为例，配置 toolchains ，如下图所示。需要说明的是：

-   如果你的 cmake 是系统自带的，那么 cmake 路径选择 `/usr/bin/cmake` ，如果是编译安装的，那么选择 `/usr/local/bin/cmake` 。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-10-02-RM-Tutorial-2-Install-OpenCV.assets/image-20211002172153968.png?raw=true" alt="image-20211002172153968" style="zoom:50%;" />

示例代码：链接: _<https://pan.baidu.com/s/1MDLwgGJ57cG3NfxDAfZASg>_ 提取码: cph8

测试方式：点击 IDE 右上角运行或命令行进入项目目录：

```shell
mkdir build
cmake ..
make
./example
```

如果出现一张苹果的图片表示安装成功。



### Python

打开终端，输入以下命令：

```shell
pip install opencv-python opencv-contrib-python -i https://pypi.tuna.tsinghua.edu.cn/simple
```

如果你有 conda 环境的话，可以先创建一个新的环境：

```shell
conda env list
conda create -n opencv python=3.9
conda activate opencv
```

以 Pycharm IDE 为例，如果非系统默认的环境，请记得配置项目设置-python 解释器的路径，例如 conda 环境的解释器路径一般都为 ``/<anaconda3 or miniconda3>/envs/<env_name>/bin/python3` 。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-10-02-RM-Tutorial-2-Install-OpenCV.assets/image-20211002171346104.png?raw=true" alt="image-20211002171346104" style="zoom:50%;" />

示例代码：链接: _<https://pan.baidu.com/s/1AlPkGtZ-4HkRjhwuFB86eQ>_ 提取码: v7dj

测试方法：点击 IDE 右上角运行或者命令进入项目目录

```shell
# 如果有 conda 环境记得先激活
python3 main.py
```

如果出现一张苹果的图片表示安装成功。



p.s. ：这篇[教程](https://harry-hhj.github.io/posts/Install-OpenCV/)讲述了如何编译安装。



## 三、备注

Clion 和 Pycharm 的安装教程在对应的安装包中都有提供，这里给出申请学生免费账号的方法。首先进入[官方申请网站](https://www.jetbrains.com/idea/buy/#discounts?billing=yearly)，选择 For students and teachers 下的 learn more ，用自己的学校邮箱申请，然后打开邮箱内的确认邮件。然后创建自己的 JetBrains Account ，在软件安装完之后的 activate 过程中输入账号密码就可以使用了。

注意：目前交大邮箱只能通过人工认证的方式验证。



----

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

