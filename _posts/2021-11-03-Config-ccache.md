---
title: Config ccache
author: Harry-hhj
date: 2021-11-03 21:15:00 +0800
categories: [Tutorial, ccache]
tags: [install, c/c++, ccache]
math: true
mermaid: false
pin: false
image:
  src: https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2021-11-03-Config-ccache.assets/ccache.jpg
  width: 639
  height: 639
---



# Ccache

这篇博客介绍一个小工具 `ccache` ，可以提高再次编译的速度。其原理是通过吧项目的源文件用 `ccache` 编译器编译，然后缓存编译生成的信息，从而在下一次编译时，利用这个缓存加快编译的速度，目前支持的语言有 `C` 、 `C++` 、 `Objective-C` 、 `Objective-C++` ，如果找不到 `ccache` 编译器，还是会选择系统默认的编译器来编译源文件。

接下来讲述 ccache 的利用过程。

## 一、安装

这里介绍 Ubuntu 的安装方法。

首先通过 `apt` 安装：

```shell
sudo apt install ccache
```

安装完后我们不能直接使用，需要先进行配置:

```shell
sudo gedit ~/.bashrc
```

在新打开的文档末尾回车，添加如下语句，注意 `<username>` 要改成你的用户名。

```shell
export CCACHE_DIR="/home/<username>/.ccache"
export CC="ccache gcc"
export CXX="ccache g++"
export PATH="$PATH:/usr/lib/ccache"
```

`Ctrl+S` 或点击 `Save` 按钮保存，然后需要更新 `.bashrc` 使其生效。

```shell
source ~/.bashrc
```

我们可以根据硬盘空间设置 `ccache` 允许使用的最大缓存空间， `<xx>` 修改为数字：

```shell
ccache -M <xx>G
```





## 二、使用

### 1. CMake

对于采用 CMake 的应用，只需要将以下的代码加入到命令 `project()` 行以后即可：

```cmake
find_program(CCACHE_PROGRAM ccache)
if (CCACHE_PROGRAM)
    set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE "${CCACHE_PROGRAM}")
endif ()

add_compile_definitions(PROJECT_DIR="${PROJECT_SOURCE_DIR}")
```



### 2. Xcode

参考参考教程：[ccache - 让Xcode编译速度飞起来](https://www.cnblogs.com/fishbay/p/7217398.html)。









<br/>

**如果觉得本教程不错或对您有用，请前往项目地址 [https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io](https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io) 点击 Star :) ，这将是对我的肯定和鼓励，谢谢！**

<br/>



## 三、参考教程

1.   [ubuntu配置ccache](https://www.jianshu.com/p/10cc47a49e81)
2.   [ccache - 让Xcode编译速度飞起来](https://www.cnblogs.com/fishbay/p/7217398.html)



---

作者：Harry-hhj，Github主页：[传送门](https://raw.githubusercontent.com/Harry-hhj)

