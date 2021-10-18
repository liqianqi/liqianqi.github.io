---
title: RM 教程 1 —— Linux 教程
author: Harry-hhj
date: 2021-09-24 16:00:00 +0800
categories: [Course, RM]
tags: [getting started, robomaster, ubuntu]
math: false
mermaid: false
pin: false
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-24-RM-Tutorial-1-Linux-Introduction.assets/IMG_4633.JPG?raw=true
  width: 639
  height: 639
---



# RM 教程 1 —— Linux 教程

> 机械是血肉，电控是大脑，视觉是灵魂。

---

## 一、Why Linux & Why Ubuntu

Ubuntu 是一个十分流行并且好用的 Linux 桌面发行版本。截止到目前，Ubuntu 已经发行了 Ubuntu 20.04 的版本，并且其稳定性和支持已经很不错了。你可以在[这里](https://cn.ubuntu.com/download)下载各个版本的 Ubuntu 系统镜像文件，虚拟机的话一般下载桌面版本。

常言道，不要重复造轮子。在实际大型项目的开发过程中，总是不可避免的会使用到大量的第三方库，而一旦用到第三方库，就不可避免的会遇到依赖的问题，这个问题在编写 C/C++ 程序的时候尤为明显。在 Windows 下使用 Visual Studio 开发 C++ 程序时，每创建一个工程都必须不厌其烦地挨个设置每个第三方库的头文件目录、库文件目录、以及库文件名。这样的事情是极为繁琐的。

这时使用 Ubuntu 系统和 Cmake ，可以让你感受到无与伦比的遍历。所以，**使用 Ubuntu 系统的第一个好处就是开发环境配置方便**。

然而 Linux 的桌面发行版层出不穷，为何偏偏要采用 Ubuntu 呢？

这是因为 Ubuntu 作为最受欢迎的 Linux 桌面发行版之一，**几乎所有软件包都会原生支持在 Ubuntu 上的安装**，同时由于使用的人多，社区很广，**遇到问题在网络上也总能搜索到 Ubuntu 下的解决方法**。想象一下，如果安装一个不太热门的 Linux 发行版，想在上面安装 CUDA （一个极为流行的 GPU 编程库），然而官方没有原生支持，自行安装过程中的各种坑，在网络上又难以搜索到，这将会是一件多么令人恼火的事情。

总之，视觉部**推荐使用 Ubuntu 20.04 系统作为基本的开发环境**。



## 二、Ubuntu 基础知识

网上有比较全面的 Ubuntu [入门介绍](https://wiki.ubuntu.org.cn/新手入门指引)，这里不做过多的介绍，重点说一下实际使用过程中最为经常使用到的地方。

### 1.  Ubuntu 硬盘与文件目录结构

区别于 Windows 系统，每个硬盘分区单独一个盘符，不同分区间相互独立，Linux 下所有硬盘分区要么直接作为根目录，要么是根目录下的一个子目录。如

> 硬盘分区 1 挂载到根目录，即：/
>
> 硬盘分区 2 挂载到根目录的子目录，如：/data

在没有其他挂载的情况下，目录 `/` ，下面的所有文件（除了目录 `/data` ）都是保存在硬盘分区 1 中。而目录 `/data` ，下面的所有文件都是保存在硬盘分区 2 中。

### 2. Ubuntu 常用文件目录及其作用

-   `/home` ：该目录下保存不同账户的用户文件。假如你的 Ubuntu 有一个叫 `user` 的账户，那么 `/home/user` 下就保存着 `user` 账户的用户文件。如果还有一个叫 `foo` 的账户，那么 `/home/foo` 下就保存着 `foo` 账户的用户文件。
-   `/root` ：该目录下保存着 `root` 账户的用户文件。`root` 账户是 Ubuntu 中的一个特殊账户，拥有最高读写权限，类似与 Windows 中的管理员。
-   `/etc` ：该目录下保存着各种软件的配置信息。
-   `/usr` ：该目录下通常保存用户安装的各个软件、开发包等。
-   `/proc` ：该目录下都是虚拟文件，用于监控系统的运行状态。
-   `/dev` ：该目录下也是虚拟文件，用于保存各个设备驱动。
-   `/mnt` ：该目录下通常保存外部存储设备。如 U 盘等设备，通常可以在该目录下访问。

### 3. Ubuntu 账户

账户相当于是标记了这台电脑的不同使用者，当多人公用一台电脑时，可以通过不同账户来划分权限，这种情况在服务器上最为常见，因为服务器通常都会有很多个用户。但在个人电脑上，则通常仅有一个账户。

每个账户，可以属于一个或多个组，就好比将多个同类的用户归为一类，同样是方便进行权限管理。

### 4. Ubuntu 权限管理

这里的权限包括文件权限和用户权限。

通常来说，一个文件有 9 个权限可以设置，而这 9 个权限可以分为 3 类，分别是文件所有者权限，组权限和其他用户权限。其中这三类中，每类都包含 3 个权限，即读、写、执行，分别简写为 `r` 、 `w` 、`x` 。由于读、写、执行可以用 3 个二进制比特表示，所以这三个权限可以用一个 八进制数表示，而一共有 3 类权限，所以一个文件的权限可以由三个八进制数表示。我们可以使用命令 ``ls -l`` 来查看当前目录下所有文件的权限。

我们可以通过 ```chmod``` 命令修改文件的权限，基本用法是 ```chmod <权限> <文件名>``` ，比如 ```chmod 755 ./run``` 。在上面我们提到，一个文件的权限可以由 3 个八进制数表示，这里就是一个典型的例子。

由于有权限限制，在默认的用户权限下，我们通常只能修改目录 `/home` 下对应用户文件夹里的文件，而其他地方的文件都是无法修改的。为了获取修改任意文件的权限，我们可以使用 `sudo` 命令。该命令会使得用户获得临时的 root 权限，也就是类似于 Windows 下的管理员权限。这时我们就可以修改那些原本不能修改的文件了。**注意：如果使用 sudo 命令创建文件，创建出的文件的所有者将是 root 用户，也就是意味着在用户权限下不能修改它。**所以，**非必要情况下，尽量不使用 sudo 命令。**

### 5. APT 包管理工具

apt 是 Ubuntu 中的一个软件，负责管理系统中安装的各类软件包，开发包。包括但不限于安装：可执行软件、开发库（头文件，链接库等）、运行库（动态链接库）。其基本命令有：

```bash
apt search <包名>		# 搜索某个包
apt update		# 更新包数据库
apt upgrade		# 升级包
apt install <包名>		# 安装某个包
apt remove <包名>		# 删除某个包
```

主要常用的命令就是上面几个。由于apt安装的包，默认并不是安装到用户目录，也就是意味着在安装/删除包时，需要 root 权限。所以，**实际使用 apt 命令时还需要在前面加上 sudo 。**



## 三、常用 Linux 命令

请读者进入 Ubuntu 系统，并打开终端：

-   方式一：按下 Command 键，搜索 Terminal ，回车
-   方式二：Ctrl + Alt + T ，这个快捷键和系统打开方式有关，比如原生系统、虚拟机，还和电脑键盘有关

一个新打开的终端应该如下图所示，从现在开始你应该适应一个只有字符组成的世界。

![image-20210924162231521](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-24-RM-Tutorial-1-Linux-Introduction.assets/image-20210924162231521.png?raw=true){: width="2428" height="1552" style="max-width: 90%" }

你会发现你刚进入时，你的默认工作目录是 `~` ，即 `/home/<username>` ，不信你可以验证一下：

### 1. pwd

```shell
pwd
```

你会发现输出是 `/home/<username>` 。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-24-RM-Tutorial-1-Linux-Introduction.assets/image-20210924162333813.png?raw=true" alt="image-20210924162333813" style="zoom:25%;" />

这时你希望进入到文档的目录，于是你需要用到：

### 2. cd

```shell
cd ~/Documents
```

仔细观察，左侧的路径已经改变了。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-24-RM-Tutorial-1-Linux-Introduction.assets/image-20210924162413441.png?raw=true" alt="image-20210924162413441" style="zoom:25%;" />

那么在这个文件夹里存在什么呢？我们可以这样查看：

### 3. ls

```shell
ls
```

打印出的结果如下图。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-24-RM-Tutorial-1-Linux-Introduction.assets/image-20210924162440262.png?raw=true" alt="image-20210924162440262" style="zoom:25%;" />

如果你的系统是新的，那么你可能看不到任何结果。这空空如也的目录，我们改如何放入我们的东西呢？

### 4. mkdir

```shell
mkdir demo
```

我们创建一个 `demo` 目录，用于存放以后要用的文件。我们进入此目录中：

```shell
cd demo
```

此时我们有一个想法需要记录，我们需要创建一个 `test.txt` 文件。

### 5. touch

```shell
touch test.txt
```

`ls` 看一下，此时文件已经创建好了，那么我们怎么输入我们的想法呢？

### 6. gedit

```shell
gedit test.txt
```

注意 `gedit` 与其说是一个命令，不如说是一个软件。我们会打开一个图形化编辑窗口，在其中随意输入内容，保存并关闭。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-24-RM-Tutorial-1-Linux-Introduction.assets/image-20210924162621187.png?raw=true" alt="image-20210924162621187" style="zoom:25%;" />

那么我们刚刚的操作是否成功， `test.txt` 中是否存在了我们希望的内容呢？当然可以再次打开文件，但我们有更简单的方式：

### 7. cat

```shell
cat test.txt
```

在终端会输出文件内容，而无需图形化界面。

我们再次编辑刚刚的文件，将内容修改为：

```cpp
#include <iostream>

int main() {
  std::cout << "Hello!" << std::endl;
  return 0;
}
```

保存并关闭。这是一个 C++ 文件，但此时文件的后缀不太正确，我们先将文件重命名：

### 8. mv

```shell
mv test.txt test.cpp
```

`ls` 看一下，文件名称已经改变了。注意， `mv` 的本意是移动文件，但是当移动前后位置相同，且指定了移动后的名字时，我们可以将其用于重命名。此时我们编译它：

```shell
g++ test.cpp -o test
```

产生了可执行文件 `test` 。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-24-RM-Tutorial-1-Linux-Introduction.assets/image-20210924162748913.png?raw=true" alt="image-20210924162748913" style="zoom:25%;" />

运行它，但我们希望把结果记录下来：

### 9. >

```shell
./test > test.log
```

程序输出被重定向到了 `test.log` 中，不信你 `cat test.log` 看看是不是这样的。

好了，看到效果后，我们已经不需要这些文件了，于是我们可以删掉它们了。

### 10. rm

````shell
rm test.log
````

`ls` 看一下 `test.log` 已经被删除了，但一个个删太麻烦了，我们来点更快的：

```shell
rm -rf *
```

此时会把当前目录下的所有东西都删除（慎用，或许我不该教你的）。



Tips：如果你想偷懒复制粘贴，但又对 Ubuntu 不熟悉，或许我该提示你一般在终端中复制的快捷键是 `shift+crtl+c` ，粘贴的快捷键是 `shfit+ctrl+v` ，而 `ctrl+c` 其实是终止。



这里旨在让读者熟悉 RM 日常需求中高频使用的命令，更多命令用法请查看[这篇教程](https://harry-hhj.github.io/posts/Linux-Commands/)。





<br/>

**如果觉得本教程不错或对您有用，请前往项目地址 [https://github.com/Harry-hhj/Harry-hhj.github.io](https://github.com/Harry-hhj/Harry-hhj.github.io) 点击 Star :) ，这将是对我的肯定和鼓励，谢谢！**

<br/>



----

作者：唐欣阳，github主页：[传送门](https://github.com/xinyang-go)

第二作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

