---
title: Install-dual-OS_Win10
author: Harry-hhj
date: 2021-08-26 16:40:00 +0800
categories: [Tutorial, 双系统]
tags: [getting started]
math: true
mermaid: true
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/win10.jpeg?raw=true
  width: 1202
  height: 676
---



# 安装 Win10 双系统

>   本篇教程是基于 MacOS 系统安装 Win10 双系统的，当然也适用于 Win10 安装 Win10 双系统，如果你愿意的话，只需要去除和  MacOS 有关的特定内容就可以了。



硬件要求：

- Macbook pro 2019（只要不是2015年前的苹果电脑应该都可以）
- Windows 虚拟机或主机（我使用的是虚拟机）
- 一块硬盘，至少 64 GB，接口 USB 3.0 及以上（机械硬盘或者 SSD ，推荐 SSD ，笔者使用的是 SamsumgSSD9801TB ）
- 外接 USB 鼠标和外接 USB 键盘，外接 USB 无线网卡（可选）
- 一个U盘或机械硬盘（ exFAT 格式），至少 1.5 TB ，也可以放在 SSD 的非系统分区中
- 电脑供电电源（过程非常耗电，请插电源操作）
- USB-C 拓展坞（可选，根据接口需要）

软件要求：

- 启动转换助理（苹果系统自带）
- WTG 辅助工具（附件[^attachment]）
- Windows 10 镜像（附件[^attachment]）

为了便于区分两个 Win 系统，以下将用于制作系统盘的 Windows 系统称作`宿主系统`，将新制作的 Windows 系统称作`目标系统`。



## 前言

Macbook pro 的存储空间是非常宝贵的，因为苹果的硬盘速度虽然高，价格也非常贵。而且，我也很不喜欢装个系统把硬盘弄糟，甚至把 Mac 系统弄坏，毕竟 Mac OS 才是我日常生活的主力。在经历了两年的虚拟机之后，我终于下决心试试装双系统了。

备注：每次用 DiskGenius 操作分区后都需要右击左侧磁盘选择`保存分区表`，推荐对所有主引导记录的硬盘如此操作。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/1.png?raw=true" alt="image-20210813174437934" style="zoom:50%;" />

## 科普

### 机械硬盘和固态硬盘（SSD）和 U 盘

#### 机械硬盘和固态硬盘

本质不同：机械硬盘本质是电磁存储，固态则是半导体存储。

防震抗摔性不同：机械硬盘很怕率，固态抗震。

数据存储速度不同：固态读写速度比机械快。

功耗不同：固态硬盘的功耗比机械硬盘低。

重量不同：固态硬盘的重量比机械硬盘轻。

噪音不同：机械硬盘有噪音，固态硬盘没有噪音。

#### 移动硬盘和U盘

SSD 硬盘与 U 盘都采用了 Flash （闪存）作为储存介质，他们的区别如下：

SSD 采用 SATA 接口主控，而绝大部分优盘采用USB接口主控。

由于 U 盘和固态硬盘之间所使用的主控芯片、 Flash 颗粒的数量和缓存容量不同造成二者存储速度存在巨大差异。固态硬盘采用了多颗 Flash 颗粒组成，而其内部也是采用了类似于 RAID 的写入方式，可同时在不同的颗粒上写入或者读取数据。而U盘通常就是单通道的写入，所以在性能上完全没有办法和固态硬盘相提并论。

U盘和固态硬盘所使用的Flash都有一定的写入次数**寿命**。一旦当写入次数达到这个数量之后，那么就无法再写入数据了，也就**意味着U盘或者固态硬盘的损坏**。SSD （固态硬盘）据的主控芯片均具备了一种平均写入数据的算法，以延长使用寿命。而 U 盘就是不具备平均写入数据功能，所以一旦 U 盘用来反复读写数据话，是非常容易造成损坏的。

容量与价格不同。



### 文件系统

#### FAT32

这一种格式是任何USB存储设备都会预装的文件系统，属 Windows 平台的传统文件格式，兼容性很好。即便 FAT32 格式兼容性好，但它**不支持 4 GB 以上的文件**，因此对于很多镜像文件或大视频文件之类的也会有无可奈何的状况。

#### NTFS

NTFS 格式却是 Windows 平台应用最广泛的文件格式，它的优点在于能够支持大容量文件和超大分区，且集合了很多高级的技术，其中包括长文件名、压缩分区、数据保护和恢复等等的功能。但它会减短闪存的寿命。

#### exFAT

exFAT 格式才是最适合 U 盘的文件格式，它是微软为了闪存设备特地设计的文件系统，是 U 盘等移动设备最好的选择之一。SSD 和 U 盘同为闪存，但SSD还是用NTFS格式为好！





好了，下面正式进入教程。

## 准备工作 - WindowsSupport

Macbook 的硬件在 Windows 上是不能直接使用的，因此需要获得相应的驱动程序，这个苹果系统已经有现成的了，我们下载就行。

在**苹果系统**中打开“启动转换助理”，如下图：

![启动转换助理](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/2.jpg?raw=true)

打开后在左上角点击`操作`-`下载Windows支持软件`，下载完毕后，将文件转存到一个 U 盘或机械硬盘上备用（注意容量，未压缩约为 1.26 GB ），以便后续文件传输。

![img](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/3.jpg?raw=true)

![WindowsSupport](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/4.jpg?raw=true)

然后将待用的硬盘（SSD）全盘格式化为 exFAT 格式，不要含有多个分区，供后续使用。



## Windows镜像文件

网上有很多，如果你是交大学生，附件[^attachment]`SW_DVD5_Win_Pro_Ent_Edu_N_10_1803_64BIT_ChnSimp_-2_MLF_X21-79700` 中也包含了，可通过交大网络激活。

激活方式：需要连接校园网或 VPN ，用管理员身份打开`命令提示符`（`右击`选择`以管理员身份运行`），进入 `C:\Windows\System32` 文件夹中，输入以下命令：

```bash
cscript slmgr.vbs /skms kms.sjtu.edu.cn
cscript slmgr.vbs /ato
```

激活成功。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/5.png?raw=true" alt="image-20210813181627309" style="zoom:33%;" />



## WinToGo 辅助工具

将上述的 Windows 镜像文件复制到宿主系统（虚拟机）中，双击打开，这时系统会显示有个 DVD 驱动器，此时先不要操作。

![在这里插入图片描述](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/6.jpg?raw=true)

使用附件[^attachment]中的 wtga5590 ，双击 `exe` 文件，注意网上 4.8 版本的 wtga 不能选择 windows 版本，所以不要使用。打开软件后，第一个候选框选择上图中的 `DVD驱动器/sousrces/install.wim` 。然后选择版本为 `企业版` ，最后选择你的硬盘。右侧高级选项选择 `传统` + `UEFI+GPT` ，其他可以选择默认。最后选择 `部署` 。需要分区的朋友可以在 `分区` 里进行设置，需要注意给系统盘留出足够的空间。也可以按之后的教程进行操作。

![img](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/7.jpg?raw=true)

会弹出一个窗口提示你整个硬盘会被格式化，让你确认，点击 `是` 即可。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/8.jpg?raw=true" alt="在这里插入图片描述" style="zoom:50%;" />

 耐心等待制作完成。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/9.jpg?raw=true" alt="在这里插入图片描述" style="zoom:50%;" />

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/10.jpg?raw=true" alt="img" style="zoom:40%;" />



## 关闭安全选项

在 MacOS 系统启动时，同时安装 `Command` + `R`  ，进入到恢复助理的界面，选择知道密码的用户，输入密码后选择左上角的 `实用工具` ，选择里面的 `允许从外部介质启动` 。同时将安全启动那里选择 `无安全性` 。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/11.jpg?raw=true" alt="img" style="zoom:50%;" />

不然可能会出现以下的错误：

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/12.jpg?raw=true" alt="img" style="zoom:40%;" />



## 安装驱动

关机，在开机时按住 `option` 键，选择 EFI 启动磁盘，然后就进入了 Windows 的启动界面，注意 第一次进入会**自动重启**，所以在重启时还需要按住 `option` 键。然后就像正常的 Windows 一样初始化就行了，唯一要注意的是**此时 Mac 里自带的网卡、键盘和触控板统统不能使用**，需要外接 USB 设备，这就是要准备 USB 鼠标键盘和无线网卡的原因。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/13.jpg?raw=true" alt="在这里插入图片描述" style="zoom:10%;" />

设置好后进入Win10 桌面，这时候鼠标键盘等依旧不能使用需要安装之前下载好的 `Windows支持软件` 。找到你U盘里复制过来的支持软件，打开文件夹找到 `setup.exe` ，双击打开安装 BootCamp 驱动。安装完后，如果遇到没有声音，使用搜索栏搜索 `Apple Software Update`，更新后应该能解决问题，再不行就多安装几次，再不行点击带❌的音量，自动搜索驱动、重启等，应该是能解决的。

安装完驱动后成后已经可以正常使用键盘、鼠标、网卡等等。但是你可能发现无法自定义设置键盘背光、触控板功能等，这是因为 `BootCamp` 锁定了外置磁盘的控制面板设置。这时下载之前说过的 `BootCamp` 控制面板工具，名字为 `AppleControlPanel.exe` 。之后在目录 `C:\Windows\System32` 找到 `AppleControlPanel.exe` 这个文件，把它替换为之前下载的同名文件。这时候在进入 `Boot Camp` 控制面板就可以自定义设置触控板功能了。

如果没有找到  `Boot Camp` 控制面板，就去 Mac 上重新下载 WindowsSupport 安装，就能解决。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/14.png?raw=true" alt="image-20210813185033085" style="zoom:70%;" />

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/15.png?raw=true" alt="image-20210813185245170" style="zoom:40%;" />



## 分区分卷

如果硬盘过大时我们会希望将一部分硬盘分理出系统盘作为数据盘使用或留作他用，这是我们的操作如下：

- 首先在宿主系统中的搜索框输入`此电脑`，然后`右键`点击`管理`，如下图：

  <img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/16.png?raw=true" alt="image-20210813165746461" style="zoom:40%;" />

- 在新出现的窗口点击`存储`->`磁盘管理`，找到目标磁盘（根据磁盘大小），`右击`选择`压缩卷`，将系统盘的容量压缩到你认为合适的大小，这里我预留了 `350 GB` （仅供参考）。

- 将多出来的空间`右击`选择`新加卷`，格式选 `exFAT`，即可。

- 在目标系统中进入同样的界面，重新`删除卷`，并`新建卷`，这么做的目的是为了**分配盘符**，不然会被隐藏，在目标系统中不可见。如果希望进行更细致的格式化可以使用附件[^attachment]中的 `DiskGenius` 重新格式化，可以选择簇大小（对簇大小的理解就是：簇大小越大，读写速度越快，但小文件浪费的空间也更多，如果不知道直接选择默认就行）。如下图：

  ![image-20210813172303745](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/17.png?raw=true)

  （图中我留了 256GB 的未分配空间，操作的方法是使用 `DiskGenius` 对 DATA 盘`新建分区`，再通过`磁盘管理`来`删除卷`就行了。）

- 最终的结果如下图所示：

  <img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/18.png?raw=true" alt="image-20210813173623009" style="zoom:50%;" />



## 结果

最终我的 Windows 界面如下：

![image-20210813185431706](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-8-26-Install-dual-OS_Win10.assets/19.png?raw=true)

恭喜成功！



## 参考教程

1. [为MacBook Pro制作WTG系统盘](https://blog.csdn.net/weixin_42708321/article/details/104412941)
2. [在我的U盘上装了 win to go瞬间感觉相见恨晚（WTG安装最详细教程）](https://sspai.com/post/62895)
3. [ssd固态硬盘和U盘的区别是什么呢？](https://zhidao.baidu.com/question/430192719.html)
4. [U盘FAT32、NTFS、exFAT格式的区别，你都知道么？](https://baijiahao.baidu.com/s?id=1646072976790941345&wfr=spider&for=pc)



如有问题欢迎来交流！

<br/>

---

作者：Harry-hhj

github主页：_<https://github.com/Harry-hhj>_



[^attachment]: _<1>_

