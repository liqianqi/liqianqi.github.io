---
title: Skills about MacOS
author: Harry-hhj
date: 2021-09-20 18:40:00 +0800
categories: [Tutorial, MacOS]
tags: [skills]
math: false
mermaid: false
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-20-Skills-about-MacOS.assets/MacOS.jpg?raw=true
  width: 
  height: 
---



# MacOS 使用技巧

## 开盖自动开机

```bash
sudo nvram AutoBoot=%00  # 关闭
sudo nvram AutoBoot=%03  # 打开
```

从关闭到打开后，会自动打开开机声音。



## 开机声音

```bash
sudo nvram BootAudio=%00  # 关闭
sudo nvram BootAudio=%01  # 打开
```


## 查看硬盘寿命

```bash
smartctl -a /dev/disk1
```


## 设置截图保存位置

```bash
defaults write com.apple.screencapture location /path/  # 默认~/Desktop
defaults write com.apple.screencapture type jpg  # 默认png
```



## 遇到没有声音的问题

打开活动监视器，找到 `coreaudiod` ，强制退出。



## SSH public key 存放位置

```bash
~/.ssh/known_hosts
```


## SSH 免 RSA key fingerprint

```bash
-o "StrictHostKeyChecking no"
```



## 切换 zsh 和 bash

环境变量配置：

-   bash 的环境变量是 `.bash_profile` 文件
-   zsh 的环境变量是 `.zshrc` 文件

从一个交互式终端的角度来讲， zsh 更为强大，但是作为脚本解释器， bash 更加符合 posix 标准，因此，建议读者日常使用 zsh （配合 [oh-my-zsh](https://link.jianshu.com/?t=https://github.com/robbyrussell/oh-my-zsh) ），但是使用 bash 做脚本解释器。

```bash
chsh -s /bin/zsh
chsh -s /bin/bash
```

注意不要加 `sudo` ，此时为切换 root 用户默认解释器。

更多关于 zsh 的内容可以查看[这篇教程](https://www.zhihu.com/question/21418449)。



## 隐藏终端主机名

```bash
sudo vim /etc/zshrc
```

修改 `PS1` ，例如：

```bash
PS1="%F{green}%n:%F{cyan}%~%F{green}%F{white} %# "
```





## 参考资料

1.   [优化mac下的terminal的zsh路径显示](https://blog.csdn.net/qq_38992249/article/details/116406664)



---

作者：Harry-hhj，github主页：[传送门](