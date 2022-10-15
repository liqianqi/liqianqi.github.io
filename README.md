# liqianqi's personal Blog

网站主页：点击[这里](https://liqianqi.github.io)访问。

声明一句：这个模板使用[上海交通大学前视觉组组长](https://Harry-hhj.github.io)的，我只是复用一下模板试一试效果。
我刚刚用上，目前是在学习搭建过程中，以后会自己搭一套自己的博客。
最下方是他的主页

博客正在搭建中。缺少：评论、美化。



## 一、环境安装

安装完环境后（参考此篇[教程](https://harry-hhj.github.io/posts/Building-your-Blog/)），在本地试运行查看 post 效果，确认无误后提交 PR ：

```bash
bundle
bundle exec jekyll s
```



## 二、博客规范

博客采用 Markdown 语言编写，支持所有 Markdown 语法，允许插入 html 语言。

### 1）博客命名

博客名称采用 `<year>-<month>-<date>-<name-of-Blog>` 的方式命名，放置于 `_posts/` 目录下，其中 `<name-of-Blog>` **不允许有空格，用 `-` 代替**。请尽量简洁并符合英文语法。此名称中的 `<year>-<month>-<date>` 仅用于区别不同的博客，方便识别，**分别用 4 、 2 、 2 个数字表示**，如 `2000-01-01` 。



### 2）博客头部

每篇博客开始都需要配置 config 选项，**必须放置于博客最开始，采用 yaml 语法**。一个通用的头部如下，创建博客可以复制以下代码使用：

```yaml
---
title: 
author: 
date: 2000-01-01 00:00:00 +0800
categories: []
tags: []
math: false
mermaid: false
toc: true
pin: false
image:
  src: 
  width: 
  height: 

---
```

说明如下：

```yaml
---
title: PyTorch Basics		# 博客标题，可以与 md 命名不同，但应当相似，可以使用空格
author: Harry-hhj		# 博客作者，可以写下你的 Github 用户名
date: 2021-08-29 11:12:00 +0800		# 时戳，用于网页排列博客的真正时间戳，指明时区
categories: [Tutorial, PyTorch]		# 分类，见下分类规范
tags: [getting started, computer science, pytorch]		# 标签，见下标签规范
math: false		# 是否使用 mathjax 公示，不使用请关闭，加速网页加载
mermaid: false		# 是否使用 mermaid 绘图（https://github.com/mermaid-js/mermaid），不使用请关闭，加速网页加载
toc: true  # 默认，开启右侧网页导航栏与否
pin: false		# 是否置顶，请设为 false ，仅我可以选择置顶与否
image:		# 放置于博客标题下的图片
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-08-29-Pytorch-Basics.assets/pytorch.jpeg?raw=true
  width: 480
  height: 230

---
```



### 3）图片和小附件

每篇博客 `<name-of-blog>.md` 所使用的图片和小附件放置于 `_posts` 目录下的同名目录 `<name-of-blog>.assets` 下。这样规定的原因是为了便于管理和更新。

最终提交的图片的路径**不是本地的相对路径**，而是 GitHub 上的预览路径。在最后上传时，应将博客中**所有的图片路径**改为 `https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/<name-of-blog>.assets/<name-of-image>?raw=true` ，请不要遗漏 `?raw=true` 。（如果不明白可以使用相对路径并在博客中注明，我来修改）

图片允许添加说明文字，格式如下：

```markdown
![img-description](/path/to/image)
_Image Caption_
```

为了避免图片加载造成网页内容位移影响体验，可以指定图片大小预留加载空间，例如：

```markdown
![Desktop View](/assets/img/sample/mockup.png){: width="700" height="400" }
```

可以指定图片的位置，默认是居中，此时**不再允许添加说明文字**：

```markdown
![Desktop View](/assets/img/sample/mockup.png){: .normal }
![Desktop View](/assets/img/sample/mockup.png){: .left }
![Desktop View](/assets/img/sample/mockup.png){: .right }
```



### 4）大附件

大型附件（如超过 100 M）请使用其他平台传输，并给出链接和密码，如可使用百度网盘或阿里云盘等。请确保链接永久有效且来源可靠。（我可能会定期下载并更新为我的链接来确保文件不会丢失）



### 5）分类规范

分类必须使用首字母大写，允许使用空格，最多为两级。

目前已经有的一级为（可以扩充更新，请使用 PR 或 Issue ）：

-   Essay：随笔，用于分享日常生活或人生感悟或文学创作等，内容不限
-   Tutorial：教程，用于分享技术，二级类别一般用英语专业名词命名，分类前先查看已有分类
-   Question：问答类教程



### 6）标签规范

标签统一采用小写，不允许大写。

目前已有的标签规范有：

-   `getting started` ：任何入门级教程或知识分享型教程【1】
-   `tools` ：任何有关工具、软件、语言等如何使用的教程【1】
-   `skills` ：关于使用技巧，主要在于提高使用效率而非讲解如何使用【1】
-   `questions` ：问题回答型文章【1】
-   `diary` ：日记、随笔【1】
-   `robomaster` ：与 RoboMaster 赛事有关的文章【1】
-   `computer science` ：和计算机科学技术相关的任何文章【2】
-   `infomation security` ：和信息安全有关的任何文章【2】
-   一般技术文章都会带有与其核心技术点相关的英语专业名词【3】

一般标签数量**小于等于 3 个**为宜，上面同一 【x】表示这些标签之间一般相互冲突。



<br/>


----

作者：Harry-hhj

Github主页：https://github.com/Harry-hhj
