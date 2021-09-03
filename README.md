# Harry's personal Blog

网站主页：点击[这里](https://harry-hhj.github.io)访问。



博客正在搭建中。缺少：评论、流量统计、美化。



## 一、环境安装

安装完环境后（参考此篇[教程](https://harry-hhj.github.io/posts/Building-your-Blog/)），在本地试运行查看 post 效果，确认无误后提交 PR ：

```bash
bundle
bundle exec jekyll s
```



## 二、博客规范

博客采用 Markdown 语言编写，允许插入 html 语言。

1.   博客名称采用 `<year>-<month>-<date>-<name/of/Blog>` 的方式命名，其中 `<name/of/Blog>` 不允许有空格，用 `-` 代替。

2.   将 `<name/of/blog>.md` 与同名目录 `<name/of/blog>.assets` 共同存放于 `_post` 目录下，其中 `<name/of/blog>.assets` 目录用于存放图片和附件。

3.   每篇博客的最开始处需要配置 config 选项，请参考其他博客顶部。

     -   必须： `title` 用于指定博客标题，一般是文件名去除日期的剩余部分
     -   必须： `author` 处签署作者花名
     -   必须：`date`：创建的准确日期，须指明时区
     -   必须：`categories`：指定文章类别，统一大写开头。一篇文章只能属于一个大类，可以二层分级，目前开设的类别有：
         -   Essay：随笔，用于分享日常生活或人生感悟或文学创作等，内容不限
         -   Tutorial：教程，用于分享技术，二级类别一般用英语专业名词命名，分类前先查看已有分类

     -   必须：`tags`：指定文章标签，格式要求较弱，必须小写开头，标签不具有层级关系，现有规定如下：

         -   只要涉及安装环境类教程必须带有 `getting started`
         -   计算机类带有 `computer`
         -   感悟类带有 `feelings`
         -   日记类带有 `diary`

     -   非必须：`toc`：指定文章是否开启右侧导航

         ```yaml
         toc: true  # 默认
         toc: false
         ```

     -   非必须：`comments`：评论功能，暂不支持

         ```yaml
         comments: true  # 默认
         comments: false
         ```

     -   非必须：`math`：如果你的博客带有如 MathJax 公式，需要手动开启，默认关闭节省性能

         ```yaml
         math: true
         ```

     -   非必须：`mermaid`：如果你的博客使用 [**Mermaid**](https://github.com/mermaid-js/mermaid) 绘图功能，需要手动开启

         ```yaml
         mermaid: true
         ```

         使用 ```` ```mermaid ```` 和 ```` ``` ```` 括起。

     -   非必须：`image`：博客初始图片

         ```yaml
         image:
           src: /path/to/image/file
           width: 1000   # 像素
           height: 400   # 像素
           alt: image alternative text
         ```

     -   不可使用：`pin`：置顶，不开放

4.   支持所有 Markdown 语法



## 三、技巧

### 图片

最终提交的图片的路径**不是本地的相对路径**，而是 GitHub 上的预览路径，这一点我会在最终提交时修改，不必费心。虽然可以将图片放入 assets 文件夹下，无需修改路径，但为了方便修改，最终决定如此规定。

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





<br/>

觉得本博客不错的请点个 star :) ~

----

作者：Harry-hhj

Github主页：https://github.com/Harry-hhj

