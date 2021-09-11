---
title: Building your Blog too
author: Harry-hhj
date: 2021-08-24 08:00:00 +0800
categories: [Tutorial, Jekyll]
tags: [getting started, jekyll]
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-08-24-Building-your-Blog.assets/devices-mockup.png?raw=true
  width: 850
  height: 585
---

按照本篇教程的步骤，搭建属于你自己的 `jekyll-theme-chirpy` 吧～（其他样式请按照相应的 `README.md` 进行操作）



# 搭建 GitHub 博客

## 第一步 安装环境

按照 [Jekyll Docs](https://jekyllrb.com/docs/installation/) 官方教程完成环境的搭建，该教程是全英文教程，如果能看懂建议根据该教程操作（因为该教程会保持最新），遇到问题后再来参考本教程。如果对英文教程不熟悉的，可以使用 Chrome 浏览器打开后进行页面翻译。注意：官方教程中有些操作是在普通的句子中而非列表的形式给出，所以在查阅时无比认真阅读每一句话，不要跳句！

## Ubuntu

安装 Ruby 和其他依赖：

```bash
sudo apt install ruby-full build-essential zlib1g-dev
```

将 gem 安装到当前用户下（不要以 root 用户安装），以下的命令将在 `~/.bashrc` 中添加环境变量来指定 gem 的安装路径：

```bash
echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

查看 `ruby` 的版本： `ruby -v` ，确保其版本高于或等于 `2.5.0` 。

最后，通过 gem 安装 `jekyll` 和 `bundler` ：

```bash
gem install jekyll bundler
```

查看 `RubyGems` 的版本： `gem -v` ，有版本输出代表安装成功。



（可选）安装最新版本的 `gcc` 和 `g++` 。

首先添加 ppa 到库，这一步实际是添加了一个 ubuntu 的源：

````bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
````

安装新版 `gcc/g++` ，目前最新的版本号是 11 ：

```bash
sudo apt install -y gcc-11 g++-11
```

这里有个小技巧，加个参数 `-y` 表示不询问并同意，这样就不用在之后输入 `y` 确认了。

查看 Ubuntu 原来的 `gcc/g++` 版本，并记住（Ubuntu 20.04 默认应该是 9）：

````bash
gcc -v
g++ -v  # 如果你的电脑安装了 g++ 的话
````

默认情况下 Ubuntu 没有安装 g++ ，如果你希望 gcc/g++ 保持配对，可以通过 `gcc -v` 查看版本下载对应的 g++ 。

我们需要将标准的 gcc/g++ 连接到我们期望的 gcc/g++ 程序，有两种连接方式建立连接：

-   `ln` 命令创建软链接

    -   ```bash
        cd /usr/bin
        sudo rm gcc
        sudo ln -s gcc-11 g++
        sudo rm g++
        sudo ln -s g++-11 g++
        ```

-   通过 `update-alternatives` 建立文件关联

    -   记住你刚刚查看到的原始版本号，以下命令中的 `<版本号>` 用下图中你查询到的数字替代：

        ![image-20210825174125342](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-08-24-Building-your-Blog.assets/image-20210825174125342.png?raw=true)

    -   ```bash
        # 首先让系统知道我们安装了多个 gcc 版本
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-<版本号> <版本号*10>
        sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 110
        # 使用交互方式的命令选择默认使用的版本，默认使用 auto 选择模式，系统将默认使用优先级最高的，无需修改直接按回车（enter）：
        sudo update-alternatives --config gcc
        
        # g++ 同理
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-<版本号> <版本号*10>
        sudo update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-11 110
        sudo update-alternatives --config g++
        ```

    -   如果不小心有个命令写错了，那么（没关系，我也因为偷懒使用历史命令修改的时候少改了地方错过）：

        ```bash
        # 如果是优先级写错了，重新 --install 就行了
        # 如果是把 g++ 装到了 gcc 里或其他类似情况，然后重新 --install 就可以了：
          sudo update-alternatives --remove <装错的链接名gcc或g++> <装错的路径>
        ```

    -   切换 gcc/g++ 版本：

        ```bash
        sudo update-alternatives --config gcc
        sudo update-alternatives --config g++
        ```





## 第二步 克隆仓库

有两种搭建方式，这里只介绍第二种，因为我们的目的就是为了借助 Github Page 来发布博客：

-   从 RubyGems 安装：易于更新，隔离不相关的项目文件，可以专注于编写。
-   从 GitHub 克隆：自定义开发方便，但更新难，只适合web开发者。

首先前往原项目[仓库](https://github.com/cotes2020/jekyll-theme-chirpy)，点击屏幕右侧的 `fork` 按钮：

![image-20210825182140392](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-08-24-Building-your-Blog.assets/image-20210825182140392.png?raw=true)

这样，我们的代码库里就有了一个相同的仓库，唯一不同的这个仓库的所有权属于我们自己。找到这个仓库并点击进入。

点击 `Code` ，复制网址，然后打开 `Terminal` 进入项目目录， `git clone <网址>` ，如果还没有装过 `git` 先安装一下：

````bsah
sudo apt install git
````

安装 gem 依赖：

```bash
cd jekyll-theme-chirpy
bundle
```

然后执行脚本：

```bash
bash tools/init.sh
```

>    如果你不打算在 Github Pages 部署博客，那么在这个命令后加上参数 `--no-gh` .

这一步所执行的操作是：

-   删除 `.travis.yml` 、`_post` 文件夹下的文件、`docs` 文档
-   如果使用了 `--no-gh` 参数，那么目录 `.github` 会被删除，否则 `.github/workflows/pages-deploy.yml.hook` 被删除 `.hook` 后缀，然后删除 `.github` 中的其他文件和目录
-   自动 commit 本次操作



## 第三步 修改设置

主要修改 `_config.yml` 中的变量，例如：

-   `url`：如果你需要部署在 Github Pages 上，那么请一定将 url 设置为 `https://<your-github-username>.github.io`
-   `avatar`：头像，可以使用本地图片，放在 `assets/img/favicons/` 目录下，并把此处网址改为相对路径
-   `timezone`
-   `lang`：按照 _<http://www.lingoes.net/en/translator/langcode.htm>_ 上的缩写设置，中文为 `zh`

修改完后，你可以现在本地预览效果，然后再决定是否保存更改并同步到远程仓库中：

```bash
bundle exec jekyll s
```

或者使用 Docker 运行网站：

```bash
docker run -it --rm \
  --volume="$PWD:/srv/jekyll" \
  -p 4000:4000 jekyll/jekyll \
  jekyll serve
```

打开浏览器访问 _<http://localhost:4000>_ 。



## 第四步 Github Pages 部署

请先务必确保 url 设置正确。

首先重新命名 Github 上的远程仓库的名字，修改为 `<your-github-username>.github.io` （不要加尖括号）。

然后重新关联本地项目和远程仓库：

```
git remote set-url https://<your-token>.github.com/<...>.github.io.git
```

GitHub 目前已经不支持密码登陆。至于 GitHub token 的使用方法，请参考此[教程](https://blog.csdn.net/u014175572/article/details/55510825)。由于比较简单，这里就不细讲了。如果你对 token 不懂，那么就按照以下步骤操作，这是一种比较不安全的做法，不符合安全理念和最小权限原则：点击网页右上角头像，选择 Settings ，点击左侧 `Developer settings` ，点击左侧 `Personal access tokens` ，点击 `generate new token` ，选择相应的权限（最简单的方法全选，日期设为不限），把生成的 `token` 复制到上面的 `your-token` 中就行了。注意，此 `token` 只能看到一次，请妥善保管，可重复使用。

设置完毕后通过 `git push` 将本地的修改提交到远程仓库，这将触发 `GitHub Actions workflow` ，一旦操作完成，会产生一个新的分支 `gh-pages` 。在网页上点击项目的 `settings` ，找到 `Pages` ，选择 `gh-pages` 分支作为 `publishing source` ，如下图：

![image-20210825214236259](https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-08-24-Building-your-Blog.assets/image-20210825214236259.png?raw=true)

点击 Github 上的链接就可以访问你的博客啦！



### 第五步 设置百度统计

之所以不使用官方推荐的 Google Analysis ，是因为在国内大部分地区都没有办法直接访问，这种统计也就失效了。因此，我选择了更加简便的 Baidu Analysis 来代替，一样可以达到效果。

首先前往[百度统计](https://tongji.baidu.com/)注册一个账号，注册完成后新建一个网站。

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-08-24-Building-your-Blog.assets/image-20210905112057891.png?raw=true" alt="image-20210905112057891" style="zoom:50%;" />

之后会要填写以下信息：

-   网站域名：`<your/github/username>.github.io`
-   网站首页：`https://<your/github/username>.github.io`
-   剩下的随便写

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-08-24-Building-your-Blog.assets/image-20210905112219946.png?raw=true" alt="image-20210905112219946" style="zoom:40%;" />

完后会会自动跳转到获取代码（如何没有自己点击跳转），选择复制代码。

接下来修改个人博客项目，需要修改三个地方：

1.   修改 `_config.yml` ，在其中加入：

     ```yaml
     baidu-analysis: <your-token>
     ```

     其中 <your-token> 就是那段复制的代码中 ```hm.src = "https://hm.baidu.com/hm.js?..." 后的那串字符。```

2.   在 `_includes` 目录下新建 `baidu-analysis.html` 文件，在其中输入：

     ```html
     <!--
       The BA snippet
     -->
     <script>
     var _hmt = _hmt || [];
     (function() {
       var hm = document.createElement("script");
       hm.src = "https://hm.baidu.com/hm.js?{{ site.baidu-analysis }}";
       var s = document.getElementsByTagName("script")[0]; 
       s.parentNode.insertBefore(hm, s);
     })();
     </script>
     ```

     这里无需修改，直接复制即可。

3.   在页头模版页面中安装百度统计，这里我选择了 `head.html` ，实际查看代码我发现 Google Analysis 是放在 `js-selector.html` 中的，但是该部分属于 `body` 而百度统计官方建议放入 `head` ，因此在 `head.html` 最后的 `</head>` 前加入以下代码：

     ```html
     <!-- BA -->
     {% if site.baidu-analysis %}
       {% include baidu-analysis.html %}
     {% endif %}
     ```

     如果你对代码整洁度有很高的要求（其实我就算是，只是配置更新太慢了就算了），你可以尝试在  `js-selector.html` 中的最后一个 `if` 语句块中加入以上代码。

至此就完成网站统计的配置了，由于更新 `_config.yml` 体现在网站上的时间比较久，因此需要耐心等待，笔者等了一个晚上。去百度统计点击首页代码状态检查，如果显示 `代码安装正确` ，那么恭喜你，你可以查看你的个人博客的访问情况了。



<br/>

如果本篇教程中有任何不对的地方，欢迎联系我指正！（评论功能可能尚在开发中）



---

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)