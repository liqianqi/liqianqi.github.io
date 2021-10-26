---
title: Markdown Fomula
author: Harry-hhj
date: 2021-09-03 18:00:00 +0800
categories: [Tutorial, MathJax]
tags: [tools, latex, markdown]
math: true
mermaid: true
image:
  src: https://raw.githubusercontent.com/Harry-hhj/Harry-hhj.github.io/master/_posts/2021-09-03-Markdown-Formula.assets/markdown.jpeg
  width: 900
  height: 600
---



# Markdown 公式编辑教程

Markdown 中的数学公式，其背后是由 MathJax 提供支持的。MathJax 是一个开源的 web 数学公式渲染器，由 JS 编写而成，提供和 LaTex 一样的公式书写方式。

一般公式分为两种，可以理解为一种特殊的代码块：

-   行内公式：由 `$` 将公式代码块括起

    -   效果：$\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt$

    -   源代码：

        ```latex
        $\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt$
        ```

-   行间公式：由 `$$` 将公式代码块括起

    -   效果：
        $$
        \Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt
        $$
        
    -   源代码：
    
        ```latex
        $$\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt$$
        ```



## 一、希腊字母

| 大写       | code       | 小写       | code       | 名称    |
| ---------- | ---------- | ---------- | ---------- | ------- |
| $A$        | `A`        | $\alpha$   | `\alpha`   | alpha   |
| $B$        | `B`        | $\beta$    | `\beta`    | beta    |
| $\Gamma$   | `\Gamma`   | $\gamma$   | `\gamma`   | gamma   |
| $\Delta$   | `\Delta`   | $\delta$   | `\delta`   | delta   |
| $E$        | `E`        | $\epsilon$ | `\epsilon` | epsilon |
| $Z$        | `Z`        | $\zeta$    | `\zeta`    | zeta    |
| $H$        | `H`        | $\eta$     | `\eta`     | eta     |
| $\Theta$   | `\Theta`   | $\theta$   | `\theta`   | theta   |
| $I$        | `I`        | $\iota$    | `\iota`    | iota    |
| $K$        | `K`        | $\kappa$   | `\kappa`   | kappa   |
| $\Lambda$  | `\Lambda`  | $\lambda$  | `\lambda`  | \lambda |
| $M$        | `M`        | $\mu$      | `\mu`      | \mu     |
| $N$        | `N`        | $\nu$      | `\nu`      | nu      |
| $\Xi$      | `\Xi`      | $\xi$      | `\xi`      | xi      |
| $O$        | `O`        | $\omicron$ | `\omicron` | omicron |
| $\Pi$      | `\Pi`      | $\pi$      | `\pi`      | pi      |
| $P$        | `P`        | $\rho$     | `\rho`     | rho     |
| $\Sigma$   | `\Sigma`   | $\sigma$   | `\sigma`   | sigma   |
| $T$        | `T`        | $\tau$     | `\tau`     | tau     |
| $\Upsilon$ | `\Upsilon` | $\upsilon$ | `\upsilon` | upsilon |
| $\Phi$     | `\Phi`     | $\phi$     | `\phi`     | phi     |
| $X$        | `X`        | $\chi$     | `\chi`     | chi     |
| $\Psi$     | `\Psi`     | $\psi$     | `\psi`     | psi     |
| $\Omega$   | `\Omega`   | $\omega$   | `\omega`   | omega   |



## 二、符号位置

### （1）上下标

上标：`^{}`

下标：`_{}`

当上下标仅为一个元素时，`{}` 可以省略。

举例：`x_i^2` 表示 $x_i^2$

### （2）底部符号

在符号底部写符号： `\underset{}{}` 

在符号顶部写符号： `\overset{}{}` 

举例： `\underset{0\le j \le k-1}{\arg \min}` 表示 $\underset{0\le j \le k-1}{\arg \max}$ ， `a\overset{?}=b` 表示 $a\overset{?}=b$

### （3）底部换行

在符号下部换行： `\understack{}`

举例： `\sum_{\substack{0 \le i \le n \\ 0 \le j \le n}} A_{ij}` 表示 $\sum_{\substack{0 \le i \le n \\ 0 \le j \le n}} A_{ij}$



## 三、括号

### 1. 小括号、方括号

使用原始的 `()` 、`[]` 

举例：`(2+3)[4*4]` 表示 $(2+3)[4*4]$。

### 2. 大括号

需使用转义符号 `\` 阻止解析：`\{\}` 或使用 `\lbrace` 和 `rbrace`

举例：`\{a*b\}:a*b` 或 `\lbrace a*b \rbrace:a*b` 表示 $\lbrace a*b \rbrace:a*b$

### 3. 尖括号

`\langle` 和 `\rangle` 

举例：`\langle x \rangle` 表示 $\langle x \rangle$

### 4. 上取整

`\lceil` 和 `rceil`

举例：`\lceil x \rceil` 表示 $\lceil x \rceil$

### 5. 下取整

`\lfloor` 和 `\rceil`

举例：`\lceil x \rceil` 表示 $\lceil x \rceil$



## 四、求和、乘积、求导、积分

### 1. 求和

`\sum_{}^{}` ，其下标表示求和下限，上标表示求和上限

使用 `\limits` 控制下标是在符号右侧还是上下侧

举例：`\sum_1^n` 表示 $\sum_1^n$ ， `\sum\limits_{i=1}` 表示 $\sum\limits_{i=1}$

### 2. 乘积

`\prod_{}^{}`  ，其下标表示乘积下限，上标表示乘积上限

使用 `\limits` 控制下标是在符号右侧还是上下侧

举例：`\prod_i^n` 表示 $\prod_i^n$ ，`\prod\limits_i^n` 表示 $\prod\limits_i^n$

### 3. 求导

导数： `\cfrac{\mathrm{d}y}{\mathrm{d}x}`

偏导数： `\cfrac{\partial y}{\partial x}`

举例：导数 $\cfrac{\mathrm{d}y}{\mathrm{d}x}$ ，偏导数 $\cfrac{\partial y}{\partial x}$

### 4. 积分

`\int_{}^{}` ，其下标表示积分下限，上标表示积分上限

举例：`\int_0^\infty` 表示 $\int_0^\infty$

多重积分：`\iint` 表示 $\iint$ 、 `\iiint` 表示 $\iiint$ 、 `\iiiint` 表示 $\iiiint$

### 5. 其他

`\bigcup` 表示 $\bigcup$

`\bigcap` 表示 $\bigcap$



## 五、分式与根式

### 1. 分式

`\frac{}{}` 或 `{}\over{}`

举例：`\frac ab` 表示 $\frac ab$ ，`a \over b` 表示 $a \over b$

### 2. 连分数

`\cfrac` 而不要使用 `\frac`

举例：`x=a_0 + \cfrac {1^2}{a_1 + \cfrac {2^2}{a_2 + \cfrac {3^2}{a_3 + \cfrac {4^2}{a_4 + \cdots}}}}` 表示 $x=a_0 + \cfrac {1^2}{a_1 + \cfrac {2^2}{a_2 + \cfrac {3^2}{a_3 + \cfrac {4^2}{a_4 + \cdots}}}}$

### 3. 根式

`\sqrt[]{}` ，其中 `[]` 表示根式的次数， `{}` 表示根式的内容

举例：`\sqrt[4]{\frac xy}` 表示 $\sqrt[4]{\frac xy}$



## 六、多行表达式

### 1. 分类表达式

`\begin{cases}` 和 `\end{cases}` ，其中使用 `\\` 换行，使用 `&` 指示需要对齐的位置，`\ ` 表示空格。

举例：

```latex
f(n)
\begin{cases}
\cfrac n2, &if\ n\ is\ even\\
3n + 1, &if\  n\ is\ odd
\end{cases}
```

表示
$$
f(n)
\begin{cases}
\cfrac n2, &if\ n\ is\ even\\
3n + 1, &if\  n\ is\ odd
\end{cases}
$$
使用 `\\[2ex]` 代替 `\\` （相当于 `\\[1ex]`）增大分类之间的垂直距离，以此类推。



### 2. 方程

`\begin{equation}` 和 `\end{equation}`

举例：

```latex
\begin{equation}
a = b + c - d
\end{equation}
```

表示
$$
\begin{equation}
a = b + c - d
\end{equation}
$$


### 3. 多行表达式

`\begin{split}` 和 `\end{split}`

举例：

```latex
\begin{equation}\begin{split} 
a&=b+c-d \\ 
&\quad +e-f\\ 
&=g+h\\ 
& =i 
\end{split}\end{equation}
```

表示
$$
\begin{equation}\begin{split} 
a & = b + c - d \\ 
  & \quad + e - f \\ 
  & = g + h \\ 
  & = i 
\end{split}\end{equation}
$$


### 4. 方程组

`\left \{` 与 `\right .` 和 `\begin{array}` 和 `\end{array}` 配合使用，或使用 `\beign{cases}` 与 `\end{cases}`

举例：

```latex
\left \{ 
\begin{array}{c}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{array}
\right .
```

表示
$$
\left \{ 
\begin{array}{c}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{array}
\right .
$$



## 七、特殊函数与符号

### 1. 三角函数

`\sin x` 表示 $\sin x$

`\cos x` 表示 $\cos x$

`\arctan x` 表示 $\arctan x$

以此类推

### 2. 比较符号

小于 $\lt$ ：`\lt`

大于 $\gt$：`\gt`

小于等于 $\le$：`\le`

大于等于 $\ge$ ：`\ge`

不等于 $\ne$ ：`\ne`

### 3. 集合关系与运算

并集 $\cup$ ：`\cup`

交集 $\cap$ ：`\cap`

差集 $\setminus$ ：`\setminus`

子集 $\subset$ ：`\subset`

父集 $\supset$ ：`\supset`

属于 $\in$ ：`\in`

不属于 $\notin$ ：`\notin`

包含 $\subseteq$ ：`\subseteq`

非包含 $\subsetneq$ ：`\subsetneq`

空集 $\emptyset$ ：`\emptyset`

空 $\varnothing$ ：`\varnothing`

### 4. 排列组合

二项式 $\binom{n+1}{2k}$ ：`\binom{n+1}{2k}` 或 `{n+1 \choose 2k}`

### 5. 箭头

$\to$ ：`\to`

$\rightarrow$ ：`\rightarrow`

$\leftarrow$ ： `\leftarrow`

$\leftrightarrow$ ： `\leftrightarrow`

$\uparrow$ ： `\uparrow`

$\downarrow$ ： `\downarrow`

$\updownarrow$ ： `updownarrow`

$\Rightarrow$ ： `\Rightarrow`

$\Leftarrow$ ： `\Leftarrow`

$\Leftrightarrow$ ： `\Leftrightarrow`

$\Uparrow$ ： `\Uparrow`

$\Downarrow$ ： `\Downarrow`

$\Updownarrow$ ： `Updownarrow`

$\longrightarrow$ ： `\longrightarrow`

$\longleftarrow$ ： `\longleftarrow`

$\longleftrightarrow$ ： `\longleftrightarrow`

$\Longrightarrow$ ： `\Longrightarrow`

$\Longleftarrow$ ： `\Longleftarrow`

$\Longleftrightarrow$ ： `\Longleftrightarrow`

$\xleftarrow[T]{n=0}$ ： `\xleftarrow[]{}`

$\xrightarrow[T]{n>0}$ ： `\xrightarrow[]{}`

$\mapsto$ ： `\mapsto`

更多箭头符号可以查看[这篇教程](https://blog.csdn.net/liyuanbhu/article/details/51473886)。

### 6. 逻辑运算符

$\land$： `\land`

$\lor$： `\lor`

$\lnot$： `\lnot`

$\forall$： `\forall`

$\exists$： `\exists`

$\top$： `\top`

$\bot$： `\bot`

$\vdash$： `\vdash`

$\vDash$： `\vDash`

### 7. 定义

$\triangleq$ ： `\triangleq`

### 8. 操作符

$\star$： `\star`

$\ast$： `\ast`

$\oplus$：`\oplus`

$\circ$： `\circ`

$\bullet$： `\bullet`

### 9. 等于

$\approx$： `\approx`

$\sim$： `\sim`

$\cong$： `\cong`

$\equiv$： `\equiv`

$\prec$： `\prec`

### 10. 范围

$\infty$： `infty`

$\aleph_o$： `aleph_o`

$\nabla$ ： `\nabla`

$\partial$： `\partial`

$\Im$： `\Im`

$\Re$： `\Re`

### 11. 模运算

$\pmod p$： `pmod p`

### 12. 点

$\ldots$： `ldots`

$\cdots$： `cdots`

$\cdot$： `cdot`



## 八、顶部符号

$\hat x$： `\hat{}`

$\widehat{xy}$： `\widehat{}`

$\overline x$： `\overline{}`

$\vec x$：`\vec{}`

$\overrightarrow x$：`\overrightarrow{}`

$\dot x$ ： `\dot{}`

$\ddot x$： `\ddot{}`



## 九、表格

`\begin{array}{列样式}` 和 `\end{array}`

TODO：列样式

```latex
\begin{array}{c|lcr}
n & \text{Left} & \text{Center} & \text{Right} \\
\hline
1 & 0.24 & 1 & 125 \\
2 & -1 & 189 & -8 \\
3 & -20 & 2000 & 1+10i \\
\end{array}
```

效果：
$$
\begin{array}{c|lcr}
n & \text{Left} & \text{Center} & \text{Right} \\
\hline
1 & 0.24 & 1 & 125 \\
2 & -1 & 189 & -8 \\
3 & -20 & 2000 & 1+10i \\
\end{array}
$$



## 十、矩阵

### 1. 基本

`\begin{matrix}` 和 `\end{matrix}`

```latex
\begin{matrix}
1 & x & x^2 \\
1 & y & y^2 \\
1 & z & z^2 \\
\end{matrix}
```

效果：
$$
\begin{matrix}
1 & x & x^2 \\
1 & y & y^2 \\
1 & z & z^2 \\
\end{matrix}
$$

### 2. 加括号

可以使用 `\left` 和 `\right` 配合表示括号符号，也可以使用特殊矩阵：

`pmatrix`： $$\begin{pmatrix}1&2\\3&4\\\end{pmatrix}$$

`bmatrix`： $$\begin{bmatrix}1&2\\3&4\\\end{bmatrix}$$

`Bmatrix`： $$\begin{Bmatrix}1&2\\3&4\\\end{Bmatrix}$$

`vmatrix`： $$\begin{vmatrix}1&2\\3&4\\\end{vmatrix}$$

`Vmatrix`： $$\begin{Vmatrix}1&2\\3&4\\\end{Vmatrix}$$

### 3. 省略元素

$\cdots$ ： `\cdots`

$\vdots$ ： `\vdots`

$\ddots$ ：`\ddots` 

举例：

```latex
\begin{pmatrix}
1&a_1&a_1^2&\cdots&a_1^n\\
1&a_2&a_2^2&\cdots&a_2^n\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
1&a_m&a_m^2&\cdots&a_m^n\\
\end{pmatrix}
```

效果：
$$
\begin{pmatrix}
1&a_1&a_1^2&\cdots&a_1^n\\
1&a_2&a_2^2&\cdots&a_2^n\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
1&a_m&a_m^2&\cdots&a_m^n\\
\end{pmatrix}
$$

### 4. 增广矩阵

`\left` 与 `\right` 和 `\begin{array}` 与 `\end{array}`

举例：

```latex
\left[
\begin{array}{cc|c}
1&2&3\\
4&5&6
\end{array}
\right]
```

效果：
$$
\left[
\begin{array}{cc|c}
1&2&3\\
4&5&6
\end{array}
\right]
$$



## 十一、公式标记与引用

`\tag{}` 来标记公式，如果以后需要引用此公式，还需要加上 `\label{}` 在 `\tag{}` 之后。

举例：

```latex
a := x^2 - y^3 \tag{10}\label{10}
```

效果：
$$
a := x^2 - y^3 \label{10}
$$
`\stackrel{\eqref{}}{}` 引用公式。

举例：

```latex
a + y^3 \stackrel{\eqref{10}}= x^2
```

效果：
$$
a + y^3 \stackrel{\eqref{10}}= x^2
$$
`\stackrel{\ref{}}{}` 不带括号引用。

举例

```latex
a + y^3 \stackrel{\ref{10}}= x^2
```

效果：
$$
a + y^3 \stackrel{\ref{10}}= x^2
$$



## 十二、字体

### 1. 黑板粗体字

`\mathbb` 或 `\Bbb` ，此字体常用来表示实数、整数、有理数、复数和大写字母。

举例：

```latex
\mathbb CHNQRZ
\Bbb CHNQRZ
```

效果：$$\mathbb CHNQRZ$$



### 2. 黑体字

`\mathbf`

举例：

```latex
\mathbf CHNQRZ
```

效果：$$\mathbf CHNQRZ$$



### 3. 打印机字体

`\mathtt`

举例：

```latex
\mathtt ABCDEFGHIJKLMNOPQRSTUVWXYZ
```

效果：$$\mathtt ABCDEFGHIJKLMNOPQRSTUVWXYZ$$



### 4. 罗马字体

`\mathrm`

举例：

```latex
\mathrm abcdefghijklmnopqrstuvwxyz
```

效果：$$\mathrm abcdefghijklmnopqrstuvwxyz$$



### 5. 手写字体

`\mathscr`

举例：

````latex
\mathscr abcdefghijklmnopqrstuvwxyz
````

效果：$$\mathscr abcdefghijklmnopqrstuvwxyz$$



### 6. Fraktur 字母（一种德国字体）

`\mathfrak`

举例：

```latex
\mathfrak ABCDEFGHIJKLMNOPQRSTUVWXYZ
```

效果：$$\mathfrak ABCDEFGHIJKLMNOPQRSTUVWXYZ$$



## 十三、颜色

`\color{}`

举例：

| 颜色   | 源码                        | 效果                      |
| ------ | --------------------------- | ------------------------- |
| 黑色   | ```\color{black}{text}```   | $$\color{black}{text}$$   |
| 灰色   | ```\color{grey}{text}```    | $$\color{grey}{text}$$    |
| 银色   | ```\color{silver}{text}```  | $$\color{silver}{text}$$  |
| 白色   | ```\color{white}{text}```   | $$\color{white}{text}$$   |
| 褐红色 | ```\color{maroon}{text}```  | $$\color{maroon}{text}$$  |
| 红色   | ```\color{red}{text}```     | $$\color{red}{text}$$     |
| 黄色   | ```\color{yellow}{text}```  | $$\color{yellow}{text}$$  |
| 绿黄色 | ```\color{lime}{text}```    | $$\color{lime}{text}$$    |
| 橄榄色 | ```\color{olive}{text}```   | $$\color{olive}{text}$$   |
| 绿色   | ```\color{green}{text}```   | $$\color{green}{text}$$   |
| 深蓝绿 | ```\color{teal}{text}```    | $$\color{teal}{text}$$    |
| 水绿色 | ```\color{aqua}{text}```    | $$\color{aqua}{text}$$    |
| 蓝色   | ```\color{blue}{text}```    | $$\color{blue}{text}$$    |
| 海军蓝 | ```\color{navy}{text}```    | $$\color{navy}{text}$$    |
| 紫色   | ```\color{purple}{text}```  | $$\color{purple}{text}$$  |
| 浅莲红 | ```\color{fuchsia}{text}``` | $$\color{fuchsia}{text}$$ |



## 十四、其他

### 1. 空格

MathJax 通过**内部策略**管理自己的公式内的空间，因此公式中无论空多少格最后都不会有效果，可以插入`\,` 增加些许间隙，插入 `\;` 插入较宽的间隙，`\quad` 与 `\qquad` 会增肌更大的间隙。

### 2. 括号

任何运算符后的 `{}` 在只有单个字符时可以省略，但此时如果为单个字母则需加入空格分隔。如：`\frac12` 表示 $\frac12$ ，`\frac ab` 表示 $\frac ab$。

### 3. 转义

对于 MathJax 的保留字符，如 `{% raw %}\{% endraw %}` 、 `{% raw %}{}{% endraw %}` 等，如果不是为了解析，那么需要在前面添加转义字符 `{% raw %}\{% endraw %}` 。



<br/>

**如果觉得本教程不错或对您有用，请前往项目地址 [https://github.com/Harry-hhj/Harry-hhj.github.io](https://github.com/Harry-hhj/Harry-hhj.github.io) 点击 Star :) ，这将是对我的肯定和鼓励，谢谢！**

<br/>



## 十五、参考文献

1.   [Mathjax公式教程](https://blog.csdn.net/dabokele/article/details/79577072)



-----

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

