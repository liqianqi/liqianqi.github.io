---
title: Linux Commands
author: Harry-hhj
date: 2021-09-22 22:00:00 +0800
categories: [Tutorial, Linux]
tags: [tools, linux]
math: false
mermaid: false
pin: false
image:
  src: https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-22-Linux-Commands.assets/linux.jpeg?raw=true
  width: 480
  height: 272
---



# Linux 常用命令

## 一、目录

| 命令                     | 意义                       |
| ------------------------ | -------------------------- |
| [`shutdown`](#基本命令)  | （立即/定时）关机，重启    |
| [`poweroff`](#基本命令)  | 立即关机                   |
| [`reboot`](#基本命令)    | 立即重启                   |
| [`man`](#基本命令)       | 帮助                       |
| [`cd`](#目录操作)        | 进入目录                   |
| [`ls`](#目录操作)        | 枚举目录                   |
| [`mkdir`](#目录操作)     | 创建目录                   |
| [`rm`](#目录操作)        | 删除目录/文件              |
| [`mv`](#目录操作)        | 移动目录/文件，重命名      |
| [`cp`](#目录操作)        | 拷贝目录/文件              |
| [`touch`](#)             | 新建目录/文件              |
| [`rm`](#文件操作)        | 删除目录/文件              |
| [`vi/vim`](#文件操作)    | 命令行操作文件             |
| [`cat`](#文件操作)       | 查看文件                   |
| [`more`](#文件操作)      | 分页查看文件               |
| [`less`](#文件操作)      | 随意查看文件               |
| [`tail`](#文件操作)      | 查看文件尾部，适用实时更新 |
| [`chmod`](#文件操作)     | 改变目录/文件全新啊        |
| [`chown`](#文件操作)     | 改变拥有者                 |
| [`tar`](#压缩/解压)      | 打包/解包/压缩/解压        |
| [`grep`](#查找)          | 搜索过滤                   |
| [`find`](#查找)          | 递归搜索                   |
| [`locate`](#查找)        | 搜索路径                   |
| [`whereis`](#查找)       | 命令路径                   |
| [`which`](#查找)         | 命令位置                   |
| [`su`](#用户切换)        | 用户切换                   |
| [`sudo`](#用户切换)      | 单次 root 权限             |
| [`service`](#系统服务)   | 配置服务                   |
| [`chkconfig`](#系统服务) | 开机自启                   |
| [`crontab`](#定时任务)   | 定时任务                   |
| [`pwd`](#其他)           | 打印当前路径               |
| [`ps`](#其他)            | 查看进程信息               |
| [`kill`](#其他)          | 杀死进程                   |
| [`ifconfig`](#其他)      | 查看网卡配置               |
| [`ping`](#其他)          | 查看连接情况               |
| [`netstat`](#其他)       | 查看端口                   |
| [`clear`](#其他)         | 清空输出                   |
| [`>`](#其他)             | 重定向符                   |
| [`|`](#其他)             | 管道                       |



## 二、<span id="基本命令">基本命令</span>

### 1. 关机

`shutdown` or `poweroff`

```shell
shutdown -h now		# 立即关机
shutdown -h 5			# 5 分钟后关机
poweroff           # 立即关机
```



### 2. 重启

`shutdown` or `reboot`

```shell
shutdown -r now		# 立即重启
shutdown -r 5			# 5 分钟后重启
reboot					 # 立即重启
```



### 3. 帮助

`--help` or `man`

```shell
shutdown --help
man shutdown			# 打开命令说明书之后，使用按键 q 退出
```



## 三、<span id="目录操作">目录操作</span>

### 1. 切换

`cd [<destination>]`

```shell
cd /			# 切换到根目录
cd /usr		# 切换到根目录下的 usr 目录
cd ..			# 切换到上级目录
cd ~       # 切换到 home 目录
cd				# 同上
cd -       # 切换到上次访问的目录
```



### 2. 查看

`ls [-al] [<target>]`

```shell
ls			  # 查看当前目录下的所有目录和文件
ls -a			# 查看当前目录下的所有目录和文件（包含隐藏文件）
ls -l			# 查看当前目录下的所有目录和文件（列表，包含更多信息）
ll				# 同上
ls /usr		 # 查看指定目录 /usr 下的所有目录和文件
```



### 3. 增删改查

#### 1) 创建目录

`mkdir <folder_name>`

```shell
mkdir aaa					# 在当前目录下创建一个名为 aaa 的目录
mkdir /usr/aaa		 # 在指定目录 /usr 下创建一个名为 aaa 的目录
```



#### 2) 删除目录

`rm [-rf] <target>`

为了方便实用你可以一直带上 `-rf` 参数，不管你的对象是目录还是压缩包还是文件。

```shell
rm -r aaa			# 递归删除当前目录下的 aaa 目录，有些文件会询问是否确认删除
rm -rf aaa		# 递归删除当前目录下的 aaa 目录，不询问

# 以下命令务必慎用！！！
rm -rf *			# 将当前目录下的所有目录和文件全部删除
rm -rf /*			# 将根目录下的所有文件全部删除（只是为了让你明白后果，你不可能会用到这个命令的！）
```



#### 3) 目录复制/移动

`mv <source> <destination>`

除了移动之外，还具有重命名的效果。

```shell
mv <old_path/folder_or_file> <new_path>		# 将 folder_or_file 从 <old_path> 移动到 <new_path>
mv aaa bbb 												# 将目录/文件 aaa 重命名为 bbb
```

`cp [-r] <source> <destination>`

```shell
cp <path/of/file> <target_folder> 		# 将文件 <path/of/file> 拷贝到 <target_folder>
cp -r <path/of/folder> <target_path>	# 将目录 <path/of/folder> 拷贝到 <target_path>
```



#### 4) 搜索目录

`find <folder> <opt> <param>`

```shell
find /usr/tmp -name 'a*'		# 查找 /usr/tmp 目录下的所有以 a 开头的目录或文件
```



## 四、<span id="文件操作">文件操作</span>

### 1. 增删该查

#### 1) 新建文件

`touch <name>`

```shell
touch a.txt			# 在当前目录下新建 a.txt 文件
touch a					# 在当前目录下新建 a 目录
```



#### 2) 删除文件

`rm [-f] <name>`

```shell
rm <file> 		# 删除文件 file
rm -f <file>	#	删除文件 file 且不询问
```



#### 3) 修改文件

`vi` or `vim`

```shell
vi <file_name>
vim <file_name>
```

基本上 vi 可以分为三种状态：

-   命令模式 Command mode ：控制屏幕光标的移动，字符、字或行的删除，查找，移动复制某区段

    -   光标移动 $\leftarrow$ 、 $\rightarrow$ 、 $\uparrow$ 、$\downarrow$ ，分别对应 `h` 、`l` 、 `k` 、 `j` 

    -   删除当前行： `dd` 

    -   查找： `/<string>`

        <img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-22-Linux-Commands.assets/image-20210923120223984.png?raw=true" alt="image-20210923120223984" style="zoom:30%;" />

    -   进入编辑模式：

        -   `i` ：在光标所在字符前开始插入
        -   `a ` ：在光标所在字符后开始插入
        -   `o` ：在光标所在行的下面另起一新行插入

    -   进入底行模式： `:`

-   插入模式 Insert mode ：只有在插入模式下才能做文字输入

    -   回到命令模式：`[ESC]`

-   底行模式 last line mode ：将文件保存或退出，也可以设置编辑环境，如寻找字符串、列出行号等

    -   退出： `:q`
    -   强制退出： `:q!`
    -   保存并退出： `:wq`

注意只有命令模式才能随意进入其他两种模式。



#### 4) 查看文件

`cat` or `more` or `less` or `tail`

```shell
cat /etc/bash.bashrc				# 一次性显示文件全部内容
more /etc/bash.bashrc				# 以一页一页的形式显示文件，按 b 后退，按 space 前进
less /etc/bash.bashrc				# 随意浏览文件，支持翻页和搜索
tail -10 /etc/bash.bashrc		# 显示 /etc/bash.bashrc 的最后 10 行
tail -f <filename>					# 把 filename 文件里的最尾部的内容显示在屏幕上，并且不断刷新，适用于查阅正在改变的日志文件
```



### 2. 修改权限

Linux 下的权限类型有三种： `r` 代表可读， `w` 代表可写， `x` 代表该文件是一个可执行文件， `-` 代表不可读或不可写或不可执行文件。

Linux 下的权限粒度有三类： `User` 、 `Group` 、 `Other` 。 User 表示某一登陆用户， Group 表示和 User 同组的其他用户，其他用户都属于 Other 。举个例子，公司的一个员工是 User ，公司的同事就是 Group ，而公司之外的人就是 Other 。

在 Linux 中一般用 10 个字符表示一个文件/目录的权限，格式如下：

```text
【表示文件类型，d 表示是目录，l 是符号链接文件】-/d/s/p/l/b/c | 【User 的权限】-rwx/-rwx/-rwx ｜ 【Group 的权限】-rwx/-rwx/-rwx ｜ 【Other 的权限】-rwx/-rwx/-rwx
```

例如下图中，

<img src="https://github.com/Harry-hhj/Harry-hhj.github.io/blob/master/_posts/2021-09-22-Linux-Commands.assets/image-20210923122037473.png?raw=true" alt="image-20210923122037473" style="zoom:30%;" />

红色部分表示 `vim.txt` 是一个文件， User 和 Group 可读可写，但不可执行，Group 可读，但不可修改删除、不可执行。

补充知识：权限也可以用数字来表示，成为 8421 表示法。规则是 $\text r = 4, \text w = 2, \text x = 1$ ，如上图中可以表示为 $664$ 。

补充知识：[附加权限位](https://blog.csdn.net/u013197629/article/details/73608613)。

`chmod ` or `chown [-R] user[:group] file`

```shell
chmod u+x a.txt			# 仅 User 追加执行权限
chmod a+x a.txt			# 所有用户追加执行权限
chmod 100 a					# 仅 User 可执行 a ，所有人不可读不可修改

chown tom:users file d.key		# 设置文件 d.key 的拥有者设为 users 群体的 tom
chown -R James:users  *				# 设置当前目录下与子目录下的所有文件的拥有者为 users 群体的 James
```



## 五、<span id="压缩/解压">压缩/解压</span>

### 1. 打包压缩

Linux 中的打包文件的后缀为 `.tar` ，压缩文件的后缀为 `.gz` ，打包并压缩的文件后缀为 `.tar.gz` 。

`tar [-zcvf]`

参数说明：

-   `z` ：调用 gzip 压缩命令进行压缩
-   `c` ：打包
-   `v` ：显示运行过程
-   `f` ：指定文件名

```shell
tar -cvf <target.tar> <sources>		# 仅打包不压缩 <sources> 的全部文件及目录，生成 <target.tar>
tar -zcvf <target.tar.gz> <sources>	# 打包 <sources> 中的全部文件及目录，并压缩为 <target.tar.gz>
```



### 2. 解压

`tar [-zxvf] <target>`

参数说明：

-   `z` ：调用 gzip 压缩命令进行解压

-   `x` ：解包
-   `v` ：显示运行过程
-   `f` ：指定文件名
-   `-C` ：指定解压的位置

```shell
tar -xvf <source.tar>									# 将 <source.tar> 解压到当前目录下
tar -xvf <source.tar> -C <destination>	# 将 <source.tar> 解压到指定目录 <destination> 下
```



## 六、<span id="查找">查找</span>

### 1. grep

强大的文本搜索工具

```shell
ps -ef | grep sshd | grep -v grep		# 查找指定服务进程，排除gerp身
ps -ef | grep sshd								# -c 查找指定进程个数
grep <file> -E [-v] <regex>								# 在文件 file 中查找包含 regex 的行并输出， -v 表示不包含
```



### 2. find

默认搜索当前目录及其子目录，并且不过滤任何结果（也就是返回所有文件），将它们全都显示在屏幕上。

```shell
find . -name "*.log" -ls  		# 在当前目录查找以.log结尾的文件，并显示详细信息。 
find /root/ -perm 600					# 查找/root/目录下权限为600的文件 
find . -type f -name "*.log"		# 查找当目录，以.log结尾的普通文件 
find . -type d | sort					# 查找当前所有目录并排序 
find . -size +100M						# 查找当前目录大于100M的文件
```



### 3. locate

让使用者可以很快速的搜寻某个路径。

默认每天自动更新一次，可以在使用 `locate` 之前，先使用 `updatedb` 命令，手动更新数据库。可能需要先安装命令： ``apt install mlocate`` 。

```shell
locate /etc/sh	# 搜索etc目录下所有以sh开头的文件 
locate pwd			# 查找和 pwd 相关的所有文件
```



### 4. whereis

定位可执行文件、源代码文件、帮助文件在文件系统中的位置。

```shell
whereis ls		# 将和 ls 文件相关的文件都查找出来
```



### 5. which

在 PATH 变量指定的路径中，搜索某个系统命令的位置，并且返回第一个搜索结果。

```shell
which pwd			# 查找pwd命令所在路径 
```



## 七、<span id="用户切换">用户切换</span>

### 1. su

用于用户之间的切换。但是切换前的用户依然保持登录状态。如果是 root 向普通或虚拟用户切换不需要密码，反之普通用户切换到其它任何用户都需要密码验证。

```shell
su test			# 切换到test用户，但是路径还是/root目录
su - test		# 切换到test用户，路径变成了/home/test
su					# 切换到root用户，但是路径还是原来的路径
su -				# 切换到root用户，并且路径是/root
exit				# 退出返回之前的用户
```



### 2. sudo

为所有想使用root权限的普通用户设计的。可以让普通用户具有临时使用root权限的权利。只需输入自己账户的密码即可。

配置文件：

```shell
sudo vi /etc/sudoers
sudo visudo
```

配置案例如下：

```shell
hadoop  ALL=(ALL)   ALL			# 允许 hadoop 用户以 root 身份执行各种应用命令，需要输入 hadoop 用户的密码
hadoop  ALL=NOPASSWD:  /bin/ls, /bin/cat	# 只允许 hadoop 用户以 root 身份执行 ls 、cat 命令，并且执行时候免输入密码
```



## 八、<span id="系统服务">系统服务</span>

```shell
service iptables status		# 查看 iptables 服务的状态
service iptables start		# 开启 iptables 服务
service iptables stop			# 停止 iptables 服务
service iptables restart	# 重启 iptables 服务
 
chkconfig iptables off		# 关闭 iptables 服务的开机自启动
chkconfig iptables on			# 开启 iptables 服务的开机自启动
```



## 九、<span id="网络管理">网络管理</span>

### 1. 主机名配置

```shell
vi /etc/sysconfig/network
```

```shell
NETWORKING=yes
HOSTNAME=node1
```



### 2. IP 地址配置

```shell
vi /etc/sysconfig/network-scripts/ifcfg-eth0
```



### 3. 域名映射

`/etc/hosts` 文件用于在通过主机名进行访问时做 ip 地址解析之用。所以，你想访问一个什么样的主机名，就需要把这个主机名和它对应的 ip 地址。

```shell
vi /etc/hosts
```

```shell
# 在最后加上
192.168.52.201  node1
192.168.52.202  node2
192.168.52.203  node3
```



## 十、<span id="定时任务">定时任务</span>

`crontab [-u <user>] <file>` or `crontab [-u user] [-e|-l|-r]`

通过 `crontab` 命令，可以在固定间隔时间，执行指定的系统指令或 shell 脚本。时间间隔的单位可以是分钟、小时、日、月、周及以上的任意组合。

首先你需要先安装 `crontab` ：

```shell
apt install crontabs
```

服务操作说明：

```shell
service crond start   	# 启动服务 
service crond stop    	# 关闭服务 
service crond restart 	# 重启服务
```

参数说明：

-   `[-u <user>]` ：用来设定某个用户的 crontab 服务
-   `<file>` ：crontab 的任务列表文件
-   `-e` ：编辑某个用户的 crontab 文件内容。如果不指定用户，则表示编辑当前用户的 crontab 文件。
-   `-l` ：显示某个用户的 crontab 文件内容。如果不指定用户，则表示显示当前用户的 crontab 文件内容。
-   `-r` ：删除定时任务配置，从 `/var/spool/cron` 目录中删除某个用户的 crontab 文件，如果不指定用户，则默认删除当前用户的 crontab 文件。

```shell
crontab file [-u user]		# 用指定的文件替代目前的 crontab
crontab -l [-u user]			# 列出用户目前的 crontab
crontab -e [-u user]			# 编辑用户目前的 crontab
```

配置：

-   第 1 列表示分钟 1～59  ，每分钟用 `*` 或者 `*/1` 表示
-   第 2 列表示小时 0～23 （ 0 表示 0 点）
-   第 3 列表示日期 1～31  
-   第 4 列表示月份 1～12  
-   第 5 列标识号星期 0～6 （ 0 表示星期天）  
-   第 6 列要运行的命令

```shell
# 每分钟执行一次date命令 
*/1 * * * * date >> /root/date.txt
 
# 每晚的21:30重启apache。 
30 21 * * * service httpd restart
 
# 每月1、10、22日的4 : 45重启apache。  
45 4 1,10,22 * * service httpd restart
 
# 每周六、周日的1 : 10重启apache。 
10 1 * * 6,0 service httpd restart
 
# 每天18 : 00至23 : 00之间每隔30分钟重启apache
0,30   18-23    *   *   *   service httpd restart
# 晚上11点到早上7点之间，每隔一小时重启apache
*  23-7/1    *   *   *   service httpd restart
```



## 十一、<span id="其他">其他</span>

### 1. 重定向

`>` or `>>` or `2>&1`

输出重定向到一个文件或设备。

```shell
ls > a.txt		# 将 ls 结果输出到 a.txt 文件
echo "This the end of the file." >> a.txt 	# 在 a.txt 末尾追加 This the end of the file. 这句话
./main 2>&1 main.log		# 将 main 运行时的标准输出和标准错误都输出到 main.log 中
```



### 2. 管道

`|`

将一个命令的输入变成另一个命令的输出。

```shell
find . -type f -readable -regex '.*\.c\|.*\.h' | xargs -I {} grep -c -H 'hello' {}		# 从当前目录开始递归寻找可读的 .c 和 .h 结尾的文件，查看文件并输出具有 hello 的总行数
```



### 3. 查看当前路径

```shell
pwd		# 打印出当前路径
```



### 4. 查看进程

```shell
ps -ef		# 输出所有进程信息
```



### 5. 结束进程

```shell
kill [-9] <pid>		# 杀死进程号为 pid 的进程， -9 表示强制。
```



### 6. 网络通信

```shell
ifconfig		# 查看网卡信息，一般用来看 dhcp 的 ip 地址
ping <ip>		# 查看与机器 <ip> 的连接情况
netstat -an	# 查看当前系统端口
```



### 7. 关闭防火墙

```shell
chkconfig iptables off
service iptables stop
```



### 8. 清屏

```shell
clear
```

快捷键： `ctrl+l`



## 十二、参考资料

1.   [Linux常用命令](https://blog.csdn.net/qq_23329167/article/details/83856430/)
2.   [Linux 命令大全](https://www.runoob.com/linux/linux-command-manual.html)
3.   [Linux权限详解（chmod、600、644、666、700、711、755、777、4755、6755、7755）](https://blog.csdn.net/u013197629/article/details/73608613)
4.   [Linux中重定向](https://www.cnblogs.com/crazylqy/p/5820957.html)
5.   [Linux Shell管道详解](http://c.biancheng.net/view/3131.html)



-----

作者：Harry-hhj，github主页：[传送门](https://github.com/Harry-hhj)

