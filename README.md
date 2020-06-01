# Detecting Text in Natural Image with Connectionist Text Proposal Network
# [ctpn_text_detection_pro](https://github.com/wandaoyi/ctpn_text_detection_pro)

- [论文地址](https://arxiv.org/abs/1609.03605)
- [我的 CSDN 博客](https://blog.csdn.net/qq_38299170/article/details/106006594) 
本项目使用 python3, keras 和 tensorflow 相结合。本模型基于 FPN 网络 和 vgg16 背骨，使用 文本连接机制提议网络(CTPN) 直接对卷积层文本序列进行定位。

```bashrc
ctpn_50000.ckpt:
- 链接：https://pan.baidu.com/s/12Or2Ty9OJG4Mev7ZOPHulA 
- 提取码：wpaa 

vgg_16.ckpt：
- 链接：https://pan.baidu.com/s/1_PptE3o6z72qOx7Klpe8oA 
- 提取码：u1yc

train data:
- 链接：https://pan.baidu.com/s/1yVcyXcp6hVG11RG4tvFJ2g 
- 提取码：0oo2

```

# Getting Started

### 训练数据(网络数据下载)
* 将上面 baidu 云盘下载的数据放 config.py 指定的文件，即可以训练。
* ./dataset/image 内为需要训练的图像
* ./dataset/label 为 上面 image 文件夹中对应的 label.txt 文件
* 当然，最好是使用自己的数据来做训练，因为每个业务的数据特征都有所不同，
* 而一个模型不可能是万能的，训练自己的模型符合自己的业务才是王道。
* 观看数据 image 和 label 的详情请参考 dataset.py 的 main 方法
* 然后自己制作自己的数据，就 OK 了。


### 参考 config.py 文件配置。
* 下面文件中 def __init__(self) 方法中的配置文件，基本都是来自于 config.py


### 测试看效果:
* ctpn_test.py 下载好 ctpn_50000.ckpt 模型，
* 随便找点数据，设置好配置文件，直接运行看结果吧。


### 准备训练数据(自己数据制作)
* 准备好 图像 和 相应的 .xml 文件到指定文件夹后
* 一键运行 prepare.py 文件，就 OK 了。
* 之后，可以 放部分数据到 dataset.py 指定的文件中，
* 或修改 config.py 指向你的文件夹路径，也可以的。
* 最后，一键运行 dataset.py 看效果


### 数据训练:
* ctpn_train.py 直接运行代码，观察 loss 情况。
* 设置 config.py 中的参数，保存模型。


### 多线程 训练:
* 参考 dataset.py 中的 get_batch(), 本项目是可以多线程训练的，
* 但是 windows 不支持 多核启动，直接丢到 Linux 下，就可以解决的了。
* 强烈建议使用多线程训练，不然，实在太慢了，效率太低了。


* 本项目，操作上，就三板斧搞定，不搞那么复杂，吓到人。



