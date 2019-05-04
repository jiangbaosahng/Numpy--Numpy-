# Numpy--用Numpy搭建一个神经网络-
## 第一部分
一个简单的神经网络结构包括，输入层，输出层以及隐藏层。隐藏层里面包含激活函数。
![Image text](https://github.com/jiangbaosahng/Numpy--Numpy-/blob/master/image/timg.jpg)

神经网络简单来说就是一个复杂的函数。拿最基本的函数来举例子：

Y= W*X+b

其中，输入层是X，Y是输出层，而隐藏层确定的是W，即权重。

权重的更新和确定需要梯度下降、激活函数等机制。

在第一部分中，我们使用的激活函数包括两个部分，其一是tanh函数，另一个是softmax 函数。

tan(x)的特点时候在[-2,2]这个区间内有较大的变化。其他输入不会有太大的变化。
图像如下：

![Image text](https://github.com/jiangbaosahng/Numpy--Numpy-/blob/master/image/timg.jpg)

softmax函数图像和表达式如下（其加和等于1）：

![Image text](https://github.com/jiangbaosahng/Numpy--Numpy-/blob/master/image/u%3D1310602997%2C3054892262%26fm%3D26%26gp%3D0.jpg)

```
import numpy as np
import math
```
上面是正常用import导入包，这没有什么好说的。
