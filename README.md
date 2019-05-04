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

接下来定义激活函数，我们搭建的神经网络只有一个隐藏层，所以可以使用两个不同的激活函数。这两个激活函数就是前面提到的，tanh以及softmax函数。

代码如下：
```
def tanh(x):
    return np.tanh(x)
    
def softmax(x):
    exp=np.exp(x-x.max())
    return exp/exp.sum()
```

因为指数函数变化较大吗，所以我们用x-x.max 缩小其变化范围，这对结果不影响。

以后我们使用的图片会是 28*28像素手写数据集里面的，所以其有10个输入分别是数字[1~10]。

```
dinensions=[28*28,10]
activation=[tanh,softmax]
distribution=[{'b':[0,0]},
              {'b':[0,0],'w':[-math.sqrt(6/(dinensions[0]+dinensions[1])),math.sqrt(6/(dinensions[0]+dinensions[1]))]}]
```

上面代码中，dinensions和activation是两个列表，可以看出，activation列表里面是两个函数，所以这也说明了列表里面可以包含的数据类型十分繁多。

distribution 列表里面对应的是字典数据，分别对应神经网络第一层的参数和第二层的参数取值范围。之所以只有两成，因为我们在前期只是搭建一个简单的神经网络。w的取值范围生成先不要问为什么。


第一层不存在权重（参数）W。

```
def init_parameters_b(layer):
    dist=distribution[layer]['b']
    return np.random.rand(dinensions[layer])*(dist[1]-dist[0])+dist[0]  #使得生成的随机数在 b 的区间内

def init_parameters_w(layer):
    dist=distribution[layer]['w']
    return np.random.rand(dinensions[layer-1],dinensions[layer])*(dist[1]-dist[0])+dist[0]  #使得生成的随机数在 b 的区间内
```
上面代码是对b和w这两个参数初始化，因为我们输入的是28*28个数字，输出的是10个数字。所以第一层的 b 也有28*28个数字组成。根据矩阵的乘法规则，第二层的时候，w的维度只有是28*28行，10列才能满足输出的10个数字。因此第二层的b是10个数字。

dinensions[XXXX] 意思是取切片，dinensions[1] 取得是10，dinensions[0]，取得是28*28。

又因为np.random.rand（）这一函数输出值的范围在【0,1】，括号里面的参数（即dinensions[layer]只是确保输出的数字个数满足要求），所以为了让输出的值在一开始设置的 b 的区间内，我们设置先乘(dist[1]-dist[0])然后加上dist[0]。dist[1]和dist[0]分别对应参数的上下线。

```
def init_parameters():
    parameters=[]
    for i in range(len(distribution)):
        layer_parameters={}
        for j in distribution[i].keys(): 
            if j=='b':
                layer_parameters['b']=init_parameters_b(i)
                continue
            if j=='w':
                layer_parameters['w']=init_parameters_w(i)
                continue
        parameters.append(layer_parameters)
    return parameters
```
上面代码是将三个参数的初始化集成达到一个函数里面。

先定义一个空列表（不要写错成空字典）是为了将三个参数统一输出。

注：字典类型不能用append，列表可以用，列表.append（字典） 也是可以的。

然后从零开始遍历distribution。

用if循环语句，目的是把参数全部包含进来。

第二层for循环和if语句是判断，并正确添加参数。

```
parameters=init_parameters()
```
将参数赋值给新的变量。
```
def predict(img,parameters):
    I0_in=img+parameters[0]['b']
    I0_out=activation[0](I0_in)
    I1_in=np.dot(I0_out,parameters[1]['w']+parameters[1]['b'])
    I1_out=activation[1](I1_in)
    return I1_out
```

定义输出函数，思路是这样的：
输入数据后，根据函数：y=wx+b，进行变换，第一层w全为1。然后经过激活函数（第一个激活函数是tanh，所以用activation[0]），得出第一层的输入I0_out。
然后进入第二层，第一层的输入作为输入，根据函数：y=wx+b，进行变换，第二层的w为parameters[1]['w']，第二层的b为parameters[1]['b']。然后再经过激活函数softmax，得到输出。

```
predict(np.random.rand(784),parameters).argmax()
```

随便输入一个784维数据（像素），都可以输出一个图片标签。


待解决问题：
argmax的用法？？


