# MXNet的新接口Gluon

## 为什么要开发Gluon的接口

在MXNet中我们可以通过`Sybmol`模块来定义神经网络，并组通过`Module`模块提供的一些上层API来简化整个训练过程。那MXNet为什么还要重新开发一套Python的API呢，是否是重复造轮子呢？答案是否定的，Gluon主要是学习了Keras、Pytorch等框架的优点，支持动态图（Imperative）编程，更加灵活且方便调试。而原来MXNet基于Symbol来构建网络的方法是像TF、Caffe2一样静态图的编程方法。同时Gluon也继续了MXNet在静态图上的一些优化，比如节省显存，并行效率高等，运行起来比Pytorch更快。

## 更加简洁的接口

我们先来看一下用Gluon的接口，如果创建并组训练一个神经网络的，我们以mnist数据集为例：

```python
import mxnet as mx
import mxnet.ndarray as nd
from mxnet import gluon
import mxnet.gluon.nn as nn
```

## 数据的读取

首先我们利用Gluon的data模块来读取mnist数据集

```python
def transform(data, label):
    return data.astype('float32') / 255, label.astype('float32')

minist_train_dataset = gluon.data.vision.MNIST(train=True, transform=transform)
minist_test_dataset = gluon.data.vision.MNIST(train=False, transform=transform)
```

```python
batch_size = 64
train_data = gluon.data.DataLoader(dataset=minist_train_dataset, shuffle=True, batch_size=batch_size)
test_data = gluon.data.DataLoader(dataset=minist_train_dataset, shuffle=False, batch_size=batch_size)
```

```python
num_examples = len(train_data)
print(num_examples)
```

## 训练模型

这里我们使用Gluon来定义一个LeNet

```python
# Step1 定义模型
lenet = nn.Sequential()
with lenet.name_scope():
    lenet.add(nn.Conv2D(channels=20, kernel_size=5, activation='relu'))
    lenet.add(nn.MaxPool2D(pool_size=2, strides=2))
    lenet.add(nn.Conv2D(channels=50, kernel_size=5, activation='relu'))
    lenet.add(nn.MaxPool2D(pool_size=2, strides=2))
    lenet.add(nn.Flatten())
    lenet.add(nn.Dense(128, activation='relu'))
    lenet.add(nn.Dense(10))
# Step2 初始化模型参数
lenet.initialize(ctx=mx.gpu())
# Step3 定义loss
softmax_loss = gluon.loss.SoftmaxCrossEntropyLoss()
# Step4 优化
trainer = gluon.Trainer(lenet.collect_params(), 'sgd', {'learning_rate': 0.5})
```

```python
def accuracy(output, label):
     return nd.mean(output.argmax(axis=1)==label).asscalar()
def evaluate_accuracy(net, data_iter):
    acc = 0
    for data, label in data_iter:
        data = data.transpose((0,3,1,2))
        data = data.as_in_context(mx.gpu())
        label = label.as_in_context(mx.gpu())
        output = net(data)
        acc += accuracy(output, label)
    return acc / len(data_iter)
```

```python
import mxnet.autograd as ag
epochs = 5
for e in range(epochs):
    total_loss = 0
    for data, label in train_data:
        data = data.transpose((0,3,1,2))
        data = data.as_in_context(mx.gpu())
        label = label.as_in_context(mx.gpu())
        with ag.record():
            output = lenet(data)
            loss = softmax_loss(output, label)
        loss.backward()
        trainer.step(batch_size)
        total_loss += nd.mean(loss).asscalar()
    print("Epoch %d, test accuracy: %f, average loss: %f" % (e, evaluate_accuracy(lenet, test_data), total_loss/num_examples))
```

## 背后的英雄 nn.Block

我们前面使用了`nn.Sequential`来定义一个模型，但是没有仔细介绍它，它其实是`nn.Block`的一个简单的形式。而`nn.Block`是一个一般化的部件。整个神经网络可以是一个`nn.Block`，单个层也是一个`nn.Block`。我们可以（近似）无限地嵌套`nn.Block`来构建新的`nn.Block`。`nn.Block`主要提供3个方向的功能：
1. 存储参数
2. 描述`forward`如何执行
3. 自动求导 

所以`nn.Sequential`是一个`nn.Block`的容器，它通过`add`来添加`nn.Block`。它自动生成`forward()`函数。一个简单实现看起来如下：

```python
class Sequential(nn.Block):
    def __init__(self, **kwargs):
        super(Sequential, self).__init__(**kwargs)
    def add(self, block):
        self._children.append(block)
    def forward(self, x):
        for block in self._children:
            x = block(x)
        return x
```

知道了`nn.Block`里的魔法后，我们就可以自定我们自己的`nn.Block`了，来实现不同的深度学习应用可能遇到的一些新的层。

在`nn.Block`中参数都是以一种`Parameter`的对象，通过这个对象的`data()`和`grad()`来访问对应的数据和梯度。

```python
my_param = gluon.Parameter('my_params', shape=(3,3))
my_param.initialize()
(my_param.data(), my_param.grad())
```

每个`nn.Block`里都有一个类型为`ParameterDict`类型的成员变量`params`来保存所有这个层的参数。它其际上是一个名称到参数映射的字典。

```python
pd = gluon.ParameterDict(prefix='custom_layer_name')
pd.get('custom_layer_param1', shape=(3,3))
pd
```

## 自义我们自己的全连接层

当我们要实现的功能在Gluon.nn模块中找不到对应的实现时，我们可以创建自己的层，它实际也就是一个`nn.Block`对象。要自定义一个`nn.Block`以，只需要继承`nn.Block`，如果该层需要参数，则在初始化函数中做好对应参数的初始化（实际只是分配的形状），然后再实现一个`forward()`函数来描述计算过程。

```python
class MyDense(nn.Block):
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        with self.name_scope():
            self.weight = self.params.get(
                'weight', shape=(in_units, units))
            self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)
```

## 审视模型的参数

我们将从下面三个方面来详细讲解如何操作gluon定义的模型的参数。
1. 初始化
2. 读取参数
3. 参数的保存与加载

从上面我们们在mnist训练一个模型的步骤中可以看出，当我们定义好模型后，第一步就是需要调用`initialize()`对模型进行参数初始化。

```python
def get_net():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.Dense(4, activation='relu'))
        net.add(nn.Dense(2))
    return net
net = get_net()
net.initialize()
```

我们一直使用默认的`initialize`来初始化权重。实际上我们可以指定其他初始化的方法，`mxnet.initializer`模块中提供了大量的初始化权重的方法。比如非常流行的`Xavier`方法。

```python
#net.initialize(init=mx.init.Xavier())
x = nd.random.normal(shape=(3,4))
net(x)
```

我们可以`weight`和`bias`来访问Dense的参数，它们是`Parameter`对象。

```python
w = net[0].weight
b = net[0].bias
print('weight:', w.data())
print('weight gradient', w.grad())
print('bias:', b.data())
print('bias gradient', b.grad())
```

我们也可以通过`collect_params`来访问`Block`里面所有的参数（这个会包括所有的子Block）。它会返回一个名字到对应`Parameter`的dict。既可以用正常`[]`来访问参数，也可以用`get()`，它不需要填写名字的前缀。

```python
params = net.collect_params()
print(params)
print(params['sequential18_dense0_weight'].data())
print(params.get('dense0_bias').data()) #不需要名字的前缀
```

## 延后的初始化

如果我们仔细分析过整个网络的初始化，我们会有发现，当我们没有给网络真正的输入数据时，网络中的很多参数是无法确认形状的。

```python
net = get_net()
net.collect_params()
```

```python
net.initialize()
net.collect_params()
```

我们注意到参数中的`weight`的形状的第二维都是0, 也就是说还没有确认。那我们可以肯定的是这些参数肯定是还没有分配内存的。

```python
net(x)
net.collect_params()
```

当我们给这个网络一个输入数据后，网络中的数据参数的形状就固定下来了。而这个时候，如果我们给这个网络一个不同shape的输入数据，那运行中就会出现崩溃的问题。

## 模型参数的保存与加载 

`gluon.Sequential`模块提供了`save`和`load`接口来方便我们对一个网络的参数进行保存与加载。

```python
filename = "mynet.params"
net.save_params(filename)
net2 = get_net()
net2.load_params(filename, mx.cpu())
```

## Hybridize

从上面我们使用gluon来训练mnist，可以看出，我们使用的是一种命令式的编程风格。大部分的深度学习框架只在命令式与符号式间二选一。那我们能不能拿到两种泛式全部的优点呢，事实上这一点可以做到。在MXNet的GluonAPI中，我们可以使用`HybridBlock`或者`HybridSequential`来构建网络。默认他们跟`Block`和`Sequential`一样是命令式的。但当我们调用`.hybridize()`后，系统会转撚成符号式来执行。

```python
def get_net():
    net = nn.HybridSequential()
    with net.name_scope():
        net.add(
            nn.Dense(256, activation="relu"),
            nn.Dense(128, activation="relu"),
            nn.Dense(2)
        )
    net.initialize()
    return net

x = nd.random.normal(shape=(1, 512))
net = get_net()
net(x)
```

```python
net.hybridize()
net(x)
```

注意到只有继承自HybridBlock的层才会被优化。HybridSequential和Gluon提供的层都是它的子类。如果一个层只是继承自Block，那么我们将跳过优化。我们可以将符号化的模型的定义保存下来，在其他语言API中加载。

```python
x = mx.sym.var('data')
y = net(x)
print(y.tojson())
```

可以看出，对于`HybridBlock`的模块，既可以把NDArray作为输入，也可以把`Symbol`对象作为输入。当以`Symbol`作为输出时，它的结果就是一个`Symbol`对象。

```python

```
