---
layout:     post
title:      Softmax and CrossEntropy Loss in PYTorch
subtitle:   
date:       2019-12-05 12:00:00
author:     "tengshiquan"
header-img: "img/post-py.jpg"
catalog: true
tags:
    - pytorch
---



# Softmax and CrossEntropy Loss in PYTorch

从其他框架转用pytorch做多分类问题, 会发现pytorch里面对 CrossEntropyLoss 的封装过于简洁不够灵活, 比如输入无法设置参数是否是结果log的,是否是经过softmax的, y是否one-hot类型的

交叉熵的Loss的公式, 简洁记法:

$$
Loss = - Y \cdot log \hat Y
$$

cross-entropy loss有很好的性质, 可以很简单的去反向传播. 

$$
\frac{\partial Loss}{\partial z_i} =  \hat y_i - y_i
$$

$z_i$ 是网络的最后一层节点的输出, 但不经过softmax



##### CrossEntropyLoss

This criterion combines `nn.LogSoftmax()` and `nn.NLLLoss()` in one single class. 

CrossEntropyLoss默认输入就是不经过 LogSoftmax的;  则网络的输出是没有经过 softmax的. 那么直接选最大的用就行.   



下面一个例子在pytorch中使用交叉熵loss解决xor问题

```python
import torch

def cross_entropy_xor():
    x = xor_input = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float)
    y = xor_output = torch.tensor([0, 1, 1, 0])

    model = torch.nn.Sequential(
        torch.nn.Linear(2, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    learning_rate = 0.1
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

    for e in range(1000):
        y_pred = model(x)
        loss = loss_fn(y_pred, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(model(x))

cross_entropy_xor()
```

```python
tensor([[ 2.6031, -1.3967],
        [-1.2459,  1.5899],
        [-1.7162,  2.9551],
        [ 2.8321, -1.4759]], grad_fn=<AddmmBackward>)
```



如果网络必须有Softmax层, 则不能使用CrossEntropyLoss, 要用log后用 NLLLoss (negative log-likelihood)

1. net(x) + CrossEntropyLoss
2. net(x) + LogSoftmax + NLLLoss
3. net(x) + Softmax + log + NLLLoss

这三种方式的loss, 最后结果是一样的

```python
batch_size, input_size, n_classes = 64, 8, 4
x = torch.randn(batch_size, input_size)
y = torch.randint(n_classes, size=(batch_size,), dtype=torch.long)

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, n_classes),
)
loss_fn = torch.nn.CrossEntropyLoss()
loss1 = loss_fn(model(x), y)
print("loss1", loss1)

y_onehot = torch.nn.functional.one_hot(y, num_classes=n_classes)
pred_softmax = (torch.nn.functional.softmax(model(x), dim=1))
formula_loss = -(y_onehot * torch.log(pred_softmax)).sum() / len(x)
print("formula_loss", formula_loss)
# loss1.backward()
# grad = -torch.t(torch.mm(torch.t(x), (y_onehot - pred_softmax)) / len(x))

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, n_classes),
    torch.nn.LogSoftmax(dim=1)
)
loss_fn = torch.nn.NLLLoss()
loss2 = loss_fn(model(x), y)
print("loss2", loss2)

torch.manual_seed(0)
model = torch.nn.Sequential(
    torch.nn.Linear(input_size, n_classes),
    torch.nn.Softmax(dim=1)
)
loss_fn = torch.nn.NLLLoss()
loss3 = loss_fn(torch.log(model(x)), y)
print("loss3", loss3)
```

