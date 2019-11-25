---
layout:     post
title:      梯度下降算法 Gradient Descent Optimizers
subtitle:   An overview of gradient descent optimization algorithms
date:       2019-01-01 12:00:00
author:     "tengshiquan"
header-img: "img/post-gd.jpg"
catalog: true
tags:
    - gradient descent
    - mathine learning
    - neural network
    - deep learning
    - Optimizer
---



# Gradient Descent Optimizers

整理一下梯度优化算法

**主要的一阶梯度算法，包括SGD, Momentum, Nesterov Momentum, AdaGrad, RMSProp, Adam**

手动指定学习速率: SGD,Momentum,Nesterov Momentum

自动调节学习速率: AdaGrad, RMSProp, Adam

二阶方法, 如牛顿法, 在实践中, 在高纬度数据集上计算不可行.



## Gradient descent variants

#### Batch gradient descent

$$
\theta = \theta - \eta \cdot \nabla_\theta J( \theta)
$$

```python
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
```

**优点:**

- cost fuction若为凸函数，能够保证收敛到全局最优值；若为非凸函数，能够收敛到局部最优值

**缺点:**

- 由于每轮迭代都需要在整个数据集上计算一次，所以批量梯度下降可能非常慢
- 训练数较多时，需要较大内存
- 批量梯度下降不允许在线更新模型，例如新增实例。 update model *online*



#### Stochastic gradient descent

Stochastic gradient descent (SGD) in contrast performs a parameter update for *each* training example.算法每读入一个数据，便立刻计算cost fuction的梯度来更新参数：

$$
\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i)}; y^{(i)})
$$

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad
```

当慢慢减小 learning rate时，SGD 和 BGD 的收敛性是一样的。

**优点:**

- 算法收敛速度快(在Batch Gradient Descent算法中, 每轮会计算很多相似样本的梯度, 这部分是冗余的)
- 可以在线更新
- 有几率跳出一个比较差的局部最优而收敛到一个更好的局部最优甚至是全局最优 keep overshooting

**缺点:**

- 容易收敛到局部最优，并且容易被困在鞍点
- SGD 因为更新比较频繁, 方差大(每次更新可能并不会按照正确的方向进行, 不稳)，会造成 cost function 有严重的震荡. 



<img src="/img/Optimizer.assets/sgd_fluctuation.png" alt="img" style="zoom: 50%;" />



![https://datascience-enthusiast.com/figures/kiank_sgd.png](/img/Optimizer.assets/kiank_sgd.png)

这个图的等高线是基于所有数据生成的. batchGD是在每个点都向着该点梯度最大的方向下降; SGD可能是随机游走,  也可能开倒车. **random walk** 



#### Mini-batch gradient descent

对每个minibatch执行


$$
\theta = \theta - \eta \cdot \nabla_\theta J( \theta; x^{(i:i+n)}; y^{(i:i+n)})
$$

```python
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad
```

优点:

- 可以降低参数更新时的方差，收敛更稳定，
- 另一方面可以充分地利用深度学习库中高度优化的矩阵操作来进行更有效的梯度计算。



![https://datascience-enthusiast.com/figures/kiank_minibatch.png](/img/Optimizer.assets/kiank_minibatch.png)

Common mini-batch sizes range between 50 and 256

现在深度学习中的SGD，一般指的就是mini-batch gradient descent



### Challenges

Vanilla mini-batch gradient descent , 并不能保证良好的收敛性, 同时有其他一些问题:

- 选择一个合理的学习速率很难。如果学习速率过小，则会导致收敛速度很慢。如果学习速率过大，可能在极值点附近振荡或者发散. 
- Learning rate schedules 试图在训练中改变学习速率，如退火, 先设定大一点的学习率然后按照预定的策略来减小lr, 或者当两次迭代之间的变化低于某个阈值后，就减小 learning rate. 无论策略或者阈值，都需要事先定义，因而无法适应数据集的特点. 
- 对所有的参数每次更新都是使用相同的学习率。如果数据特征是稀疏的或者每个特征有着不同的频率，那么我们可能并不想以同一个大小的学习率来更新全部参数, 希望能对那些很少出现的特征应该使用一个相对较大的学习率
- 对于非凸目标函数，容易陷入那些次优的局部极值点, 鞍点. 



## Momentum

从山上往下滚球, 如果没有惯性的话, 可以视为球每到一个地方都是静止的, 然后从那个点的梯度进行下降, 所以很容易就陷入一些小坑,即局部最小值. 而加上了惯性以后, 可以视为球有速度, 可以冲破一些比较小的坑; 表现上, 在斜率比较小的长斜坡的方向上,得以加速, 并可能越过一些小坑; 在坡度很陡的山谷, 可以减小来回的震荡

直观上讲就是，要是当前时刻的梯度与历史时刻梯度方向相似，这种趋势在当前时刻则会加强；要是不同，则当前时刻的梯度方向减弱。 **类似滚动的小球，增加的惯性可以起到更平滑和加速的作用，抑制振荡并使我们穿过狭窄的山谷，小驼峰和局部极小。**

SGD有个问题,就是每次迭代计算的梯度含有比较大的噪音. 而Momentum方法可以比较好的缓解这个问题,尤其是**在面对小而连续的梯度但是含有很多噪声的时候,可以很好的加速学习**.

**特点:** 
前后梯度方向一致时,能够加速学习 
前后梯度方向不一致时,能够抑制震荡



![img](/img/Optimizer.assets/20180516112034267.png)

<img src="/img/Optimizer.assets/momentum.gif" alt="img" style="zoom: 25%;" />

<img src="/img/Optimizer.assets/no-momentum.gif" alt="img" style="zoom:25%;" />

把上一把参数的该变量作为v , 乘以 Momentum

$$
\begin{align} 
\begin{split} 
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta) \\ 
\theta &= \theta - v_t 
\end{split} 
\end{align}
$$

动量项$\gamma$ 一般选0.9 

```python
# Momentum update
v = gamma * v + learning_rate * grad(x) # integrate velocity
x -= v
```



如果每次迭代得到的梯度都是g, 那么最后得到的v的稳定值为 $\frac{\eta \cdot g }{1-\gamma}$ , 也就是说,Momentum最好情况下能够将学习率加速为$\frac{1}{1-\gamma}$ 倍.一般 $\gamma$  的取值有0.5,0.9,0.99这几种



## Nesterov accelerated gradient NAG

从山顶往下滚的球只会身不由己地前进. 更好的方式应该是在遇到倾斜向上之前应该减慢速度。这样可能就不会把之前积攒的速度冲进上面的一个局部最小.  

根据公式, 并不看当前点的梯度, 而是看当前点按照之前的动量趋势前进的位置的梯度, 位置上提前了一些
$$
\begin{align} 
\begin{split} 
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta_t - \gamma v_{t-1} ) \\ 
\theta_{t+1} &= \theta_t - v_t 
\end{split} 
\end{align}
$$



![](/img/Optimizer.assets/nesterov.jpeg)

主要是防止 overshoot 

<img src="/img/Optimizer.assets/pAwIf.png" alt="CM vs NAG example" style="zoom:67%;" />

下图解释为什么

<img src="/img/Optimizer.assets/1*6MEi74EMyPERHlAX-x2Slw.png" alt="img" style="zoom:70%;" />



```python
x_ahead = x - gamma * v
# evaluate dx_ahead (the gradient at x_ahead instead of at x)
v = gamma * v + learning_rate * grad(x_ahead)
x -= v
```



参考 https://zhuanlan.zhihu.com/p/22810533, 可以推导为 

$$
\begin{align} 
\begin{split} 
v_t &= \gamma v_{t-1} + \eta \nabla_\theta J( \theta ) + \gamma  \eta[\nabla J(\theta) - \nabla J(\theta_{t-1}) ] \\ 
\theta &= \theta - v_t 
\end{split} 
\end{align}
$$

直观含义就很明显了：如果这次的梯度比上次的梯度变大了，那么有理由相信它会继续变大下去，那我就把预计要增大的部分提前加进来；如果相比上次变小了，也是类似的情况。**所以NAG本质上是多考虑了目标函数的二阶导信息，怪不得可以加速收敛了！其实所谓“往前看”的说法，在牛顿法这样的二阶方法中也是经常提到的，比喻起来是说“往前看”，数学本质上则是利用了目标函数的二阶导信息。**





# Adaptive Learning Rate

An adaptive learning rate can be observed in AdaGrad, AdaDelta, RMSprop and Adam.



## Adagrad 自适应梯度算法

基本思想是 对每个参数用不同的学习率.  学习率在一开始比较大，用于快速梯度下降。随着优化过程的进行，对于已经下降很多的参数，则减缓学习率，对于还没怎么下降的变量，则保持一个较大的学习率.

非常适合处理稀疏数据

算法, 全局学习率逐参数的除以历史梯度平方和的平方根. 

对SGD,  对i个参数, 在t时刻, 更新公式

$$
\theta_{t+1, i} = \theta_{t, i} - \eta \cdot g_{t, i}
$$

Adagrad对每一个参数使用不同的学习速率

$$
\theta_{t+1,i} = \theta_{t,i}
    -\frac{\eta}
    {
        \sqrt
        {
            \epsilon +
            \sum_{\tau=1}^{t}
            \left( \nabla J(\theta_{\tau,i}) \right) ^2
        }
    } \nabla J(\theta_{t,i})
$$

 $\epsilon$是一个平滑参数，为了使得分母不为0 .  另外如果分母不开根号，算法性能会很糟糕。



```python
def adagrad(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(v) for k, v in model.items()}

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for k in grad:
            cache[k] += grad[k]**2
            model[k] += alpha * grad[k] / (np.sqrt(cache[k]) + eps)

    return model
```



Adagrad主要优势在于它能够为每个参数自适应不同的学习速率，一般的人工都是设定为0.01。
Adagrad的缺点是在训练的中后期，分母上梯度平方的累加将会越来越大，从而梯度趋近于0，使得训练提前结束。 经验表明，在普通算法中也许效果不错，但在深度学习中，深度过深时会造成训练提前结束。



## Adadelta

Adadelta是对Adagrad的改进，主要是为了克服Adagrad的两个缺点

- 其学习率是单调递减的，训练后期学习率非常小
- 其需要手工设置一个全局的初始学习率

为了解决第一个问题，Adadelta只累积过去 w 窗口大小的梯度; 同时为了避免低效地存储过去w个梯度和, 使用了 **running average** ,当前均值只与历史均值以及当前值有关:

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g^2_t
$$

$\gamma$ 类似momentum term, around 0.9. 

$$
\Delta \theta_t = - \dfrac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_{t}
$$
















 



### RMSProp   均方根传播

RMSProp通过引入一个衰减系数，让r每回合都衰减一定比例，类似于Momentum中的做法。

1.[AdaGrad]算法的改进。鉴于神经网络都是非凸条件下的，RMSProp在非凸条件下结果更好，改变梯度累积为指数衰减的移动平均以丢弃遥远的过去历史。

2.经验上，RMSProp被证明有效且实用的深度学习网络优化算法。





## Adam

**Adaptive Moment Estimation (Adam)**

Adam uses **Momentum** and **Adaptive Learning Rates** to converge faster.

随机梯度下降保持单一的学习率更新所有的权重，学习率在训练过程中并不会改变。而 Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。

- 适应性梯度算法（AdaGrad）为每一个参数保留一个学习率以提升在稀疏梯度（即自然语言和计算机视觉问题）上的性能。
- 均方根传播（RMSProp）基于权重梯度最近量级的均值为每一个参数适应性地保留学习率。这意味着算法在非稳态和在线问题上有很有优秀的性能。

Adam 算法同时获得了 AdaGrad 和 RMSProp 算法的优点。









![Animation of how the newer optimizers compare in terms of convergence.](/img/Optimizer.assets/saddle.gif)





# References

An overview of gradient descent optimization algorithms https://ruder.io/optimizing-gradient-descent/

Momentum-Based & Nesterov Accelerated Gradient Descent https://towardsdatascience.com/learning-parameters-part-2-a190bef2d12

[Why Momentum Really Works] https://distill.pub/2017/momentum/  特效不错

http://cs231n.github.io/neural-networks-3/ 

https://blog.csdn.net/u012328159/article/details/80311892



