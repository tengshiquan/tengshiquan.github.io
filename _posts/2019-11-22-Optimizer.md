---
layout:     post
title:      梯度下降优化算法 Gradient Descent Optimizers
subtitle:   An overview of gradient descent optimization algorithms
date:       2019-11-22 12:00:00
author:     "tengshiquan"
header-img: "img/post-gd.jpg"
catalog: true
tags:
    - gradient descent
    - mathine learning
    - neural network
    - deep learning
    - Optimiztion
---



# Gradient Descent Optimizers

整理一下梯度优化算法

**主要的一阶梯度算法，包括SGD, Momentum, Nesterov Momentum, AdaGrad, RMSProp, Adam**

手动指定学习速率: SGD,Momentum,Nesterov Momentum

自动调节学习速率: AdaGrad, RMSProp, Adam

二阶方法如牛顿法, 实践中计算代价太高，不适合大数据。



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



### Momentum

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

```python
def momentum(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]

    return model
```

如果每次迭代得到的梯度都是g, 那么最后得到的v的稳定值为 $\frac{\eta \cdot g }{1-\gamma}$ , 也就是说,Momentum最好情况下能够将学习率加速为$\frac{1}{1-\gamma}$ 倍.一般 $\gamma$  的取值有0.5,0.9,0.99这几种



### Nesterov accelerated gradient NAG

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

```python
def nesterov(model, X_train, y_train, minibatch_size):
    velocity = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        model_ahead = {k: v + gamma * velocity[k] for k, v in model.items()}
        grad = get_minibatch_grad(model_ahead, X_mini, y_mini)

        for layer in grad:
            velocity[layer] = gamma * velocity[layer] + alpha * grad[layer]
            model[layer] += velocity[layer]

    return model
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

直观含义就很明显了：如果这次的梯度比上次的梯度变大了，那么有理由相信它会继续变大下去，那我就把预计要增大的部分提前加进来；如果相比上次变小了，也是类似的情况。**所以NAG本质上是多考虑了目标函数的二阶导信息，可以加速收敛**





## Adaptive Learning Rate

An adaptive learning rate can be observed in AdaGrad, AdaDelta, RMSprop and Adam.



### Adagrad 自适应梯度算法

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



### Adadelta

Adadelta是对Adagrad的改进，主要是为了克服Adagrad的两个缺点

- 其学习率是单调递减的，训练后期学习率非常小
- 其需要手工设置一个全局的初始学习率

##### Idea1: Accumulate Over Window 

Adadelta只累积过去 w 窗口大小的梯度; 同时为了避免低效地存储过去w个梯度和, 使用了 **running average** ,当前均值只与历史均值以及当前值有关, **数加权平均（Exponentially weighted average, exponentially decaying average）**:

$$
E[g^2]_t = \gamma E[g^2]_{t-1} + (1 - \gamma) g^2_t
$$

$\gamma$ 类似momentum term, around 0.9. 

$$
\Delta \theta_t = - \dfrac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_{t}
$$

分母可以看成 root mean squared (RMS)

$$
\Delta \theta_t = - \dfrac{\eta}{RMS[g]_{t}} g_t
$$

##### Idea2: Correct Units with Hessian Approximation

假定参数x有一个虚拟的量纲单位, 则之前的方法, 增量 $\Delta x$ 的单位与x 不一致,  在SGD中, 该单位与梯度g成比例, 而不是x本身. (假设cost function是无单位的)

$$
\text{units of }\Delta x \propto \text{units of } g \propto  \frac{\partial f}{\partial x} \propto \frac{1}{ \text{units of }x}
$$

相反, 二阶方法如牛顿法, 使用了 Hessian 矩阵或者其近似矩阵的 , 匹配了单位; 可以看出牛顿法无需设置学习率, 使用的是hessian矩阵的逆.

$$
\Delta x \propto H^{-1}g \propto \frac{\frac{\partial f}{\partial x}}{\frac{\partial^2 f}{\partial x^2}} \propto \text{units of } x
$$

因为二阶方法单位是匹配的,那么利用牛顿法的公式可以得到

$$
\Delta x = \frac{\frac{\partial f}{\partial x}}{\frac{\partial^2 f}{\partial x^2}}
\Rightarrow  \frac{1}{\frac{\partial^2 f}{\partial x^2}} = \frac{\Delta x}{\frac{\partial f}{\partial x}}
$$

现在考虑上面RMS更新公式分子的单位, 需要与x匹配, 利用$\Delta x$ ,t时刻前的w个窗口的加权平均, 则有

$$
\Delta x_t  = - \dfrac{RMS[\Delta x]_{t-1}}{RMS[g]_{t}} g_{t}
$$

1. 不再需要学习率
2. 分子比分母慢一个时间步, 可能增加系统鲁棒性, 如在梯度出现剧变的时候, 分母会突然变大,而分子还没反应,所以学习率就小.
3. 该公式只用了一阶信息. 分子相当于一个加速项,类似于momentun. 分母与adagrad类似,会是学习率慢慢下降





### RMSProp   均方根传播

RMSprop 是 Geoff Hinton 提出的一种自适应学习率方法。 RMS 是均方根的意思

RMSprop 与 AdaDelta 的第一种形式相同

RMSProp在非凸条件下结果更好. 经验上，RMSProp被证明有效且实用的深度学习网络优化算法。

$$
\begin{align} 
\begin{split} 
E[g^2]_t &= 0.9 E[g^2]_{t-1} + 0.1 g^2_t \\ 
\theta_{t+1} &= \theta_{t} - \dfrac{\eta}{\sqrt{E[g^2]_t + \epsilon}} g_{t} 
\end{split} 
\end{align}
$$

Hinton 建议 $\gamma=0.9, \eta=0.001$

```python
def rmsprop(model, X_train, y_train, minibatch_size):
    cache = {k: np.zeros_like(v) for k, v in model.items()}
    gamma = .9

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for k in grad:
            cache[k] = gamma * cache[k] + (1 - gamma) * (grad[k]**2)
            model[k] += alpha * grad[k] / (np.sqrt(cache[k]) + eps)

    return model
```



### Adam

**Adaptive Moment Estimation (Adam)**

Adam uses **Momentum** and **Adaptive Learning Rates** to converge faster.

Adam 就是在 RMSprop 的基础上加了 bias-correction 和 momentum，

Adam 像RMSprop一样使用指数衰减的加权平方梯度$v_t$,  同时像momentum一样,使用指数衰减的加权梯度$m_t$

$$
\begin{align} 
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\ 
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 
\end{align}
$$

$m_t$与$v_t$分别是梯度的加权和带权有偏方差，初始为0的向量. Adam的作者发现他们偏向于逼近0向量, 特别是在刚开始的时间步内, 还有特别是在衰减因子比较小的时候(如$\beta_1$,$\beta_2$接近于1). 为了改进这个问题, 对$ m_t$与$v_t$进行偏差修正(bias-corrected)：

$$
\begin{align} 
\hat{m}_t &= \dfrac{m_t}{1 - \beta^t_1} \\ 
\hat{v}_t &= \dfrac{v_t}{1 - \beta^t_2}
\end{align}
$$

最终，Adam的更新方程为：

$$
\theta_{t+1} = \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

论文中建议默认值：$β_1=0.9, β_2=0.999, \epsilon =10^{−8}$, 论文中将Adam与其它的几个自适应学习速率进行了比较，效果均要好。
- 适应性梯度算法（AdaGrad）为每一个参数保留一个学习率以提升在稀疏梯度（即自然语言和计算机视觉问题）上的性能。
- 均方根传播（RMSProp）基于权重梯度最近量级的均值为每一个参数适应性地保留学习率。这意味着算法在非稳态和在线问题上有很有优秀的性能。



Momentum通过对梯度mean的估计，使得梯度在**纵向上的正负步长逐步抵消，横向上的步长逐步累积**，从而减少了震荡，加快了学习速率。可以认为：**对于mean的这种估计，其实是从噪音中提取到了有效信号，因为信号的方向在一段时间内是不会变的。同时抵消了噪音，比如白噪音，它的均值为0，把多个白噪音累加后，它们是正负相互抵消的**当然实际情况噪音可能会非常复杂，如果步长太大，或者衰减太小，还是会出现震荡甚至发散的情况。

对于波动比较大的梯度，它的方差肯定是很大的，所以它用梯度去除以二阶距的开方，作为梯度下降的梯度，从而也使得它在纵轴上的步长减小了，同时相对的增加了横轴的步长. 

这个也是mt和vt的设计来源。同时因为mt和vt涉及了所有历史梯度信息，所以他们都能较好处理梯度稀疏的情况。

随机变量的一阶矩和二阶矩。模型的梯度是一个随机变量，一阶矩表示梯度均值，二阶矩表示其方差，一阶矩来控制模型更新的方向，二阶矩控制步长(学习率)。用moveing average来对一阶矩和二阶矩进行估计。bias correct是为了缓解初始一阶矩和二阶矩初始为0带来的moving average的影响。

目的是提高sgd的稳定性。假设gradient为随机变量D，这个算法本质就是使得optimizing variables向着**mean(D)/std(D)**移动。那么问题就是怎么估计mean(D)跟std(D)这两个expectation。Adam用的办法就是normalized moving average。就是你有一个sequence d1, d2,d3,.....都是D的sample，由$x_i = \beta *(x_{i-1}) +(1 - \beta ) * d_i$ ，得到 $E(X_t)/(1- \beta^t))=E(D)$.  $std(D)=E(D^2)$同理。这种ema的应用在统计里是非常非常常用的smoothing technique了 (from 知乎)


正是由于采用了**m/sqrt(v)**的形式，梯度本身的大小被消除了，假设你在目标函数上乘以k，那么这个k会同时出现在分子分母上，从而在梯度更新时被消除了。

根据原文作者的意思，在机器学习中，我们经常可以方便的获知最优参数所处的范围，从而据此设置alpha的取值大小，所以Adam的这种特性可以方便你设置alpha。

**根据我们在Momentum和RMSProp中的描述，m/sqrt(v)可以看做是对梯度“信噪比”（SNR）的估计，其中m是信号量，而v是噪音量。当噪音大时，步长小；噪音小时，步长大。这正是我们需要的特性，从而使得Adam能够较好的处理噪音样本，另外，当我们接近最优点时，噪音通常会变得很大，或者说真正的信号变得很微弱，信噪比通常会接近0，这样步长就会变得较小，从而天然起到退火效果(annealing)**



##### INITIALIZATION BIAS CORRECTION

$$
v_t = (1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} \cdot g_i^2
$$

$$
\begin{align}
\mathbb E[v_t] &= \mathbb E \Big [ (1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} \cdot g_i^2 \Big]
\\  &= \mathbb E[g_t^2] \cdot (1 - \beta_2) \sum_{i=1}^t \beta_2^{t-i} + \zeta
\\ &= \mathbb E[g_t^2] \cdot (1 - \beta_2^t) + \zeta
\end{align}
$$





![image-20191129170421175](/img/Optimizer.assets/image-20191129170421175.png)

```python
def adam(model, X_train, y_train, minibatch_size):
    M = {k: np.zeros_like(v) for k, v in model.items()}
    R = {k: np.zeros_like(v) for k, v in model.items()}
    beta1 = .9
    beta2 = .999

    minibatches = get_minibatch(X_train, y_train, minibatch_size)

    for iter in range(1, n_iter + 1):
        t = iter
        idx = np.random.randint(0, len(minibatches))
        X_mini, y_mini = minibatches[idx]

        grad = get_minibatch_grad(model, X_mini, y_mini)

        for k in grad:
            M[k] = beta1 * M[k] + (1. - beta1) * grad[k]
            R[k] = beta2 * R[k] + (1. - beta2) * grad[k]**2

            m_k_hat = M[k] / (1. - beta1**(t))
            r_k_hat = R[k] / (1. - beta2**(t))

            model[k] += alpha * m_k_hat / (np.sqrt(r_k_hat) + eps)

    return model
```

```python
# from cs231
eps = 1e-8
beta1 = 0.9
beta2 = 0.999

# t is your iteration counter going from 1 to infinity
m = beta1*m + (1-beta1)*dx
mt = m / (1-beta1**t)
v = beta2*v + (1-beta2)*(dx**2)
vt = v / (1-beta2**t)
x += - learning_rate * mt / (np.sqrt(vt) + eps)
```



当取$\beta_1 =0$ , $\beta_2$ 取一个非常接近1的值的时候 ,  然后 $ \alpha$ 为一个退火的版本$\alpha_t = \alpha \cdot t^{-1/2}$,  这时可以看出 adagrad是adam的一个特例.



### AdaMax

在 Adam 中，$v_t$ 的更新规则:  

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) |g_t|^2
$$

可以将基于 $\ell_2$范数的更新规则泛化到基于$\ell_p$范数的更新规则, 同时改系数$\beta_2$为$\beta_2^p$

$$
v_t = \beta_2^p v_{t-1} + (1 - \beta_2^p) |g_t|^p 
\\ = (1- \beta^p_2) \sum_{i=1}^t \beta_2^{p(t-i)} \cdot |g|^p
$$

虽然这样的变体会因为 p 的值较大而在数值上变得不稳定，这也是为什么$\ell_1$和$\ell_2$范数在实践中最常用. 但是$\ell_\infty$ 却通常表现为极端稳定, 因此, 作者提出adamax.  引入$u_t= \lim_{p \to \infty}(v_t)^{1/p}$ 来代替之前的 $\sqrt{\hat{v}_t} + \epsilon$

$$
\begin{align} 
u_t &= \lim_{p \to \infty} \Big( (1- \beta^p_2) \sum_{i=1}^t \beta_2^{p(t-i)} \cdot |g|^p \Big)^{1/p}
\\ & = \lim_{p \to \infty} \Big( (1- \beta^p_2) \sum_{i=1}^t \beta_2^{p(t-i)} \cdot |g|^p \Big)^{1/p}
\\ & = \lim_{p \to \infty} (1- \beta^p_2)^{1/p} \Big(  \sum_{i=1}^t \beta_2^{p(t-i)} \cdot |g|^p \Big)^{1/p}
\\ & = \lim_{p \to \infty} \Big(  \sum_{i=1}^t \beta_2^{p(t-i)} \cdot |g|^p \Big)^{1/p}
\\ & = \max(\beta_2^{t-1}|g_1|, \beta_2^{t-2}|g_2|, \dots, \beta_2^{t-1}|g_{t-1}|, |g_t|) 
\\ & = \max(\beta_2 \cdot u_{t-1}, |g_t|) 
\end{align}
$$

此方法对学习率的上限提供了一个更简单的范围 $\mid \Delta t \mid \leq \alpha$

![image-20191129163631762](/img/Optimizer.assets/image-20191129163631762.png)



### Nadam

Nadam = NAG + Adam

之前的NAG为

$$
\begin{align} 
\begin{split} 
g_t &= \nabla_{\theta_t}J(\theta_t - \gamma m_{t-1})\\ 
m_t &= \gamma m_{t-1} + \eta g_t\\ 
\theta_{t+1} &= \theta_t - m_t 
\end{split} 
\end{align}
$$

下面不再利用两次动量更新, 改为直接用 look-ahead的动量来更新当前参数, 利用mt来look ahead. 

$$
\begin{align} 
\begin{split} 
g_t &= \nabla_{\theta_t}J(\theta_t)\\ 
m_t &= \gamma m_{t-1} + \eta g_t\\ 
\theta_{t+1} &= \theta_t - (\gamma m_t + \eta g_t) 
\end{split} 
\end{align}
$$



下面修改adam的mt更新公式



可以看出，Nadam对学习率有了更强的约束，同时对梯度的更新也有更直接的影响。一般而言，在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果。



### AMSGrad

自适应学习算法已经成为标准, 但仍然有些情况无法收敛: 物体识别以及机器翻译. 可能还不如SGDwith momentum.

作者指出 梯度平方的指数加权移动平均 是adaptive方法方法比较差的原因. 引入exponential average的原因是为了解决adagrad方法的学习率最后会消失的问题,但是只使用过去一小段时间梯度(short-term memory)的情况在某些问题中变成了一个问题.

使用Adam算法会有一种情况, 有些minibatch提供了一个很大的, 有信息量的梯度, 但这些minibatch出现的几率很小, exponential average减小了这些梯度的影响, 最后造成了较差的收敛性. 作者提供了一个简单的凸优化的例子. 

为了解决这个问题, 作者提出了AMSGrad算法. 使用过去平方梯度$v_t$的最大值, 而不是指数加权. 

$$
\hat{v}_t = \text{max}(\hat{v}_{t-1}, v_t)
$$

然后, AMSGrad会造成step size不会增加, 从而避免了Adam的问题.  为了简单, 去掉了Adam的偏差修正部分, 最终更新公式如下:

$$
\begin{align} 
\begin{split} 
m_t &= \beta_1 m_{t-1} + (1 - \beta_1) g_t \\ 
v_t &= \beta_2 v_{t-1} + (1 - \beta_2) g_t^2\\ 
\hat{v}_t &= \text{max}(\hat{v}_{t-1}, v_t) \\ 
\theta_{t+1} &= \theta_{t} - \dfrac{\eta}{\sqrt{\hat{v}_t} + \epsilon} m_t 
\end{split} 
\end{align}
$$

AMSGrad在某些问题上的表现会不如Adam算法. 






## 总结

<img src="/img/Optimizer.assets/opt2.gif" alt="img" style="zoom: 67%;" />

<img src="/img/Optimizer.assets/opt1.gif" alt="img" style="zoom: 67%;" />



- 对于稀疏数据，尽量使用学习率自适应的优化方法，即 Adagrad, Adadelta, RMSprop, Adam, 不用手动调节，而且最好采用默认值
- 最近很多论文都是使用原始的SGD梯度下降算法，并且使用简单的学习速率退火调整（无动量项）. SGD通常训练时间更长，容易陷入鞍点，但是在好的初始化和学习率调度方案的情况下，结果更可靠
- 如果在意更快的收敛，并且需要训练较深较复杂的网络时，推荐使用学习率自适应的优化方法。例如对于RNN之类的网络结构,Adam速度快,效果好,而对于CNN之类的网络结构,SGD +momentum 的更新方法要更好（常见国际顶尖期刊常见优化方法）.
- Adadelta，RMSprop，Adam是比较相近的算法，在相似的情况下表现差不多。
- 在想使用带动量的RMSprop，或者Adam的地方，大多可以使用Nadam取得更好的效果
- 其实还有很多方面会影响梯度下降算法，如梯度的消失与爆炸，梯度下降算法目前无法保证全局收敛还将是一个持续性的数学难题。
- Adam略优于RMSprop，因为其在接近收敛时梯度变得更加稀疏。整体来讲，Adam 是最好的选择。



### 其他的SGD优化策略

- Shuffling and Curriculum Learning

  为了使得学习过程更加无偏，应该在每次迭代中随机打乱训练集中的样本。
  另一方面，在很多情况下，我们是逐步解决问题的，而将训练集按照某个有意义的顺序排列会提高模型的性能和SGD的收敛性，如何将训练集建立一个有意义的排列被称为 Curriculum Learning 

- Batch normalization
  
  为了方便训练，我们通常会对参数按照0均值1方差进行初始化，随着不断训练，参数得到不同程度的更新，这样这些参数会失去0均值1方差的分布属性，这样会降低训练速度和放大参数变化随着网络结构的加深。
  Batch normalization在每次mini-batch反向传播之后重新对参数进行0均值1方差标准化。这样可以使用更大的学习速率，以及花费更少的精力在参数初始化点上。Batch normalization充当着正则化、减少甚至消除掉Dropout的必要性。

- Early stopping
  
  在验证集上如果连续的多次迭代过程中损失函数不再显著地降低，那么应该提前结束训练

- Gradient noise
  
  即在每次迭代计算梯度中加上一个高斯分布$N(0, \sigma^2_t)$的随机误差，即

  $$
  g_{t, i} = g_{t, i} + N(0, \sigma^2_t)
  $$

  再将高斯误差的方差进行退火：

  $$
  \sigma^2_t = \dfrac{\eta}{(1 + t)^\gamma}
  $$

   对梯度增加随机误差会增加模型的鲁棒性，即使初始参数值选择地不好，并适合对特别深层次的负责的网络进行训练。其原因在于增加随机噪声会有更多的可能性跳过局部极值点并去寻找一个更好的局部极值点，这种可能性在深层次的网络中更常见。





## Parallelizing and distributing SGD 分布式SGD

如果你处理的数据集非常大，并且有机器集群可以利用，那么并行或分布式SGD是一个非常好的选择，因为可以大大地提高速度。SGD算法的本质决定其是串行的(step-by-step)。因此如何进行异步处理便是一个问题。虽然串行能够保证收敛，但是如果训练集大，速度便是一个瓶颈。如果进行异步更新，那么可能会导致不收敛。下面将讨论如何进行并行或分布式SGD，并行一般是指在同一机器上进行多核并行，分布式是指集群处理。

- Hogwild
  
- Downpour SGD
  
- Delay-tolerant Algorithms for SGD
  
- TensorFlow
  
- Elastic Averaging SGD







# References

An overview of gradient descent optimization algorithms https://ruder.io/optimizing-gradient-descent/

Momentum-Based & Nesterov Accelerated Gradient Descent https://towardsdatascience.com/learning-parameters-part-2-a190bef2d12

[Why Momentum Really Works] https://distill.pub/2017/momentum/  特效不错

http://cs231n.github.io/neural-networks-3/ 

https://blog.csdn.net/u012328159/article/details/80311892

https://blog.csdn.net/u010899985/article/details/81836299

https://blog.csdn.net/heyongluoyao8/article/details/52478715

https://www.cnblogs.com/neopenx/p/4768388.html adadelta

https://zhuanlan.zhihu.com/p/22252270

http://louistiao.me/notes/visualizing-and-animating-optimization-algorithms-with-matplotlib/ 

https://segmentfault.com/a/1190000012668819?utm_source=tag-newest

