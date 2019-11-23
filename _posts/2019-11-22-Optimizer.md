---
layout:     post
title:      Gradient Descent Optimizers
subtitle:   An overview of gradient descent optimization algorithms
date:       2019-01-01 12:00:00
author:     "tengshiquan"
header-img: "img/post-gd.jpg"
catalog: true
tags:
    - git
---





# Gradient Descent Optimizers

https://mlfromscratch.com/optimizers-explained/#/

[Why Momentum Really Works](https://distill.pub/2017/momentum/).

http://cs231n.github.io/neural-networks-3/ 有代码实现



## The objective of Machine Learning algorithm



**主要是一阶的梯度法，包括SGD, Momentum, Nesterov Momentum, AdaGrad, RMSProp, Adam**

手动指定学习速率: SGD,Momentum,Nesterov Momentum

自动调节学习速率: AdaGrad, RMSProp, Adam





## Stochastic Gradient Descent

in BGD, we have to calculate the cost for all training examples in the dataset.  Exactly this is the motivation behind SGD.

The equation for SGD is used to update parameters in a neural network – we use the equation to update parameters in a backwards pass, using backpropagation to calculate the gradient $\nabla$:
$$
\theta = \theta - \eta \cdot 
    \overbrace{\nabla_\theta J(\theta; \, x, \, y)}^{\text{Backpropagation}}
$$


## Momentum

### Motivation for momentum



<img src="Optimizer.assets/ball-1.gif" alt="When optimizing the cost function for a weight, we might imagine a ball rolling down a hill amongst many hills. We hope that we get to some form of optimum." style="zoom:50%;" />

<img src="Optimizer.assets/image-17.png" alt="Ball stuck on a hilly 2D curve. " style="zoom: 25%;" />

In the above case, we are stuck at a local minimum, and the motivation is clear –  we need a method to handle these situations, perhaps to never get stuck in the first place.



### Explanation of momentum

在SGD基础上扩展,  加个动量,以及相关的权重; 

参数 $\gamma$ called the momentum

动量用的是参数改变的增量, Last change (last update) to θ is called  $v_t$. 上次参数的增量

由**当前梯度**以及**上次的增量(动量, 惯性)**两部分加权组成

This time element increases the momentum of the ball by some amount. This amount is called gamma γ, which is usually initialized to 0.9. But we also multiply that by the **previous update** $v_t$.
$$
\theta_t = \theta_{t} - \eta\nabla J(\theta_{t}) + \gamma v_{t}
$$


> Theta $\theta$ at time step t equals $\theta_t$ minus the learning rate, times the gradient of the objective function J with respect to the parameter $\theta_t$, plus a momentum term gamma $\gamma$, times the change to $\theta$ at the **last time** step t-1.

### Momentum Term

要让球滚下来滚的快, 需要惯性, 积累速度  accumulates more speed for each epoch

下面公式右边漏了 $\gamma$ 
$$
v_{t} = \eta\nabla J(\theta_{t-1}) + v_{t-1}
$$
summation  这个公式有点问题, 看个大概意思就行
$$
\theta_t = \theta_{t} - \eta\nabla J(\theta_{t}) + \gamma \sum_{\tau=1}^{t}
    \eta\nabla J(\theta_{\tau})
$$
<img src="Optimizer.assets/momentum.gif" alt="img" style="zoom: 25%;" />![img](Optimizer.assets/no-momentum.gif)

<img src="https://mlfromscratch.com/content/images/2019/10/no-momentum.gif" alt="img" style="zoom:25%;" />



### Different Notation: A second explanation

the Delta symbol $Δ$ to indicate change:

这里 $\rho$是动量,   这个公式比较好
$$
\Delta w_t = \epsilon \nabla E(w) + \rho \Delta w_{t-1}
$$
改成上面的notation
$$
\Delta \theta_t = \eta \nabla J(\theta_t) + \gamma \Delta \theta_{t-1}
$$

$$
\theta_t = \theta_t - \Delta \theta_t
$$



### pros and cons

缺点: 如果动量太大, 很可能在一个局部最小点边缘一直来回波动



## Adam

**Adaptive Moment Estimation (Adam)**

Adam uses **Momentum** and **Adaptive Learning Rates** to converge faster.



![Animation of how the newer optimizers compare in terms of convergence.](Optimizer.assets/saddle.gif)



### Adaptive Learning Rate

An adaptive learning rate can be observed in AdaGrad, AdaDelta, RMSprop and Adam.

AdaDelta has the same update rule as RMSprop.

The adaptive learning rate property is also known as **Learning Rate Schedules**

Part of the intuition for adaptive learning rates, is that we start off with big steps and finish with small steps – almost like mini-golf. 学习率先大后小



#### AdaGrad: Parameters Gets Different Learning Rates

Adaptive Gradients (AdaGrad) provides us with a simple approach, for **changing the learning rate over time**. This is important for adapting to the differences in datasets, since we can get small or large updates, according to how the learning rate is defined.


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














### Momentum



![](https://cdn-images-1.medium.com/max/1600/1*hJSLxZMjYVzgF5A_MoqeVQ.jpeg)



上面的SGD有个问题,就是每次迭代计算的梯度含有比较大的噪音. 而Momentum方法可以比较好的缓解这个问题,尤其是**在面对小而连续的梯度但是含有很多噪声的时候,可以很好的加速学习**.Momentum借用了物理中的动量概念,即前几次的梯度也会参与运算.为了表示动量,引入了一个新的变量v(velocity).v是之前的梯度的累加,但是每回合都有一定的衰减.

直观上讲就是，要是当前时刻的梯度与历史时刻梯度方向相似，这种趋势在当前时刻则会加强；要是不同，则当前时刻的梯度方向减弱。 **类似滚动的小球，增加的惯性可以起到更平滑和加速的作用，抑制振荡并使我们穿过狭窄的山谷，小驼峰和局部极小。**



**具体实现:** 

需要:**学习速率 ϵ, 初始参数 θ, 初始速率v, 动量衰减参数α** 

每步迭代过程: 
$$
\begin{align}
& \hat g \leftarrow +\frac{1}{m}\nabla_\theta \sum_i L(f(x_i;\theta),y_i)\\
& v\leftarrow\alpha v-\epsilon\hat g \\
& \theta\leftarrow\theta+v
\end{align}
$$
其中参数α表示每回合速率v的衰减程度.同时也可以推断得到,如果每次迭代得到的梯度都是g,那么最后得到的v的稳定值为
$$
\frac{\epsilon\lVert g\rVert}{1-\alpha}
$$
也就是说,Momentum最好情况下能够将学习速率加速$\frac{1}{1-\alpha}$ 倍.一般 α  的取值有0.5,0.9,0.99这几种.当然,也可以让 α  的值随着时间而变化,一开始小点,后来再加大.不过这样一来,又会引进新的参数.

**特点:** 
前后梯度方向一致时,能够加速学习 
前后梯度方向不一致时,能够抑制震荡



### Nesterov Momentum

 从山顶往下滚的球会盲目地选择斜坡。更好的方式应该是在遇到倾斜向上之前应该减慢速度。

这是对之前的Momentum的一种改进,大概思路就是,先对参数进行估计,然后使用估计后的参数来计算误差

**具体实现:** 
需要:**学习速率 ϵ, 初始参数 θ, 初始速率v, 动量衰减参数α** 
每步迭代过程: 

1. 从训练集中的随机抽取一批容量为m的样本$\left\{ x_1,\ldots,x_m \right\}$,以及相关的输出$y_i$ 
2. 计算梯度和误差,并更新速度v和参数θ

$$
\begin{align}
& \hat g \leftarrow +\frac{1}{m}\nabla_\theta \sum_i L(f(x_i;\theta+\alpha v),y_i)\\
& v\leftarrow\alpha v-\epsilon\hat g \\
& \theta\leftarrow\theta+v
\end{align}
$$

注意  在估算 $\hat g$  的时候,参数变成了$\theta+\alpha v$





神经网络研究员早就意识到学习率肯定是难以设置的超参数之一，因为它对模型的性能有显著的影响。 损失通常高度敏感于参数空间中的某些方向，而不敏感于其他。动量算法可以在一定程度缓解这 些问题，但这样做的代价是引入了另一个超参数。 



### AdaGrad  适应性梯度算法

简单来讲，设置全局学习率之后，每次通过 全局学习率逐参数的除以历史梯度平方和的平方根，使得每个参数的学习率不同 . 基本思想是对每个变量用不同的学习率，这个学习率在一开始比较大，用于快速梯度下降。随着优化过程的进行，对于已经下降很多的变量，则减缓学习率，对于还没怎么下降的变量，则保持一个较大的学习率

1. 简单来讲，设置全局学习率之后，每次通过，全局学习率逐参数的除以历史梯度平方和的平方根，使得每个参数的学习率不同 
2. 效果是：在参数空间更为平缓的方向，会取得更大的进步（因为平缓，所以历史梯度平方和较小，对应学习下降的幅度较小） 
3. 缺点是,使得学习率过早，过量的减少
4. 在某些模型上效果不错。

 

**具体实现:**  

需要:**全局学习速率 ϵ, 初始参数 θ, 数值稳定量δ**  

中间变量: 梯度累计量r(初始化为0)  

每步迭代过程: 

> 取m个样本，计算 $y_i$
>
> 计算梯度:  $\hat g \leftarrow +\frac{1}{m}\nabla_\theta \sum_i L(f(x_i;\theta),y_i)$
>
> 累积平方梯度: $r\leftarrow r+\hat g\odot \hat g$
>
> $\triangle \theta = -\frac{\epsilon}{\delta+\sqrt{r}}\odot \hat g$	
>
> $\theta\leftarrow\theta+\triangle \theta$



**优点:** 
能够实现学习率的自动更改。如果这次梯度大,那么学习速率衰减的就快一些;如果这次梯度小,那么学习速率衰减的就慢一些。

**缺点:** 
任然要设置一个变量ϵ
经验表明，在普通算法中也许效果不错，但在深度学习中，深度过深时会造成训练提前结束。

 



### RMSProp   均方根传播

RMSProp通过引入一个衰减系数，让r每回合都衰减一定比例，类似于Momentum中的做法。

1.[AdaGrad]算法的改进。鉴于神经网络都是非凸条件下的，RMSProp在非凸条件下结果更好，改变梯度累积为指数衰减的移动平均以丢弃遥远的过去历史。

2.经验上，RMSProp被证明有效且实用的深度学习网络优化算法。







### Adam

随机梯度下降保持单一的学习率更新所有的权重，学习率在训练过程中并不会改变。而 Adam 通过计算梯度的一阶矩估计和二阶矩估计而为不同的参数设计独立的自适应性学习率。

- 适应性梯度算法（AdaGrad）为每一个参数保留一个学习率以提升在稀疏梯度（即自然语言和计算机视觉问题）上的性能。
- 均方根传播（RMSProp）基于权重梯度最近量级的均值为每一个参数适应性地保留学习率。这意味着算法在非稳态和在线问题上有很有优秀的性能。

Adam 算法同时获得了 AdaGrad 和 RMSProp 算法的优点。











