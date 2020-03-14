---
layout:     post
title:      Steepest Descent & Conjugate Gradient
subtitle:   
date:       2020-03-01 12:00:00
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



# Steepest Descent & Conjugate Gradient



## Steepest Descent  最陡梯度下降

大概思路:  

1. 对于任意函数, 要通过迭代方式求极小值.  任意取一点, 然后求该点的梯度, 沿着梯度反方向移动, 显然是该点最陡的方向;  
2. 对一般GD算法, 步长就是学习率; 在这个算法里面,  步长是 计算出来的.  什么样的步长最适合?  沿着该点梯度反方向走, 一直走到该方向函数值由下降转为上升的点, 即一直走到与目前方向与那个点等高线相切.  显然本次方向也就与下一次迭代的方向垂直了. 轨迹是zigzag .   
3. 另一个角度,下一步的迭代方向如果跟前一步的不垂直, 那说明前一步还有分量在该点上可以继续改进.
4. 怎么求这个点, 显然这个点是该次迭代方向上的一个极值点, 同时又是步长这个单变量的函数, 可以令方向导数为0来求.    **Line search**    注意区分, 该点的梯度以及该点的方向导数. 
5. 如果起始点选在特征向量方向上, 则可以一步走到目标; 对二维, 两个特征向量之间一个方向, 拉伸后是45°, 是收敛步数最多的.



<img src="http://fourier.eng.hmc.edu/e176/lectures/GradientDescent.png" alt="GradientDescent.png" style="zoom: 67%;" />



<img src="http://fourier.eng.hmc.edu/e176/lectures/GradientDescent1.png" alt="GradientDescent1.png" style="zoom: 67%;" />



- The **minimum** of the objective function $f(x)$ can be achieved from an initial guess $x_0$, followed by an iterative process during which the estimated solution is moved along the negative direction of the derivative $f'(x)=df(x)/dx$: 

$$
x_{n+1}=x_n-\delta \; f'(x_n)
$$



- $f(x_1,\cdots,x_N)=f({\mathbf x})$ is a scalar function of $N$ variables ${\mathbf x}=[x_1,\cdots,x_N]^T$. The gradient vector ${\mathbf g}_f$ of function $f({\mathbf x})$ is

$$
 \mathbf{g}_{f(\mathbf{x})}=\nabla f(\mathbf{x})=\frac{d f(\mathbf{x})}{d \mathbf{x}}=\left[\frac{\partial f(\mathbf{x})}{\partial x_{1}}, \cdots, \frac{\partial f(\mathbf{x})}{\partial x_{N}}\right]^{T}
$$



- In each iteration, we need to find the optimal step size 标量 $\delta_n$ that minimizes the function

$$
f_{n+1}=f({\mathbf x}_{n+1})=f({\mathbf x}_n-\delta_n\, {\mathbf g}_n)
$$

- its derivative with respect to $\delta$ should be zero. By the chain rule, we have

$$
\frac{d}{d \delta_{n}} f\left(\mathbf{x}_{n+1}\right)=\frac{d}{d \mathbf{x}} f\left(\mathbf{x}_{n+1}\right) \cdot \frac{d}{d \delta_{n}}\left(\mathbf{x}-\delta_{n} \mathbf{g}_{n}\right)=\nabla f\left(\mathbf{x}_{n+1}\right)^{T}\left(-\mathbf{g}_{n}\right)=-\mathbf{g}_{n+1}^{T} \mathbf{g}_{n}=0
$$

- This result indicates that at $\mathbf x_{n+1}$ the direction of steepest descent $ \mathbf g_{n+1}$ is always perpendicular to the previous direction $\mathbf g_n$, i.e., $\mathbf g_{n+1}\perp \mathbf g_n$.  显然垂直




- If the function $f({\mathbf x})$ is **quadratic** 二次方程  

$$
f({\mathbf x})=\frac{1}{2}{\mathbf x}^T{\mathbf A}{\mathbf x}-{\mathbf b}^T{\mathbf x}+c 
$$

- gradient vector:  $${\mathbf g}=\frac{d}{d{\mathbf x}}f({\mathbf x})={\mathbf A}{\mathbf x}-{\mathbf b}$$

- The iteration for the minimization is  $${\mathbf x}_{n+1}={\mathbf x}_n-\delta_n{\mathbf g}_n$$

- The **optimal step size** $\delta_n$ that minimizes $$f({\mathbf x}_{n+1})=f({\mathbf x}_n-\delta_n{\mathbf g}_n)$$ can be found based on the fact 

  $$
  \begin{aligned}
\mathbf{g}_{n+1}^{T} \mathbf{g}_{n} &=\left(\mathbf{A} \mathbf{x}_{n+1}-\mathbf{b}\right)^{T} \mathbf{g}_{n}=\left[\mathbf{A}\left(\mathbf{x}_{n}-\delta_{n} \mathbf{g}_{n}\right)-\mathbf{b}\right]^{T} \mathbf{g}_{n} \\
&=\left(\mathbf{A} \mathbf{x}_{n}-\mathbf{b}\right)^{T} \mathbf{g}_{n}-\delta_{n}\left(\mathbf{A} \mathbf{g}_{n}\right)^{T} \mathbf{g}_{n}=\mathbf{g}_{n}^{T} \mathbf{g}_{n}-\delta_{n} \mathbf{g}_{n}^{T} \mathbf{A} \mathbf{g}_{n}=0
\end{aligned}
  $$
  
- 得到 $\delta_{n}$

$$
  \delta_n=\frac{ {\mathbf g}_n^T{\mathbf g}_n}{ {\mathbf g}_n^T{\mathbf A}{\mathbf g}_n }
$$





http://fourier.eng.hmc.edu/e176/lectures/NumericalMethods/node18.html





## Conjugate Gradient  共轭梯度

http://fourier.eng.hmc.edu/e176/lectures/NumericalMethods/node19.html

<img src="http://fourier.eng.hmc.edu/e176/lectures/ConjugateGradient.png" alt="ConjugateGradient.png" style="zoom:80%;" />

form https://en.wikipedia.org/wiki/Conjugate_gradient_method

![img](/img/2020-03-01-ConjugateGradient.assets/220px-Conjugate_gradient_illustration.svg.png)



https://zhuanlan.zhihu.com/p/64227658  

说的比较简单  但这篇里面用r表示搜索方向, 一般r都是表示残差, 负梯度. 公式部分不用细看

所谓共轭梯度法, 最大的优势就是每个方向都走到了极致, 也即是说寻找极值的过程中绝不走曾经走过的方向,那么 n 空间的函数极值也就走 n 步就解决了.假如是二维空间, 那就直走两步, 跟最速下降法比优势是不言而喻的.

- 满足什么条件才叫某个方向彻底走到了极致呢?
- 如果说**当前的误差跟上一步的方向是正交的**是不是意味着这个方向再也不需要走了?
- 所谓迭代, 就是根据当前位置不断地得到新的方向和新的步长, 这样走下去.
- 我们将下一步要走的方向记作: $r_t$ 
- 那前面的误差与上一步方向正交意思就是: $r_{t-1}^{T}{e_t}=0$
- 那我们要是做到了这点方向就把握住了 ! 我们就走这个误差方向.
- 但实际上我们根本找不到这个方向, 要是那么清楚方向一开始一步到位不就是了?
- 但我们有一个等价的做法, 那就是**共轭正交**: $r_{t-1}^{T}A{e_t}=0$
- 其中矩阵 $A$ 是一个常对称矩阵, 也就是作用在右边向量的一个线性变换罢了.

算法核心: 方向 + 步长


- 所谓迭代, 就是根据当前位置不断地得到新的方向和新的步长, 这样走下去.
- 下面我们来确定这个步长 $\alpha _{t}$ :
- 要保证 $r_{t}^{T}A{e_{t+1}}=0$ .
- 利用关系 $${e_{t}}={x^{*}}-{x_{t}}\Rightarrow {e_{t+1}}={x^{*}}-{x_{t+1}}={e_{t}}+{x_{t}}-{x_{t+1}}$$
- 可得: 

$$
\begin{align}   & r_{t}^{T}A{e_{t+1}}=r_{t}^{T}A\left[ {e_{t}}+{x_{t}}-{x_{t+1}} \right] \\   & \ \ \ \ \ \ \ \ \ \ \ \ \ \ =r_{t}^{T}A\left[ {e_{t}}-{\alpha_{t}}{e_{t}} \right] \\   & \ \ \ \ \ \ \ \ \ \ \ \ \ \ =r_{t}^{T}A{e_{t}}-{\alpha_{t}}r_{t}^{T}A{e_{t}}=0 \\  \end{align}
$$

- $$\Rightarrow {\alpha_{t}}=\frac{r_{t}^{T}A{e_{t}}}{r_{t}^{T}Ar}$$

- 已知 $ \nabla f\left( x \right)=Ax-b $

- 
  $$
  \Rightarrow {\alpha_{t}}=\frac{r_{t}^{T}A{e_{t}}}{r_{t}^{T}A{r_{t}}}=\frac{r_{t}^{T}A\left( {x^{*}}-{x_{t}} \right)}{r_{t}^{T}A{r_{t}}}=\frac{r_{t}^{T}\left( b-A{x_{t}} \right)}{r_{t}^{T}A{r_{t}}}=-\frac{r_{t}^{T}\nabla f\left( {x_{t}} \right)}{r_{t}^{T}A{r_{t}}}
  $$

- 消除了前面未知误差$e$, 能确定这一步的步长了.



现在来确定方向 $r_t$ :

- 至于第一个方向怎么选取呢? 理论上是随便选, 不过当然选负梯度方向是最好.

- 给一组向量, 要得到正交化的向量, **施密特正交化**

  define the [projection](https://en.wikipedia.org/wiki/Projection_(linear_algebra)) [operator](https://en.wikipedia.org/wiki/Operator_(mathematics)) by 

  
  $$
  {\displaystyle \mathrm {proj} _{\mathbf {u} }\,(\mathbf {v} )={\langle \mathbf {v} ,\mathbf {u} \rangle  \over \langle \mathbf {u} ,\mathbf {u} \rangle }{\mathbf {u} },}
  $$
  

  The Gram–Schmidt process then works as follows:
  $$
  \begin{aligned}
  &\mathbf{u}_{1}=\mathbf{v}_{1}, \quad \mathbf{e}_{1}=\frac{\mathbf{u}_{1}}{\left\|\mathbf{u}_{1}\right\|}\\
  &\begin{aligned}
  \mathbf{u}_{2} &=\mathbf{v}_{2}-\operatorname{proj}_{\mathbf{u}_{1}}\left(\mathbf{v}_{2}\right), & \mathbf{e}_{2} &=\frac{\mathbf{u}_{2}}{\left\|\mathbf{u}_{2}\right\|} \\
  \mathbf{u}_{3} &=\mathbf{v}_{3}-\operatorname{proj}_{\mathbf{u}_{1}}\left(\mathbf{v}_{3}\right)-\operatorname{proj}_{\mathbf{u}_{2}}\left(\mathbf{v}_{3}\right), & \mathbf{e}_{3} &=\frac{\mathbf{u}_{3}}{\left\|\mathbf{u}_{3}\right\|} \\
  \mathbf{u}_{4} &=\mathbf{v}_{4}-\operatorname{proj}_{\mathbf{u}_{1}}\left(\mathbf{v}_{4}\right)-\operatorname{proj}_{\mathbf{u}_{2}}\left(\mathbf{v}_{4}\right)-\operatorname{proj}_{\mathbf{u}_{3}}\left(\mathbf{v}_{4}\right), & \mathbf{e}_{4} &=\frac{\mathbf{u}_{4}}{\left\|\mathbf{u}_{4}\right\|} \\
  & \vdots & & \vdots \\
  \mathbf{u}_{k} &=\mathbf{v}_{k}-\sum_{j=1}^{k-1} \operatorname{proj}_{\mathbf{u}_{j}}\left(\mathbf{v}_{k}\right), & \mathbf{e}_{k} &=\frac{\mathbf{u}_{k}}{\left\|\mathbf{u}_{k}\right\|}
  \end{aligned}
  \end{aligned}
  $$
  
- 不过**现在的正交是共轭正交**了, 但基本上差不多是一个意思, 按照施密特正交化的原理去推导出共轭正交的一组向量就是了.

- 这样方向将被如此确定: 

$$
{r_{t}}=-\nabla f\left( {x_{t}} \right)+\sum\limits_{i<t}{\frac{r_{i}^{T}A\nabla f\left( {x_{t}} \right)}{r_{i}^{T}A{r_{i}}}{r_{i}}}
$$





## 共轭梯度法详细推导分析

https://blog.csdn.net/weixin_37895339/article/details/84640137

算法求解速度较快，虽然比梯度下降法复杂，但是比二阶方法简单。


$$
r_k = -(Ax_k -b)   \quad 负梯度, 残差 \\ 
e_k = x^* - x_k 	\quad 误差
$$


虽然梯度下降法的每一步都是朝着局部最优的方向前进的，但是它在不同的迭代轮数中会选择非常近似的方向，说明这个方向的误差并没通过一次更新方向和步长更新完，在这个方向上还存在误差，因此参数更新的轨迹是锯齿状。共轭梯度法的思想是，选择一个优化方向后，本次选择的步长能够将这个方向的误差更新完，在以后的优化更新过程中不再需要朝这个方向更新了。由于每次将一个方向优化到了极小，后面的优化过程将不再影响之前优化方向上的极小值，所以理论上对N维问题求极小只用对N个方向都求出极小就行了。为了不影响之前优化方向上的更新量，需要每次优化方向共轭正交。假定每一步的优化方向用$p_k$ 表示，可得共轭正交

$$
p_iAp_j = 0  , i \neq j
$$

由此可得，每一步优化后，当前的误差和刚才的优化方向共轭正交。

$$
p_k A e_{k+1} = 0
$$

#### 1.优化方向确定

假定第一次优化方向为初始负梯度方向

$$
   p_1 = r_1 = b-Ax_1
$$

 每一次优化方向与之前的优化方向正交，采用Gram-Schmidt方法进行向量正交化，每次优化方向根据**当前步的梯度**得出,  当前点的梯度, 减去了之前所有步的p分量, 即**将当前梯度共轭正交化**

$$
 p_k = r_k-\sum_{i \lt k}\frac{p_i^TAr_k}{p_i^TAp_i}p_i
$$


 

#### 2.优化步长的选取

假定第k步的优化步长为 $\alpha_k$ ,  这里对于p, 可以是任意选择的, 如果是每点的梯度, 则是最速下降法.

方法一： 

$$
f\left( {x_{t+1}} \right)=f\left( {x_{k}}+{\alpha_{k}}{p_{k}} \right)=g\left( {\alpha _{k}} \right)
$$

对 $\alpha_k$ 求导令导数为0可得

$$
\alpha_k=\frac{p_k ^Tr_k}{p_k^TAp_k}
$$

方法二：   

$ e_{k+1} = x^{*}-x_{k+1} = x^{*}-x_{k}+x_{k}-x_{k+1} = e_{k}-a_{k} p_{k}  $   该式可以叠加到第一项

$$
\begin{align}
\begin{aligned}
p_{k}^{T} A e_{k+1} &=p_{k}^{T} A\left(x^{*}-x_{k+1}\right) \\
&=p_{k}^{T} A\left(x^{*}-x_{k}+x_{k}-x_{k+1}\right) \\
&=p_{k}^{T} A\left(e_{k}-a_{k} p_{k}\right) \\
&=p_{k}^{T} A e_{k}-a_{k} p_{k}^{T} A p_{k}=0 \\ \\
\to a_{k} &=\frac{p_{k}^{T} A e_{k}}{p_{k}^{T} A p_{k}} \\
&=\frac{p_{k}^{T} A\left(x^{*}-x_{k}\right)}{p_{k}^{T} A p_{k}} \\
&=\frac{p_{k}^{T}\left(A x^{*}-A x_{k}\right)}{p_{k}^{T} A p_{k}} \\
&=\frac{p_{k}^{T}\left(b-A x_{k}\right)}{p_{k}^{T} A p_{k}} \\
&=\frac{p_{k}^{T} r_{k}}{p_{k}^{T} A p_{k}}
\end{aligned}
\end{align}
$$

#### 推论

1. 第k步计算的梯度 $r_k$  和前k-1步的优化向量$ \{p_i\}_{i=1}^{k-1}$正交。 
2. 第k步计算的梯度 $r_k$  和前k-1步的梯度$ \{r_i\}_{i=1}^{k-1}$正交。 
3. 第k步计算的梯度 $r_k$  和前k-2步的优化向量 $\{p_i\}_{i=1}^{k-2}$ 共轭正交。
  

工程中可以利用推论来进行算法优化. 







## An Introduction to the Conjugate Gradient Method Without the Agonizing Pain

来自论文 'An Introduction to the Conjugate Gradient Method Without the Agonizing Pain'

笔记, 在线代方面扩展了很多知识点. 



- **正定** :  对任意非0向量x,  matrix $A$ is **positive-definite** if, for every nonzero vector $x$

$$
x^{T} A x>0 \tag{2}
$$
- **二次型**: A **quadratic form** is simply a scalar, quadratic function of a vector with the form (该定义有问题, 不该有一次项!!)


$$
f(x)=\frac{1}{2} x^{T} A x-b^{T} x+c  \tag{3}
$$
- 二次型（quadratic form）：n个变量的二次多项式称为二次型，即在一个多项式中，未知数的个数为任意多个，但每一项的次数都为2的多项式。同时, A可以由f(x)构造出对称的, 所以下面只需考虑正定就行.  
  
  $$
\begin{aligned}
f\left(x_{1}, x_{2}, \cdots, x_{n}\right)=a_{11} x_{1}^{2}+2 a_{12} x_{1} x_{2} & +\cdots+2 a_{1 n} x_{1} x_{n}  \\
+a_{22} x_{2}^{2} & +\cdots+2 a_{2 n} x_{2} x_{n} \\
  & +\cdots+a_{n n} x_{n}^{2}  \\
  = \sum_{i=1}^{n} \sum_{j=1}^{n} a_{i j} x_{i} x_{j} \\
  \text{where} \quad  a_{ij} =a_{ji}, \quad  2a_{i j} x_{i} x_{j}  &=a_{i j} x_{i} x_{j}+a_{j i} x_{j} x_{i}
  \end{aligned}
  $$
  
- 

- **前提** A是 **对称正定矩阵** , 才有等效问题, 求二次f(x)的最小值 等价于 求解 Ax = b.   if $A$ is **symmetric** and **positive-definite**, $f(x)$ is **minimized** by the solution to $A x=b$     

- 如果A是正定矩阵, 则f(x)的形状像 碗. Because $A$ is **positive-definite**, the surface defined by $f(x)$ is shaped like a paraboloid bowl. 

<img src="/img/2020-03-01-ConjugateGradient.assets/image-20200303023857580.png" alt="image-20200303023857580" style="zoom:50%;" />

- 下文使用这个example :
$$
  A=\left[\begin{array}{ll}
  3 & 2 \\
  2 & 6
  \end{array}\right], \quad b=\left[\begin{array}{r}
  2 \\
  -8
  \end{array}\right], \quad c=0 \tag{4}
$$

- **梯度**: The **gradient** of a quadratic form is defined to be
$$
f^{\prime}(x)=\left[\begin{array}{c}
\frac{\partial}{\partial x_{1}} f(x) \\
\frac{\partial }{\partial x_{2}} f(x) \\
\vdots \\
\frac{\partial}{\partial x_{n}} f(x)
\end{array}\right]  \tag{5}
$$
- 梯度就是最大增量的方向. 极值点 : 梯度为0 . The gradient is a vector field that, for a given point $x$, points in the direction of greatest increase of $f(x)$.  One can minimize $f(x)$ by setting $f^{\prime}(x)$ equal to zero. 

- 写成矩阵形式
$$
f^{\prime}(x)=\frac{1}{2} A^{T} x+\frac{1}{2} A x-b  \tag{6}
$$
- 对称 If $A$ is **symmetric**, this equation reduces to  $f^{\prime}(x)=A x-b$

- 令$f^{\prime}(x)=0$, 求解x Setting the gradient to zero, we obtain $A x=b$  the linear system we wish to solve. Therefore, the solution to $A x=b$ is a critical point of $f(x) $. 

- A对称正定, 一次线性求解与二次极值 问题等价  If $A$ is **positive-definite** as well as **symmetric**, then this solution is a **minimum** of $f(x),$ so $A x=b$ can be solved by finding an $x$ that minimizes $f(x)$. (If $A$ is not symmetric, then Equation 6 hints that CG will find a solution to the system $\frac{1}{2}\left(A^{T}+A\right) x=b .$ Note that $\frac{1}{2}\left(A^{T}+A\right)$ is symmetric).

- Why do **symmetric positive-definite** matrices have this nice property? Consider the relationship between $f$ at some **arbitrary point** $p$ and at the **solution point** $x=A^{-1} b .$ From Equation 3 one can show (Appendix C1) that if $A$ is **symmetric** (be it positive-definite or not), 若对称, 可以分解出 eAe
$$
  f(p)=f(x)+\frac{1}{2}(p-x)^{T} A(p-x)  \tag{8}
$$
  If $A$ is **positive-definite** as well, then by Inequality $2,$ the latter term is positive for all $p \neq x$. It follows that $x$  is a global minimum of $f$ .  再正定,  则存在最小值, 就是Ax=b的解. 不正定, 该解不是极值,也不存在极值. 

- ![image-20200303152058580](/img/2020-03-01-ConjugateGradient.assets/image-20200303152058580.png)
二次型对应的原函数f(x)各种形状: (a) **Quadratic form** for a **positive-definite** matrix. (b) For a **negative-definite** matrix. (c) For a **singular (and positive-indefinite) matrix**. A line that runs through the bottom of the valley is the set of solutions. (d) For an **indefinite** matrix. Because the solution is a saddle point, Steepest Descent and CG will not work. In three dimensions or higher, a singular matrix can also have a saddle.



##### The Method of Steepest Descent 最陡梯度法

这里用迭代求二次型的极值点,  利用梯度为一次函数, 引入了一些概念后面一直用到

- 从任意点出发, 滑向底部 . In the method of Steepest Descent, we start at an arbitrary point $x_{(0)}$ and slide down to the bottom of the paraboloid. We take a series of steps $x_{(1)}, x_{(2)}, \ldots$ until we are satisfied that we are close enough to the solution $x$

- 方向是负梯度. When we take a step, we choose the direction in which $f$ decreases most quickly, which is the direction opposite $f^{\prime}\left(x_{(i)}\right) .$ According to Equation $7,$ this direction is $-f^{\prime}\left(x_{(i)}\right)=b-A x_{(i)}$

- **误差 离x真值** The **error** $e_{(i)}=x_{(i)}-x$ is a **vector** that indicates how far we are **from the solution x**. 

- **残差 离拟合目标b的距离**  The **residual** $r_{(i)}=b-A x_{(i)}$ indicates how far we are **from the correct value of $b$**. 

- 关键公式  $r_{(i)}=-A e_{(i)}$,   **residual** as being the error transformed by $A$ into the same space as $b$. 

- **residual 就是该点负梯度**. More importantly, $r_{(i)}=-f^{\prime}\left(x_{(i)}\right),$  **residual** as the **direction of steepest descent**.   residual本来就是梯度, 只是一次函数里面有了error, 然后有了r=-Ae转换

- <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200303160404132.png" alt="image-20200303160404132" style="zoom:80%;" />

- Suppose we start at $x_{(0)}=[-2,-2]^{T} .$ Our first step, along the direction of steepest descent, will fall l somewhere on the solid line in Figure 6(a). In other words, we will choose a point $x_{(1)} = x_{(0)} + \alpha r_{(0)}$ .

  步长多大? how big a step should we take ?

- 抛物线就是 linesearch平面与二次型的交线. A **line search** is a procedure that chooses $\alpha$ to minimize $f$ along a line.  Figure $6(\mathrm{b})$ we are restricted to choosing a point on the intersection of the vertical plane and the paraboloid. Figure $6(\mathrm{c})$ is the parabola defined by the intersection of these surfaces.  

- 求抛物线极值点的$\alpha$值,得出**两次方向正交**.  $\alpha$ minimizes $f$ when the directional derivative $\frac{d}{d \alpha} f\left(x_{(1)}\right)$ is equal to zero. By the chain rule, $\frac{d}{d \alpha} f\left(x_{(1)}\right)=f^{\prime}\left(x_{(1)}\right)^{T} \frac{d}{d \alpha} x_{(1)}=f^{\prime}\left(x_{(1)}\right)^{T} r_{(0)} .$ Setting this expression to zero, we find that $\alpha$ should be chosen so that $r_{(0)}$ and $f^{\prime}\left(x_{(1)}\right)$ are orthogonal (see Figure 6(d)).

-  <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200303161713234.png" alt="image-20200303161713234" style="zoom:67%;" />  Figure 7 显示search line上个点的梯度. 

- 抛物线上任意点的斜率等于search line上相应点的梯度的投影(虚线箭头).   The slope of the parabola (Figure $6(\mathrm{c})$ ) at any point is equal to the magnitude of the projection of the gradient onto the line (Figure 7 , dotted arrows). $f$ is minimized where the projection is zero - where the gradient is orthogonal to the search line.

- 计算步长 To determine $\alpha,$ note that $f^{\prime}\left(x_{(1)}\right)=-r_{(1)},$ and we have
$$
  \begin{aligned}
  r_{(1)}^{T} r_{(0)} &=0 \\
  \left(b-A x_{(1)}\right)^{T} r_{(0)} &=0 \\
  \left(b-A\left(x_{(0)}+\alpha r_{(0)}\right)\right)^{T} r_{(0)} &=0 \\
  \left(b-A x_{(0)}\right)^{T} r_{(0)}-\alpha\left(A r_{(0)}\right)^{T} r_{(0)} &=0 \\
  \left(b-A x_{(0)}\right)^{T} r_{(0)} &=\alpha\left(A r_{(0)}\right)^{T} r_{(0)} \\
  r_{(0)}^{T} r_{(0)} &=\alpha r_{(0)}^{T}\left(A r_{(0)}\right) \\
  \alpha &=\frac{r_{(0)}^{T} r_{(0)}}{r_{(0)}^{T} A r_{(0)}}
  \end{aligned}
$$
- <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200305035431526.png" alt="image-20200305035431526" style="zoom: 67%;" />

- **Steepest Descent**算法 Putting it all together, the method of **Steepest Descent** is:
  
$$
  \begin{aligned}
  r_{(i)} &=b-A x_{(i)}   \quad \quad \quad \quad \quad  (10)\\
  \alpha_{(i)} &=\frac{r_{(i)}^{T} r_{(i)}}{r_{(i)}^{T} A r_{(i)}} \quad \quad \quad \quad  \quad  (11)\\
  x_{(i+1)} &=x_{(i)}+\alpha_{(i)} r_{(i)} \quad \quad \quad  \quad  (12)
  \end{aligned}
$$

- 计算优化  The algorithm, as written above, requires two matrix-vector multiplications per iteration. The computational cost of Steepest Descent is dominated by matrix-vector products; fortunately, one can be eliminated. By premultiplying both sides of Equation 12 by $-A$ and adding $b,$ we have
  
$$
  r_{(i+1)}=r_{(i)}-\alpha_{(i)} A r_{(i)} \tag{13}
$$

- Although Equation 10 is still needed to compute $r_{(0)}$, Equation 13 can be used for every iteration thereafter. The product $A r$, which occurs in both Equations 11 and 13 , need only be computed once. The disadvantage of using this recurrence is that the sequence defined by Equation 13 is generated without any feedback from the value of $x_{(i)},$ so that accumulation of floating point roundoff error may cause $x_{(i)}$ to converge to some point near $x$. This effect can be avoided by periodically using Equation 10 to recompute the correct residual.

  

具体算法 
$$
\begin{aligned}  
& i \Leftarrow 0 \\
& r \Leftarrow b-A x \\
&\delta \Leftarrow r^{T} r \\
&\delta_{0} \Leftarrow \delta \\
&\text { While } i<i_{\max } \text { and } \delta>\varepsilon^{2} \delta_{0} \text { do } \\
&\quad\quad q \Leftarrow A r \\
&\quad\quad \alpha \Leftarrow \frac{\delta}{r^{T} q} \\
&\quad\quad x \Leftarrow x+\alpha r \\
&\quad\quad \text { If } i \text { is divisible by } 50 \\
&\quad\quad\quad\quad r \Leftarrow b-A x \\
&\quad\quad \text { else } \\
&\quad\quad\quad\quad r \Leftarrow r-\alpha q \\
&\quad\quad \delta \Leftarrow r^{T} r \\
&\quad\quad i \Leftarrow i+1 \\
\end{aligned} 
$$


##### Eigenvectors and Eigenvalues 用特征值特征向量解释

- 矩阵的**特征向量**就是矩阵作用到该向量上,不会发生旋转.  An **eigenvector** $v$ of a matrix $B$ is a nonzero vector that **does not rotate** when $B$ is applied to it (except perhaps to point in precisely the opposite direction). $v$ may change length or reverse its direction, but it won't turn sideways. 

- **特征值**. there is some scalar constant $\lambda$ such that $B v=\lambda v .$ The value $\lambda$ is an **eigenvalue** of $B$

- 特征向量可拉伸. For any constant $\alpha$, the vector $\alpha v$ is also an eigenvector with eigenvalue $\lambda$

- 迭代就是对向量不停地做线性变换.  Why should you care? Iterative methods often depend on applying $B$ to a vector over and over again. 

-  按特征值缩放.  When $B$ is repeatedly applied to an eigenvector $v$, one of two things can happen. 

  - If $\vert \lambda \vert<1,$ then $B^{i} v=\lambda^{i} v$ will vanish as $i$ approaches infinity . 
  - If $\vert \lambda \vert >1,$ then $B^{i} v$ will grow to infinity. Each time $B$ is applied, the vector grows or shrinks according to the value of $\vert \lambda \vert$

- **对称矩阵,有n个线性无关特征向量.** 特征向量不唯一, 矩阵对应的特征值集合是唯一的.  If $B$ is **symmetric** (and often if it is not), then there exists a set of $n$ **linearly independent eigenvectors** of $B,$ denoted $v_{1}, v_{2}, \ldots, v_{n} .$ This set is not unique, because each eigenvector can be scaled by an arbitrary nonzero constant. Each eigenvector has a corresponding **eigenvalue**, denoted $\lambda_{1}, \lambda_{2}, \ldots, \lambda_{n} .$ These are **uniquely** defined for a given matrix. The eigenvalues may or may not be equal to each other; for instance, the eigenvalues of the identity matrix $I$ are all one, and every nonzero vector is an eigenvector of $I$ . 

- 将B作用到非特征向量, 可将该向量拆成特征向量的线性组合: What if $B$ is applied to a vector that is not an eigenvector?  think of a vector as a sum of other vectors whose behavior is understood. Consider that the set of eigenvectors $\lbrace v_{i} \rbrace$ forms a basis for $\mathbb{R}^{n}$ (because a symmetric $B$ has $n$ eigenvectors that are linearly independent). Any $n$ -dimensional vector can be expressed as a linear combination of these eigenvectors, and because matrix-vector multiplication is distributive, one can examine the effect of $B$ on each eigenvector separately.

- ![image-20200305015201046](/img/2020-03-01-ConjugateGradient.assets/image-20200305015201046.png)

- 如图, 一个分量收敛, 一个分量发散, 组合后也发散. In Figure 11, a vector $x$ is illustrated as a sum of two eigenvectors $v_{1}$ and $v_{2} .$ Applying $B$ to $x$ is equivalent to applying $B$ to the eigenvectors, and summing the result. On repeated application, we have $B^{i} x=B^{i} v_{1}+B^{i} v_{2}=\lambda_{1}^{i} v_{1}+\lambda_{2}^{i} v_{2} .$ If the magnitudes of all the eigenvalues are smaller than one, $B^{i} x$ will **converge** to zero, because the eigenvectors that compose $x$ converge to zero when $B$ is repeatedly applied. If one of the eigenvalues has magnitude greater than one, $x$ will **diverge** to infinity. 

- This is why numerical analysts attach importance to the **谱半径 spectral radius** of a matrix:
  
$$
  \rho(B)=\max \left|\lambda_{i}\right|, \quad \lambda_{i} \text { is an eigenvalue of } B
$$

  If we want $x$ to converge to zero quickly, $\rho(B)$ should be less than one, and preferably as small as possible.  想收敛,  谱半径越小越好

- 非对称矩阵没有n个线性无关的特征向量.  **nonsymmetric matrices** that do not have a full set of $n$ independent eigenvectors. These matrices are known as **退化 defective** , a name that betrays the well-deserved hostility they have earned from frustrated linear algebraists. the behavior of defective matrices can be analyzed in terms of **generalized eigenvectors and generalized eigenvalues**. The rule that $B^{i} x$ converges to zero if and only if all the generalized eigenvalues have magnitude smaller than one still holds, but is harder to prove.

-  **正定矩阵的特征值都是正数**.  the **eigenvalues** of a **positive-definite** matrix are all **positive**. This fact can be proven from the definition of eigenvalue:   
  By the definition of positive-definite, the $v^{T} B v$ is positive (for nonzero $v$ ). Hence, $\lambda$ must be positive also. 
$$
  \begin{aligned}
  B v &=\lambda v \\
  v^{T} B v &=\lambda v^{T} v
  \end{aligned}
$$

- **Jacobi Method** for solving $A x=b$

-  拆分为对角阵以及余阵. The matrix $A$  is **split** into two parts: $D$ , whose **diagonal** elements are identical to those of $A$, and whose **off-diagonal** elements are zero; and $E$, whose diagonal elements are zero, and whose off-diagonal elements are identical to those of $A$. Thus, ​ $A = D + E$. We derive the Jacobi Method:
$$
  \begin{aligned}
  A x &=b \\
  D x &=-E x+b \\
  x &=-D^{-1} E x+D^{-1} b \\
  x &=B x+z, \quad \text { where } \quad B=-D^{-1} E, \quad z=D^{-1} b
  \end{aligned} \tag{14}
$$

- 迭代求解: **驻点**;  Because $D$ is diagonal, it is easy to invert. This identity can be converted into an iterative method by forming the recurrence    **stationary point**
  
$$
  x_{(i+1)}=B x_{(i)}+z \tag{15}
$$

- 也有其他拆分矩阵的方法, 用于求解线性方程. choosing a different $D$ and $E$  -- derived the **Gauss-Seidel** method, or the method of **Successive Over-Relaxation (SOR)**. 

- 需要B的谱半径比较小, 这样收敛快.  Our hope is that we have chosen a splitting for which $B$ has a **small spectral radius**. Here, I chose the Jacobi splitting arbitrarily for simplicity.

- error 序列的迭代公式. apply the principle of thinking of a vector as a sum of other, well-understood vectors. Express each iterate $x_{(i)}$ as the sum of the exact solution $x$ and the error term $e_{(i)} .$ Then, Equation 15 becomes
  
$$
  \begin{aligned}x_{(i+1)} &=B x_{(i)}+z \\&=B\left(x+e_{(i)}\right)+z \\&=B x+z+B e_{(i)} \\&=x+B e_{(i)} \\ \therefore e_{(i+1)} &=B e_{(i)}
  \end{aligned}
$$
- 起始点选择无所谓, 只要矩阵B的变换能让 error越来越小 Each iteration does not affect the "correct part" of $x_{(i)}$ (because $x$ is a stationary point); but each iteration does affect the error term. It is apparent from Equation 16 that if $\rho(B)<1$, then the error term  $e_{(i)}$  will converge to zero as $i$ approaches infinity. Hence, the initial vector  $x_{(0)}$ has no effect on the inevitable outcome!  

- 影响收敛速度的元素, 初始位置, 矩阵B的谱半径 the choice of $x_{(0)}$ does affect the number of iterations required to converge to $x$ within given tolerance. However, its effect is less important than that of the **spectral radius** $\rho(B),$ which determines the speed of convergence. 

- Jacobi 方法一般不收敛 .  $B$ is not generally symmetric (even if $A$ is), and may even be defective. However, the rate of convergence of the Jacobi Method depends largely on $\rho(B)$,which depends on $A$ . Unfortunately , Jacobi does not converge for every $A$ , or even for every positive-definite $A$.

- ![image-20200304014013300](/img/2020-03-01-ConjugateGradient.assets/image-20200304014013300.png)

- 特征多项式
  
$$
  \begin{array}{l}
  A v=\lambda v=\lambda I v \\
  (\lambda I-A) v=0
  \end{array}
$$


  Eigenvectors are nonzero, so $\lambda I-A$ must be singular. Then,
$$
  \operatorname{det}(\lambda I-A)=0
$$
  The determinant of $\lambda I-A$ is called the **characteristic polynomial**. It is a degree $n$ polynomial in $\lambda$ whose roots are the set of eigenvalues. 

- 举例求特征向量以及特征值: The characteristic polynomial of $A$ (from Equation 4) is
$$
  \operatorname{det}\left[\begin{array}{cc}
  \lambda-3 & -2 \\
  -2 & \lambda-6
  \end{array}\right]=\lambda^{2}-9 \lambda+14=(\lambda-7)(\lambda-2)
$$
  and the eigenvalues are 7 and 2 .​ 

- To find the eigenvector associated with $\lambda=7$

  
$$
  \begin{aligned}
  (\lambda I-A) v=\left[\begin{array}{cc}
  4 & -2 \\
  -2 & 1
  \end{array}\right]\left[\begin{array}{c}
  v_{1} \\
  v_{2}
  \end{array}\right] &=0 \\
  \therefore 4 v_{1}-2 v_{2} &=0
  \end{aligned}
$$
  Any solution to this equation is an eigenvector; say, $v=[1,2]^{T} .$ By the same method, we find that $[-2,1]^{T}$ is an eigenvector corresponding to the eigenvalue $2$ .

- **直接想象一个特征向量与坐标轴重合的例子,就很容易理解.  二次型的特征向量, 就是f(x)图像椭球体的两个轴, 两个特征值分别对应该方向的陡度steepness.**  陡度可以理解为, 该方向走的长度 * 陡度 =上升的高度.这里注意, 特征值对应的是二次, 所以是长度的平方.  即两个方向从极值点走到相同的高度, 那所走路径的平方*特征值是一样的, 即 eAe 相等.     Figure  12, we see that these eigenvectors coincide with the axes of the familiar ellipsoid, and that a larger eigenvalue corresponds to a steeper slope.  (Negative eigenvalues indicate that $f$ decreases along the axis, as in Figures $5(b)$ and $5(d) .$ )

-  **Jacobi Method** 求解例子. Using the constants specified by Equation $4,$ we have
$$
\begin{aligned}
  x_{(i+1)} &=-\left[\begin{array}{cc}
  \frac{1}{3} & 0 \\
  0 & \frac{1}{6}
  \end{array}\right]\left[\begin{array}{cc}
  0 & 2 \\
  2 & 0
  \end{array}\right] x_{(i)}+\left[\begin{array}{cc}
  \frac{1}{3} & 0 \\
  0 & \frac{1}{6}
  \end{array}\right]\left[\begin{array}{c}
  2 \\
  -8
  \end{array}\right] \\
  &=\left[\begin{array}{cc}
  0 & -\frac{2}{3} \\
  -\frac{1}{3} & 0
  \end{array}\right] x_{(i)}+\left[\begin{array}{c}
  \frac{2}{3} \\
  -\frac{4}{3}
  \end{array}\right]
  \end{aligned}
$$
- The **eigenvectors** of $B$ are $[\sqrt{2}, 1]^{T}$ with **eigenvalue** $-\sqrt{2} / 3,$ and $[-\sqrt{2}, 1]^{T}$ with eigenvalue $\sqrt{2} / 3$ .
- 这里B的特征向量与A的不一样. These are graphed in Figure $13(\mathrm{a}) ;$ note that they do not coincide with the eigenvectors of $A,$ and are not related to the axes of the paraboloid. 
- 如图13, error序列不断被矩阵B变换走向收敛.  Figure $13(\mathrm{b})$ shows the **convergence** of the **Jacobi method**. The mysterious path the algorithm takes can be understood by watching the **eigenvector components** of each successive error term (Figures $13(\mathrm{c}),(\mathrm{d}),$ and (e). Figure $13(\mathrm{f})$ plots the eigenvector components as arrowheads. These are converging normally at the rate defined by their eigenvalues, as in Figure 11  可以看出, 两个error的特征分量以B的特征值的速率 走向收敛, 即 $e_0$的两个分量分别*$(-\sqrt{2} / 3,\sqrt{2} / 3)$ , 然后合成$e_1$...
- ![image-20200305004809207](/img/2020-03-01-ConjugateGradient.assets/image-20200305004809207.png)

##### Convergence Analysis of Steepest Descent 收敛分析

![image-20200304025041322](/img/2020-03-01-ConjugateGradient.assets/image-20200304025041322.png)

- 先考虑特例, residual是特征向量, 即起始点在轴上. 一步到位  consider the case where $e_{(i)}$ is an eigenvector with eigenvalue $\lambda_{e} .$ Then, the **residual** $r_{(i)}=-A e_{(i)}=-\lambda_{e} e_{(i)}$ is also an eigenvector.  
  
$$
  \begin{aligned}
  e_{(i+1)} &=e_{(i)}+\frac{r_{(i)}^{T} r_{(i)}}{r_{(i)}^{T} A r_{(i)}} r_{(i)} \\
  &=e_{(i)}+\frac{r_{(i)}^{T} r_{(i)}}{\lambda_{e} r_{(i)}^{T} r_{(i)}}\left(-\lambda_{e} e_{(i)}\right) \\
  &=0
  \end{aligned}
$$

  - 如果某个点在椭球体的一个轴上, 即**error** 以及 **residual** 都在该轴上,就能一步到达最小点. The point $x_{(i)}$ lies on one of the **axes of the ellipsoid**, and so the residual points directly to the center of the ellipsoid. Choosing $\alpha_{(i)}=\lambda_{e}^{-1}$ gives us instant convergence. 
  
- 为了分析一般情况, 将error表示为标准正交特征向量的组合.   For a more general analysis , must express $e_{(i)}$ as a linear combination of eigenvectors, and we shall furthermore require these eigenvectors to be **标准正交 orthonormal**. 

- **若A是对称矩阵, 则有n个正交特征向量.**  It is proven in Appendix C2 that if $A$ is **symmetric**, there exists a set of $n$ **orthogonal eigenvectors** of $A$.    

- **正交特征向量$v$**选单位长度 As we can scale eigenvectors arbitrarily,  choose so that each eigenvector is of unit length. This choice gives us the useful property that   
$$
  v_{j}^{T} v_{k}=\left\{\begin{array}{ll}
  1, & j=k \\
  0, & j \neq k
  \end{array}\right.  \tag{17}
$$

- 现在error表示为正交向量的线性组合,系数为长度. Express the error term as a linear combination of eigenvectors  
$$
  e_{(i)}=\sum_{j=1}^{n} \xi_{j} v_{j} \tag{18}
$$
  where $\xi_{j}$ is the **length of each component** of $e_{(i)}$ .

-  From Equations 17 and 18 we have the follow owing identities:  
  
$$
\begin{aligned}r_{(i)} &=-A e_{(i)}=-\sum_{j} \xi_{j} \lambda_{j} v_{j}   \quad\quad \quad  (19)

  \\\left\|e_{(i)}\right\|^{2}=e^{T}_{(i)}  e_{(i)} &=\sum_{j} \xi_{j}^{2} \quad \quad \quad \quad \quad \quad \quad \quad \quad \quad (20)\\e_{(i)}^{T} A e_{(i)} &=\left(\sum_{j} \xi_{j} v_{j}^{T}\right)\left(\sum_{j} \xi_{j} \lambda_{j} v_{j}\right) \\&=\sum_{j} \xi_{j}^{2} \lambda_{j} \quad \quad \quad \quad \quad \quad \quad \quad \quad (21)
  \\\left\|r_{(i)}\right\|^{2}=r_{(i)}^{T} r_{(i)} &=\sum_{j} \xi_{j}^{2} \lambda_{j}^{2}   \quad \quad \quad \quad \quad \quad \quad \quad \quad (22)
  \\r_{(i)}^{T} A r_{(i)} &=\sum_{j} \xi_{j}^{2} \lambda_{j}^{3} \quad \quad \quad \quad \quad \quad \quad \quad \quad (23)
  \end{aligned}
$$

- Equations 20 and 22 are just **Pythagoras’ Law**.    勾股定理

- residual r 也可以利用Ae的正交分解  Equation 19 shows that $r_{(i)}$ too can be expressed as a sum of eigenvector components, and the length of these components are $-\xi_{j} \lambda_{j}$ .  

- 由 Equation 12 
  
$$
  \begin{aligned}
  e_{(i+1)} &=e_{(i)}+\frac{r_{(i)}^{T} r_{(i)}}{r_{(i)}^{T} A r_{(i)}} r_{(i)} \\
  &=e_{(i)}+\frac{\sum_{j} \xi_{j}^{2} \lambda_{j}^{2}}{\sum_{j} \xi_{j}^{2} \lambda_{j}^{3}} r_{(i)} \quad \quad \quad \quad \quad \quad \quad \quad \quad (24)
  \end{aligned}
$$

  

- 现在考虑另外一个特殊情况, 所有特征值都相同.  Now let's examine the case where $e_{(i)}$ is arbitrary, but all the eigenvectors have a common eigenvalue $\lambda$.  

  - 因为特征值一样, 所以二次型的等高线就是同心圆.  不管起点在哪, residual都是指向圆心.  

  - by choosing $\alpha_{(i)}=\lambda_{e}^{-1}$.  也可以一步到达.  下面证明, 由公式24

  - $$
  \begin{aligned}
  e_{(i+1)} &=e_{(i)}+\frac{\lambda^{2} \sum_{j} \xi_{j}^{2}}{\lambda^{3} \sum_{j} \xi_{j}^{2}}\left(-\lambda e_{(i)}\right) \\
  &=0
  \end{aligned}
  $$
  - ![image-20200304115817647](/img/2020-03-01-ConjugateGradient.assets/image-20200304115817647.png) 

- 但一般情况, A的特征值不相等且非0.  

- 由公式24, 分数部分可以看成是$\lambda_j^{-1}$ 的加权平均. error正交分解后, weights $\xi_{j}^{2}$  确保error 的那些较长的正交分量的优先级.  The weights $\xi_{j}^{2}$ ensure that longer components of $e_{(i)}$ are given precedence.  

- 因此在迭代中, 有些较短的error的正交分量可能会短暂的增加 As a result, on any given iteration, some of the shorter components of $e_{(i)}$ might actually increase in length (though never for long).  

- 粗糙 与 光滑 For this reason, the methods of **Steepest Descent** and **Conjugate Gradients** are called **roughers**. By contrast, the Jacobi Method is a **smoother**, because every eigenvector component is reduced on every iteration.  



##### General Convergence 一般情况的收敛性

- **能量范数** , 显然这个公式与普通的范数比,矢量中间有个矩阵A的变换.    To bound the convergence of Steepest Descent in the general case, we shall define the **energy norm**
  $\Vert e \Vert_{A}=\left(e^{T} A e\right)^{1 / 2}$.  
- <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200305031312203.png" alt="image-20200305031312203" style="zoom:50%;" />
- 图中两个特征向量的能量范数大小一样.   可以用个简单的二次型快速验证.  图中, 特征向量的长度是一次的, 范数不开根号是二次的,特征值是二次项的系数. 
- 这里比欧式距离好用. This norm is easier to work with than the **Euclidean norm**, and is in some sense a more natural norm; examination of Equation 8 shows that **minimizing** $\Vert e_{(i)}\Vert_{A}$ is **equivalent** to  **minimizing** $f\left(x_{(i)}\right)$ .问题等价, f(x) 在某些方向求最小值, 就是error分量eAe的范数的最小值. 
-  
$$
  \begin{aligned}
  \left\|e_{(i+1)}\right\|_{A}^{2} &=e_{(i+1)}^{T} A e_{(i+1)} \\
  &=\left(e_{(i)}^{T}+\alpha_{(i)} r_{(i)}^{T}\right) A\left(e_{(i)}+\alpha_{(i)} r_{(i)}\right) \quad(\text { by Equation } 12) \\
  &\left.=e_{(i)}^{T} A e_{(i)}+2 \alpha_{(i)} r_{(i)}^{T} A e_{(i)}+\alpha_{(i)}^{2} r_{(i)}^{T} A r_{(i)} \quad \text { (by symmetry of } A\right) \\
  &=\left\|e_{(i)}\right\|_{A}^{2}+2 \frac{r_{(i)}^{T} r_{(i)}}{r_{(i)}^{T} A r_{(i)}}\left(-r_{(i)}^{T} r_{(i)}\right)+\left(\frac{r_{(i)}^{T} r_{(i)}}{r_{(i)}^{T} A r_{(i)}}\right)^{2} r_{(i)}^{T} A r_{(i)} \\
  &=\left\|e_{(i)}\right\|_{A}^{2}-\frac{\left(r_{(i)}^{T} r_{(i)}\right)^{2}}{r_{(i)}^{T} A r_{(i)}} \\
  &=\left\|e_{(i)}\right\|_{A}^{2}\left(1-\frac{\left(r_{(i)}^{T} r_{(i)}\right)^{2}}{\left(r_{(i)}^{T} A r_{(i)}\right)\left(e_{(i)}^{T} A e_{(i)}\right)}\right) \\
  & =\left\|e_{(i)}\right\|_{A}^{2}\left(1-\frac{\left(\sum_{j} \xi_{j}^{2} \lambda_{j}^{2}\right)^{2}}{\left(\sum_{j} \xi_{j}^{2} \lambda_{j}^{3}\right)\left(\sum_{j} \xi_{j}^{2} \lambda_{j}\right)}\right) \quad \quad \text { (by Identities } 21,22,23 ) \\
  &=\left\|e_{(i)}\right\|_{A}^{2} \omega^{2}, \quad \omega^{2}=1-\frac{\left(\sum_{j} \xi_{j}^{2} \lambda_{j}^{2}\right)^{2}}{\left(\sum_{j} \xi_{j}^{2} \lambda_{j}^{3}\right)\left(\sum_{j} \xi_{j}^{2} \lambda_{j}\right)}
  \end{aligned}
$$
- 现在找$\omega$上限,影响收敛速度.  The analysis depends on finding an upper bound for $\omega$ . 

- To demonstrate how the <u>weights and eigenvalues affect convergence</u>, I shall derive a result for $n=2$.   Assume that $\lambda_{1} \geq \lambda_{2}$.  

- 谱条件数 The **spectral condition number** of $A$ is defined to be $\kappa=\lambda_{1} / \lambda_{2} \geq 1 .$ 

- error的坡度取决于起始点的选择.  The **slope** of $e_{(i)}$ (relative to the coordinate system defined by the eigenvectors ), which **depends on the starting point**, is denoted $\mu=\xi_{2} / \xi_{1}$ . We have
  
$$
  \begin{aligned}
  \omega^{2} &=1-\frac{\left(\xi_{1}^{2} \lambda_{1}^{2}+\xi_{2}^{2} \lambda_{2}^{2}\right)^{2}}{\left(\xi_{1}^{2} \lambda_{1}+\xi_{2}^{2} \lambda_{2}\right)\left(\xi_{1}^{2} \lambda_{1}^{3}+\xi_{2}^{2} \lambda_{2}^{3}\right)} \\
  &=1-\frac{\left(\kappa^{2}+\mu^{2}\right)^{2}}{\left(\kappa+\mu^{2}\right)\left(\kappa^{3}+\mu^{2}\right)}
  \end{aligned}  \tag{26}
$$

- The value of $\omega$, which determines the rate of convergence of Steepest Descent, is graphed as a function of $\mu$ and $\kappa$ in Figure 17. 
- ![image-20200304172047492](/img/2020-03-01-ConjugateGradient.assets/image-20200304172047492.png)
- The graph confirms my two examples. If $e_{(0)}$ is an eigenvector, then the slope $\mu$ is zero (or infinite);  $\omega$ is zero, so convergence is instant. If the eigenvalues are equal, then the condition number $\kappa$ is one; again,  $\omega$ is zero.
- ![image-20200304175938948](/img/2020-03-01-ConjugateGradient.assets/image-20200304175938948.png)
- Figure 18 illustrates examples from near each of the four corners of Figure 17 
- 由图像可知道 u=k 的时候, 有一条山脊线, 即上限, 收敛最慢   Holding $\kappa$ constant (because $A$ is fixed), a little basic calculus reveals that Equation 26 is maximized when $\mu = \pm \kappa$.  In Figure 17, one can see a faint ridge defined by this line. 
- 直观, 对二次型, 最好的方向是特征向量方向, 最差的是在两个特征向量夹角中间的方向. Figure 19 plots worst-case starting points for our sample matrix $A$. These starting points fall on the lines defined by $\xi_2 / \xi_1 = \pm \kappa$.     即 $\xi_2 / \xi_1 = \lambda_{1} / \lambda_{2}$  直观来看, 对二维的情况, 即搜索方向是拆分为基于特征向量, 然后各方向系数是特性值反过来的话, 最慢. 如果不反过来, 那就更靠近陡的方向,收敛快.  
- <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200305033258116.png" alt="image-20200305033258116" style="zoom:67%;" />
- upper bound for $\omega$ (corresponding to the worst-case starting points) is found by setting $\mu^{2}=\kappa^{2}:$
  
$$
  \begin{aligned}\omega^{2} & \leq 1-\frac{4 \kappa^{4}}{\kappa^{5}+2 \kappa^{4}+\kappa^{3}} \\&=\frac{\kappa^{5}-2 \kappa^{4}+\kappa^{3}}{\kappa^{5}+2 \kappa^{4}+\kappa^{3}} \\&=\frac{(\kappa-1)^{2}}{(\kappa+1)^{2}} \\\omega & \leq \frac{\kappa-1}{\kappa+1}\end{aligned}  \tag{27}
$$

- The more *ill-conditioned* the matrix (that is, the larger its condition number $\kappa$ ), the slower the convergence of Steepest Descent.

- 更一般的. In Section $9.2,$ it is proven that Equation 27 is also valid for $n>2,$ if the condition number of a symmetric, positive-definite matrix is defined to be the ratio of the largest to smallest eigenvalue. 
$$
  \kappa=\lambda_{\max } / \lambda_{\min }
$$

- 收敛.  The convergence results for Steepest Descent are
$$
  \left\|e_{(i)}\right\|_{A} \leq\left(\frac{\kappa-1}{\kappa+1}\right)^{i}\left\|e_{(0)}\right\|_{A}, \text { and }
$$

$$
\begin{aligned}
  \frac{f\left(x_{(i)}\right)-f(x)}{f\left(x_{(0)}\right)-f(x)} & =\frac{\frac{1}{2} e_{(i)}^{T} A e_{(i)}}{\frac{1}{2} \sum_{(0)}^{T} A e_{(0)}} \quad \text { (by Equation } 8 ) \\
  & \leq\left(\frac{\kappa-1}{\kappa+1}\right)^{2 i}
  \end{aligned}
$$



#### The Method of Conjugate Directions  共轭方向

##### Conjugacy 共轭性

- Steepest Descent 算法经常在重复的方向上面走, 不能在某个方向一步到位. **Steepest Descent** often finds itself taking steps in the same directionas earlier steps (see Figure 8 ). 

- 希望能拆成n个正交的方向, 然后依次在各个正交方向上走到位,一个方向只走一次. Wouldn't it be better if, every time we took a step, we <u>got it right the first time</u>? Here's an **idea**: let's pick a set of **orthogonal** *search directions* $d_{(0)}, d_{(1)}, \ldots, d_{(n-1)} .$ In each search direction, we'll take <u>exactly one step</u>, and that step will be just the right length to line up evenly with $x$. After $n$ steps, we'll be done. 

- <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200305040044731.png" alt="image-20200305040044731" style="zoom:67%;" />

- 图示拆成正交的, 每个方向一步到位.   Figure 21 illustrates this idea, using the coordinate axes as **search directions**. The first (horizontal) step leads to the correct $x_{1}$ -coordinate; the second (vertical) step will hit home. Notice that $e_{(1)}$ is **orthogonal** to $d_{(0)}$.

- 问题就分两步: 方向怎么找, 步长怎么定.

-  In general, for each step we choose a point
$$
  x_{(i+1)}=x_{(i)}+\alpha_{(i)} d_{(i)}
$$

- 怎么求步长,利用$e_{(i+1)}$ 与$d_{(i)}$的正交性,  即$e_{(i+1)}$绝对不该含有$d_{(i)}$的分量.   To find the value of $\alpha_{(i)}$, use the fact that $e_{(i+1)}$ should be orthogonal to $d_{(i)}$,  so that we <u>need never step in the direction of $d_{(i)}$ again.</u> Using this condition, we have
  
$$
  \begin{aligned}
  d_{(i)}^{T} e_{(i+1)} &=0 \\
  d_{(i)}^{T}\left(e_{(i)}+\alpha_{(i)} d_{(i)}\right) &=0 \quad \text {(by Equation 29 } ) \\
  \alpha_{(i)} &=-\frac{d_{(i)}^{T} e_{(i)}}{d_{(i)}^{T} d_{(i)}}
  \end{aligned} \tag{30}
$$

- 上式中需要error, 但如果知道的话, 就知道了目标点, 就不用迭代来解决了. Unfortunately, we haven't accomplished anything, because we can't compute $\alpha_{(i)}$ without knowing $e_{(i)}$ and if we knew $e_{(i)}$, the problem would already be solved.

- 怎么定分解方向, 先方案再讲为什么.
  
-  **A-orthogonal** 对A共轭, **共轭** :  注意, 下面公式里两个搜索方向向量下标不一样, 是两个不同的向量. 矢量1与 (A矩阵变换过的矢量2)正交.  The solution is to make the search directions $A$ -orthogonal instead of orthogonal. Two vectors $d_{(i)}$ and $d_{(j)}$ are **A-orthogonal**, or **conjugate**, if
$$
  d_{(i)}^{T} A d_{(j)}=0
$$

- <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200305043759155.png" alt="image-20200305043759155" style="zoom:67%;" />

- 图22显示了A正交的向量, 可以理解为正交的拉伸.  但这个从上面公式看不出, 因为A是值作用于其中一个向量的.   Figure 22( a ) shows what  $A$-orthogonal vectors look like. Imagine if this article were printed on bubble gum, and you grabbed Figure 22( a ) by the ends and stretched it until the **ellipses** appeared **circular**.拉伸, 使得椭圆变成正圆, 相关向量也变成正交了. The vectors would then appear orthogonal, as in Figure $22(b)$  

- Our new requirement is that $e_{(i+1)}$ be **$A$-orthogonal** to $d_{(i)}$ (see Figure 23(a)). 

- 下面要寻找该A-正交分解方向上的最小值.  Not coincidentally, this orthogonality condition is equivalent to finding the minimum point along the search direction $d_{(i)}$, as in Steepest Descent. 

- To see this, set the directional derivative to zero:   
  
$$
  \begin{aligned}
  \frac{d}{d \alpha} f\left(x_{(i+1)}\right) &=0 \\
  f^{\prime}\left(x_{(i+1)}\right)^{T} \frac{d}{d \alpha} x_{(i+1)} &=0 \\
  -r_{(i+1)}^{T} d_{(i)} &=0 \\
  d_{(i)}^{T} A e_{(i+1)} &=0
  \end{aligned}
$$
- Following the derivation of Equation 30, 上下同乘以A , here is the expression for $\alpha_{(i)}$ when the search directions are $A$ -orthogonal:
  
$$
  \begin{aligned}\alpha_{(i)} &=-\frac{d_{(i)}^{T} A e_{(i)}}{d_{(i)}^{T} A d_{(i)}} \\&=\frac{d_{(i)}^{T} r(i)}{d_{(i)}^{T} A d_{(i)}}\end{aligned} \tag{32}
$$
- 上式, 如果serach方向选为梯度方向, 则与Steepest Descent算法一样. 注意, 梯度方向并不是**$A$-orthogonal** . Unlike Equation 30 , we can calculate this expression. Note that if the **search vector** were the residual, this formula would be identical to the formula used by Steepest Descent.

- ![image-20200305045136410](/img/2020-03-01-ConjugateGradient.assets/image-20200305045136410.png)

- 下面证明的确是n步内结束. 如果 error 可以表达为n个A正交的分量的线性组合. To prove that this procedure really does compute $x$ in $n$ steps, express the error term as a linear combination of search directions; namely,
  
$$
  e_{(0)}=\sum_{j=0}^{n-1} \delta_{j} d_{(j)} \tag{33}
$$
- 系数的处理. The values of $\delta_{j}$ can be found by a mathematical trick. Because the search directions are $A$ -orthogonal, it is possible to eliminate all the $\delta_{j}$ values but one from Expression 33 by premultiplying the expression by $d_{(k)}^{T} A:$
  
$$
\begin{aligned}d_{(k)}^{T} A e_{(0)} &=\sum_{j} \delta_{(j)} d_{(k)}^{T} A d_{(j)} \\d_{(k)}^{T} A e_{(0)} & =\delta_{(k)} d_{(k)}^{T} A d_{(k)} \quad \text { (by A-orthogonality of } d \text { vectors } ) \\\delta_{(k)} &=\frac{d_{(k)}^{T} A e_{(0)}}{d_{(k)}^{T} A d_{(k)}} \\& =\frac{d_{(k)}^{T} A\left(e_{(0)}+\sum_{i=0}^{k-1} \alpha_{(i)} d_{(i)}\right)}{d_{(k)}^{T} A d_{(k)}} \quad \text { (分子+号右边部分乘以括号外的系数为0 , by } A \text { -orthogonality of } d \text { vectors } ) \\& =\frac{d_{(k)}^{T} A e_{(k)}}{d_{(k)}^{T} A d_{(k)}} \quad \text { (By Equation 29) }    \quad\quad\quad\quad {(34)}
  \end{aligned}
$$
- x的序列由$\alpha_{(i)} d_{(i)}$来累加生成, 正好是$\delta_{(j)} d_{(j)}$的反过程, 知道就行.   By Equations 31 and $34,$ we find that $\alpha_{(i)}=-\delta_{(i)} .$ This fact gives us a new way to look at the error term. As the following equation shows, the process of building up $x$ component by component can also be viewed as a process of cutting down the error term component by component (see Figure $23(\mathrm{b})$ ).
  
$$
\begin{aligned}
  e_{(i)} &=e_{(0)}+\sum_{j=0}^{i-1} \alpha_{(j)} d_{(j)} \\
&=\sum_{j=0}^{n-1} \delta_{(j)} d_{(j)}-\sum_{j=0}^{i-1} \delta_{(j)} d_{(j)} \\
  &=\sum_{j=i}^{n-1} \delta_{(j)} d_{(j)}
  \end{aligned} \tag{35}
$$
- After $n$ iterations, every component is cut away, and $e_{(n)}=0 ;$ the proof is complete.



#####  Gram-Schmidt Conjugation 施密特正交化

- 下面需要的就是找到  $A$ -orthogonal search directions $$\left\{d_{(i)}\right\}$$ 
- 一个简单的生成序列的方法,  a simple way to generate them, called a  **conjugate Gram-Schmidt process**
- 先找一组线性无关的向量, 坐标轴也行. Suppose we have a set of $n$ linearly independent vectors $u_{0}, u_{1}, \ldots, u_{n-1} .$ The coordinate axes will do in a pinch, although more intelligent choices are possible. 
- ![image-20200306184634738](/img/2020-03-01-ConjugateGradient.assets/image-20200306184634738.png)
- 先定一个起始方向, 然后, 后面的都减去前面的该方向上的$A$-orthogonal分量, 一直迭代完所有维度.   To construct $d_{(i)}$, take $u_{i}$ and subtract out any components that are not $A$-orthogonal to the previous $d$ vectors (see Figure 24). In other words, set
$$
\begin{array}{l}
d_{(0)}=u_{0}, \text { and for } i>0, \text { set } \\
\qquad d_{(i)}=u_{i}+\sum_{k=0}^{i-1} \beta_{i k} d_{(k)}
\\ \text{  where the }\beta_{i k} \text{ are defined for }i>k 
\end{array}
$$
- To find their values, use the same trick used to find $\delta_{j}$ :
$$
\begin{aligned}
d_{(i)}^{T} A d_{(j)} &=u_{i}^{T} A d_{(j)}+\sum_{k=0}^{i-1} \beta_{i k} d_{(k)}^{T} A d_{(j)} \\
0 &=u_{i}^{T} A d_{(j)}+\beta_{i j} d_{(j)}^{T} A d_{(j)}, \quad i>j  \quad (\text{by A-orthogonality of d vectors})\\
\beta_{i j} &=-\frac{u_{i}^{T} A d_{(j)}}{d_{(j)}^{T} A d_{(j)}} 
\end{aligned}
$$
- 复杂度 The difficulty with using **Gram-Schmidt conjugation** in the method of **Conjugate Directions** is that all the old search vectors must be kept in memory to construct each new one, and furthermore $\mathcal{O}\left(n^{3}\right)$ operations are required to generate the full set.
- <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200307061126266.png" alt="image-20200307061126266" style="zoom: 33%;" />
- 高斯消去法, 如果由坐标轴建立起共轭方向序列, 类似于高斯消元  In fact, if the search vectors are constructed by conjugation of the axial unit vectors, Conjugate Directions becomes equivalent to performing **Gaussian elimination** (see Figure 25).
- 共轭方向是正交的一种拉伸. An important key to understanding the method of **Conjugate Directions** (and also CG) is to notice that Figure 25 is just a **stretched** copy of Figure 21! 
- 沿着共轭方向, 相当于拉伸后的正交方向. Remember that when one is performing the method of Conjugate Directions (including CG), one is simultaneously performing the method of Orthogonal Directions in a **stretched (scaled)** space.



##### Optimality of the Error Term 分析error

- Conjugate Directions has an interesting property: it finds at every step the **best solution within the bounds of where it's been allowed to explore**. 

- 搜索方向构成的线性子空间, 则error就在这里面 Where has it been allowed to explore? Let $$\mathcal{D}_{i}$$ be the $i$-dimensional subspace span $$\left\{d_{(0)}, d_{(1)}, \dots, d_{(i-1)}\right\}$$ ;  the value $e_{(i)}$ is chosen from $$e_{(0)}+\mathcal{D}_{i}$$ 

- 另外一个推导思路:  What do I mean by "best solution"? I mean that Conjugate Directions chooses the value from $$e_{(0)}+\mathcal{D}_{i}$$ that minimizes $$\left\|e_{(i)}\right\|_{A}$$ (see Figure 26 ). 

- In fact, some authors derive CG by trying to minimize $$\left\|e_{(i)}\right\|_{A}$$ within $$e_{(0)}+\mathcal{D}_{i}$$.

- <img src="/img/2020-03-01-ConjugateGradient.assets/image-20200307070616512.png" alt="image-20200307070616512" style="zoom: 33%;" />

- 用能量范数分析 In the same way that the error term can be expressed as a linear combination of search directions(Equation 35), its **energy norm** can be expressed as a **summation**.

  






https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf