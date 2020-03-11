---
layout:     post
title:      Reinforcement Learning and Optimal Control
subtitle:   Note on "Reinforcement Learning and Optimal Control" (2019)
date:       2020-01-28 12:00:00
author:     "tengshiquan"
header-img: "img/post-bg-sunset.jpg"
catalog: true
tags:
    - AI
    - Reinforcement Learning
    - Dynamic Programming
    - Optimal Control

---

# Reinforcement Learning and Optimal Control



## Exact Dynamic Programming  精确动态规划

reinforcement learning,   approximate dynamic programming = neuro-dynamic programming

### DETERMINISTIC DYNAMIC PROGRAMMING  确定性动态规划

#### Deterministic Problems 确定性问题

- discrete-time dynamic system : 状态转移以及cost都是确定的,  系统的状态转移函数

$$
x_{k+1} = f_k(x_k, u_k), \quad  k = 0, 1,\dots,N-1,
$$

​	 The set of all possible $x_k$ is called the state space at time k , $x_k$ 以及 $u_k$ 都是集合, 取决于k;  generality 是DP methodology的强项. 

- total cost of a control sequence $\lbrace u_{0}, \ldots, u_{N-1} \rbrace$:

$$
J\left(x_{0} ; u_{0}, \ldots, u_{N-1}\right)=g_{N}\left(x_{N}\right)+\sum_{k=0}^{N-1} g_{k}\left(x_{k}, u_{k}\right)
$$
​		where $g_{N}\left(x_{N}\right)$ is a terminal cost incurred at the end of the process.  

- optimal value:

$$
J^{*}\left(x_{0}\right)=\min _{u_{k} \in U_{k}\left(x_{k}\right) \atop k=0, \ldots, N-1} J\left(x_{0} ; u_{0}, \ldots, u_{N-1}\right)
$$

<img src="/img/2020-01-28-OptControl.assets/image-20200221000851697.png" alt="image-20200221000851697" style="zoom:67%;" />

 

##### Discrete Optimal Control Problems 离散最优控制问题

<img src="/img/2020-01-28-OptControl.assets/E49CEA9C-4932-4277-BA29-F83EF42298B7.png" alt="E49CEA9C-4932-4277-BA29-F83EF42298B7" style="zoom: 67%;" />

Figure 1.1.2 Transition graph for a deterministic finite-state system.  将每个路径的长度取cost的大小, 则**确定性有限问题的最优代价问题等价于该图的最短路径问题**.  最后的节点t是虚拟的,表示结束.

图中节点的stage是基于k的, 每个路径的节点数不一定一样长, 可以补为一样长, 然后后补的路径的长度为0. 每个stage的那一列的node都是不同的state.   TSP问题可以用上图来表示.

##### Continuous-Spaces Optimal Control Problems 连续空间最优化控制问题

state  belongs to a Euclidean space .  

例子:   A Linear-Quadratic Problem , system equation is linear, the cost function is quadratic. 
Linear-quadratic problems with no constraints on the state or the control admit a nice analytical solution. 有解析解. Linear-quadratic problems  with  linear constraints on the state and/or the control,  be solvable not only by DP but also by quadratic programming methods. 除了DP, 还可以用二次规划方式求解.

Generally deterministic optimal control problems with continuous state and control spaces (in addition to DP) admit a solution by nonlinear programming methods, such as gradient, conjugate gradient, and Newton’s method, which can be suitably adapted to their special structure.  采用非线性规划方法（如梯度法、共轭梯度法和牛顿法）求解，这些方法可以适应其特殊的结构。



#### The Dynamic Programming Algorithm

最优原则: 一个序列是最优的,则从某个时间点的往后的尾部子问题也必然是最优的;   DP的核心思想.  如果求的是近似解, 则这里也可以略放松, 就是某个时间之后走的是很好的解, 则整体也会是好的解.

**principle of optimality**:   the tail of an optimal sequence is optimal for the tail subproblem

Let $\left\{u_{0}^{*}, \ldots, u_{N-1}^{*}\right\}$ be an **optimal control sequence**, which together with $x_{0}$ determines the corresponding state sequence $\left\{x_{1}^{*}, \ldots, x_{N}^{*}\right\}$ via the system equation.  Consider the subproblem whereby we start at $x_{k}^{*}$ at time $k$ and wish to minimize the "**cost-to-go**" from time $k$ to time $N$
$$
g_{k}\left(x_{k}^{*}, u_{k}\right)+\sum_{m=k+1}^{N-1} g_{m}\left(x_{m}, u_{m}\right)+g_{N}\left(x_{N}\right)
$$
over $\left\{u_{k}, \ldots, u_{N-1}\right\}$ with $u_{m} \in U_{m}\left(x_{m}\right), m=k, \ldots, N-1 .$ Then the truncated optimal control sequence $\left\{u_{k}^{*}, \ldots, u_{N-1}^{*}\right\}$ is optimal for this subproblem.



DP algorithm DP算法思想: it proceeds sequentially, by solving all the tail subproblems of a given time length, using the solution of the tail subproblems of shorter time length.   基于最优尾部子问题. 



##### Finding an Optimal Control Sequence by DP

从最后的时间点逆向求解最优, 所以下面的loop,在程序实现时最好是倒序











#### AN OVERVIEW OF INFINITE HORIZON PROBLEMS

$$
J_{\pi}\left(x_{0}\right)=\lim _{N \rightarrow \infty} \underset{k=0,1, \ldots}{E}\left\{\sum_{k=0}^{N-1} \alpha^{k} g\left(x_{k}, \mu_{k}\left(x_{k}\right), w_{k}\right)\right\}
$$

两类:

1. **Stochastic shortest path problems** (**SSP** for short). Here, $\alpha=1$ but there is a special cost-free termination state;  长度是随机的, 可能有无限长, 但有一个终结状态, 只要能走到该状态就能结束序列.

2. Discounted problems.   $\alpha < 1$ . 

   discounted问题可以转换为SSP问题.









### Appendix: Mathematical Analysis

We will make heavy use of the DP operators $T$ and $T_{\mu},$ particularly for the discounted problem:
$$
\begin{align}{}
{(T J)(i)=} & {\min _{u \in U(i)} \sum_{j=1}^{n} p_{i j}(u) \Big(g(i, u, j)+\alpha J(j) \Big), \quad i=1, \ldots, n} \\
{(T_\mu J)(i)=}  & {\sum_{i=1}^{n} p_{i j} \big (\mu(i)\big) \bigg (g \Big (i, \mu(i), j \Big )+\alpha J(j) \bigg ), \quad i=1, \ldots, n}
\end{align}
$$
A key property is the **monotonicity** of these operators, i.e.,  **单调性**
$$
T J \geq T J^{\prime}, \quad T_{\mu} J \geq T_{\mu} J^{\prime}, \quad \text{for all } J \text{ and } J^{\prime} \text{ with }J \geq J^{\prime}
$$

**常数T后discount** Also for the discounted problem, we have the "**constant shift**" property, which states that if the functions $J$ is increased uniformly by a constant $c$ then the functions $T J$ and $T_{\mu} J$ are also increased uniformly by the constant $\alpha c$



#### 4.13.1 Proofs for Stochastic Shortest Path Problems

**Proposition 4.2.1**: (**Convergence of VI**) Given any initial conditions $J_{0}(1), \ldots, J_{0}(n),$ the sequence $\left\{J_{k}(i)\right\}$ generated by the VI algorithm
$$
J_{k+1}(i)=\min _{u \in U(i)}\left[p_{i t}(u) g(i, u, t)+\sum_{j=1}^{n} p_{i j}(u)\left(g(i, u, j)+J_{k}(j)\right)\right]
$$
converges to the optimal cost $J^{*}(i)$ for each $i=1, \ldots, n$



**Proposition 4.2.2**: (**Bellman's Equation**) The optimal cost function $J^{*}=\left(J^{*}(1), \ldots, J^{*}(n)\right)$ satisfies for all $i=1, \ldots, n,$ the equation
$$
J^{*}(i)=\min _{u \in U(i)}\left[p_{i t}(u) g(i, u, t)+\sum_{j=1}^{n} p_{i j}(u)\left(g(i, u, j)+J^{*}(j)\right)\right]
$$
and in fact it is the unique solution of this equation.



**Proposition 4.2.3**: (**VI and Bellman's Equation for Policies**) For any stationary policy $\mu,$ the corresponding cost function $J_{\mu}=$ $\left(J_{\mu}(1), \ldots, J_{\mu}(n)\right)$ satisfies for all $i=1, \ldots, n$ the equation
$$
J_{\mu}(i)=p_{i t}(\mu(i)) g(i, \mu(i), t)+\sum_{j=1}^{n} p_{i j}(\mu(i))\left(g(i, \mu(i), j)+J_{\mu}(j)\right)
$$

and is in fact the unique solution of this equation. Furthermore, given any initial conditions $J_{0}(1), \ldots, J_{0}(n),$ the sequence $\left\{J_{k}(i)\right\}$ generated by the VI algorithm that is specific to $\mu$
$$
J_{k+1}(i)=p_{i t}(\mu(i)) g(i, \mu(i), t)+\sum_{j=1}^{n} p_{i j}(\mu(i))\left(g(i, \mu(i), j)+J_{k}(j)\right)
$$
converges to the cost $J_{\mu}(i)$ for each $i$

**Proposition 4.2.4**: (**Optimality Condition**) A stationary policy $μ$ is optimal if and only if for every state $i, μ(i)$ attains the minimum in Bellman’s equation (4.7).



证明是压缩映射, 对cost , 走了一轮DP, 会选其中cost小的action, 总cost会变小, 所以是个压缩映射.

**Proposition 4.2.5**: (**Contraction Property of the DP Operator** ) The DP operators $T$ and $T_{\mu}$ of Eqs. (4.8) and (4.9) are contraction mappings with respect to some weighted norm
$$
\|J\|=\max _{i=1, \ldots, n} \frac{|J(i)|}{v(i)}
$$
defined by some vector $v=(v(1), \ldots, v(n))$ with positive components. 这里v不是状态的总回报,是用于加权的, 下面的公式里面似乎去掉也可以.  In other words, there exist positive scalar $\rho<1$ and $\rho_{\mu}<1$ such that for any two $n$-dimensional vectors $J$ and $J^{\prime},$ we have
$$
\left\|T J-T J^{\prime}\right\| \leq \rho\left\|J-J^{\prime}\right\|, \quad\left\|T_{\mu} J-T_{\mu} J^{\prime}\right\| \leq \rho_{\mu}\left\|J-J^{\prime}\right\|
$$
**Proof**: We first define the vector $v$ using the problem of Example 4.2.1. In particular, we let $v(i)$ be the maximal expected number of steps to termination starting from state $i .$ 这里v(i)是4.2.1.这个特殊问题的最大走的步数的预期. From Bellman's equation in Example 4.2.1, we have for all $i=1, \ldots, n,$ and stationary policies $\mu$
$$
v(i)=1+\max _{u \in U(i)} \sum_{j=1}^{n} p_{i j}(u) v(j) \geq 1+\sum_{j=1}^{n} p_{i j}(\mu(i)) v(j), \quad i=1, \ldots, n
$$
Thus we obtain for all $\mu$


$$
\sum_{j=1}^{n} p_{i j}(\mu(i)) v(j) \leq v(i)-1 \leq \rho v(i) \quad i=1, \dots, n
$$
where $\rho$ is defined by
$$
\rho=\max _{i=1, \ldots, n} \frac{v(i)-1}{v(i)}
$$
since $v(i) \geq 1$ for all $i,$ we have $\rho<1$ We will now show that Eq. ( 4.97) implies the desired contraction property. Indeed, consider the operator $T_{\mu},$ which when applied to a vector $J=(J(1), \ldots, J(n))$ produces the vector $T_{\mu} J=\left(\left(T_{\mu} J\right)(1), \ldots,\left(T_{\mu} J\right)(n)\right)$
defined by
$$
\left(T_{\mu} J\right)(i)=p_{i t}(\mu(i)) g(i, \mu(i), t)+\sum_{j=1}^{n} p_{i j}(\mu(i))(g(i, \mu(i), j)+J(j))
$$

for all $i=1, \ldots, n .$ We have for all $J, J^{\prime},$ and $i$
$$
\begin{aligned}
\left(T_{\mu} J\right)(i) &=\left(T_{\mu} J^{\prime}\right)(i)+\sum_{j=1}^{n} p_{i j}(\mu(i))\left(J(j)-J^{\prime}(j)\right) \\
&=\left(T_{\mu} J^{\prime}\right)(i)+\sum_{j=1}^{n} p_{i j}(\mu(i)) v(j) \frac{\left(J(j)-J^{\prime}(j)\right)}{v(j)} \\
& \leq\left(T_{\mu} J^{\prime}\right)(i)+\sum_{j=1}^{n} p_{i j}(\mu(i)) v(j)\left\|J-J^{\prime}\right\| \\
& \leq\left(T_{\mu} J^{\prime}\right)(i)+\rho v(i)\left\|J-J^{\prime}\right\|
\end{aligned}
$$
where the last inequality follows from Eq. $(4.97) .$ By minimizing both sides over all $\mu(i) \in U(i),$ we obtain $(T J)(i) \leq\left(T J^{\prime}\right)(i)+\rho v(i)\left\|J-J^{\prime}\right\|, \quad i=1, \ldots, n$
Thus we have
$$
\frac{(T J)(i)-\left(T J^{\prime}\right)(i)}{v(i)} \leq \rho\left\|J-J^{\prime}\right\|, \quad i=1, \ldots, n
$$
Similarly, by reversing the roles of $J$ and $J^{\prime},$ we obtain
$$
\frac{\left(T J^{\prime}\right)(i)-(T J)(i)}{v(i)} \leq \rho\left\|J-J^{\prime}\right\|, \quad i=1, \ldots, n
$$
By combining the preceding two inequalities, we have
$$
\frac{\left|(T J)(i)-\left(T J^{\prime}\right)(i)\right|}{v(i)} \leq \rho\left\|J-J^{\prime}\right\|, \quad i=1, \ldots, n
$$
and by maximizing the left-hand side over $i,$ the contraction property $\| T J- T J^{\prime}\|\leq \rho\| J-J^{\prime} \|$ follows. $\quad \mathbf{Q} . \mathbf{E . D .}$



#### 4.13.2 Proofs for Discounted Problems



#### 4.13.3 Convergence of Exact and Optimistic Policy Iteration

**Proposition 4.5.1**: (**Convergence of Exact PI**) For both the SSP and the discounted problems, the exact PI algorithm generates an improving sequence of policies [ i.e., $J_{\mu^{k+1}}(i) \leq J_{\mu^{k}}(i)$ for all i and k] and terminates with an optimal policy.

Proof: For any $k,$ consider the sequence generated by the VI algorithm for policy $\mu^{k+1}:$
$$
J_{N+1}(i)=\sum_{j=1}^{n} p_{i j}\left(\mu^{k+1}(i)\right)\left(g\left(i, \mu^{k+1}(i), j\right)+\alpha J_{N}(j)\right), \quad i=1, \ldots, n
$$
where $N=0,1, \ldots,$ and
$$
J_{0}(i)=J_{\mu^{k}}(i), \quad i=1, \ldots, n
$$
From Eqs. ( 4.34) and $(4.33),$ we have
$$
\begin{aligned}
J_{0}(i) &=\sum_{j=1}^{n} p_{i j}\left(\mu^{k}(i)\right)\left(g\left(i, \mu^{k}(i), j\right)+\alpha J_{0}(j)\right) \\
& \geq \sum_{j=1}^{n} p_{i j}\left(\mu^{k+1}(i)\right)\left(g\left(i, \mu^{k+1}(i), j\right)+\alpha J_{0}(j)\right) \\
&=J_{1}(i)
\end{aligned}
$$
for all i. By using the above inequality we obtain

$$
\begin{aligned} J_{1}(i) &=\sum_{j=1}^{n} p_{i j}\left(\mu^{k+1}(i)\right)\left(g\left(i, \mu^{k+1}(i), j\right)+\alpha J_{0}(j)\right) \\ & \geq \sum_{j=1}^{n} p_{i j}\left(\mu^{k+1}(i)\right)\left(g\left(i, \mu^{k+1}(i), j\right)+\alpha J_{1}(j)\right) \\ &=J_{2}(i) \end{aligned}
$$

for all $i,$ and by continuing similarly we have
$$
J_{0}(i) \geq J_{1}(i) \geq \cdots \geq J_{N}(i) \geq J_{N+1}(i) \geq \cdots, \quad i=1 \ldots, n
$$

since by Prop. $4.3 .3, J_{N}(i) \rightarrow J_{\mu^{k+1}}(i),$ we obtain $J_{0}(i) \geq J_{\mu^{k+1}}(i)$ or
$$
J_{\mu^{k}}(i) \geq J_{\mu^{k+1}}(i), \quad i=1, \dots, n, \quad k=0,1, \dots
$$
Thus the sequence of generated policies is improving, and since the number of stationary policies is finite, we must after a finite number of iterations, say $k+1,$ obtain $J_{\mu^{k}}(i)=J_{\mu^{k+1}}(i)$ for all $i .$ Then we will have equality throughout in Eq. $(4.98),$ which means that
$$
J_{\mu^{k}}(i)=\min _{u \in U(i)} \sum_{j=1}^{n} p_{i j}(u)\left(g(i, u, j)+\alpha J_{\mu^{k}}(j)\right), \quad i=1, \ldots, n
$$
Thus the costs $J_{\mu^{k}}(1), \ldots, J_{\mu^{k}}(n)$ solve Bellman's equation, and by Prop.
$4.3 .2,$ it follows that $J_{\mu^{k}}(i)=J^{*}(i)$ and that $\mu^{k}$ is optimal. $\quad \mathbf{Q . E . D .}$

**Proposition 4.5.2**: (**Convergence of Optimistic PI**) For the discounted problem, the sequences $\left\{J_{k}\right\}$ and $\left\{\mu^{k}\right\}$ generated by the optimistic PI algorithm satisfy
$$
J_{k} \rightarrow J^{*}, \quad J_{\mu^{k}} \rightarrow J^{*}
$$
**Proof**: First we choose a scalar $r$ such that the vector $\bar{J}_{0}$ defined by $\bar{J}_{0}=$ $J_{0}+r e,$ satisfies $T \bar{J}_{0} \leq \bar{J}_{0}$ [here and later, $e$ is the unit vector, i.e., $e(i)=1$ for all $i] .$ This can be done since if $r$ is such that $T J_{0}-J_{0} \leq(1-\alpha) r e$ we have
$$
T \bar{J}_{0}=T J_{0}+\alpha r e \leq J_{0}+r e=\bar{J}_{0}
$$
where $e=(1,1, \ldots, 1)^{\prime}$ is the unit vector. 

With $\bar{J}_{0}$ so chosen, define for all $k, \bar{J}_{k+1}=T_{\mu^{k}}^{m_{k}} \bar{J}_{k}$.  Then since we have
$$
T(J+r e)=T J+\alpha r e, \quad T_{\mu}(J+r e)=T_{\mu}+\alpha r e
$$
for any $J$ and $\mu,$ it can be seen by induction that for all $k$ and $m= 0,1, \ldots, m_{k}$,  the vectors $J_{k+1}=T_{\mu^{k}}^{m} J_{k}$ and $\bar{J}_{k+1}=T_{\mu^{k}}^{m} \bar{J}_{k}$ differ by a multiple of the unit vector, namely
$$
r \alpha ^{m_0+\dots+m_{k-1}+m}e
$$
It follows that if $J_{0}$ is replaced by $\bar{J}_{0}$ as the starting vector in the algorithm, the same sequence of policies $\left\{\mu^{k}\right\}$ will be obtained; i.e., for all $k,$ we have $T_{\mu^{k}} \bar{J}_{k} = T  \bar{J}_{k}$. Moreover,we have $\lim_{k \to \infty}(\bar J_k - J_k) = 0$ .

Next we will show that $J^{*} \leq \bar{J}_{k} \leq T^{k} \bar{J}_{0}$ for all $k,$ from which convergence will follow. Indeed, we have $T_{\mu^{0}} \bar{J}_{0}=T \bar{J}_{0} \leq \bar{J}_{0},$ from which we obtain
$$
T_{\mu^{0}}^{m} \bar{J}_{0} \leq T_{\mu^{0}}^{m-1} \bar{J}_{0}, \quad m=1,2, \ldots
$$
so that
$$
T_{\mu^{1}} \bar{J}_{1}=T \bar{J}_{1} \leq T_{\mu^{0}} \bar{J}_{1}=T_{\mu^{0}}^{m_{0}+1} \bar{J}_{0} \leq T_{\mu^{0}}^{m_{0}} \bar{J}_{0}=\bar{J}_{1} \leq T_{\mu^{0}} \bar{J}_{0}=T \bar{J}_{0}
$$
This argument can be continued to show that for all $k,$ we have $\bar{J}_{k} \leq T \bar{J}_{k-1}$ so that
$$
\bar{J}_{k} \leq T^{k} \bar{J}_{0}, \quad k=0,1, \ldots
$$
On the other hand, since $T \bar{J}_{0} \leq \bar{J}_{0},$ we have $J^{*} \leq \bar{J}_{0},$ and it follows that successive application of any number of operators of the form $T_{\mu}$ to $\bar{J}_{0}$ produces functions that are bounded from below by $J^{*} .$ Thus,
$$
J^{*} \leq \bar{J}_{k} \leq T^{k} \bar{J}_{0}, \quad k=0,1, \ldots
$$
By taking the limit as $k \rightarrow \infty,$ we obtain $\lim _{k \rightarrow \infty} \bar{J}_{k}(i)=J^{*}(i)$ for all $i$
and since $\lim _{k \rightarrow \infty}\left(\bar{J}_{k}-J_{k}\right)=0,$ we obtain
$$
\lim _{k \rightarrow \infty} J_{k}(i)=J^{*}(i), \quad i=1, \ldots, n
$$
Finally, from the finiteness of the state and control spaces, it follows that there exists $\epsilon>0$ such that if $\max _{i}\left|J(i)-J^{*}(i)\right| \leq \epsilon$ and $T_{\mu} J=T J$
so that $\mu$ is optimal. since $J_{k} \rightarrow J^{*},$ this shows that $\mu^{k}$ is optimal for all sufficiently large $k$ . $\mathbf{Q . E . D .}$



#### 4.13.4 Performance Bounds for One-Step Lookahead, Rollout, and Approximate Policy Iteration

We first prove the basic performance bounds for $\ell$-step lookahead schemes and discounted problems.

**Proposition 4.6.1**: (**Limited Lookahead Performance Bounds**) 

(a) Let $\tilde{\mu}$ be the $\ell$ -step lookahead policy corresponding to $\tilde{J}$. Then
$$
\left\|J_{\tilde{\mu}}-J^{*}\right\| \leq \frac{2 \alpha^{\ell}}{1-\alpha}\left\|\tilde{J}-J^{*}\right\|
$$
where $\|\cdot\|$ denotes the maximum norm ( 4.15)

(b) Let $\tilde{\mu}$ be the one-step lookahead policy obtained by minimization in the equation
$$
\begin{aligned}
  \hat{J}(i)=& \min _{u \in U(i)} \sum_{j=1}^{n} p_{i j}(u)(g(i, u, j)+\alpha \tilde{J}(j)), \quad i=1, \ldots, n \\
 
 &\text { where } \bar{U}(i) \subset U(i) \text { for all } i=1, \ldots, n .\\   \text { Assume} & \text{ that for some } 
\text { constant } c, \text { we have } \\
 &  \hat{J}(i) \leq \tilde{J}(i)+c, \\
  \text { Then }& \\
&  J_{\tilde{\mu}}(i) \leq \tilde{J}(i)+\frac{c}{1-\alpha},  \quad i=1, \ldots, n
\end{aligned}
$$
Proof: (a) In the course of the proof, we will use the contraction property of $\left.T \text { and } T_{\mu} \text { (cf. Prop. } 4.3 .5\right) .$ Using the triangle inequality, we write for every $k$
$$
\left\|T_{i}^{k} J^{*}-J^{*}\right\| \leq \sum_{m=1}^{k}\left\|T_{i j}^{m} J^{*}-T_{i j}^{m-1} J^{*}\right\| \leq \sum_{m=1}^{k} \alpha^{m-1}\left\|T_{\tilde{\mu}} J^{*}-J^{*}\right\|
$$
By taking the limit as $k \rightarrow \infty$ and using the fact $T_{\mu}^{k} J^{*} \rightarrow J_{\tilde{\mu}},$ we obtain
$$
\left\|J_{i k}-J^{*}\right\| \leq \frac{1}{1-\alpha}\left\|T_{\tilde{\mu}} J^{*}-J^{*}\right\|
$$
Denote $\hat{J}=T^{\ell-1} \tilde{J} .$ The rightmost expression of Eq. ( 4.103) is estimated by using the triangle inequality and the fact $T_{\tilde{\mu}} \hat{J}=T \hat{J}$ as follows:
$$
\begin{aligned}
\left\|T_{\tilde{\mu}} J^{*}-J^{*}\right\| & \leq\left\|T_{\tilde{\mu}} J^{*}-T_{\tilde{\mu}} \hat{J}\right\|+\left\|T_{\tilde{\mu}} \hat{J}-T \hat{J}\right\|+\left\|T \hat{J}-J^{*}\right\| \\
&=\left\|T_{\tilde{\mu}} J^{*}-T_{\tilde{\mu}} \hat{J}\right\|+\left\|T \hat{J}-T J^{*}\right\| \\
& \leq 2 \alpha\left\|\hat{J}-J^{*}\right\| \\
&=2 \alpha\left\|T^{\ell-1} \tilde{J}-T^{\ell-1} J^{*}\right\| \\
& \leq 2 \alpha^{\ell}\left\|\tilde{J}-J^{*}\right\|
\end{aligned}
$$

By combining the preceding two relations, we obtain Eq.(4.99)
(b) Let us denote by $e$ the unit vector whose components are all equal to 1. Then by assumption, we have
$$
\tilde{J}+c e \geq \hat{J}=T_{\tilde{\mu}} \tilde{J}
$$
Applying $T_{\tilde{\mu}}$ to both sides of this relation, and using the monotonicity and constant shift property of $T_{\tilde{\mu}},$ we obtain
$$
T_{\tilde{\mu}} \tilde{J}+\alpha c e \geq T_{\tilde{\mu}}^{2} \tilde{J}
$$
Continuing similarly, we have,
$$
T_{\tilde{\mu}}^{k} \tilde{J}+\alpha^{k} c e \geq T_{\tilde{\mu}}^{k+1} \tilde{J}, \quad k=0,1,\dots
$$
Adding these relations, we obtain
$$
\tilde{J}+\left(1+\alpha+\cdots+\alpha^{k}\right) c e \geq T_{\tilde{\mu}}^{k+1} \tilde{J}, \quad k=0,1, \ldots
$$
Taking the limit as $k \rightarrow \infty,$ and using the fact $T_{\tilde{\mu}}^{k+1} \tilde{J} \rightarrow J_{\tilde{\mu}},$ we obtain the desired inequality $(4.102)$ .   $\mathrm{Q.E.D.}$



We next show the basic cost improvement property of rollout.

**Proposition 4.6.2**: (**Cost Improvement by Rollout**) Let $\tilde{\mu}$ be the rollout policy obtained by the one-step lookahead minimization 
$$
\min _{u \in \bar{U}(i)} \sum_{j=1}^{n} p_{i j}(u)\left(g(i, u, j)+\alpha J_{\mu}(j)\right)
$$
where $\left.\mu \text { is a base policy [cf. Eq. }(4.100) \text { with } \tilde{J}=J_{\mu}\right]$ and we assume that $\mu(i) \in \bar{U}(i) \subset U(i)$ for all $i=1, \ldots, n .$ Then $J_{\tilde{\mu}} \leq J_{\mu}$
**Proof**: Let us denote
$$
\hat{J}(i)=\min _{u \in \bar{U}(i)} \sum_{j=1}^{n} p_{i j}(u)\left(g(i, u, j)+\alpha J_{\mu}(j)\right)
$$
We have for all $i=1, \ldots, n$
$$
\hat{J}(i) \leq \sum_{j=1}^{n} p_{i j}(u)\left(g(i, \mu(i), j)+\alpha J_{\mu}(j)\right)=J_{\mu}(i)
$$
