---
layout:     post
title:      读书笔记 <强化学习导论>
subtitle:   Personal Note on "Reinforcement Learning&#58; An Introduction"
date:       2019-11-21 12:00:00
author:     "tengshiquan"
header-img: "img/post-rl-sutton.jpg"
catalog: true
tags:
    - AI
    - DRL
    - Reinforcement Learning

---



# Note on Reinforcement Learning

有点零碎, 待有空了整理汇总一下



#### Notaion

Capital letters are used for random variables, whereas lower case letters are used for the values of random variables and for scalar functions.  大写 表示 变量， 小写 表示 变量的具体的值: 小写一般是写在 sum 后面，  大写一般是写在 E 后面

$\arg\max_a f(a)$    : 	value of a at which f(a) takes its maximal value  ,返回的是a的值









# Introduction

**reinforcement learning**, more focused on goal-directed **learning from interaction** 

**强化学习**  聚焦在  交互中进行 目标导向的学习



试错性 (trial-and-error) 的搜索以及延迟的奖赏——是强化学习两个最为重要的且最具区分度的特征.



使用 动力系统理论 (dynamical systems theory) 的概念 : 将强化学习问题作为对<u>非完全已知</u> incompletely-known 的马尔科夫决策过程 (Markov decision process, MDP) 的最优化控制.

无监督学习 (unsupervised learning), 后者常常被用于发现未标签的数据集合的潜在结构.

强化学习正试图最大化收益信号，而不是试图寻找隐藏的结构。一个代理的经验中如果能具有被发现的结构（Uncovering structure）对强化学习肯定是有益的，但其本身并没有解决强化学习要最大化收益信号的问题。



EE 探索与利用的问题





#### Elements

**Time-step**

**policy** ： s 下选择 a ;mapping from perceived states of the environment to actions to be taken

r ：即时奖励  immediate, intrinsic desirability

V(s) : 未来奖励和的期望，predictions of rewards ;长期  long-term desirability

所有强化学习算法中最为重要的组件就是高效地对值进行估计 estimating 的方法. “估值处于核心地位”  

[^_^]: 棋感， 手感， 路感。。。

**model** : 模型用于模仿环境的反应, 其能够**推断**出环境将会做出怎样的反应.   model-free 即不知道在s执行a有什么反应，即不知道r以及走到的具体状态s'，即不知道MDP的dynamics， 必须实际去sample一下才知道

**planning** : 计划: 通过在实际经历前考虑将来可能的情形, 来决定行为方式.    search model for the best policy

使用模型与计划的强化学习方法被称为有模型 (model-based) 方法; 与之相反的是更简单的使用试错的免模
型 (model-free) 方法, 试错可以视为计划的反面.





#### Tic-Tac-Toe

V(s)表示s下的赢率；切到下个状态的时候，没有显式的获得奖励r

先建立值表，输的状态V=0,赢的V=1,中间状态V=0.5, 然后开始评估.

$$
V(S_t) \leftarrow V(S_t) + \alpha[V(S_{t+1}) - V(S_t)]
$$

 $\alpha$ 步长 step-size  ，用后继的推前面的



**temporal-difference learning** method : because its  changes are based on a **difference**: $V(S_{t+1}) - V(S_t)$, between estimates at two successive times.  时序差分





# Tabular Solution Methods

表格法； s,a 可以枚举的问题 ;  精确解法

can find exact solutions, find exactly the optimal value function and the optimal policy. 



最优策略有时有非常多个，对于可以有随机的最优策略里面，抽去部分随机性，变成固定最优策略





## MAB 问题

The most important feature distinguishing reinforcement learning from other types of learning is that it uses training information that **evaluates the actions** taken rather than instructs by giving correct actions.

重要区别，不设计策略，而是从动作回报里面学到策略



- 利用 exploit: greedy 贪心
- 探索 explore  

如果整个期限很长，则前期探索多一点好；如果时间限制很短，则更多贪心！ 再不考虑其他成本的情况下

老虎机问题是 没有前后关联的状态，nonassociative，只有一个状态 问题， 所以不存在 V(s)的问题， 只研究Q(a)



#### Action-value Methods 动作-值

注意，这里也用了$q_*$,代表的是期望， 下面MDP中代表的是最优。。。 

选a后的期望奖s励记为 $q_∗ ( a ) $ ,  在t step 下选择a ，之前的奖励平均值为 $Q_t(a)$ 

$$
q_∗ ( a ) \  \doteq \mathbb E [ R_t \ | \  A_t = a ]
$$

$$
Q_t(a) = \frac{\sum_{i=1}^{t-1}{R_i \cdot   \mathbb I_{A_i=a}}}{ \sum_{i=1}^{t-1}{\mathbb I_{A_i=a}}  } = \frac{时步 t 之前采取动作 a 所获得奖赏之和}{时步 t 之前采取动作 a 的次数}
$$



**sample-average** 样本均值: 根据大数定理 law of large numbers，$Q_t(a)$ 收敛于 converges to  $q_∗(a)$.

$$
Q' \leftarrow   \frac{Q (n -1) +r}{n} = Q+ \frac{1}{n}(r - Q)
$$




##### greedy action selection method 贪心

$$
A_t \doteq \arg\max_a Q_t(a)
$$

with ties broken arbitrarily 有多个相同greedy值时，随机选  arbitrary



$\epsilon-greedy$  : near-greedy action selection rule

steps increases, every action will be sampled   infinite  times,  ensuring  all $Q_t(a)$ converge to $q_∗ ( a ) $.

the probability of **selecting** the optimal action converges to greater than $1-\epsilon$, that is, to near certainty

只要epsilon大于0，无限步数后，肯定找到所有的真实值，算法到那个时候，选择最优的几率是大于1 - epsilon 

所以有模拟退火的方法， epsilon到后面应该很小



![](/img/RL_Introduction.assets/figure_2_1.png)

![image-20181121101209997](/img/RL_Introduction.assets/image-20181121101209997.png)

Epsilon为0，则上来就随机选择一个，如果reward为负的，会再换，否则之后就一直选这个； 为1则纯随机；图上下面的 最优解 百分比就说明了;  上图是很多次游戏的叠加平均，每次游戏里面每个老虎机有1000步 



#### Incremental Implementation  增量的实现

$n \geq 2$ 开始

$$
Q_n \doteq \frac{R_1+R_2...+R_{n-1}}{n-1}
$$

以这种方式实现的话, 对内存与计算时间的需求会随着越来越多的奖赏被观察到而逐步增大.

$$
Q_{n+1} = Q_n + \frac{1}{n}[R_n - Q_n]
$$

$$
NewEstimate ← OldEstimate + StepSize[Target − OldEstimate]
$$



#### Nonstationary Problem 非静态问题 非稳态

**stationary bandit problems**   reward probabilities <u>do not change over time</u>.

**Nonstationary**   give more weight to recent rewards than to long-past rewards. One of the most popular ways of doing this is to use a constant step-size parameter. 将之前 1/n  变成 $\alpha$ .

$$
Q_{n+1} = Q_n + \alpha[R_n - Q_n] 
\\ = (1-\alpha)^nQ_1 + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i}R_i
$$

$Q_{n+1}$ 为 R 以及 Q_1 的 **weighted average** 加权平均， 因为权重和  $(1-\alpha)^n + \sum_{i=1}^{n} \alpha(1-\alpha)^{n-i}=1$ 

**exponential recency-weighted average** 指数新近加权平均



$\alpha_n(a)$ 表示经过第 n 次选择动作a 后所收到奖赏的步长参数.

sample-average : $\alpha_n(a) = \frac{1}{n}$ 

不是所有的序列选择 $\alpha_n(a)$ 都确保能收敛. 一个**随机逼近理论 (stochastic approximation theory)** 中的著名结论, 提出了能以 1 的概率保证收敛的必要条件 :  

$$
\sum_{i=1}^{\infty} \alpha_n(a) = \infty  \ and \ \sum_{i=1}^{\infty} \alpha_n^2(a) < \infty
$$

对步长恒定的情形, 即$\alpha_n(a)= \alpha $  不能满足第二个条件, 这意味着估计值不会完全收敛, 而是继续随新收到的奖赏变化. 这是非稳态正需要的, 而实际上非稳态问题在强化学习中是最为常见的.



#### Optimistic Initial Values

1. $Q_1(a)$ 造成的 bias 偏差 ； 样本均值法，所有a选择一次后，偏差消失；恒定步长，偏差随时间减小

2. 乐观初始值 可以鼓励探索，在静态问题上表现得高效, 但不通用. 任何初始值的方法都不适用于动态问题

<img src="/img/RL_Introduction.assets/figure_2_3.png">

蓝色的前期的尖峰是因为， 去掉r本身的随机性，则q=5，e=0，前10把肯定能找到最优解，在那个时候都选了最优，然后迭代一下，q变小了，慢慢都变小



#### Upper-Confidence-Bound  Action Selection 上置信

select among the non-greedy actions according to their potential for actually being optimal, taking into account both how close their estimates are to being maximal and the uncertainties in those estimates.

non-greedy在随机选的时候，没有考虑 距离差 ； 几率

$$
A_t \doteq \arg\max_a \bigg[\  Q_t(a) + c \sqrt{ \frac{\ln {T} }{ N_t(a)}} \  \bigg] \
$$

square-root term is a measure of the uncertainty or variance in the estimate of a’s value. 

有**较低的估计值**或**已经被频繁选择**过的动作, 将会**随时间推移**减少被选择的频率.

对动态问题，对有巨大状态空间的问题， 效果不好



#### Gradient Bandit Algorithms 梯度老虎机算法

$H_t(a)$表示a的 numerical preference 数值上的偏好

$$
Pr\{ A_t=a \} \doteq \frac{e^{H_t(a)}}{\sum_{b=1}^{k} e^{H_t(b) }}  \doteq \pi_t(a)
$$

**stochastic gradient ascent**

$$
H_{t+1}(A_t) \doteq H_t(A_t) + \alpha(R_t - \bar R_t) (1- \pi_t(A_t))  \\
H_{t+1}(a) \doteq H_t(a) - \alpha(R_t - \bar R_t)   \pi_t(a)  \ \ \  \forall\ a  \neq A_t
$$

$\bar R_t$ 为t以及之前所有奖励的平均， 做为比较奖励的baseline  ； 没有baseline，效果很差



#### Associative Search (Contextual Bandits)

即有了状态 state

#### Summary

![image-20181127101005278](/img/RL_Introduction.assets/image-20181127101005278.png)





## Finite Markov Decision Processes  MDP

Agent–Environment

Agent  代理 智能体

**Environment**

state  $S_t \in  \mathcal S$

action  $A_t \in  \mathcal A(s)$

reward  $R_{t+1}  \in  \mathcal R$  for an actoin

return :   gain  for episode

trajectory :  $S_0,A_0, R_1, …… S_{t}, A_{t},  \to R_{t+1} , S_{t+1}$   注意下标



有限MDP 中， （S,A ,R）是有限集， R，S 都有明确的概率分布， 仅依赖于前一状态以及动作

$$
p(s',r|s,a) \doteq Pr(S_t = s',R_t = r| S_{t-1} =s,A_{t-1} = a)
$$

**function p** defines the **dynamics** of the MDP.  动态空间；p映射：   $\mathcal S \times \mathcal R \times \mathcal S \times \mathcal A \to [0,1]$

 

一个state 有**Markov property 马尔科夫性**  说明该state include information about all aspects
of the past agent–environment interaction that make a diffserence for the future



**状态转移概率** (state-transition probability)

$$
p(s'|s,a) \doteq Pr(S_t = s’| S_{t-1} =s,A_{t-1} = a) = \sum_{r \in \mathcal R}p(s',r|s,a)
$$

expected rewards

$$
r(s,a) \doteq \mathbb E[R_t |S_{t-1} = s,A_{t-1}=a ]  = \sum_{r \in \mathcal R} r \sum_{s' \in \mathcal S}p(s',r|s,a)
$$

$$
r(s,a,s') \doteq \mathbb E[R_t |S_{t-1} = s,A_{t-1}=a,S_{t} = s' ]  = \sum_{r \in \mathcal R} r \frac{p(s',r|s,a)}{p(s'|s,a)}
$$



#### Goals and Rewards

**reward hypothesis**: maximization of the expected value of the cumulative sum of a received scalar signal (called reward).

The use of a reward signal to formalize the idea of a goal is one of the **most distinctive features** of reinforcement learning.



#### Returns and Episodes

maximize the expected **returns** 回报  reward sequence

$$
G_t \doteq R_{t+1} +  R_{t+2} +R_{t+3} + ... +R_{T}
$$

T is a final time step 终止时间步 

**Episode**  **(trial)** 一集  一节， 结束于   末状态 (terminal state) ; 

**episodic tasks** : set of all nonterminal states, denoted $\mathcal S$ ;  set of all states , denoted $\mathcal S^+$. 



**continuing tasks** 持续式任务;  T = $\infty$

Discount  折扣 ， $\gamma$ 折扣率 discount rate , =0 短视 ； 不过加折扣，则最终G肯定是无穷大，无法比较策略的优劣了

$$
G_t \doteq  R_{t+1} +  \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = \sum_{k=0}^\infty \gamma^k
R_{t+k+1} 
\\ = R_{t+1} + \gamma G_{t+1}
$$



##### Unified Notation for Episodic and Continuing Tasks

统一公式；设定 absorbing state 吸收状态， 在吸收状态里面r=0 ; 

$$
G_t \doteq   \sum_{k=t+1}^T \gamma^{k-t-1} R_{k}
$$

**transition diagram : 这里的 S，R 都是大写的** 纯时序

![image-20181128095818603](/img/RL_Introduction.assets/image-20181128095818603.png)





#### Policies and Value Functions

公式里面， 大写的表示变量， 小写的表示变量具体的值。

对于带具体时间步下标的 R G S Q 都是大写  $V(S_t) \ \  v(s)$  时间步一般都是t开始， T结束

s下各种a，任何各种s’， 随后各种 r

**deterministic**  :  policy的属性 s下a固定

Nonstationary ：环境的属性， r 对于 时间的稳定性 ！r本身是随机分布的事实是一直存在的，只需简单平均

**policy** : a mapping from states to probabilities of selecting each possible action. 



agent is following policy $\pi$  at time t , $\pi(a \vert s)$  is the probability that $A_t = a$ if $S_t = s$  几率函数，跟p一样



state-value function for policy $\pi$ ：the expected return when starting in s and following $\pi$

$$
v_\pi(s) \doteq \mathbb E_\pi[G_t|S_t =s] = \mathbb E_\pi \bigg [ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg | S_t=s \bigg ] , \forall s \in \mathcal S
$$


action-value function for policy $\pi$ : 

expected return starting from s, taking the action a, and thereafter following $\pi$ ；a之后按策略

$$
q_\pi(s,a) \doteq \mathbb E_\pi[G_t|S_t =s,A_t =a] = \mathbb E_\pi \bigg [ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg | S_t=s ,A_t =a \bigg ]
$$



**Monte Carlo methods**  $v_\pi$ and $q_\pi$ can be estimated from experience. 

**parameterized function approximator** 函数拟合 用少于状态数的参数，来拟合出 V,Q



##### Bellman equation for $v_\pi$

$$
v_\pi(s) \doteq \mathbb E_\pi[G_t|S_t =s] = \sum_a \pi(a|s) \sum_{s',r}p(s',r|s,a) \Big[ r+\gamma v_\pi(s')  \Big]
$$

$$
q_\pi(s,a) \doteq \mathbb E[R_{t+1} + \gamma v_\pi(S_{t+1}) \ |\ S_t =s,A_t =a] 
 = \sum_{s',r}p(s',r|s,a) \Big[ r+\gamma v_\pi(s')  \Big]
$$

**backup diagrams** 备份图 ，这里的 s,r 都是小写的！ 白圈 s ， 黑点 s,a ， 路径 r

V(s) 收敛的过程，其实就是 各条路径的 r值的 反向累加收敛的过程

<img src="/img/RL_Introduction.assets/image-20181127101210945.png" width="30%">



<img src="/img/RL_Introduction.assets/image-20190105160000548.png" width="70%">

<img src="/img/RL_Introduction.assets/image-20190105160150063.png" width="70%">



**backed-up value** : looking ahead to a successor state (or state–action pair), using the value of the successor and the reward along the way to compute.





The Bellman equation averages over all the possibilities, weighting each by its probability of occurring. It states that the value of the start state must equal the (discounted) value of the expected next state, plus the reward expected along the way.  在各种概率上，状态的值 等于 下一状态的值+奖赏 的期望值

图中的状态结点并不一定表示不同的状态; 只是时序上的往下排列



#### Optimal Policies and Optimal Value Functions

**partial order**ing over policies: $\pi \geq \pi'$ if and only if $v_\pi(s) \geq v_{\pi'}(s) , \forall s \in \mathcal S$

**optimal policy**  $\pi_*$ 只是定义

$$
v_*(s) \doteq \max_\pi v_\pi(s) 
\\ q_*(s,a) \doteq \max_\pi q_\pi(s,a)
$$

the **state–action pair** (s,a), this function gives the expected return for taking action a in state s and thereafter following an optimal policy.  s下选a之后的最高奖励 是 之后状态的最佳才行

两者关系

$$
q_*(s,a)  = \mathbb E_\pi  [ R_{t+1} +   \gamma  v_*(S_{t+1})  \ | \ S_t=s ,A_t =a  ]
$$



下面两个bellman optimal 都有 max_a，关键是递推关系

Bellman optimality equation  for $v_*$   s之后选最大s‘
$$
v_*(s) = \max_{a \in \mathcal A(s)} q_{\pi_*}(s,a) 
\\ = \max_a \mathbb E[R_{t+1} + \gamma v_*(S_{t+1}) | S_t=s,A_t =a]
\\ = \max_a \sum_{s',r} p(s',r|s,a) [r+ \gamma  v_*(s')]
$$

state value under an optimal policy must equal the expected return for the best action from that state.



Bellman optimality equation  for $q_*$  选a之后最大是 再选最优a'  只有这里是QQ

$$
q_*(s,a) = \mathbb E \Big[R_{t+1} + \gamma \max_{a'} q_*(S_{t+1},a') \Big|S_t=s,A_t =a  \Big]
\\ = \sum_{s',r} p(s',r|s,a) \Big[ r+ \gamma \max_{a'} q_*(s',a')   \Big]
$$


![image-20181127101416813](/img/RL_Introduction.assets/image-20181127101416813.png)

Max弧线表示选择最优



对于有限 MDP 而言, 针对 v∗ 的贝尔曼最优性方程有不依赖于策略的唯一解. 贝尔曼最优性方程实际上是方程组, 每一个状态都对应于一个方程, 因此如果有 n 个状态的话,那么就在 n 个方程中有 n 个未知数. 如果环境的动态 p 是已知的, 那么原则上说我们可以从各种各样的解非线性方程组的方法中选择一个来求得 v∗. 同样, 也可以解类似的方程组来求出 q∗.   ~~递推回去求解~~ 

显式地求解贝尔曼最优性方程往往不可行，在大规模问题中。 要采用近似的方法

通过 v∗, 最优的长期期望回报转为了可以在每个状态上通过局部、立即的计算而获得的量. 

动作值函数实际上已 经存储了所有单步搜索的结果. 其将最优的长期期望回报作为 “可以在每一个状态-动作对 上立即获得的本地的量” 提供.   It provides the optimal expected long-term return as a value that is locally and immediately available for each state–action pair.





The **online** nature of reinforcement learning makes it possible to approximate optimal policies in ways that put more effort into learning to make good decisions for frequently encountered states, at the expense of less effort for infrequently encountered states. This is one key property that distinguishes reinforcement learning from other approaches to approximately solving MDPs.

强化学习的在线特性使得可以以这样的方式来近似最优策略: 在经常遇到的状态上投入更多的精力, 使得能在这些状态上做出好的决策——以在不常遇到的状态上投入更少精力为代价. 

 





## Dynamic Programming  DP 动态规划

DP : refers to a collection of algorithms ,compute optimal policies given a model of  (MDP). 

 limited in reinforcement : 	assumption of a perfect model and computational expense



一般不适用于 continuous 连续问题  exact solutions are possible only in special cases.

A common way of obtaining approximate solutions for tasks with continuous states and
actions is to quantize the state and action spaces and then apply finite-state DP methods.



key idea of DP: use of **value functions** to organize and structure the search for good policies.  

V -> P;  两个 Bellman optimality equations



#### Policy Evaluation (Prediction)

**policy evaluation**  prediction problem : **compute** the state-value function $v_\pi$ for an arbitrary policy $\pi$.

思路：bellman改迭代 ， 逼近， 去近似 $V_\pi$

If the environment’s dynamics are completely known,  上式 a system of $\vert S \vert$ simultaneous linear equations, $\vert S \vert$ 个联立的线性方程组，则bellman方程可以直接求解； DP 是用迭代的方法求解  **iterative solution ** 


目标：求解 sequence of approximate value functions V(s) : $\mathcal S^+ \to \mathbb R$

迭代公式  k是迭代的下标， k+1新值， k 旧值 ，其实做成上标比较好

$$
v_{k+1}(s) \doteq \mathbb E_\pi \big[R_{t+1} + \gamma v_k(S_{t+1})| S_t = s \big] 
\\ = \sum_a \pi(a|s) \sum_{s',r}p(s',r|s,a) \big [r+ \gamma v_k(s') \big ]
$$

其中， r 是真实值， v(s') 是估的 ； 通过 bellman公式， 下一个状态的旧值 以及r 来更新 s， 收敛到真实值



**iterative policy evaluation**  迭代策略估计 :  $k \to \infty $ ,  sequence ${v_k}  \to v_\pi $, 只要保证$v_\pi$ 存在。

**expected update** 预期内的更新 : Each iteration of iterative policy evaluation updates the value of **every** state once to produce the new approximate value function $v_{t+1}$.   每轮迭代，更新一次所有状态的v

<u>之所以expected: based on an expectation over all possible next states rather than on a sample next state.</u> 

 

编程时候，用两个array表示v，一新一旧；

 **in place**  就地  原位：不使用临时变量 ,只用一个array， 立刻更新到旧值上，depending on the order in which the states are updated, sometimes new values are used instead of old ones .  converges faster  

a **sweep** through the state space.  一次状态空间的更新扫荡

For the **in-place** algorithm, the order in which states have their values updated during the sweep has a significant influence on the rate of convergence.  遍历的顺序影响很大



![image-20181125214035332](/img/RL_Introduction.assets/image-20181125214035332.png)

loop 里面的 v 只是临时记录下 旧值， 然后好判断是不是已经收敛到足够的精度





#### Policy Improvement

对所有状态， 很可能一个策略不会游走到所有的状态，但要对所有的状态都有应对！！
新策略只在s状态改变了a，找到一个q值大的，但可能造成后续的s’以及r序列跟之前的完全不一样
但是q是代表之后所有的收益，所以如果是已经收敛好的，则肯定新策略更优

**deterministic policies**:  $\pi(s)$ ,action taken in state s under deterministic policy ，返回的选中的a值；a是固定的，比如四个方向都可以的，固定选一个方向的叫固定策略，只要找到一个最优解就行；  但转移到的下一个状态可能是随机的

$$
q_\pi(s,a) \doteq \mathbb E[R_{t+1} + \gamma v_\pi(S_{t+1}) \ |\ S_t =s,A_t =a] 
\\ = \sum_{s',r}p(s',r|s,a) \Big[ r+\gamma v_\pi(s')  \Big]
$$

key criterion is whether this is greater than or less than $v_\pi(s)$.

If  greater， 则选a，然后再按原策略进行 比一直按原策略好

**policy improvement theorem**： 对所有具体状态s，都有$\pi '  好于或者等于  \pi $ ，哪怕只改一点点，其他的用原来的也是新的策略；不用关心该s下改了a以后，走向不一样

$$
q_\pi(s, \pi(s)) \geq v_\pi(s) ,  \ \forall s \in \mathcal S
\\ v_{\pi '}(s) \geq v_\pi(s) 
$$

感觉只能局部优化， 单步优化， 因为只是看一步的r是不是更优

对表格法的问题， 可以遍历

greedy 选择新a最大q的策略公式

$$
\pi'(s) \doteq \arg\max_a q_\pi(s,a) 
\\ =\arg\max_a  \sum_{s',r}p(s',r|s,a) \Big[ r+\gamma v_\pi(s')  \Big]
$$

等到收敛了，必定是找到了最优解



**stochastic policies**: $\pi(a\vert s)$  probability of taking action a in state s under stochastic policy  这边是几率值

对所有的能取到 maximum 的actions 概率一样， 其他 submaximal 取值小的都概率0





#### Policy Iteration

一个策略的V收敛以后， 肯定能找到很多s的a是可以改进的， 改进，然后再计算收敛

计算好$V_{\pi_0}$, 找到的新策略$\pi_1$，然后计算$V_{\pi_1}$是以之前计算的V(s)数组作为基础的，不是从头算起， 所以V迭代一次h还未收敛就能用于计算新的策略$\pi_3$

或者一开始的V是0，一步评估， 一步选a，都会最终收敛， 只要每个状态都sweep一遍 ，每次loop

<img src="/img/RL_Introduction.assets/image-20181125222414257.png" width="80%">

![image-20181126000739105](/img/RL_Introduction.assets/image-20181126000739105.png)





#### Value Iteration

One drawback to policy iteration is that each of its iterations involves policy evaluation, which may itself be a protracted iterative computation requiring multiple sweeps through the state set.  策略评估浪费时间

the policy evaluation step of policy iteration can be truncated in several ways without losing the convergence guarantees of policy iteration. One important special case is when policy evaluation is stopped after just one sweep (one update of each state). This algorithm is called value iteration  策略评估可以被缩短而不影响收敛性；一个重要的特例是，更新一次状态的策略评估后就停止，这就是值迭代(上面的算法，policy evaluation 的外loop只执行一次，即评估一次，然后就是对每个s选最大v)

Value iteration effectively combines, in each of its sweeps, one sweep of policy evaluation and one sweep of policy improvement.    直接选MAX


$$
v_{k+1}(s) \doteq \max_a   \mathbb E_\pi[R_{t+1} + \gamma v_k(S_{t+1})|S_t = s,A_t = a]
\\= \max_a \sum_{s',r}p(s',r|s,a) \Big[ r+\gamma v_k(s')  \Big]
$$

另外一个视角容易理解 <u>turning the Bellman optimality equation into an update rule</u>； 以及比较 backup 图

一个是评估的sweep，一个是improvement的sweep，两个合成了一个sweep了，无显式$\pi(s)$ 

过程显然与s状态空间的遍历顺序有关系， 但最终结果都是收敛的

![image-20181126003016353](/img/RL_Introduction.assets/image-20181126003016353.png)

Faster convergence:interposing multiple policy evaluation sweeps between each policy improvement sweep 评估若s干sweep，然后改进策略一sweep 感觉是能最快迭代的

In general,the entire class of **truncated policy iteration** algorithms can be thought of as sequences of sweeps, some of which use policy evaluation updates and some of which use value iteration updates.

总的方向上，V值是一直积累往更大的方向收敛的， 要么直接选下面更大的一个，要么慢慢的选更大的，方向总是定的； policy iteration 以及 value  iteration 都是基于V(S)收敛的方法



#### Asynchronous Dynamic Programming

之前是讨论每个策略evalue sweep多少次就improve的问题；现在讨论每次sweep里面 遍历 state set 的顺序问题

两者结合 -> **asynchronous truncated policy iteration**

A major drawback to the DP methods: operations over the entire state set of the MDP, require sweeps of the state set.

Asynchronous DP algorithms are **in-place** iterative DP algorithms that are not organized in terms of systematic sweeps of the state set.  不用系统的刷新状态；These algorithms update the values of states in any order whatsoever, using whatever values of other states happen to be available. 能用啥就用啥；The values of some states may be updated several times before the values of others are updated once. To converge correctly, however, an asynchronous algorithm must continue to update the values of all the states: it can’t ignore any state after some point in the computation. 要收敛必须刷新到**所有状态**在后期阶段  Asynchronous DP algorithms allow great flexibility in selecting states to update. 灵活性



有一些规模很大的问题， 就算规则是很简单的，例如围棋，更不用说规则复杂的； 就算找到上层的规则，对于底层的各个状态仍然无法全部的遍历到，这算不算数学上那种无法验证的问题 π中含有任意长度的连续数字9；怎么样去取舍某些状态，才能在全面性取得很好的效果？？一些知识的价值就是对这些最边边角角的状态的探索



可以自己选择刷新的顺序，或者跳过一些不重要的状态; 让v值的 传播更有效率

To solve a given MDP, run an DP algorithm at the same time that an agent is actually experiencing the MDP. The agent’s experience can be used to determine the states to which the DP algorithm applies its updates. At the same time,the latest value and policy information from the DP algorithm can guide the agent’s decision making.



#### Generalized Policy Iteration  GPI

广义的策略迭代

核心就是看怎么去评估， 选策略肯定是greedy

只要最终update 了所有状态， 足够多的次数， 随便什么顺序去update， 只要选大的方向没错， 最终收敛；可以看backup图来理解， 选的策略只是图中某几条路径，  评估这个策略到收敛，就是这几条路径的值收敛，但也会推动选择其他a改变策略的时候，让其他策略的评估更准，最终都收敛。

DP算法，对于全状态的遍历，其实是r值从后期状态往前期状态的反向累加的迭代过程，上面的不同算法，只是叠加迭代的路径选择不同而已， 最好的就是直接选max

- 对所有状态 state set 顺序遍历， 关键是evaluation  sweep的次数
  1. 策略的V评估到收敛 选V(s)最大  Policy Iteration  选最大即优化一次策略，选能让各个状态取最大的V的a   
  2. 策略的V评估一次 选V最大  Value Iteration
  3. 策略的V评估若干次 选V最大   **truncated policy iteration** 
- 任意顺序尽快的遍历更新 state   -> Asynchronous DP





![image-20181129104149661](/img/RL_Introduction.assets/image-20181129104149661.png)



<img src="/img/RL_Introduction.assets/image-20181126205345116.png" width="30%">

<img src="/img/RL_Introduction.assets/image-20181126205422750.png" width="50%">





#### Efficiency of Dynamic Programming

worst time : polynomial in the number of states and actions ； 比较快

Linear programming 更快，但适用的 空间更小

DP is sometimes thought to be of limited applicability because of the **curse of dimensionality**

solve MDPs with millions of states；

PI， VI 哪个更优?

On problems with large state spaces, asynchronous DP methods are often preferred. 





#### Summary

**bootstrapping**: update estimates on the basis of other estimates







## Monte Carlo Methods  MC  蒙特卡洛

**estimating** value functions and discovering optimal policies ； **averaging sample returns**

**model-free** :  do not assume complete knowledge of the environment.

require only **experience**:  sample sequences of states, actions, and rewards from actual or simulated interaction with an environment.

**simulated** experience : 需要一个只输出sample的模型即可，不像DP一样需要各种可能状态全分布的model

for **episodic tasks** :   sample, 用的是episode结尾返回的回报, 过程可能是随机乱选的

Monte Carlo methods can thus be **incremental** in an **episode-by-episode** sense, but not in a **step-by-step** (**online**) sense. 按照一个个episode来增量update，而不是一步一步 :   offline

Because all the action selections are undergoing learning, the problem becomes **nonstationary** from the point of view of the earlier state

handle **nonstationarity** : adapt the idea of general policy iteration (GPI), DP computed value functions from knowledge of the MDP, here we **learn** value functions from sample returns with the MDP.



On-line : step-by-step   可以把episode的mdp过程看成一个line

Off-line : episode-by-episode



#### Monte Carlo Prediction

learning V(s) for a given policy ：average the returns observed after visits to that state. As more returns are observed, the average should **converge** to the expected value.



Each occurrence of state s in an episode is called a **visit** to s.   s may be visited multiple times in the same episode；the first time it is visited in an episode the **first visit** to s. 



这里要特别注意，timestep的S 以及具体状态值的s

S0 S1 S2 S3 ..S10 ; S1 = s, S3=s, 其他的不管， 则，  V(s) 有两个 G(1)  以及 G(3)

**first-visit** MC method estimates $v_\pi(s)$ as the average of the returns following first visits to s,

**every-visit** MC method  : averages the returns following all visits to s. 

Both converge ； 如果一个episode里面，保证状态不会重复出现， 则二者一样

![image-20181125170933779](/img/RL_Introduction.assets/image-20181125170933779.png)

这里算G的时候，是采用倒序的方式，然后对每个$S_t$记录第一次出现时候的G，最后再求平均；所以必须在一个episode结束以后才能算G，off-line



对backup diagrams：  MC 一个sample ，一个路径走到底；DP 每个状态只考虑所有可能的下一步，然后考虑每个状态

estimates for each state are independent : not bootstrap

可以只搞 states of interest



#### Monte Carlo Estimation of Action Values

Without a model, state values alone are not sucient. must estimate the value of each action in order for suggesting a policy.

求 Q(s,a)与求V(s)基本一样，**<u>model-free要求Q</u>**

对model-free的问题，不知道s,a -> r,s' 是啥，必须要记录a。不然虽然能逼近出V，但无法找到相应的策略，其实也能，再找出相应的转移函数；不采用遍历的方式(在s下遍历a)，则只能采用s下随机抽样到所有a的方式来逼近V(s)，如果使用max，只可以求出V*的；但是因为不知道model，还是需要记录下取Vmax的那个a，其实就是求Q



The only complication is that many state–action pairs may never be visited。 对于策略的状态值，对于每个状态，策略都有应对,只是说不一定能达到某个状态s。每个状态下的动作a有偏好，有些策略根本不会选择，如果采用sample的方式将不会有结果





**maintaining exploration** 问题：  如果是固定策略，有些动作访问不到，必须想办法探索到

1.  specifying **exploring starts** : every pair has a nonzero probability of being selected as the start 每个s-a都有几率做开头 ；这种方式在直接与环境交互的情况下(model-free)，无法实现

2. stochastic policy with a nonzero probability of selecting all actions in each state 这个方法稍微现实一些



#### Monte Carlo Control

**Control** 问题 **给定一个策略π，求解近似最优策略π∗**  E I E I E I...

to approximate optimal policies  去拟合最优策略

![image-20181127171148894](/img/RL_Introduction.assets/image-20181127171148894.png)

Policy improvement:

$$
\pi(s) \doteq \arg\max_a q(s,a)
$$

$$
q_{\pi_k}(s, \pi_{k+1}(s)) = q_{\pi_k}(s,  \arg\max_a  q_{\pi_k}(s, a)) = \max_a q_{\pi_k}(s, a) 
\\ \  \geq q_{\pi_k}(s, \pi_{k}(s)) \ \ \geq \   v_{\pi_k}(s)
$$



两个保证收敛的假设:

1. exploring starts
2. policy evaluation could be done with an infinite number of episodes. 需要无限多个episode



去掉第二个假设，解决需要无限多episode的方法

1. approximating  近似，评估误差的界限以及概率，然后采用做够多的步数保证误差很小， 不现实

2. evaluation时放弃Policy收敛就开始改进，不用太多episode，差不多就行，但如果要比较近似，还是要很多episode；按GPI，评估就一步sweep, 得到Value Iteration；

   in-place version Value Iteration alternate between improvement and evaluation steps for single states 

利用2的方法，得到算法 on-policy

Monte Carlo policy iteration it is natural to alternate between evaluation and improvement on an episode-by-episode basis. After each episode, the observed returns are uesed for policy evaluation, and improved at all the states visited in the episode.	  

蒙特卡洛算法，比较自然的可以使用 边评估边改进的策略来求最佳策略。每次计算的节奏是按照episode来，每sample一个episode就 Evalue Improve 一次；因为这个算是是改进正在采样的算法， 所以是on-policy

![image-20181127180146950](/img/RL_Introduction.assets/image-20181127180146950.png)

这里求G也是倒序，所以是  offline , onpolicy





#### Monte Carlo Control without Exploring Starts

现在讨论 avoid assumption of exploring starts  
**开始不能从所有状态出发，则要选策略要加入随机，保证遍历全部**

只有采样的算法， sample=generate data，才区分 on-off

On-policy methods attempt to evaluate or improve the policy that is used to make decisions;  MC ES

Off-policy methods evaluate or improve a policy different from that used to generate the data. 



on-policy control methods the policy is generally **soft**, meaning that $\pi(a\vert s) > 0 \ \  \forall s \in  \mathcal S \ and \ \forall a \in \mathcal A(s) $, 每个s，a都有机会 

$soft$ :  $\pi(a\vert s) >  0$  每个a都有机会取到，  比如exploring start 就保证了这个条件

$\epsilon-soft$ :   $\pi(a\vert s) \geq  \frac{\epsilon}{\vert \mathcal A(s)\vert }$ for all s,a ;  $ 1 \geq \epsilon >0$  即所有的都有几率选到   这样跟上面的soft没有太大区别，但是最小的概率有了下限，比soft要大

$\epsilon-greedy$ :   就是 $\epsilon$ 几率选全部a ， $1- \epsilon $ 的几率选最优的a



![image-20181127215306007](/img/RL_Introduction.assets/image-20181127215306007.png)

GPI does not require that the policy be taken all the way to a greedy policy, only that it be moved toward a greedy policy.

any $\epsilon-greedy$ policy with respect to $q_\pi$  is an improvement over any $\epsilon-soft$ policy $\pi$

$$
\begin{aligned}
q_\pi(s,\pi'(s)) &=  \sum_a \pi'(a|s) q_\pi(s,a)
\\ &=  \frac{\epsilon}{|\mathcal A(s)|} \sum_a q_\pi(s,a) + (1 - \epsilon) \max_a q_\pi(s,a)
\\& \geq  \frac{\epsilon}{|\mathcal A(s)|} \sum_a q_\pi(s,a) + (1 - \epsilon) \sum_a \frac{\pi(a|s) - \frac{\epsilon}{|\mathcal A(s)|} }{1 - \epsilon} q_\pi(s,a)
\\ &= v_\pi(s)
\end{aligned}
$$

等号成立的时候，就是 $\pi$ 以及 $\pi'$ 都是 soft策略里面最优策略的时候

最简单理解上面公式  一个简单情况， 任意 q都相等， 但是soft策略 $\pi$，则有

$$
1 \geq |A|\pi     \iff    1 \ge \frac{|A|\pi -\epsilon}{1-\epsilon} \iff 1 \geq \sum_a \frac{ \pi- \frac{\epsilon}{|\mathcal A(s)|}}{1-\epsilon}
$$

宽松策略下不停采样，迭代找到最优q值的序列，也就是找到了宽松策略下的最优解

就是找到一个 最优的 $\epsilon - greedy$ 策略， 无法找到 true value的最优解； 只要把 epsilon

取一个比较大的值， 比如 0.9 就可以看出来，能找到最好的策略，在10%的时候选最优a，90%还是随机



#### Off-policy Prediction via Importance Sampling

一个 一步的 MDP， 两个action ， 一个G是0，一个G是1， 采样是平均几率，则MC的话，会得到 0.5 的G平均值

目标策略是选action0 几率0.2，action1 几率 0.8 ，则最终期望是 0.8 ； 通过importance sample也能算出来

但如果采样数很少的时候， 则方差还是蛮大的  ,   如果走了5,  3次action1， 2次action0 ， 则target期望是 （0.8/0.5)*3/5 = 0.96 ,  如果是 weighted 方式 (0.8/0.5) * 3  /( (0.8/0.5) * 3 + (0.2/0.5) *2 ) = 0.86

control算法， 则是greedy 则直接选 action1， 所以算出 target （1/0.5)*3/5 = 1.2 , weighted 方式算出 1，下面的算法里面，发现该 a 不属于 greedy actions ，则直接break





从 epsilon-greedy 无法收敛到 真值 这一角度引入 off-policy，  importance sampling

所有的control算法都面临这样一个dilemma，seek to learn action values conditional  on subsequent optimal behavior, but they need to behave non-optimally in order to explore all actions (to find the optimal actions).
学习action value的时候，需要使用下一步最佳动作值；但为了找到最优动作，需要explore action的空间， exploring本身是非最优的。 

dilemma: explore all actions (to find the optimal actions)

**on-policy** approach:  search for a **near-optimal** policy that still explores  之前的MC算法都是 接近最优算法

一个解决方法是 使用两套策略， 一个用于学习，一个用于EE

**Off-policy : learning is from data “off” the target policy**

**Target policy**: learned about

**Behavior policy** :  to generate behavior



on-policy : simple

Off-policy: **greater variance and slower to converge** 方差大，收敛慢 ;    more powerful and general;  **include on-policy** ；  还可以直接从人类专家的数据中学习



Prediction:

本章研究 固定的 target， behavior策略的 prediction

$\pi$ is the target policy, b is the behavior policy,   both   **fixed** and given.

**assumption of coverage**:  require  $ \pi(a\vert s) > 0 $  implies    $b(a\vert s) > 0 $  即target如果会选到一个动作， behavior必须也要能采样到

b must be stochastic in states where it is not identical to $\pi$  b策略是随机的并且与pi不一样 ， pi策略可以是deterministic ;  在control 的情况下， pi 策略一般用的是当前value 估值下的 deterministic greedy 算法； 本节是prediction问题，所以 pi策略是不变的



Almost all off-policy methods utilize importance sampling

given $S_t$ ,   下面都是从t开始， 发生一个 trajectory的几率

$$
Pr\{A_t,S_{t+1},....,S_T \ |\ S_t,A_{t:T-1} \sim \pi  \} = \prod_{k=t}^{T-1} \pi(A_k|S_k)p(S_{k+1}|S_k,A_k)
$$

**importance sampling** :  estimating expected values under one distribution given samples from another

**importance sampling ratio** :  该值可以大于1或者小于1， 取值 0 - 无穷，所以会造成方差很大

$$
\rho_{t:T-1} \doteq \frac{\prod_{k=t}^{T-1} \pi(A_k|S_k) p(S_{k+1}|s_k,A_k) }{\prod_{k=t}^{T-1} b(A_k|S_k) p(S_{k+1}|s_k,A_k)}
=\prod_{k=t}^{T-1} \frac{\pi(A_k|S_k)}{b(A_k|S_k)}
$$

$$
\mathbb E[G_t|S_t = s ] = v_b(s)
\\ \mathbb E [\rho_{t:T-1} G_t|S_t = s ] = v_\pi(s)
$$



现在让时间步下标跨越 trajectory，方便索引某个s，比如第一个trajectory终止在100， 则第二个trajectory从101开始;然后定义一个访问了s所有下标集合 : **the set of all time steps in which state s is visited, denoted $\mathcal T(s)$**   所有状态s被访问到的time step集合 (every-visit) ;   first-visit的方式则值放入第一个该s的time step

let  T(t) denote the first time of termination following time t , and $G_t$ denote the return after t  up through T(t). Then $\lbrace G_t\rbrace_{ t \in \tau(s) }$  are the returns that pertain to state s, and  $\lbrace \rho_{t:T(t)-1}\rbrace_{t \in \tau(s)}$ are the corresponding importance-sampling ratios 

下面两种求 $V_{\pi}$ 的方法， 一种直接按出现次数求平均；    两个公式的 分子部分都是加权求G

关键是 两种方式求出来的收敛值是一样的 ； 对于 target 与 behavior 策略一致的情况， 两个公式是一样的

**ordinary importance sampling**

$$
V(s) \doteq \frac{\sum_{t \in \mathcal T(s)}  \rho_{t:T(t)-1}G_t}{ |\mathcal T(s)| }
$$

**weighted importance sampling**

$$
V(s) \doteq \frac{\sum_{t \in \mathcal T(s)}  \rho_{t:T(t)-1}G_t}{ \sum_{t \in \mathcal T(s)}  \rho_{t:T(t)-1}}
$$

weighted 公式，分母的解释， 所有从s出发的trajectory的几率的求和  ， 如果是 behavior 策略是纯随机，则分母的期望就跟上面的公式一样



对 first-visit

weighted 的方式， 是**有偏 biased**的， 对于只有一个观察值的时候，加权求和的结果是这个观察值本身。所以这个方法的值函数期望是$v_{b}(s)$而不是$v_{\pi}(s)$ 。

不加权求和的结果期望是无偏的，但是它有可能很极端，也就是方差会很大。 比如ratio是10的时候，说明观察到的G要是乘以10才是target的结果；有些情况比如轨迹序列本身和目标策略产生的序列很接近，但仍然可能产生远离期望的结果。正式的说就是，ordinary importance sampling的期望是无偏的，但是方差是有可能没有上限的因为相对比值是没有上限的。相反weighted importance sampling的方法期望是有偏的但是方差会逐渐趋近于0，即使比例有可能是无穷大。所以在实际应用中一般会选择weighted importance sampling。不过ordinary 的方法很容易用到近似算法中，第二部分会讲到.

对 every-visit

？？？ 这两种估计算法都是有偏的并且随着采样次数的增多偏差都会收敛到0

实际操作中， every-visit is prefered，因为这样不需要记录某个状态是否被访问过。





##### Example 5.4: Off-policy Estimation of a Blackjack State Value

![image-20190616213012403](/img/RL_Introduction.assets/image-20190616213012403.png)

Weighted importance sampling produces lower error estimates of the value of a single blackjack state from off-policy episodes.    加权重要性采样比 普通的重要性采样，  mse 要小



##### Example 5.5: Infinite Variance

The estimates of ordinary importance sampling will typically have infinite variance, and thus unsatisfactory convergence properties, whenever the scaled returns have infinite variance—and this can easily happen in off-policy learning when trajectories contain loops.  常规重要性采样 通常造成**无限大的方差**， 收敛性不可接受， 当采样轨迹中包含 loop；

例子，target policy 总是往左， 所以最终的G期望必定是1。 b策略是两边随机， 所以最终的G期望是0.5



![image-20190616212755767](/img/RL_Introduction.assets/image-20190616212755767.png)

Ordinary importance sampling produces surprisingly unstable estimates on the one-state MDP

Figure 5.4 shows ten independent runs of the first-visit MC algorithm using ordinary importance sampling. Even after millions of episodes, the estimates fail to converge to the correct value of 1. In contrast, the weighted importance-sampling algorithm would give an estimate of exactly 1 forever after the first episode that ended with the left action.These results are for off-policy first-visit MC.

$ Var[X]  \doteq    \mathbb E [ X^2 ] - {\bar X}^2 $  如果方差要是无穷大，则第一项需要无穷大 ，所以 下面的需要是无穷大

$$
\mathbb E_b \left[ \left( \Pi_{t=0}^{T-1} \frac{\pi(A_t|S_t)}{b(A_t|S_t)} G_0 \right)^2 \right]
$$

对于任何结尾是向右的episode， ratio都是0， 因为target不会向右， 这些episode有没啥用；现在考虑前面是若干步向左，最后一步是向左，并且走到底的episode； 因为G都是1，所以可以略掉；现在 要求方差的期望，需要考虑按episode的长度来划分， 某个长度的几率 乘以 该ratio的平方，  最终求和就是 期望。

![image-20190617223534830](/img/RL_Introduction.assets/image-20190617223534830.png)





#### Incremental Implementation  增量法实现

Monte Carlo importance sampling  prediction	

对于on-policy来说和第二章讲的方法没有区别，可以直接使用。之前平均reward，再平均G；

对于off-policy而言需要区分ordinary方法和weighted方法。
对ordinary，把每次增量的部分改为使用相对概率系数修正后，再去平均。



Suppose  sequence of returns $G_1, G_2, . . . , G_{n-1}$, starting in the same state,  each with a corresponding randdom weight $W_i$  (e.g., $W_i = \rho_{t_i:T(t_i)-1}$)  ；
此处下标i在firsit-visit 里面，其实就代表了第i个episode;  $n \geq 2$ 



上面 $V_n$ 是定义式， V的下标n 代表第几次迭代， $V_1 = 0$ $V_2 = G_1$ 才用到第一个episode

下面是迭代求解公式

ordinary importance sampling

$$
V_n \doteq  \frac{\sum_{k=1}^{n-1}W_kG_k}{n-1}
\\ V_{n+1} = V_n + \frac{1}{n}[W_nG_n - V_n]
$$


weighted importance sampling  


$$
V_n \doteq  \frac{\sum_{k=1}^{n-1}W_kG_k}{\sum_{k=1}^{n-1}W_k}, \quad n \geq 2
$$

$$
V_{n+1}  \doteq V_n + \frac{W_n}{C_n}\Big[ G_n - V_n \Big]  , \quad n\geq 1
\\ C_{n+1} = C_n + W_{n+1}
\\ C_0 \doteq  0
$$

$C_n$ 是权重的累加，需要记录下来； $C_0 = 0 , V_1 = 0,  C_1 = W_1 , V_2 = G_1有偏, C_2 = W_1+W_2,  V_3 = ... $

上面两方式的公式，差值部分 一个用的是 $W_nG_n - V_n$  一个用的是 $G_n - V_n$ 



该算法也适用于on-policy ，  target 与 behavior 一样就行 ， W=1 恒成立

下面算法里的 W 用来维护 $\rho_{t:T-1}$  ， 是与具体episode 相关的， 在一个episode里面的， 每个episode出来以后要重置； C 是对应所有采样的episode的，所以不会在episode结束归零



这里开始用Q， 对于一步MDP， Q值表与任何策略无关，只与最终的G有关， 例如 S0 a0   G0=0，  S0 a1 G1=1

则Q就是 [0,1] , 与策略的不相关； 一步以上的MDP， Q值表与策略就相关了， 例如 S a0 S0 a0 G00 =0 , S a0 S0 a1 G01= 5, S a1 S1 a0 G01 = 2,  S a1 S1 a1 G11 = 4,   S状态的Q值表就与策略相关



下面算法中， W的更新是滞后的， 是因为这里算的是Qtable， 如果算的是V，则需要提前！！！

![image-20190623212605679](/img/RL_Introduction.assets/image-20190623212605679.png)

为何这里这个是滞后的， W的更新！！  至少一步mdp这块，使用滞后的，是说得通的

 

#### Off-policy Monte Carlo Control

The policy used to generate behavior, called the **behavior** policy, may in fact be unrelated to the policy that is evaluated and improved, called the **target** policy.An advantage of this separation is that the target policy may be deterministic (e.g., greedy), while the behavior policy can continue to sample all possible actions.

现在考虑第二种形式的蒙特卡洛控制算法，也就是off-policy形式的控制算法。off-policy分为目标策略和行为策略两个策略，分开的其中一个好处是目标策略可以是确定性的，而行为策略可以不断地采样所有可能的算法。Off-policy的蒙特卡洛控制算法利用了前两节讲到的off-policy求值函数的方法，这个方法要求行为策略在每个状态执行每个动作的概率都大于0，因此行为策略需要是epsilon-soft的。

下面的图片中展示了算法的伪代码，算法基于GPI框架和weighted importance sampling用来逼近最优策略和最优动作状态值函数。其中是根据Q值得到的最优贪婪策略。Q是的估计值。其中行为策略可以是任意策略，但是需要满足能够遍历所有状态动作对的要求。使用soft策略即可。策略会在所有访问过的状态上逼近最优策略，而且在这期间行为策略b可以是变化的，即使是在一个episode中间变化。

![image-20181203110833447](/img/RL_Introduction.assets/image-20181203110833447.png)

这里 b 策略就是 behavior策略 ； target 是个greedy 所以是 1  

也跟其他MC一样是 **倒序遍历** episode

有个特别的，就是E一次了以后， 发现 A 不在 当前策略里面，则不要这个episode，因为当前评估，不是走的最优的，t更小的就不看了  

用epsilon-greedy MC就算是找到了最优解，也无法找到最终真值，  会因为epsilon的原因，数值被打折；而通过 importance sampling， 可以保证在最优路径收敛到真值









## Temporal-Difference Learning  TD 时序差分

 **difference**, $V(S_{t+1}) - V(S_t)$



![image-20181207200726512](/img/RL_Introduction.assets/image-20181207200726512.png)

![image-20181207200751304](/img/RL_Introduction.assets/image-20181207200751304.png)



![image-20181207200811613](/img/RL_Introduction.assets/image-20181207200811613.png)



![image-20181207200909428](/img/RL_Introduction.assets/image-20181207200909428.png)

<img src="/img/RL_Introduction.assets/image-20181227145356931.png" width="60%">



TD:

1. central and novel to reinforcement learning
2. combination of MC ideas and DP ideas.  without model ; without final outcome



过程分两部分：

policy evaluation : prediction

finding an optimal policy : control ； generalized policy iteration (GPI)



#### TD Prediction

TD and MC methods use experience  

**constant-$\alpha$ MC** ： every-visit MC for nonstationary : $V(S_t) \leftarrow V(S_t)  + \alpha \Big[G_t - V(S_t)  \Big]$

The simplest TD ：$V(S_t) \leftarrow V(S_t) + \alpha \Big[R_{t+1} + \gamma V(S_{t+1}) -  V(S_t)   \Big]$ 收敛到 $V_\pi$

 MC update目标是 $G_t$ ,TD 目标是 $R_{t+1}+\gamma V(S_{t+1})$

$$
v_\pi(s) \doteq \mathbb E_\pi[G_t | S_t =s] \ \ MC
\\ =  \mathbb E_\pi[R_{t+1}  + \gamma G_{t+1}| S_t =s]  \ \ 
\\ =  \mathbb E_\pi[R_{t+1}  + \gamma v_\pi(S_{t+1}) | S_t =s] \ \ DP
$$




![image-20181203185709422](/img/RL_Introduction.assets/image-20181203185709422.png)

每步step都更新一次V(s) , model-free ， 直接观察a后的结果就可以更新迭代



**bootstrapping** method : TD(0) bases its <u>update</u> in part <u>on an existing estimate</u>



The TD target is an <u>estimate</u> for both reasons: it samples the expected values  and it uses the current estimate $V$ instead of the true $v_\pi$. 

TD methods <u>combine the sampling of Monte Carlo with the bootstrapping of DP</u>. 



MC 必须走到T，才知道整个流程的R  offline



MD, TD : sample updates ; based on a single sample successor rather than on a complete distribution of all possible successors

DP : expected updates



**TD error** :  $ \delta_t \doteq R_{t+1} + \gamma V(S_{t+1})  - V(S_t) $

$\delta_t$ is the error in $V (S_t)$, available at time t + 1 



<u>the **Monte Carlo error** can be written as a sum of TD errors</u>:   episode更新

$$
G_t - V(S_t) = R_{t+1} + \gamma G_{t+1} - V(S_t) + \gamma V(S_{t+1}) - \gamma V(S_{t+1}) 
\\= \delta_t + \gamma(G_{t+1} - V(S_{t+1})) = \sum_{k=t}^{T-1} \gamma^{k-t} \delta_k
$$



#### Advantages of TD Prediction Methods

bootstrap : learn a guess from a guess 



TD > DP : model-free

TD > MC: online,  incremental



收敛性 guarantee convergence : <u>For any fixed policy $\pi$, TD(0) has been proved to converge to $v_\pi$</u>, in the mean for a constant step-size parameter if it is suciently small, and with probability 1 if the step-size parameter decreases according to the usual stochastic approximation conditions .



Markov reward process: MRP , Markov decision process without actions





##### Bias/Variance Trade-Off

Return $G_t = R_{t+1} + γR_{t+2} + ... + γ_{T−1}R_T$ is unbiased estimate of $v_π(S_t)$ 

True TD target $R_{t+1} + γv_π(S_{t+1})$ is unbiased estimate of $v_π(S_t)$ 

TD target $R_{t+1} + γV(S_{t+1})$ is biased estimate of  $v_π(S_t)$ 
TD target is much lower variance than the return: 

- Return depends on many random actions, transitions, rewards 
- TD target depends on one random action, transition, reward 





#### Optimality of TD(0)

Suppose there is available only a finite amount of experience, a common approach with incremental learning methods is to present the experience repeatedly until the method converges upon an answer.

在只有有限的外界交互经验的情况下，先把已经有的经验迭代到收敛

**batch updating** : updates are made only after processing each complete batch of training data.



Under batch updating, TD(0) converges   与  constant-$\alpha$  MC method 收敛到的值不同

constant-$\alpha$  MC method converges to values, V (s)  sample averages  , find the estimates that minimize mean-squared error on the training set 

TD is optimal in a way that is more relevant to <u>predicting returns</u>.

Batch TD(0) finds the estimates that would be exactly correct for the maximum-likelihood model of the Markov process.

we expect  answer will produce lower error on future data

例子:  对下面8个过程

A,0,B,0    B,1 B,1 B,1  B,1 B,1  B,1 B,0    =>V_B = 3/4  

V_A = 3/4   batch TD(0)   TD 算A的时候，考虑了B

V_A = 0    batch MC    因为G_A = 0 只有这一个采样，所以只能忠于数据，后面的G_B值改变也没用了



**maximum-likelihood estimate**  of a parameter is the parameter value whose probability of generating the data is greatest 。 batch  TD(0)  converges  考虑了预测在里面

**certainty-equivalence estimate** : equivalent to assuming that the estimate of the underlying process was known with certainty rather than being approximated ； batch  MC  converges  忠于观测数据

Although the nonbatch methods do not achieve either the certainty-equivalence or the minimum squared-error estimates, they can be understood as moving roughly in these directions.

TD methods converge more quickly than Monte Carlo methods



If $n = \vert S \vert$ is the number of states, then just forming the maximum-likelihood estimate of the process
may require on the order of n^2 memory, and computing the corresponding value function requires on the order of n^3 computational steps if done conventionally. In these terms it is indeed striking that TD methods can approximate the same solution using memory no more than order n and repeated computations over the training set. On tasks with large state spaces, TD methods may be the only feasible way of approximating the certainty-equivalence solution.  ? TD 比 按常规做法 省内存





#### Sarsa: On-policy TD Control

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \Big[ R_{t+1} + \gamma Q(S_{t+1},A_{t+1}) - Q(S_t,A_t)  \Big]
$$



Sarsa 算法 初始化的时候，要先用 策略(比如弱贪婪，一定要能遍历到所有的状态)先选一个A出来， 迭代的时候， 利用当前的S,A以及s'， 要再用策略选个A'出来， 这样循环

The convergence properties of the Sarsa algorithm depend on the **nature** of the policy’s dependence on Q. Sarsa converges with probability 1 to an optimal policy and action-value function as long as all state–action pairs are visited an infinite number of times and the policy converges in the limit to the greedy policy 

Sarsa  算法收敛性取决于选择a的策略p，会收敛到$q_\pi$的真值；  如果该p是贪婪算法，并且能遍历全部的q，则迭代足够多次，会收敛到  $q_*$ ，但策略本身还是贪婪的，主要选q大的，然后一点随机，只不过选q大的那个都是最优了，所以收敛到该贪婪算法的最优极限策略

关键：与Qlearing的区别，为什么要用A'， 因为是on-policy， 采样的策略是就是要优化的策略

![image-20181207203127797](/img/RL_Introduction.assets/image-20181207203127797.png)



#### Q-learning: Off-policy TD Control

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \Big[ R_{t+1} + \gamma 
\max_a Q(S_{t+1},a) - Q(S_t,A_t)  \Big]
$$

![image-20181207203351032](/img/RL_Introduction.assets/image-20181207203351032.png)

Qlearing 收敛到 q_* ， Q learning 的target policy是greedy Policy 

注意这里， 采样用的A是$\epsilon-greedy$选的,  然后经过evaluate优化的策略， 下一步，Q值表已经发生了变化 



#### Expected Sarsa

$$
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha \Big[ R_{t+1} + \gamma \mathbb E_\pi[Q(S_{t+1},A_{t+1}) \ |\ S_{t+1}  ] - Q(S_t,A_t)  \Big]
\\ \leftarrow Q(S_t,A_t) + \alpha \Big[ R_{t+1} + \gamma \sum_a \pi(a|S_{t+1})Q(S_{t+1},a)   - Q(S_t,A_t)  \Big]
$$

<img src="/img/RL_Introduction.assets/image-20190105201815259.png" width="10%">

<img src="/img/RL_Introduction.assets/image-20190105195043134.png" width="60%">









#### Maximization Bias and Double Learning

讨论的所有用于控制问题的算法都涉及到最大化target policy。例如Q learning 的target policy是greedy Policy ，而sarsa的target policy 是ϵ-greedy policy。这些算法，最大化估计了action-value，会导致很大的正向偏差。



All the **control** algorithms involve **maximization** in the construction of their **target policies**.

**maximization bias**  如果一个变量均值为0，但随机， max就会一直取大的

举个例子：考虑一个state s下所有的action对应的真实action-value  q(s,a)均为0，但是估计的action-value Q(s,a)是不确定的，可能大于0 也可能小于0。真实的action-value的最大值为0，但估计的action-value Q(s,a)的最大值是正值，这导致了正向偏差。我们称之为Maximization Bias。

![img](/img/RL_Introduction.assets/1*HmGJAGiZG8coo-B4Q7m7-g.png)



Let X1 and X2 two random variables that represent the reward of two actions at state B.

![img](/img/RL_Introduction.assets/1*47PA48y8jv8hPfIBO_aL4g.png)

Q-Learning uses Max Q(s’,a), represented in the table by Max(𝝁) ，  Max(𝝁) is not a good estimator for Max E(X). It is biased!  上表就是说明  Max(𝝁) 不是Max E(X) 的好的估计 



![image-20190326164355027](/img/RL_Introduction.assets/image-20190326164355027.png)

对Qlearning , 评估以及选择下一个a都是利用了最大的Q的估计， 所以误差较大，现在将这两个estimate分开，误差会减少。

如何避免Maximization Bias呢？考虑第二章中提到的老虎机问题，当我们在有噪声情况下估计每个action 的value时。我们之前讨论过，如果我们采用估计action-value的最大值作为实际action-value的最大值时，会产生很大的偏差。因为我们采用同一系列的采样数据来决定最优action并估计这个action的action value。假设我们将采样数据分为两组，用他们分别产生独立的估计值 Q1(a)和Q2(a),他们的实际value都是q(a)，我们使用其中一个估计 Q1(a)来决定最优action  $A^∗=\arg \max_aQ_1(a)$,另一个估计 ,这种估计方式是无偏估计$Q_2(A^∗)=Q_2(\arg \max_aQ_1(a))$，即$\mathbb E [Q_2(A^∗)]=q(A^*)$。我们可以交换两个估计的角色重复这一过程得到 $Q_1(\arg \max_aQ_2(a)$，这就是double learning 的基本思想。



在标准的 Q-学习和 DQN 中的 max 操作使用同样的值来进行**选择**和**衡量**一个行动。这实际上更可能选择过高的估计值，从而导致过于乐观的值估计。为了避免这种情况的出现，我们可以对**选择**和**衡量**进行解耦。这其实就是双 Q-学习 (van Hasselt, 2010)。

Max操作使得估计的值函数比值函数的真实值大。如果值函数每一点的值都被过估计了相同的幅度，即过估计量是均匀的，那么由于最优策略是贪婪策略，即找到最大的值函数所对应的动作，这时候最优策略是保持不变的。也就是说，在这种情况下，即使值函数被过估计了，也不影响最优的策略。强化学习的目标是找到最优的策略，而不是要得到值函数，所以这时候就算是值函数被过估计了，最终也不影响我们解决问题。然而，在实际情况中，过估计量并非是均匀的，因此值函数的过估计会影响最终的策略决策，从而导致最终的策略并非最优，而只是次优。

为了解决值函数过估计的问题，Double Q-learning 将动作的选择和动作的评估分别用不同的值函数来实现。





Double Q-learning

$$
Q_1(S_t,A_t) \leftarrow Q_1(S_t,A_t) + \alpha \Big[ R_{t+1} + \gamma 
Q_2 \Big( S_{t+1}, \arg\max_a Q_1(S_{t+1},a) \Big ) - Q_1(S_t,A_t)  \Big]
$$

![image-20181207205514082](/img/RL_Introduction.assets/image-20181207205514082.png)





#### Afterstates

一些特殊案例

afterstate value functions

Afterstates are useful when we have knowledge of an initial part of the environment’s dynamics but not necessarily of the full dynamics

目前讲述的都是通用的算法，但是显然总是有一些例外用一些特殊方法会更好。对于第一章中讲的tic-tac-toe游戏来说，使用TD算法得到的既不是状态值函数也不是动作值函数。传统的状态值函数是估算在当前状态采取动作以后会得到的反馈期望，但是在这个例子里估计的是agent采取了动作之后的状态值。把这个状态叫做afterstates，这些状态值函数叫afterstate value functions。Afterstate可以被用在那些我们知道一些环境的变化但不能完全建模的情况。

<u>使用这种设计显然是更有效率的。因为有的不同的position在采取了动作之后可能变化到同样的结果状态。这样对于传统的状态值函数这两个状态是分开的，但是对于afterstate value function就能够发现他们是相等的。</u>这个技术用在很多情境下，比如排队任务等等。本书中目前讲到的GPI和策略以及（afterstate）值函数都能够大致的应用上，也需要考虑选择on-policy和off-policy以及需要exploring等等。

 



## n-step Bootstrapping

MC是一种每episode更新一次的方法，TD是单步更新的方法，n-step Bootstrapping （步步为营）是一种介于TD和MC之间的方法，n-step更新一次。



#### n-step TD Prediction

**n-step TD methods**

<img src="/img/RL_Introduction.assets/image-20190104112558466.png" width="50%">

当在一个采样数据中选择以n步数据来更新value function 时，采用的方法为 n-step TD prediction。

Monte Carlo updates the estimate of $v_\pi(S_t)$ is updated in the direction of the complete return:

$$
G_t \doteq  R_{t+1} + \gamma R_{t+2} + \gamma^2R_{t+3}+\dots+ \gamma^{T-t-1}R_{T}
$$


TD(0) : $R_{t+1} + \gamma V(S_{t+1})$

**one-step return**: first reward plus the discounted estimated value of the next state
$$
G_{t:t+1} \doteq R_{t+1} + \gamma V_t(S_{t+1})
$$

$V_t : \mathcal S \to \mathbb R $   是 t 时刻的 对 s的评估值  在一个 episode里面； 这个下标不太好 容易乱掉，都是按次序迭代V

**n-step return**:

前面n步是新经历的R， 后面的是上次的V值

$$
G_{t:t+n} \doteq  R_{t+1} + \gamma R_{t+2}  +\dots+ \gamma^{n-1}R_{t+n}+ \gamma^{n}V_{t+n-1}(S_{t+n})
$$

n-step 的state value更新公式为 

$$
V_{t+n}(S_t) \doteq V_{t+n-1}(S_t) + \alpha \Big[ G_{t:t+n} - V_{t+n-1}(S_t)  \Big]
$$


收敛性:  当n=1的时候，左边就是0；  左边是用了一段真实R以及V ，肯定比V本身要跟接近v

$$
\max_s \bigg|\mathbb E_\pi[G_{t:t+n}|S_t=s] - v_\pi(s)\bigg|  
\leq \gamma^n \max_s \bigg| V_{t+n-1}(s)  - v_\pi(s) \bigg|
$$

![image-20190107153335633](/img/RL_Introduction.assets/image-20190107153335633.png)

以n=2为例， 则 V_0 <- R_1 + R_2 + V_2 ， V_1 <- R_2 + R_3 + V_4  。。。。。。

这个是边交互边迭代， online ， onpolicy

#### n-step Sarsa

<img src="/img/RL_Introduction.assets/image-20190105010352571.png" width="60%">


$$
G_{t:t+n} \doteq  R_{t+1} + \gamma R_{t+2}  +\dots+ \gamma^{n-1}R_{t+n}+ \gamma^{n}Q_{t+n-1}(S_{t+n},A_{t+n})
$$

$$
Q_{t+n}(S_t,A_t) \doteq Q_{t+n-1}(S_t,A_t) + \alpha \Big[ G_{t:t+n} - Q_{t+n-1}(S_t,A_t)  \Big]
$$





![image-20190107153504321](/img/RL_Introduction.assets/image-20190107153504321.png)



#### n-step Off-policy Learning





#### Off-policy Learning Without Importance Sampling: The n-step Tree Backup















## Planning and Learning with Tabular Methods


A unified view of reinforcement learning methods

- Model-base : DP and heuristic search  : rely on  **planning**   sim exp

- Model-free : MC TD  : rely on  **learning**  real exp

相似点:

- heart : compute value
- looking ahead to future events, computing a **backed-up value**, and then using it as an update target for an **approximate value function**







#### Models and Planning

**Model** :  simulate the environment and produce simulated experience.  s,a -> r,s'

- distribution models : all possibilities and their probabilities
- Sample models : one of the possibilities



![image-20181209203840966](/img/RL_Introduction.assets/image-20181209203840966.png)

对很多问题， model都无法知道有多少state





**planning(规划)** :  takes a **model** as input and produces or improves a policy ； 关键是从 model 而来

$$
Model  \xrightarrow{planning} Policy
$$


- state-space Planning
- plan-space planning  书里不讨论

$$
Model  \xrightarrow{}\ sim \ exp \ \xrightarrow{buckups}\ values \xrightarrow{} Policy
$$



both  estimation of value functions by backing-up update operations

- planning :  use <u>simulated experience</u> generated by a model

- learning : use <u>real experience</u> generated by the environment

特别的， 很多情况下， learning方法可以被planning方法替换； learning方法取实际exp作为输入，也可以改成用sim exp

Q-learing -> Q-planning   收敛到<u>该model的最优解</u>

![image-20181210151716601](/img/RL_Introduction.assets/image-20181210151716601.png)



benefits of planning in small, incremental steps : Planning in very small steps may be the most effcient approach





#### Dyna: Integrated Planning, Acting, and Learning



1. **learn model** directly from real experience 
2. use **planning** to construct a value function or policy
3. **Integrate** learning and planning into a single architecture



New information gained from the interaction may change the  model and thereby interact with planning. 模型本身就是环境的近似，所以更多与环境交互的信息可以进化model，然后让规划更优



在Planning agent中，对所获得的真实环境信息来说，至少有两个主线任务：

- model learning：将真实经验用于提高模型精确度，使得model更接近真实环境

- direct RL(Reinforcement learning): 对真实经验数据运用强化学习来提升value function 和Policy



**Dyna-Q**, a **simple** architecture integrating the major functions needed in an online planning agent. 每个组成部分都用简单的

- direct RL : one-step tabular Q-learning
- Planning : random-sample one-step tabular **Q-planning**
- model-learning : table-based and <u>assumes</u> environment is **deterministic**

<img src="/img/RL_Introduction.assets/image-20181208213007830.png" width="50%">

direct-RL 与 planning 是迭代优化的同一个 Value列表, 而且用的同一个强化学习方法

Both direct and indirect methods have advantages and disadvantages. Indirect methods often make fuller use of a limited amount of experience and thus achieve a better policy with fewer environmental interactions. On the other hand, direct methods are much simpler and are not affected by biases in the design of the model. 间接学习在更少交互的情况下得到更好策略；直接学习更简单并且不会被模型带偏



<img src="/img/RL_Introduction.assets/image-20181208224954811.png" width="60%">

中轴线为agent与环境交互的过程，会生成一系列的 real expericence，左侧direct RL 利用 real expericence 提升 value function 和Policy。右侧，model learning 利用 real expericence 来对环境建模生成model，search control 表示从model中选择 starting states和 actions来生成的simulated experience 的过程，最后，Planning 对simulated experience 运用强化学习方法。



**Real experience** Sampled from environment (true MDP)
**Simulated experience** Sampled from model (approximate MDP) 



**Search control** :  the process that selects the starting states and actions for the simulated experiences generated by the model  控制选取 model起始状态以及起始动作 用来sim



从概念上而言，Planning, acting, model learning, direct RL应该是同步进行的，但在实际运用中，acting, model learning, direct RL所占用的计算资源远小于Planning，所以调整了他们是执行顺序，先执行acting, model learning, direct RL，再进行几步Planning

![image-20181208231113755](/img/RL_Introduction.assets/image-20181208231113755.png)

虽然planning 减少了 使用episode的次数， 但本身的 n  loop 也是需要计算资源的。 但与环境交互得来的exp的代价相比要少



![image-20181210170335267](/img/RL_Introduction.assets/image-20181210170335267.png)

simple maze 这个例子中，Q-learning  第一个 episode 只能迭代最后一步的策略， 因为之前的r都是 0 ；而dyna-q 则可以迭代到很多步的策略



- Model-Free RL 
  - No model 
  - Learn value function (and/or policy) from real experience
- Model-Based RL (using Sample-Based Planning) 
  - Learn a model **from real experience**.  是learn出来的？ 不是直接配置的么
  - Plan value function (and/or policy) **from simulated experience** 
- Dyna
  - Learn a model from real experience 
  - Learn and plan value function (and/or policy) from real and simulated experience 



- Model-Free RL 
  - Advantages:
    - Can efficiently learn model by supervised learning methods
    - Can reason about model uncertainty
  - Disadvantages:
    - First learn a model, then construct a value function
      ⇒ two sources of approximation error





#### When the Model Is Wrong

Models may be incorrect because

- the environment is stochastic and only a limited number of samples have been observed, 
- the model was learned using function approximation that has generalized imperfectly,
- the environment has changed and its new behavior has not yet been observed

由于模型本身的不完善造成的误差， planning 比较容易 算到  suboptimal policy。

模型错误的表现 :  找到suboptimal policy 以后, 环境发生了改变，实际的回报会小于 suboptimal policy 的预期

##### 不容易被发现的模型错误

但是当环境中原始最优策略可以通行，环境变化，出现更好的策略时，model错误在短时间内变得不容易被发现。



<img src="/img/RL_Introduction.assets/image-20181216211221856.png" width="50%">





planning也有EE的取舍问题:  exploration 选a来改进模型  ； exploitation 选model下最优的a来获得最大回报。一方面希望足够的探索，来发现环境，另一方面，希望探索不要过大，以防止降低方法的性能。no solution that is both perfect and practical, but simple **heuristics** are often effective.





**Dyna-Q+** : 对每组state-action 距离自身最近一次出现的消失时间进行了统计，然后认为消失时间越久的state-action越容易发生变化，model就越可能出错.

思路 加个折扣；  越久没有try到的s,a ，再次被遍历到，说明模型发生变化的可能性越大，要鼓励 许久未出现的s,a， r上要倾斜  bonus reward : $r + k \sqrt {\tau}$  , $\tau$ 代表多少步没出现





#### Prioritized Sweeping  分优先级扫描

Dyna 模拟的时候，selected uniformly at random 平均随机的选择 s,a 来改进model    

更好的方案 ：  focuse on particular s–a ， 一些状态 prior to the goal ，是更有价值的；

对迷宫这个例子，以及很多奖励在末端的，从t往T迭代，中间很多状态都是0，浪费； 要迭代跟目标相关的

**backward focusing** : work back from any state whose value has changed， 即重点关注 value

agent已经收敛到了最优策略，这个时候环境改变，agent发现某个s'的估值发生变化，则在一步update之内，所有的action lead  to s 的值都会变， 所以 predecessor 状态 s 的估值都会in turn改变

根据上述**backward focusing 思想**可以反向传播获得一系列的state - action，但不是每个state - action都一样有用，一些state 的value改变比较频繁，而另一些改变比较小。value变化较大的state-action 之前的state-action的value变化大的可能性较高。在随机环境中，<u>变化的可能性与变化程度和state-action需要被更新的紧急程度有关</u>。根据紧急程度进行优先性排序，这是优先扫描（prioritized sweeping)的基本思想。

**prioritized sweeping** :  prioritize the updates according to a measure of their urgency, and perform them in order of priority.

对于游戏来说， 怎么走到后面的状态也是问题； 对游戏状态的空间，cnn的角度，应该先粗后细



![image-20181211093534302](/img/RL_Introduction.assets/image-20181211093534302.png)

将prioritized sweeping思想扩展到不确定环境下。通过不断统计每个state-action的出现次数，和下一步所有可能出现的state-action构成了model，用expected update(期望更新）取代一个 sample update(采样更新），计算所有可能的下一步的states和他们出现的概率。

prioritized sweeping只是提高规划效率的一种方法，但不是最好的方法。将其扩展到不确定环境下，便受到了期望值的限制，会浪费很多时间在计算低概率的转移上。在下一节中，我们将说到：在很多情况下，采样更新（sample updates）明显减小了计算量，其结果很接近真实的value function。 不过方差大



backward focusing只是一种侧重更新思想，还有很多其他的侧重更新方法，例如forward focusing侧重在当前Policy下经常出现的states容易到达下一步的states。

**forward focusing** : focus on states according to how easily they can be reached from the states that are visited frequently under the current policy



#### Expected vs. Sample Updates

本章剩下章节中，我们重点分析一些Planning和learning 方法结合的思想，从Expected 和 Sample Updates开始。

讨论过的方法都可以归结为对value-function的更新, 可以从三个维度来看

- update state values or action values
- estimate the value for the optimal policy or for an arbitrary given policy
- expected updates or sample updates

<img src="/img/RL_Introduction.assets/image-20181211173206383.png" width="50%">



expected update

$$
Q(s,a) \leftarrow \sum_{s',r} \hat p(s',r|s,a) \Big[ r+\gamma \max_{a'}Q(s', a') \Big]
$$
sample update Q-learning-like 

$$
Q(s,a) \leftarrow  Q(s,a) +  \alpha \Big[ R + \gamma \max_{a'}Q(S', a') -  Q(s,a) \Big]
$$



expected update的计算量比较大，而sample update的计算量比较小，但是sample update真的可以接近expected update吗？这里直接给出结论：**在大的随机branching factor，且state过多的情况下，sample updates比 expected update效果好。**





#### Trajectory sample

找路径上的相关状态比遍历全部状态要有效的多， 因为不是所有的状态都值的重视; 但时间够的话，遍历肯定是最强的

本节将比较两种分布更新方式：

Exhaustive sweep 在DP（动态规划）中，扫描所有的state（或state-action)，每次扫描更新一次所有的state（或state-action)。当state过多时，单次扫描的时间消耗过大。在很多任务中，很多state之间是没啥关系的，全部扫描意味着每个state都需要消耗同样的时间，不如重点关注某些需要的state，而忽视那些无关的state。
根据某些分布来从state 或者state-action space中进行采样。 Dyna-Q的agent 使用了均匀采样（uniform) ，但这会导致和 Exhaustive sweep 相同的一些问题，更常用的是根据 on-policy 分布进行采样，即根据当前策略（current policy） 所观测到的分布。在current policy下与model交互时，很容易就能获得该分布。在episode 问题中，从初始state出发，根据current policy，可以生成一条到terminal state 的采样数据。连续问题中，从初始state开始，可以根据current policy ，一直生成采样数据。在这两类问题中，采样数据的state转移和reward都由model提供，而采样action由当前策略（current policy）给出。这种生成采样数据的方式叫做 **trajectory sample**。

利用on-policy分布的好处是忽略了大部分没有意义的state,但是会让一部分state space一直被更新。比较on-policy分布和uniform分布的效果。在uniform实验中，我们循环遍历每组state-action，并依次更新每组state-action。在on-policy实验中，我们模拟出一些episodes：他们的初始state相同，更新那些在 ϵ-greedy 策略下出现的state-action。假设一个state对应b个可以到达的下一时刻的state，b被称作branching factor 





#### Real-time Dynamic Programming

an on-policy trajectory-sampling version of the value-iteration algorithm of dynamic programming (DP).

很多状态在最优策略下 从起始走不到， 不用考虑

Real-time Dynamic Programming简写为RTDP，是一种 on-policy trajectory-sampling 的异步动态规划value iteration 算法。异步动态规划算法不需要依次扫描state space，可以以任意顺序更新state value，与第四章中介绍的传统DP方法相比，RTDP的更新顺序由真实或仿真trajectories （state action reward序列）中的出现的state次序决定。

如果这些trajectories的起始状态（start state）为一个小子集，即只能从特定的state集合中选取start state。

- 对于 prediction 问题（对于一个给定Policy，需要估计出当前的value ），on-Policy trajectory sampling允许算法彻底跳过一些在start state集合下采用当前策略不可能到达的states，这些到不了的state和prediction 问题关系不大。

- 对于control 问题（需要找到一个optimal Policy，而不是估计当前策略的value)，同样有一些在start state集合下采用optimal Policy也无法到达的states,这些states和optimal Policy无关，因此可以不在意这些states对应什么action。此时，我们需要找到是一个optimal partial policy，即局部最优策略，即对于与optimal Policy有关的states是最优策略，但对与optimal Policy无关的states而言，可以随便选择action，甚至可以不选。



收敛条件:

on-Policy trajectory-sampling control 方法 Sarsa 需要在 exploring starts 前提下无限次遍历所有的 state-action才能找到optimal policy，RTDP也是如此。但对于某类问题RTDP可以很快找到optimal partial policy，不需要无限次遍历和optimal Policy有关的states，甚至可以永远不经过其中某些states。这类问题需要满足条件如下：

- the initial value of every goal state is zero,
- there exists at least one policy that guarantees that a goal state will be reached with probability one from any start state,
- all rewards for transitions from non-goal states are strictly negtive 所有经过的奖励都是负数，cost
- all the initial values are equal to, or greater than, their optimal values (which can be satisfied by simply setting the initial values of all states to zero).

类似 寻路算法, 最优路径搜索问题 A*

总之，RTDP相比传统DP方法而言，是一种关注局部有效state从而提升算效率的算法。 收敛快



#### Planning at Decision Time

这里介绍两类Planning：

- background Planning  一种是我们前面一直在说的基于model产生的simulated experience用以逐步提升Policy 或 value function的Planning。通过比较当前state下的value，来选择action。在当前state下，选择action之前，Planning重点在于提高value表或者value 的近似函数表达，涉及到为很多state选择action。此处，Planning所关注的不是当前state。  一直在优化策略，遇到情况直接给出响应，适合低延迟的，快速响应的问题

- decision-time Planning  每给出一个新的state就立即规划出一个action，这种Planning方式比one-step-ahead 方式更深入。通常这种Planning方式多用在不需要快速回应的任务中，如下象棋，可以允许有一定的决策时间。   遇到具体情况的时候，再算策略





#### Heuristic Search

人工智能中基于的state-space planning的传统decision-time Planning方法统称为heuristic search。

对每个出现的state，都需要考虑一个树状结构，从叶节点开始估计value function，然后回到current state（根节点），如果说，之前介绍的是one-step-ahead的DP 方法，这个就是向前看多步，然后来估计当前节点的value function。当根据计算的value function选择出当前action时，所有的backup values都被丢弃，不用存起来。

heuristic search的特点在与着眼于当前状态（current state）和当前动作（current action），以树状结构进行深层搜索，利用backup思想，来估计当前state的value function，从而选择action。

总的来说，在估计当前value function时，往前看很多步比one-step ahead 的方法要准确。

启发式对空间的搜索是基于人工规则，一般更有效率



![image-20181227112305673](/img/RL_Introduction.assets/image-20181227112305673.png)



#### Rollout Algorithms

Rollout 算法是基于Monte Carlo control的decision-time Planning方法，用到的都是从当前state出发的simulated trajectories数据。

和Monte Carlo control方法不同，rollout 算法的目标不是为了估计最优的action-value function q∗, 或者一个给定策略π的action-value function qπ，他采用Monte Carlo 来估计给定Policy （此处为rollout policy)下，每个current state的action values。根据decision-time Planning的特点，Rollout 算法立即就使用了这些action values并将其丢弃。

Rollout Algorithms的目的在于提高原有给定policy,而不是找到optimal Policy。rollout policy越好，且value estimate越精确，rollout 算法生成的policy就越好。但是value estimate的精确估计需要大量的采样，会花去大量的计算时间，因此需要平衡value estimate的精确性和规划时间。



rollout 不是学习算法，因为不记录 状态的值



#### Monte Carlo Tree Search

Monte Carlo Tree Search（MCTS) 是迄今为止，最成功的decision-time Planning算法，也是一种Rollout Algorithm，但是在Monte Carlo估计 value function的地方进行了改进，使得重点关注reward较高的trajectory的数据。对于环境模型足够简单（可以快速的进行多步仿真）的单智能体（single-agent)连续决策问题十分有效。 

MCTS的核心思想是依次关注从current state开始的可以获得较高reward的trajectories。


MCTS的工作过程如上图所示，每次迭代都包含四个步骤：

1. Selection. Starting at the root node, a tree policy based on the action values attached to the edges of the tree traverses the tree to select a leaf node.
2. Expansion. On some iterations (depending on details of the application), the tree is expanded from the selected leaf node by adding one or more child nodes reached from the selected node via unexplored actions.
3. Simulation. From the selected node, or from one of its newly-added child nodes (if any), simulation of a complete episode is run with actions selected by the rollout policy. The result is a Monte Carlo trial with actions selected first by the tree policy and beyond the tree by the rollout policy.
4. Backup. The return generated by the simulated episode is backed up to update, or to initialize,the action values attached to the edges of the tree traversed by the tree policy in this iteration of MCTS. No values are saved for the states and actions visited by the rollout policy beyond the tree. Figure 8.11 illustrates this by showing a backup from the terminal state of the simulated  trajectory directly to the state–action node in the tree where the rollout policy began (though in general, the entire return over the simulated trajectory is backed up to this state–action node).



![image-20181227180915376](/img/RL_Introduction.assets/image-20181227180915376.png)







#### Summary

Planning 需要环境模型（model)。distribution model包括所有state的转移和action 带来的reward。sample model根据概率生成一系列的state 和reward。动态规划需要distribution model来计算expected update。通常sample model比distribution model容易获得。

然后我们介绍了Planning 和learning 的密切关系，都是需要通过增程式backup更新的方式估计value function，只是Planning的作用对象是model产生的simulated experience ，而learning的作用对象为 real environment 产生的real experience。并提出了Dyna结构，将Planning 、acting、model-learning结合在一起。

本章同时也介绍了state-space planning 方法的几个不同方面的变体。 第一个方面，我们关注更新的大小，更新程度越小，Planning 方法增量计算越多，最小的更新是 one-step 采样更新，如在Dyna中。另一个方面是更新的分布，是否有侧重的进行搜索，Prioritized sweeping重点关注那些values最近有变化的state-action，这可以跳过一些无关的state-action。Real-time dynamic侧重扫描那些与current state 和Policy 有关的state-action，相比完全扫描的传统DP来说提高了算法效率。

Planning 在进行决策时也需要重点关注一些states,如在采样中会出现的state。传统heuristic search是decision-time planning的一个例子，另一个例子是rollout algorithms和 Monte Carlo Tree Search。





- Definition of return 

  Is the task episodic or continuing, discounted or undiscounted? 

- **Action values** vs. **state values** vs. **afterstate values** 

  What kind of values should be estimated? If only state values are estimated, then either a model or a separate policy (as in actor–critic methods) is required for action selection. 

- Action selection/exploration 

  How are actions selected to ensure a suitable trade-off between **exploration and exploitation**? We have considered only the simplest ways to do this: "-greedy, optimistic initialization of values, soft-max, and upper confidence bound. 

- **Synchronous** vs. **asynchronous** 

  Are the updates for all states performed simultane- ously or one by one in some order? 

- Real vs. simulated 

  Should one update based on real experience or simulated experi- ence? If both, how much of each? 

- Location of updates 

  What states or state–action pairs should be updated? Model- free methods can choose only among the states and state–action pairs actually encountered, but model-based methods can choose arbitrarily. There are many possibilities here. 

- Timing of updates 

  Should updates be done as part of selecting actions, or only after- ward? 

- Memory for updates 

  How long should updated values be retained? Should they be retained permanently, or only while computing an action selection, as in heuristic search? 





# Approximate Solution Methods

**generalization**: To make sensible decisions in such states it is necessary to generalize from previous encounters with different states that are in some sense similar to the current one.



## On-policy Prediction with Approximation

针对 on-policy prediction 问题，用function approximate 估计 state-value function的创新在于：value function 不再是表格形式，而是权重参数为w的数学表达式，即$\hat v(s,\mathbf w) \approx v_\pi(s)$。其中 $\hat v$ 可以是state的线性函数，也可以是一个多层人工神经网络（ANN），也可以是一个决策树。值得注意的是，权重$\mathbf w$的维度小于states 的数目，也就是说，一个权重可以改变多个state 的估计值（estimated value）。

Consequently, when a single state is updated, the change generalizes from that state to affect the values of many other states. Such **generalization** makes the learning potentially more powerful but also potentially more dicult to manage and understand.  泛化

将function approximate 用于强化学习可以解决部分可观测问题(partially observable problems)（即有部分state 是agent 无法获得的）
function approximate 无法用于 state的维度逐渐增加的情况，在第17章重点讨论



#### Value-function Approximation

所有的prediction方法都采用back-up value(或update target)来更新value：

- MC（蒙特卡罗）  $S_t \mapsto G_t$
- TD（0）  $S_t \mapsto R_{t+1} + \gamma \hat v(S_{t+1},\mathbf w_t) $
- n-step TD   $S_t \mapsto G_{t：t+n} $
- DP（动态规划）  $s \mapsto E_\pi[R_{t+1} + \gamma \hat v(S_{t+1},\mathbf w_t)\vert S_t = s]$



其中 $s\mapsto u$ 表示，在状态 s 下对应的**update target** 为 u。显然, value function的目标就是输入s, 输出u。如果利用表格法, 则 value 表中 s的估计值很容易趋近于 u，同时其他states的估计值不会发生变化。改变s状态的估值，其他所有状态不受影响这种方式虽然精确，对某些大规模问题效率极低，没有利用上泛化！

下面采用一种更复杂的更新方式，即在s 上的估计值更新会牵扯到其他states 的估计值更新。这种update target可以类比与机器学习中的标签，这种数据有标签的机器学习方法称为**监督学习（supervised learning)**。 when the outputs are numbers, like u, the process is often called **function approximation**。 **函数拟合** s->数值u的映射。我们传给拟合函数$s\mapsto g$   这样的训练样本。因此监督学习的方法可以用于强化学习中的value prediction 问题

但大部分监督学习方法都是假设用于训练的数据集（training set）是静态的  stationary，即大小是固定的，但对于强化学习问题，其涉及到与环境不断交互，产生新的state，这需要function approximate 的方法可以有效的从递增的训练集中在线学习。另外，强化学习的target function有时候会不断改变，如在control问题中，GPI 过程需要学习当$Policy  \pi$ 改变时的 $q_\pi$。即使Policy保持不变，由bootstrapping（步步为营）方法（如TD或DP）生成的target values也是非静态的。如果某些方法不能很轻松的处理非静态问题，就不太适合用于强化学习问题。



#### The Prediction Objective (VE)

[^_^]: 模型本身的上限就是该方法的上限了，因为模型的拟合还会有误差

目标函数

迄今为止，我们没有一个用于明确表示Prediction 问题的指标。在tabular（表格化） 问题中，因为学习得到的value function就直接等于真实的value function，因此不需要明确prediction 的质量。另外，表格化问题中的一个state的value更新不会影响其他state的value更新。但在运用function approximation 后，一个state的value 更新会影响其他很多states 的value值，这样我们不可能获得每个state 的values的真实值。假设有足够多的state ，数目多于权重数目，对一个state的准确估计意味着牺牲其他 state 的value估计精度。

要指定**state的重要程度**，使用一个分布 $\mu(s) \geq 0, \sum_s\mu(s) = 1$ 来表示对每个state value估计误差的重视程度，其中，误差为估计值 $\hat v(s,\mathbf w)$和真实值 $v_\pi(s)$的平方差。此时，prediction的目标为最小化均方根误差$\overline {VE}$: 

$$
\overline{VE} (\mathbf w)\doteq \sum_{s \in \mathcal S} \mu(s) \Big[ v_\pi(s) - \hat v(s,\mathbf w) \Big]^2
$$



Often μ(s) is chosen to be the fraction of time spent in s. Under on-policy training this is called the **on-policy distribution**.  在线策略的state概率分布  通常 μ(s)直接就选s的**出现占比** , 即任意一个step, 状态s出现的几率

on-policy distribution is a little different in that it depends on how the initial states of episodes are chosen. Let h(s) denote the probability that an episode begins in each state s, and let **$\eta(s)$ denote the number of time steps spent, on average, in state s in a single episode.** 只是在一个特定episode, 如果每个episode的steps长度都不一样, 是不是以最长的作为标准, 应该是可以的

对episode task 的on-policy而言，**h(s) 为一个轨迹里面s作为起始的概率，  $\eta(s)$ 为在轨迹里面每一步遇到s的概率, 再求和, 也可以理解为在一个K步的轨迹中, 所出现的平均步数(次数)**，利用递推公式 , $\bar s$为s前置状态, 前个状态转入s的所有可能性加起来

$$
\eta (s) = h(s) + \sum_{\bar s} \eta(\bar s) \sum_a \pi(a|\bar s) p(s|\bar s,a)
$$

则可以算出每个s的出现占比

$$
\mu(s) = \frac{\eta(s)}{\sum_{s'} \eta(s')}
$$


it is not completely clear that the VE is the right performance objective for reinforcement learning。 VE 作为评估函数是否在强化学习中是合适的还没有定论。

prediction的目标为最小化均方根误差$\overline {VE}$，即找到一个全局最优，使得存在$ \mathbf w^*$ 对所有 $\mathbf w$ 有 $\overline {VE}（ \mathbf w^*) \leq \overline {VE}(\mathbf w)$。线性函数通常可以找到这个全局最优解，但人工神经网络或决策树通常很难找到全局最优解，这些复杂的function approximation 方法通常会陷入局部最优，但对强化学习问题而言，有局部最优比无解强，这也是可以接受的。

接下来，我们将分别介绍一些简单的泛化方法，在介绍这些函数逼近之前，我们先介绍一下无论使用何种函数逼近都会涉及的求解方法：随机梯度下降（stochastic-gradient descent,SGD)和Semi-gradient方法。

 

#### Stochastic-gradient and Semi-gradient Methods

##### SGD

拟合函数 $\hat v(s,\mathbf w)$ 是 w的可导函数

随机梯度下降法（SGD），该方法常用于 function approximation中，对RL问题也十分适用。

$$
\mathbf w_{t+1} \doteq \mathbf w_t - \frac{1}{2}\alpha \nabla \Big[ v_\pi (S_t) - \hat v(S_t,\mathbf w_t) \Big] ^2  
\\ = \mathbf w_t + \alpha  \Big[ v_\pi (S_t) - \hat v(S_t,\mathbf w_t) \Big] \nabla \hat v(S_t,\mathbf w_t)
$$

**这个公式能成立的关键是 $v_\pi (S_t)$ 独立于w, 与w无关**

下面是标量函数的f 对参数w的向量展开, This derivative vector is the gradient of f with respect to w.
$$
\nabla f(\mathbf w) \doteq \Big(\frac{\partial f(\mathbf w)}{\partial w_1},...,\frac{\partial f(\mathbf w)}{\partial w_d}  \Big)^{\top }
$$

In fact, the convergence results for SGD methods assume that $\alpha$ decreases over time. If it decreases in such a way as to satisfy the standard stochastic approximation conditions (2.7), then the SGD method (9.5) is guaranteed to **converge to a local optimum**.   $\alpha$ 按条件递减, SGD保证收敛到局部最优 



下面讨论拟合的target,  $U_t$ 为估计值，不是真值v(s)的情况,  得看 $U_t$ 是不是无偏估计
$$
\mathbf w_{t+1} \doteq  \mathbf w_t + \alpha  \Big[ U_t - \hat v(S_t,\mathbf w_t) \Big] \nabla \hat v(S_t,\mathbf w_t)
$$



如果 $U_t$是一个无偏估计，即$E[U_t\vert S_t = s] = v_\pi(S_t)$，那么在$\alpha$递减情况下，$\mathbf w_t$最终会收敛到一个局部最优。 

假设这些样例中的states都是agent与环境交互产生，和 Monte Carlo算法中的采样数据一样。那么true value则为这些states value的期望值，因此Monte Carlo target  $U_t \doteq G_t$为$v_\pi(S_t)$的无偏估计，SGD算法用于 Monte carlo的算法伪代码如下图所示： 

![image-20181228222250792](/img/RL_Introduction.assets/image-20181228222250792.png)



##### <mark>semi-gradient methods</mark>

核心: 用一个bootstrap的迭代值 r+v(s') 作为 target v(s) ;   **biased 有偏估计**

使用 bootstrap estimate of $ v_\pi(S_t)$ 当作 target $U_t$.  bootstrapping target ，如 $G_{t:t+n}$ 或 DP target , 都依赖当前的参数w，所以是true value 的有偏估计, will not produce a **true gradient-descent method**.   所以这种做法只考虑了 参数w 在值估计上的作用, 而忽略了w对target的影响. 只包含了一部分的梯度 include only a part of the gradient ,  所以称为 **semi-gradient methods**.

虽然 semi-gradient 方法的收敛性, 不如 Gradient 方法可靠,  但当线性（linear function）拟合时，收敛. 有如下特点：

- 快速收敛性
- 不需要等到一个episode结束，就可以进行target value更新，可以在线学习。 continual and online

**semi-gradient TD (0)**  算法伪代码如下： 

![image-20181228222720513](/img/RL_Introduction.assets/image-20181228222720513.png)



##### **State aggregation** 状态聚合

is a simple form of generalizing function approximation in which states are grouped together, with one estimated value . 把类似的状态放一块, 然后共用一个估计值v. The value of a state is estimated as its group’s component, and when the state is updated, that component alone is updated.

不同于 minibatch SGD, 没关系

Example 9.1 1000 state的 random walk 值函数 estimating

For the state aggregation, the 1000 states were partitioned into 10 groups of 100 states each. 分成10组. 

值得注意是下面的状态分布 , 对最左边的第一个台阶, 是估值线高于真值的. 最右的最后一个台阶,低于真值.  因为最左边台阶, state100比state1出现几率高,所以值估计就偏向state100而不是state1

![image-20191224172830221](/img/RL_Introduction.assets/image-20191224172830221.png)





#### Linear Methods

线性方法是最重要的function approximation 方法之一，即将$\hat v(.,\mathbf w)$ 表征为权重向量 $\mathbf w$ 的线性函数，对每个state s 而言，有 和$\mathbf w$等维的特征向量 $\mathbf x (s) = (x_1(s),x_2(s),...,x_d(s))^T$，线性逼近的state-value函数表达如下： 

$$
\hat v(s,\mathbf w) \doteq \mathbf w^ \top \mathbf x(s) \doteq \sum_{i=1}^{d} w_ix_i(s)
$$

$$
\nabla \hat v(s, \mathbf w) = \mathbf x(s)
$$



$$
\mathbf w_{t+1} \doteq  \mathbf w_t + \alpha  \Big[ U_t - \hat v(S_t,\mathbf w_t) \Big] \mathbf  x(S_t)
$$

semi-gradient TD(0) algorithm

$$
\mathbf w_{t+1} \doteq  \mathbf w_t + \alpha  \Big( R_{t+1} + \gamma \mathbf w_t^\top\mathbf x_{t+1} -  \mathbf w_t^\top\mathbf x_{t} \Big ) \mathbf  x_t 
\\ =  \mathbf w_t + \alpha  \Big( R_{t+1}\mathbf  x_t  - \mathbf  x_t (\mathbf  x_t -\gamma \mathbf x_{t+1} ) ^\top\mathbf w_{t}   \Big )
$$

其中，$\mathbf x_t = \mathbf x(S_t)$,  一旦达到平衡点，$w_{t+1} = w_t$   ;对任意$\mathbf w_t$的下一时刻期望权重可以表达如下：

$$
\mathbb E[\mathbf w_{t+1}|\mathbf w_{t}] = \mathbf w_{t} + \alpha(\mathbf b - \mathbf A \mathbf w_{t}) 
\\ \mathbf b \doteq \mathbb E [R_{t+1} \mathbf x_t]  \in \mathbb R^d
\\ \mathbf A \doteq \mathbb E \Big[\mathbf x_t (\mathbf x_t - \gamma \mathbf x_{t+1} )^
\top \Big] \in  \mathbb R^d  \times \mathbb R^d
$$

**TD fixed point** :  上式收敛以后，linear semi-gradient TD(0)确实收敛在这个点上 

$$
\mathbf {w_{TD}} \doteq \mathbf A^{-1} \mathbf b
$$



在TD fixed point 上，证明$ \overline  {VE} $有一个上限： 

$$
\overline {VE}(\mathbf w_{TD}) \leq \frac{1}{1-\gamma} \min_{\mathbf w} \overline {VE}(
\mathbf w)
$$

TD 方法生成的$\overline{VE}$小于$\frac{1}{1-\gamma}$倍最小可能的$\overline  {VE}$的最小值（MC方法所能达到的极限）。 $\gamma$ 趋近于1 ，所以该倍数会很大。  

在on-policy分布下，MC、semi-gradient DP、semi-gradient Sarsa(0)均收敛于TD fixed point，但如果采用其他更新分布，则采用函数逼近的bootstrapping方法可能趋于无穷大，在第11章中重点讨论。



![image-20190109161139698](/img/RL_Introduction.assets/image-20190109161139698.png)







#### Feature Construction for Linear Methods

线性方法不仅收敛性好，且十分计算高效。这一切都取决于如何选择states的特征，本节将讨论特征构造方法。好的特征会告诉强化学习一些先验知识，如在处理几何问题时，所构造的特征可能是几何体的形状、颜色、大小或函数。假如研究一个移动机器人，则需要的特征可能是位置、角度、电池剩余等。

线性方法的局限在于，我们所考虑到的特征之间是彼此独立的，即没有考虑到两个特征间的联系。但对一些特征，我们需要考虑他们之间的联系，如开车转弯时，车辆速度和方向盘转角之间的关系，不能独立的说方向盘转角越大越好或者越小越好，当速度较低方向盘转角可以大一些，但车辆速度较高时大转角会很危险。接下来我们讨论几种解决特征间耦合的方案。



##### Polynomials 多项式

很多问题的states都是用数字来表示，如打高尔夫球时的states 为位置和速度，汽车租赁问题中每个停车场的汽车数目。这类问题中，将函数逼近用于强化学习问题的思想与插值或回归问题类似。因此，很多用于插值或回归的方法也可以用于强化学习。这里我们将讨论多项式方法在强化学习中的运用，虽然效果不佳，但便于我们理解。

举个例子，假设有一个强化学习问题，他的state 是一个二维的，即s = (s_1,s_2), 可以选择特征为$\mathbf x(s) = (s_1,s_2)^T$，但这种特征表达无法体现两个特征值s_1,s_2间的交互关系。可以通过构造一个高维的多项式特征向量来解决该问题： 

$\mathbf x(s) = (s_1,s_2)^\top $ =>     $\mathbf x(s) = (1,s_1,s_2,s_1s_2,s_1^2,s_2^2,s_1^2s_2,s_1s_2^2,s_1^2s_2^2)^\top $

##### Fourier Basis

另一种线性函数逼近方法基于古老的傅里叶级数，用sin和cos基函数（特征量）的加权和来表示周期函数（f(x) = f(x +\tau))。当一个需要估计的函数已知时，傅里叶级数和傅里叶变换应用十分广泛，基函数的权重很容易给出，只要有足够多的基函数，任何函数都能被逼近。在强化学习中，需要被估计的函数通常是未知的，但傅里叶基函数仍然可以很好的应用。

首先考虑一维问题，通常傅里叶级数用周期为 \tau的sin和cos函数的线性组合来表示周期为\tau的一维度周期函数。如果要研究如何在有界区间上逼近非周期函数，可以通过改变傅里叶原理中的 \tau来设置区间长度，将非周期函数转化为sin和cos函数线性组合的周期函数在该有限区间的一个周期。

另外，如果将\tau设为兴趣区间的两倍，那么只需要在[0,\frac{\tau}{2}]区间上进行函数逼近，可以只使用cos函数或只使用sin函数。如果只使用cos函数，表示任意偶函数，当有足够多的cos基函数时，可以表示[0,\frac{\tau}{2}]区间内的任意函数。sin作为奇函数，其线性组合往往是奇函数，在原点处通常不连续，所以一般在[0,\frac{\tau}{2}]区间下考虑仅使用cos函数的线性组合，但也不是绝对的，在某些情况下，用sin和cos的组合更好。

##### Coarse Coding

##### Tile Coding

##### Radial Basis Functions





#### Selecting Step-Size Parameters Manually

前两节主要介绍了一种基于线性函数的函数逼近方法，以及特征选取问题，本节将主要探讨如何手动选取学习率的问题。

很多SGD方法需要选择合适的学习率\alpha，其中一些方法可以自动选择，但大部分还是人工选取。通常选取的 t 时刻的学习率为\alpha_t = \frac{1}{t}，在MC中适用，但在TD等非静态问题或函数逼近问题中均不适用。对与线性方法，循环最小二乘（recursive least-squares)可以用于选择最优学习率，可以扩展到TD中，在第八小节中会介绍，但需要O(d^2)的计算复杂度，对高维问题不太适用。

为了直观的了解如何人工选择学习率，我们回到表格化问题（tabular case) 中,当\alpha = 1时，会彻底消除采样误差，但我们通常希望学习过程缓慢一些。当\alpha =\frac{1}{10}时，会将10次经验的value的平均值作为target value来优化，当希望运用100次经验数据时，\alpha =\frac{1}{100}。

但对函数逼近而言，没有一个对某个state经验数目的明确概念。但有一个相似性准则，假设学习需要\tau次包含相同的特征向量的经验，可以构造学习率如下



其中，\mathbf x的用同一个分布随机生成的特征向量，作为SGD的输入。如果\mathbf x长度上变化不大，则\mathbf x^T\mathbf x为一个常量。



#### Nonlinear Function Approximation: Artificial Neural Networks





#### Least-Squares TD   LSTD

在本章中讨论过的所有方法每个time step(学习率）都需要与参数成比例的计算。越计算表现越好。但本节，我们提出一种更好的线性函数逼近方法。 


直接计算 TD fix point

其中  $\epsilon$ 是个很小的量，保证矩阵可逆

$$
\mathbf  {\hat A_t} \doteq  \sum_{k=0}^{t-1} \mathbf x_k  (\mathbf x_k - \gamma \mathbf x_{k+1})^\top + \epsilon \mathbf I
\\ \mathbf  {\hat b_t} \doteq  \sum_{k=0}^{t-1}  R_{k+1} \mathbf x_k
\\  \mathbf w_t \doteq  \mathbf  {\hat A_t^{-1}}\mathbf  {\hat b_t}
$$

在该算法中，更多的数据被用上啦，但算法复杂度也增加了，semi-gradient TD(0)单次更新的计算复杂度为 O(d).

LSTD有多复杂？复杂性随着t不断增长。但可以通过增量迭代的方式减少计算复杂度。使得计算复杂度变为 $O(d^2)$。 否则一般求逆矩阵的复杂度是  $O(d^3)$

![image-20181229221545751](/img/RL_Introduction.assets/image-20181229221545751.png)

该算法，w的赋值，只有最后一步才是解，中间的记录无所谓

当然， O(d^2)相对 O(d)挺大，但是否采用LSTD取决于d的大小，学习效率的重要程度和系统其他部分的复杂程度。LSTD不需要设置学习率意味着他永远不会遗忘，但对于target policy 改变的强化学习问题或GPI 过程而言是有问题的。在研究control 问题中，LSTD通常和一些算法结合来帮助遗忘。



#### Memory-based Function Approximation

迄今我们讨论了参数化(parametric) 的逼近value function的方法，在该方法中，学习算法通过调整参数来估计整个 state spaces上的value function，每次更新，学习算法以减小逼近误差为目的，使用一组训练样例 $s\mapsto g$ 来改变参数。更新后，抛弃该训练样例，当需要估计一个state value时（该state 被称为 query state)，采用上一时刻学习算法提供的参数来求解该state下的函数。

Memory-based 函数逼近与参数化函数逼近完全不同，他将所经过的训练样例存储在记忆库中，但不进行参数更新。当需要估计query state的value时，从记忆库中提取一系列训练样例，然后进行估计。这种方式也称作**lazy learning**，因为只到系统要求提供输出结果时才进行计算。

Memory-based function是一种典型的非参数化方法（nonparametric)。逼近函数不局限于事先规定参数类型的函数形式（如线性或多项式），而是取决于训练样例本身。

有很多 memory-based 方法可以按照如何选择需要存储的训练样例，这些记忆如何被用于回应query state进行分类。这里，我们重点关注 local-learning方法，用于估计当前query state周围局部的value function。这类方法从记忆库中提取与当前quary state相关性最强的一些state，相关性常常用距离表征。当计算出query state value时，丢弃这个局部逼近结果。

最简单的memory-based 方法是**最近邻算法  nearest neighbor method**，可以很快从记忆库中找到query state中的最近state，并返回最近state的value作为query state 的估计value。略加变化，通过返回最近几个训练样例的value 的加权平均作为query state 的估计value（weighted average)，权重随距离递减。Locally weighted regression是一个类似的方法，结合了参数化函数逼近方法，以最小化误差为目标。

Memory-based方法相对于参数化方法的优点在于：

- 不局限于事先定义的函数形态，在数据多的时候可以提高精确度
- Memory-based local逼近方法，对强化学习十分适用，第8章中说明了trajectory sampling 对强化学习十分重要，memory-based local 方法关注于real 或 simulated trajectories经历过的states（或state-action pairs)的局部邻近区域。不需要考虑全局，因为有些state space中的区域不可能到达。
- Memory-based 方法需要agent的经验对current state邻近区域内的value估计立即回应。但参数化方法则需要通过增程式调整参数以进行全局逼近。

 

#### Kernel-based Function Approximation

Memory-based 方法，如weighted average和 locally weighted regression，需要分配权重。用于分配权重的函数被叫做核函数（kernel function）。在weighted average和 locally weighted regression方法中，核函数为state间的距离。一般来说，核函数不仅仅局限于距离函数，只要可以通过某种方式表示两个state间的相似性即可。在本节中，核函数k(s,s')表示s'对query state s的影响程度。

kernel regression 是一种memory-based方法，计算所有记忆库中训练样例target value的核加权平均，D是存储的训练样例集，g(s')是s’的target value,则state s的value估计值为 







#### Looking Deeper at On-policy Learning: Interest and Emphasis

本章中所讨论的方法对每个state的侧重都一样，但有些时候，需要有所侧重。如在离散episode 问题中，我们可能对episode早期的states的精度要求更高；当学习一个action-value函数时，greedy action比别的action更重要。函数逼近的资源是有限的，如果更有目的性的去关注一些量可以获得更好的性能提升。

对所有state一视同仁的原因在于我们根据on-policy分布来进行更新，该分布对semi-gradient 方法有很强的理论支持。如果不采用on-policy分布来生成数据，我们需要介绍一些新的概念。

首先，我们介绍一个非负标量方法，假设有一个叫做interest的随机变量 I_t，表示我们对当前state的感兴趣程度，如果我们完全没兴趣，则取值为0 ，如果我们充满兴趣则取值为1，一般取值都在0和1之间。兴趣值可以按照任意因果关系设置；例如可以取决于时间或者t时刻学到的参数。\overline {VE}中的分布 \mu定义为跟着target policy 出现 state 的分布，乘以inerest。另外，我们介绍一个非负随机标量变量 ，emphasis M_t,该标量乘以学习率，用于加强或减弱t时刻的学习，n-step 学习方法的权重更新表达式修改为：



#### Summary

要在人工智能或大部分工程问题中使用强化学习，就必须考虑泛化的问题。为了解决该问题，所有监督学习的函数逼近方法均可以用于强化学习的value估计。

也许最合适的监督学习方法是参数化函数逼近方法，这些方法中policy被权重参数\mathbf w 参数化。我们定义均方根误差\overline{VE}来表示on-policy分布\mu下的估计value $v_{\pi_{\mathbf w}}(s)$误差。\overline{VE}给我们指明了value-function approximation的优化方向。

为了找到一个好的权重向量，最常用的方法是随机梯度下降法（SGD）。本章中我们关注与 on-policy下的fixed policy,即Policy evaluation 或 prediction问题。针对，boostrapping方法（DP, TD(0),n-step TD等)，结合SGD,衍生出了semi-gradient方法。在这些boostrapping方法中，权重向量在更新目标中出现，所以不能用于计算梯度，也就是说SGD的结果不完全适用于这些方法。

另外，采用线性函数逼近时，semi-gradient 方法会得到很好的结果。线性函数逼近是函数逼近方法中一种易于理解的简单方法，其重点在于特征选择。可以通过多项式方法构建特征，但在在线学习中通常不考虑这种方式。根据傅里叶原理或有重叠区域的coarse coding来选择特征更加有效。Tile coding是一种计算高效的coarse coding方法。径向基函数（RBF)在一维或二维问题中可以提供一个平滑的输出。LSTD是一种高效利用数据的线性TD预测方法。非线性函数逼近主要介绍了人工神经网络，主要采用反向传播加SGD对参数进行更新，这就是有名的deep reinforcement learning（深度强化学习）。





## On-policy Control with Approximation

本章我们关注on-policy control 问题，这里采用参数化方法逼近action-value函数 $\hat q(s,a,\mathbf w) \approx q(s,a)$，其中，$\mathbf w$为权重向量。在11章中会讨论off-policy方法。本章介绍了semi-gradient Sarsa算法，是对上一章中介绍的semi-gradient TD(0)的一种扩展，将其用于逼近action value, 并用于 on-policy control。在episodic 任务中，这种扩展是十分直观的，但对于连续问题来说，我们需要考虑如何将discount (折扣系数）用于定义optimal policy。值得注意的是，在对连续任务进行函数逼近时，我们必须放弃discount ，而改用一个新的形式 ” average reward”和一个“differential” value function进行表示。

首先，针对episodic任务，我们将上一章用于state value 的函数逼近思想扩展到action value上，然后我们将这些思想扩展到 on-policy GPI过程中，用$\epsilon-greedy$来选择action，最后针对连续任务，对包含differential value的average-reward运用上述思想。



#### Episodic Semi-gradient Control

将第9章中的semi-gradient prediction 方法扩展到control问题中。这里，approximate action-value $\hat q \approx q_\pi$，是权重向量 $\mathbf w$ 的函数。在第9章中逼近state-value时，所采用的训练样例为 $S_t \mapsto U_t$，本章中所采用的训练样例为$S_t,A_t \mapsto U_t$，update target $U_t$可以是$q_\pi(S_t,A_t)$的任何逼近，无论是由MC还是n-step Sarsa获得。

对action-value prediction 的梯度下降如下：

$$
\mathbf w_{t+1} \doteq \mathbf w_t + \alpha \Big[U_t - \hat q(S_t,A_t,\mathbf w_t) \Big] \nabla \hat q(S_t,A_t,\mathbf w_t)
$$

the update for the one-step Sarsa method is    **episodic semi-gradient one-step Sarsa**

$$
\mathbf w_{t+1} \doteq \mathbf w_t + \alpha \Big[R_{t+1} +\gamma \hat q(S_{t+1},A_{t+1},\mathbf w_t) - \hat q(S_t,A_t,\mathbf w_t) \Big] \nabla \hat q(S_t,A_t,\mathbf w_t)
$$

在policy不变的前提下，该方法和TD(0)的有同样的收敛边界条件



为了构造control 问题，我们需要结合action-value prediction，policy improvement 和 action selection。目前还没有探索出十分适合连续动作空间或动作空间过大的问题的方法。另一方面，如果动作空间离散且不大的情况下，我们可以用之前章节中提到的方法。

因此，对当前state S_t下，可能执行的action a，均计算出 $\hat q(S_t,a,\mathbf w_t)$，然后选择使得 \hat q(S_t,a,\mathbf w_t)最大的动作，即greedy action $A_t^* = argmax_a\hat q(S_t,a,\mathbf w_t)$。Policy improvement，需要考虑探索问题，需要将被估计的policy变为 greedy policy 的 soft approximation，即 $\epsilon-greedy$ policy。同样，用\epsilon-greedy policy来选择action。算法伪代码如下： 

![image-20181231114947227](/img/RL_Introduction.assets/image-20181231114947227.png)





#### n-step Semi-gradient Sarsa

通过令update target $U_t = G_{t:t+n}$,可以得到n-step semi-gradient Sarsa,其中 ： 

$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1}R_{t+n}+ \gamma^n \hat q(S_{t+n},A_{t+n},\mathbf w_{t+n-1} )
$$

此时的权重向量更新公式为：

$$
\mathbf w_{t+n} \doteq \mathbf w_{t+n-1} + \alpha \Big[G_{t:t+n}  - \hat q(S_t,A_t,\mathbf w_{t+n-1}) \Big] \nabla \hat q(S_t,A_t,\mathbf w_{t+n-1})
$$


![image-20181231120420507](/img/RL_Introduction.assets/image-20181231120420507.png)



#### Average Reward: A New Problem Setting for Continuing Tasks

为了构建马尔科夫过程（MDP）的求解目标，这里我们介绍除了 **episodic setting** 和**discounted setting**的第三种 setting—— **average reward setting**。

和discounted setting类似， average reward setting也是用于agent和环境不断交互，永不停止，没有开始或者没有结束的连续问题中。与discounted setting不同的是：没有折扣系数，agent对即刻reward 和延迟reward 的重视程度一致。 average reward setting主要出现在传统动态规划中，但很少出现在强化学习中。下一小节，我们将详细讨论对于函数逼近问题，discounted setting存在的问题。因此在解决连续任务下 函数逼近时，用 average reward setting代替 discounted setting。





因为是连续问题，所以 所有的R加起来是无穷大， 除以步数t，可以算出每步的均值，作为该策略的优劣标准

因为是连续问题，所以按照某个策略，最终会进入一定的收敛状态，在该稳态一步的R的均值可以视为 该策略本身的优劣

$r(\pi)$  表示到该策略pi最终能获得的平均r

在 average reward setting中，policy $\pi$ 的优劣由reward的平均率表示：

$$
r(\pi) \doteq \lim_{h\to \infty} \frac{1}{h} \sum_{t=1}^{h}  \mathbb E[R_t | S_0,A_{0:t-1} \sim \pi] 
\\ =  \lim_{t\to \infty}  \mathbb E[R_t | S_0,A_{0:t-1} \sim \pi] 
\\ = \sum_s \mu_\pi(s) \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)r
$$

其中，$\mu_\pi$是稳态分布steady state distribution，$\mu_\pi = lim_{t\to \infty} Pr\{S_t = s \vert A_{0:t-1} \sim \pi \}$，假设$\mu_\pi$对任意$\pi$均存在，且独立于 S_0。这个假设对MDP来说被称为ergodicity (遍历）。这意味着MDP从何处开始或任何早期agent的决定所产生的影响都是暂时的。长期来说，一个state 下的期望值仅仅取决于policy和MDP状态转移概率。遍历条件是上式有极值的重要保证。

对undiscounted连续问题来说，很难区分哪种优化方法好，但根据每个time step的average reward规划出policy比较实用。也就是通过求解$r(\pi)$，计算出使得$r(\pi)$最大的\pi作为optimal Policy。

稳态分布steady state distribution是一种特殊分布，如果根据policy $\pi$选择了一个action，分布仍然不改变，因此有： 

$$
\sum_s \mu_\pi(s) \sum_a \pi(a|s) p(s'|s,a) = \mu_\pi(s')
$$

 the differential return

$$
G_t \doteq R_{t+1} - r(\pi) + R_{t+2}-  r(\pi) + \dots
$$




![image-20190101202612610](/img/RL_Introduction.assets/image-20190101202612610.png)





#### Deprecating the Discounted Setting



#### Differential Semi-gradient n-step Sarsa



#### Summary

本章我们将第九章中参数化函数逼近和semi-gradient decent 扩展到control 问题在中。首先，对episodic问题进行了简单扩展，然后对连续问题，先介绍了average-reward setting 和 differential value function。然后说明了discounted setting 不适用于连续问题中的函数逼近的原因。

average reward中的 differential value function也有Bellman等式、TD error。我们用他们构造了differential version of semi-gradient n-step Sarsa。









## Eligibility Traces 资格迹

在实际应用中，一个动作的成功或失败需要一段时间以后才能知道，所以强化信号往往是一个动作序列中很早以前

某个动作所引起的响应，这种情况称为延时强化学习问题。

为解决这种长时间延时强化学习的信度分配问题，强化学习系统的预报能力就显得很重要，所以有很好预报能力的强化学习算法会十分有用。

资格迹是所发生事件的一种临时记录。例如，状态的访问及动作的选取，资格迹使得存储参数与事件的联系作为学习的一种资格。当误差产生时，只有那些有资格的状态和动作赋予信度，资格迹有助于将事件和训练信息联系起来。作为本身，资格迹是时间信度分配的一个基本机制。



Eligibility Traces是强化学习的基本原理之一。例如TD(λ)算法，(λ)表示eligibility traces的使用情况。几乎所有TD方法，如 Q-Learning或Sarsa，都可以和eligibility traces结合起来生成更高效通用的方法。

Eligibility Traces可以用于泛化TD和MC（蒙特卡罗）方法。当用eligibility traces增强TD后，会产生一系列方法，包括两个极端的方法：MC（λ=1）和 one step TD（λ=0） 位于两者中间的方法比这两方法好。Eligibility Traces也提供了将MC方法用于在线连续问题的方式。

第7章中，我们介绍了一种连接TD和MC方法的n-step TD。Eligibility Traces的连接方式提供了一种更为优雅的算法机制，有很大的计算优势。该机制包含一个**短期记忆向量**，Eligibility Traces $\mathbf z_t∈ \mathbb R^d$，对应一个**长期权重向量** $\mathbf w_t∈ \mathbb R^d$。基本思想是$\mathbf w_t$ 中参与生成estimated value的元素对应的 $\mathbf z_t$中的元素波动并逐渐衰减。如果在trace衰减到0之前，TD error不为零时，则该$\mathbf w_t$中参与生成estimated value的元素会一直持续学习。trace-decay 参数 λ∈[0,1]，表示trace的衰退速度。

与n-step方法相比：

- Eligibility Traces方法只需要存储一个trace vector，而不是n个最新的feature vectors。

- 学习过程是连续且时间均匀的，不会有延迟。不需要在episode结束时刻立即进行所有运算。

- 学习可以立刻影响行为，不需要n step后才延迟生效。


Eligibility Traces 表明学经过的states。这种方式称为 backward views。

本章首先，从 state value和 prediction 问题说起，再将其扩展到action value和control 问题中。先讨论on-policy问题，再扩展到off-policy问题。重点讨论 linear function approximation。



等比求和:  $S = \frac{a_1(1-q^n)}{1-q}$

#### The λ-return

$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^{n} \hat v (S_{t+n}, \mathbf w_{t+n-1}) , \  0 \leq t \leq T -n
$$

对tabular learning而言，该式可作为 approximate SGD learning update 的update target。这个结论对n-step returns的平均值也成立，只要权重和为1，如$\frac{1}{2}G_{t:t+2}+\frac{1}{2}G_{t:t+4}$，这种组合return的方式和 n-step return 一样通过减小 TD error来更新，因此可以保证算法收敛。 **Averaging方式可以衍生出一系列新算法，如将 one-step 和infinite-step return平均获得 TD和MC之间的另一种算法**，通过将experience-based update 和 DP update 取平均值，可以将experience-based 和model-based 方法结合。

nstep是通过调整n，集合了后面n个R来更快的迭代； $TD(\lambda)$ 是通过整合当前若干个R以及若干个估值v来更快迭代；超参的调整还是要看具体业务

这种将简单元素取平均的更新方式叫做**compound update**。$\frac{1}{2}G_{t:t+2}+\frac{1}{2}G_{t:t+4}$就是一种compound update

 <img src="/img/RL_Introduction.assets/image-20190105164014791.png" width="10%">

$$
G_t^{\lambda} \doteq (1- \lambda) \sum_{n=1}^\infty \lambda^{n-1} G_{t:t+n}
$$

第一个TD的系数是 1- λ  ，第二个 (1-λ)λ ， 依次递推 ，λ = 0 one-step TD，λ = 1， MC

$$
G_{t:t+n} \doteq  R_{t+1} + \gamma R_{t+2}  +\dots+ \gamma^{n-1}R_{t+n}+ \gamma^{n}V_{t+n-1}(S_{t+n}) \ 关键是利用了 t+n-1 时刻的评估函数
\\ (1 - \lambda) \sum_{n=T-t}^\infty \lambda ^{n-1}G_t =   \lambda^{T-t-1}G_t \ 错位相减
$$

The backup digram for TD(λ). If λ = 0, then the overall update reduces to its first component, the one-step TD update, whereas if λ = 1, then the overall update reduces to its last component, the Monte Carlo update.

<img src="/img/RL_Introduction.assets/image-20190102203843783.png" width="50%">

<img src="/img/RL_Introduction.assets/image-20190102204758600.png" width="70%">

​	
$$
G_t^{\lambda} \doteq (1- \lambda) \sum_{n=1}^{T-t-1} \lambda^{n-1} G_{t:t+n} + \lambda^{T-t-1} G_t
$$



##### offline λ-return algorithm

现在定义以 λ-return 为update target的第一个算法——**off-line λ-return algorithm**。在episode中，权重向量不变，<u>episode结束时，整个off-line update 序列按照 semi-gradient rule更新</u>： 

$$
\mathbf w_{t+1} \doteq \mathbf w_t + \alpha \Big[ G_t^\lambda - \hat v(S_t,\mathbf w_t) \Big] \nabla \hat v(S_t, \mathbf w_t) , \ t=0,\dots,T-1
$$

λ-return 使得MC和one-step TD可以平顺过渡，以19-state random walk task(第七章例子）为例，和n-step TD 结果进行比较： 整体结果差不多，两个任务中，都是中间参数表现比较好，即n-step方法中n取中间值，λ-return算法中 λ 取中间值。



目前我们讨论的方法都叫做 forward views算法。这什么意思呢？想向我们是一个拿着望远镜的小人，在每一个state处都可以看到未来一些时刻的事情。 在t时刻需要之后所有的R

![image-20190106220054551](/img/RL_Introduction.assets/image-20190106220054551.png)

对于episodic问题，每个$G_{t:t+n}$就是一个分量，在一个episode中，从s_0开始 用 T 个分量  ，然后依次减少分量

s_t的时候看 t+1 -> T ，后面分量个数依次减少





#### TD(λ)

TD(λ) 是强化学习中最古老且应用最广泛的算法，也是第一个阐述了forward-view 和backward-view间联系的算法（It was the first algorithm for which a formal relationship was shown between a more theoretical forward view and a more computationally congenial backward view using eligibility traces.）本节我们将介绍如何逼近上一小节提到的 off-line λ-return algorithm。

TD(λ)对off-line (λ)-return算法进行了三点提升：

- 单步更新权重向量，无需等到episode结束，学习更快
- 计算在时间上均匀分布
- 不仅可用于episode问题，还适用于连续问题



> 注意， $\gamma$ 表示G的折扣，同之前；  $\lambda$ 在return里面是各种 $G_{t:t+n}$的分量 ,$\sum =1$ ， 在资格迹里面是 衰减系数

本节介绍 <u>semi-gradient version of TD(λ)</u>  with function approximation。在函数逼近时，eligiblility trace zt∈Rd和权重向量wt的元素个数相同，但权重向量有长期记忆，存在时间和系统等长，<u>zt是短期记忆，存在时间小于一个episode长度</u>。eligiblility trace存在于学习过程中，their only consequence is that they affect the weight vector, and then the weight vector determines the estimated value。看公式

在TD(λ) 中，eligiblility trace vector 在episode开始时初始化为0，每个time step 由value梯度更新，且以γλ衰减

**accumulating trace**

$$
\mathbf z_{-1} \doteq 0 
\\ \mathbf z_{t} \doteq \gamma \lambda \mathbf z_{t-1} + \nabla \hat v(S_t,\mathbf w_t),\ 0 \leq t \leq T
$$

$γ$是discount rate,  $\lambda$  是  trace-decay parameter;  eligiblility trace一直追踪<u>那些对recent state valuation有贡献的权重向量</u>  <u>**过去梯度的累加，带衰减**</u>。这里 recent 是以$γλ$界定的。(在线性函数逼近中，$∇\hat v (S_t,\mathbf w_t)$ 梯度就是特征向量$\mathbf x_t$，于是eligiblility trace vector只是所有过去输入的和，带衰减)。trace表明了当reinforcing event发生时的权重向量元素的资格，对 reinforcing event我们关注的是one-step TD errors(也就是梯度)。state-value prediction的TD error为 

$$
\delta_t \doteq R_{t+1} + \gamma \hat v(S_{t+1},\mathbf w_t) - \hat v(S_{t},\mathbf w_t)
$$

在TD(λ)中，权重更新公式为    跟semi-gradient TD (0)更新公式一样，只是梯度换成了z

$$
\mathbf w_{t+1} \doteq \mathbf w_t + \alpha \delta_t \mathbf z_t
$$


> 这里没有用到$G_{t:t+n}$ ; 因为是TD算法，不会等到episode结束，只能向过去看，找出过去的累积对现在状态影响最大的分量



![image-20190102220235412](/img/RL_Introduction.assets/image-20190102220235412.png)



TD(λ)是backward in time的，即每个时刻，我们由当前TD error，按照对当前eligiblility trace的贡献程度，将其分配给之前的states。可以想象我们是一个拿着话筒的小人。沿着state 流计算 TD error，然后对之前经过的state 回喊。  <u>往回看， 关注的是z值，而不是R</u> , 其实是 x的值

![image-20190106221848712](/img/RL_Introduction.assets/image-20190106221848712.png)

为了更好的理解backward view，从λ入手。当λ=0时， t 时刻的trace恰好是St对应的 gradient， TD(λ)更新变为 one-step semi-gradient TD updaet TD(0) 。 TD(0)只改变了当前state的TD error。当λ<1时，λ越大，所涉及的之前的states越多，但越远的state变化越小，因为对应的eligiblility trace越小、表示早期的state对当前TD error 的置信度较低。

当λ=1时，早期state的置信度每步只衰减γ，刚刚好可以生成MC算法的行为。例如 记TD error 为δt，包含一个undiscounted 的reward Rt+1。从该时刻向前追溯k 个step要打γk折，当λ=1,γ=1时 eligiblility trace 不衰减，和MC方法在undiscounted episode task表现一致，当λ=1，记作TD(1)。

TD(1)是MC方法的一种通用表达，扩展了应用范围，第5章中提到的MC方法局限于episode task，TD(1)可以用在  discounted continuing task中。另外，TD(1)也可以迭代和在线学习，MC方法的一个缺点在于只能在一个episode结束时学习。

如果MC方法在episode结束前采取了一个bad action（产生了很不好的reward），那么MC继续选择这个action的趋势不会被减弱。但on-line TD(1)，从incomplet ongoing episode 中学习n-step TD ,如果有特别好或者坏的情况发生，TD(1) control menthods 可以立即学习并在同一个episode中更改行为。



对任意λ，如果选择合适的α，那么两个算法性能没有什么区别，但如果α选择过大，λ-return algorithm只比TD(λ)坏一点。

当step-size 参数α逐渐衰减时，在linear TD(λ)被证明在on-policy case中是收敛的。满足：

$$
\overline {VE} (\mathbf w_\infty) \leq \frac{1- \gamma \lambda}{1- \gamma} \min_\mathbf w \overline {VE} (\mathbf w)
$$


#### n-step Truncated λ-return Methods

off-line λ-return algorithm 因为用了一个λ-return ，在episode结束前是一个未知量。n-step returns提供了一个方法，用estimated values代替抛弃的reward。

我们定义 truncate λ-return如下

$$
G_{t:h}^\lambda \doteq (1-\lambda)\sum_{n=1}^{h-t-1} \lambda^{n-1}G_{t:t+n} + \lambda^{h-t-1}G_{t:h}
$$

和 λ-return的公式对比，h和终止时刻 T的作用一致。但λ-return的权重是根据 true return获得。



<img src="/img/RL_Introduction.assets/image-20190103211208763.png" width="70%">



Ecient implementation ：k step $\lambda$-return

$$
G_{t:t+k}^\lambda = \hat v(S_t, \mathbf w_{t-1}) + \sum_{i=t}^{t+k-1}(\gamma \lambda)^{i-t} \delta_i' 
\\
\delta_i' \doteq R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf w_t) - \hat v(S_t, \mathbf w_{t-1})
$$




#### Redoing Updates: The Online λ-return Algorithm

选择Truncated TD(λ)中的truncation 参数n涉及到一种平衡，n应该足够大使得该方法和 off-line λ-return算法接近，但也应该足够小，使得算法能够快速更新

The idea is that, on each time step as you gather a new increment of data, you go back and redo all the updates since the beginning of the current episode. The new updates will be better than the ones you previously made


$$
h-1:\quad \mathbf w_1^1 \doteq \mathbf w_0^1 + \alpha[G_{0:1}^\lambda - \hat v(S_0,\mathbf w_0^1)] \nabla\hat v(S_0,\mathbf w_0^1)
\\ h-2:\quad \mathbf w_1^2 \doteq \mathbf w_0^2 + \alpha[G_{0:2}^\lambda - \hat v(S_0,\mathbf w_0^2)] \nabla\hat v(S_0,\mathbf w_0^2)
\\ \quad \quad \quad \quad \mathbf w_2^2 \doteq \mathbf w_1^2 + \alpha[G_{1:2}^\lambda - \hat v(S_1,\mathbf w_1^2)] \nabla\hat v(S_1,\mathbf w_1^2)
\\ h-3:\quad \mathbf w_1^3 \doteq \mathbf w_0^3 + \alpha[G_{0:3}^\lambda - \hat v(S_0,\mathbf w_0^3)] \nabla\hat v(S_0,\mathbf w_0^3)
\\ \quad \quad \quad \quad \mathbf w_2^3 \doteq \mathbf w_1^3 + \alpha[G_{1:3}^\lambda - \hat v(S_1,\mathbf w_1^3)] \nabla\hat v(S_1,\mathbf w_1^3)
\\ \quad \quad \quad \quad \mathbf w_3^3 \doteq \mathbf w_2^3 + \alpha[G_{2:3}^\lambda - \hat v(S_2,\mathbf w_2^3)] \nabla\hat v(S_2,\mathbf w_2^3)
$$

一般形式

$$
\mathbf w_{t+1}^h \doteq \mathbf w_t^h + \alpha[G_{t:h}^\lambda - \hat v(S_t,\mathbf w_t^h)] \nabla\hat v(S_t,\mathbf w_t^h) , \quad 0\leq t < h \leq T
$$


online λ-return algorithm是完全在线算法，在episode中，每步t 都使用t 时刻已有数据产生一个新的权重向量wt,主要缺点在于计算复杂度高，每次都要从头开始计算整个 episode，比 off- line λ-return方法的计算复杂度高，off-line λ-return方法仅在termination 处从头开始计算，在其他位置不更新。on-line方法效果比off-line好，有两点原因：

- during the episode  online makes  update 
- at the end of the episode, w used in bootstrapping 即评估G的时候，也被迭代了很多次

计算复杂度高



#### True Online TD($\lambda$)

on-line λ-return方法是目前性能最好的TD算法，但计算复杂度太高。有没有办法将这种forward-view算法用eligibility trace转换为有效的backward-view算法呢？答案是肯定的。这就是true on-line TD(λ)。



online λ-return algorithm can be arranged in a triangle:

$$
\begin{aligned}
& \mathbf w_0^0
\\&  \mathbf w_0^1 \quad \mathbf w_1^1 \quad 
\\&  \mathbf w_0^2 \quad \mathbf w_1^2 \quad \mathbf w_2^2
\\&  \   \vdots \quad \quad\   \vdots \quad \quad\   \vdots \quad    \ddots
\\&  \mathbf w_0^T \quad \mathbf w_1^T \quad \mathbf w_2^T \  \cdots \ \mathbf w_T^T 
\end{aligned}
$$


One row of this triangle is produced on each time step.  对角线上的w序列是真正需要的，在bootstrapping in the n-step returns of the updates 中发挥作用。 下面就是要找一个高效的计算该序列的算法。对线性拟合的情况，可以得到一个算法：

$$
\mathbf w_{t+1} \doteq \mathbf w_{t} + \alpha\delta_t\mathbf z_{t} +
\alpha(\mathbf w_{t}^\top \mathbf x_{t} - \mathbf w_{t-1}^\top \mathbf x_{t} )(\mathbf z_{t} - \mathbf x_{t})
$$

**dutch trace**

$$
\mathbf z_{t} \doteq \gamma\lambda\mathbf z_{t-1} + (1-\alpha\gamma\lambda\mathbf z_{t-1}^\top \mathbf x_{t})\mathbf x_{t}
$$

该算法可以产生和on-line λ-return 算法一样的权重向量，该方法的存储要求和 TD（λ）一样，但计算提速50%



**replacing trace**

$$
z_{i,t} \doteq \begin{cases} 
1 & \quad \text{if } x_{i,t}= 1\\	
\gamma \lambda z_{i,t-1} & \quad \text{otherwise }  
\end{cases}
$$

现在我们把replacing traces看作是dutch traces的粗糙逼近。dutch trace通常比replacing trace表现的更好而且有更清楚的理论解释。accumulating traces依然保留在那些无法使用dutch trace的非线性函数逼近的情形下。





#### Dutch Traces in Monte Carlo Learning

尽管资格迹紧密的和TD学习算法结合在一起，但实际上他们和TD并没有关系。实际上资格迹甚至可以直接用在MC算法中。之前展示了forward view的线性MC算法可以用来推导出一个等价而且计算更简单的backward-view形式的使用dutch traces的算法。这是本书中唯一显示说明forward-view 和 backward-view 等价的例子。

线性的MC 算法

$$
\mathbf w_{t+1} \doteq \mathbf w_{t} + \alpha[G - \mathbf w_{t}^\top\mathbf w_{t}]\mathbf w_{t}
$$

 eligibility不是一个局限于TD learning 的方法



#### Sarsa(λ)

extend eligibility-traces to action-value methods.  

$$
G_{t:t+n} \doteq R_{t+1} + \gamma R_{t+2} + \dots + \gamma^{n-1} R_{t+n} + \gamma^{n} \hat q (S_{t+n},A_{t+n}, \mathbf w_{t+n-1}) , \quad   t +n < T
$$

offine λ-return algorithm

$$
\mathbf w_{t+1} \doteq \mathbf w_t + \alpha \Big[ G_t^\lambda - \hat q(S_t,A_t,\mathbf w_t) \Big] \nabla \hat q(S_t, A_t,\mathbf w_t) , \quad t=0,\dots,T-1
$$

Sarsa(λ)

$$
\mathbf w_{t+1} \doteq \mathbf w_t + \alpha \delta_t \mathbf z_t
$$

$$
\delta_t \doteq R_{t+1} + \gamma \hat q(S_{t+1},A_{t+1},\mathbf w_t) - \hat q(S_{t},A_{t},\mathbf w_t)
$$

$$
\begin{aligned}
&\mathbf z_{-1} \doteq 0 
\\ &\mathbf z_{t} \doteq \gamma \lambda \mathbf z_{t-1} + \nabla \hat q(S_t,A_t,\mathbf w_t),\quad 0 \leq t \leq T
\end{aligned}
$$



<img src="/img/RL_Introduction.assets/image-20190110213818508.png" width="60%">

![image-20190110214413663](/img/RL_Introduction.assets/image-20190110214413663.png)



action-value形式的online λ-return algorithm

![image-20190110215620436](/img/RL_Introduction.assets/image-20190110215620436.png)

最后还有一个truncated版本的Sarsa( λ)算法，它在与多层ANN合用的时候能够作为一个高效的model-free的control算法。



#### Variable λ and γ

generalize the degree of bootstrapping and discounting beyond constant parameters to functions potentially dependent on the state and action。 更一般的形式



#### Watkins’s Q(λ) to Tree-Backup(λ)

近些年来提出了很多在Q-learning上进行eligibility traces扩展的方法。最早是Watkins’s Q(λ)



#### Stable Off-policy Methods with Traces

一些使用了资格迹在off-policy的情况下能够保证算法稳定。这里介绍四种在本书介绍的标准概念里最重要的四个算法，包括了通用形式的bootstrap和discounting函数。所有的算法都使用了线性函数逼近，不过对于非线性函数逼近的拓展也可以在论文中找到。









#### Conclusions

资格迹与TD error的结合提供了一个高效增量形式的在MC和TD算法之间转换和选择的方式。第七章介绍的n步算法也能做到，但是资格迹方法更加通用，学习速度更快而且共不同计算复杂度的选择。这一章主要介绍了优雅新兴的资格迹方法的理论解释。这个方法能够用于on-policy和off-policy，也能适应变化的bootstrap和discounting。这个理论的一方面是true online方法，它能够精确复制计算量特别大的理论算法的结果，而且保留了传统TD算法计算的亲和力。另一方面是对于从更易理解的forward-view的方法到更容易计算的backward-view方法的转换推导。

第五章提到MC算法可以在非马尔科夫的任务中有优势，因为它不使用bootstrap。因为资格迹方法使得TD算法更像MC算法，因此带资格迹的TD也能够得到这种优势。如果想用TD算法的一些特性而任务又是部分非马尔科夫的，就可以选择使用资格迹形式的TD。

通过调整我们可以将资格迹方法放置到MC到一步TD方法之间的任一个位置。对于应该把它放在哪还没有理论解释，但是经验告诉我们对于步骤比较长的任务使用资格迹比不使用效果更好。在两个方法中间的位置取得的效果一般比较好。虽然现在还不能够非常清楚地使用这个方法。

使用资格迹方法需要比一步TD方法更多的计算量，但是它能够带来更快的学习速度。一般将资格迹用在数据比较稀疏而且不能重复的过程中，经常用在online的应用。而对于产生数据方便的时候经常不使用资格迹







## Policy Gradient Methods

这一章介绍一个学习参数化策略的方法，这个方法中不需要考虑值函数就可以得到策略。在学习策略参数的时候有可能还会用到值函数，但是选择动作的时候不需要。  要把policy参数化为一个可微函数，然后学习这些参数。

$$
\theta_{t+1} = \theta_{t} + \alpha \widehat {\nabla J(\theta_{t})}
$$


对于所有满足此类更新形式的算法，不论是否计算值函数都叫做policy gradient methods。那些同时计算策略函数和值函数的方法又叫做actor-critic算法，actor表示学好的策略，而critic表示学好的值函数。

![image-20190114160528514](/img/RL_Introduction.assets/image-20190114160528514.png)



#### Policy Approximation and its Advantages

对于策略梯度算法，策略函数可以被任意的参数化，只要最后的函数$π(a∣s,θ)$ 是一个对于其参数可微的函数.

to ensure exploration , the policy never becomes deterministic 



##### discrete action spaces

对于那些离散且不大的动作空间，一个简单常用的参数化方法是对于每一个状态动作对都构造一个参数化数值$h(s,a,\theta)\in \mathbb R$ 作为其优先级的表示。softmax

action preferences themselves can be parameterized arbitrarily  动作优先值可以被任意地参数化。比如可以直接使用一个神经网络来逼近，或者只使用简单的线性.



一大优势是可以逼近确定性策略，而对于建立在值函数之上的ϵ-greedy算法来说不可能做到，因为总要对非最优动作分配ϵ部分的概率。 基于action value 的softmax，action value 最终收敛到的是 true value，所以最终的概率值不会是0或者1， 所以不是确定性策略

第二个优势是可以支持对于动作任意分配概率。某些问题里可能采取任意的动作才是最优的。基于动作值函数的策略就不能自然地支持这一功能。

第三个优势是某些问题里策略可能是一个更简单的函数近似的对象。最后一个优势是策略的参数化有时是强化学习算法里注入关于目标策略的先验知识的方法。



value-based的特点：

- 这类方法通常会获得一个确定的策略（deterministic policy），但很多问题中的最优策略是随机策略（stochastic policy）。（如石头剪刀布游戏，如果确定的策略对应着总出石头，随机策略对应随机出石头、剪刀或布，那么随机策略更容易获胜）
- value function 的微小变化对策略的影响很大，可能直接决定了这个action是否被选取
-  在连续或高维度动作空间不适用。 因为这类算法的本质是连续化了value function，但动作空间仍然是离散的。对于连续动作空间问题，虽然可以将动作空间离散化处理，但离散间距的选取不易确定。过大的离散间距会导致算法取不到最优action，会在这附近徘徊，过小的离散间距会使得action的维度增大，会和高维度动作空间一样导致维度灾难，影响算法的速度。

**Policy-based method 克服了上述问题，可以用于连续或高维度动作空间，且可以生成stochastic policy**





#### The Policy Gradient Theorem

理论优势: 对于连续的参数化策略，动作概率作为一个已学习参数的函数是连续变化的，不像是ϵ−greedy 有可能是突变。主要因为这个特点所以**policy-gradient方法有着更强的收敛特性**。特别是策略对于参数的连续性使得这个方法能够利用梯度上升算法。



对 no discouting episodic  case ,  定义性能评估函数， 从s0出发要获得最大回报：

$$
J(\theta) 	\doteq  v_{\pi_\theta} (s_0)
$$

对函数逼近而言，朝着确保policy提升的方向改变policy parameter是有挑战的。问题在于性能函数取决于action 选择和state distribution，这两者都受到policy parameter的影响。 给定一个state，policy parameter对action和reward的影响可以直接根据参数表达式计算出来。但policy 对state distribution 的影响是关于环境的函数，通常是未知的.  

其实, 拟合函数本身的限制就是天花板, 毕竟神经网络不是万能的

> 其他的证明，一般以 τ 的积分来做， 因为τ 本身是个具体的路径值 ，所以不需要 全导数公式

下式这个梯度与状态分布无关.  从任意状态s开始 ， 对参数$\theta$求梯度

$$
\begin{aligned}
\nabla v_\pi(s) &= \nabla \Big[\sum_a \pi(a|s) q_\pi(s,a) \Big]
\\ & = \sum_a \Big[ \nabla\pi(a|s)q_\pi(s,a) + \pi(a|s) \nabla q_\pi(s,a) \Big] \quad 
\\ & = \sum_a \bigg[ \nabla\pi(a|s)q_\pi(s,a) + \pi(a|s) \nabla  \sum_{s',r} p(s',r|s,a)  \Big( r+ v_\pi(s') \Big)  \bigg]
\\ & = \sum_a \bigg[ \nabla\pi(a|s)q_\pi(s,a) + \pi(a|s)   \sum_{s'} p(s'|s,a)   \nabla v_\pi(s')   \bigg]   \quad  可以展开递推unrolling
\\ & = \sum_a \bigg[ \nabla\pi(a|s)q_\pi(s,a) + \pi(a|s)   \sum_{s'} p(s'|s,a)    
\\ & \quad \quad \quad  \sum_{a'} \Big[ \nabla\pi(a'|s')q_\pi(s',a') + \pi(a'|s')   \sum_{s''} p(s''|s',a')   \nabla v_\pi(s'')   \Big]   \bigg]   
\\ & = \sum_{x \in \mathcal S}  \sum_{k = 0} ^\infty Pr(s \to x,k,\pi) \sum_a \nabla\pi(a|x)q_\pi(x,a) 
\end{aligned}
$$

sum_a 对所有的状态都是一样的，可以一层层拿进去, 最后可以放到最后面去。  

**核心思想: 利用bellman公式将梯度层层展开, 并利用每步reward与参数无关, 将 $Q^\pi$ 的导数转化为 $\pi$ 的导数**

$ Pr(s \to x,k,\pi) $ is the probability of transitioning  from state s to state x <u>in k steps</u> under policy $\pi$ 

Pr 表示状态s在某一步的占比，所有步加起来就是在整个路径上的占比。

 $\eta(s)$ 为在轨迹里面每一步遇到s的概率, 再求和, 也可以理解为在一个K步的轨迹中, 所出现的平均步数(次数), $\sum_{s'} \eta(s')$ 是一个正整数,即 episode的step数K

这个Pr很简洁  第二个sum 表示，从状态s出发，路径上，在某个step k ，进入所有的状态的 sum肯定是1， 所以直接算这两个sum的和是 平均总步数 K，对连续的，总和是1； 如果从状态S的维度来切分，是某个s在路径上每步出现概率的总和


$$
\begin{aligned}
\nabla J(\theta) =& \nabla v_\pi(s_0)
\\ = & \sum_s \bigg( \sum_{k=0}^\infty Pr(s_0 \to s,k,\pi) \bigg) \sum_a \nabla \pi(a|s) q_\pi(s,a)
\\ = & \sum_s \eta(s) \sum_a \nabla \pi(a|s) q_\pi(s,a)
\\ = & \sum_{s'} \eta(s') \sum_s \frac{\eta(s)}{\sum_{s'}\eta(s')}  \sum_a \nabla \pi(a|s) q_\pi(s,a)
\\ = & \sum_{s'} \eta(s') \sum_s \mu(s)  \sum_a \nabla \pi(a|s) q_\pi(s,a)
\\ \propto &  \sum_s \mu(s)  \sum_a \nabla \pi(a|s) q_\pi(s,a)  \quad \text{Q.E.D}
\end{aligned}
$$

结论：梯度与状态概率分布的导数无关， 梯度正比于下面的右边

注意, 这里 $\pi$以及q仍然是网络参数$\theta$ 的函数

$$
\nabla J(\theta)  \propto   \sum_s \bigg[\mu(s)  \sum_a  q_\pi(s,a) \nabla \pi(a|s,\theta)  \bigg ]
$$

解释： 好的策略总的思路就是 让q值比较大的s，几率上尽可能的多出现；对局就是尽量引导到对自己有利的局面；这里面还是要利用q值的拟合.   所以梯度上要引导至q比较不错的s的方向; 
另一个问题就是某个状态s出现的占比跟策略也是相关的，包括q的估值也是跟策略相关的

本质上还是链式求导，以状态s作为切分的维度， 每个维度上有多少的梯度贡献





#### REINFORCE: Monte Carlo Policy Gradient

第一个PG算法

sample gradient的期望值和actual gradient成比例

$$
\nabla J(\theta)  \propto   \sum_s \mu(s)  \sum_a  q_\pi(s,a) \nabla \pi(a|s,\theta)
 \\ = \mathbb E_\pi \bigg[  \sum_a q_\pi(S_t,a) \nabla\pi(a|S_t, \theta)  \bigg]
$$

上式，下面的部分, 左边对于右边，  按策略采样的期望 $E_\pi$ ， 每一步与s的占比分布必定是一样的 , 但$J(\theta)$ 是算整个episode的期望的 , 要注意  $E_s(x)$ 与 $E_\pi(x)$ 是有区别的

**all-actions method** 它的更新包含了所有动作 , $\hat q$ is some learned approximation to $q_\pi $  学习来的评估函数

$$
\theta_{t+1} \doteq \theta_t + \alpha \sum_a \hat q(S_t,a,\mathbf w) \nabla \pi(a|S_t,\theta)
$$

is promising and deserving of further study

不实用的点在于, q是估的，需要学习而来, 而且要用到 S_t 下所有a的 r , 调整了策略,所有的采样就要重新评估



推导 **传统 REINFORCE** 方法，去掉了q的学习, 引入 A_t ,  去掉 sum_a , 引入 策略的加权  


$$
\begin{aligned}
\nabla J(\theta)  = & \mathbb E_\pi \bigg[\sum_a \pi(a|S_t,\theta) q_\pi(S_t,a) \frac{\nabla \pi(a|S_t,\theta)}{\pi(a|S_t,\theta)}  \bigg] \quad \sum_a\pi是策略本身
\\ = & \mathbb E_\pi \bigg[  q_\pi(S_t,A_t) \frac{\nabla \pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}  \bigg]  \quad \sum_a就是E_\pi,\   a \to A_t  
\\ = & \mathbb E_\pi \bigg[ G_t \frac{\nabla \pi(A_t|S_t,\theta)}{\pi(A_t|S_t,\theta)}  \bigg]  \quad because \ \mathbb E_\pi[G_t|S_t,A_t]    = q_\pi(S_t,A_t) 定义
\end{aligned}
$$

$$
\theta_{t+1} \doteq \theta_t + \alpha G_t \frac{\nabla \pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}
$$

这个更新是有其内部逻辑的。每次增量都正比于反馈估计值Gt  也就是对于反馈值比较大的就沿着其梯度方向增大较多。反比与采取动作的概率，这样能够防止那些经常被访问的状态占据大多数更新。

**缺陷, 这种方式一个很大的问题就是 G 是基于统计出来的, 只能做一些统计方面的规则, 算不上学习, 如果是复杂规则, 复杂度到函数拟合上的,就效果很差了; 这个时候, 就需要更多的state的特征, 根据特征来学习到特定的规则,达到更好的拟合效果** 

在不考虑资源的情况下, 对所有状态都有表格法, 则基于统计的方法就完全足够了; 某些状态中肯定无法获知的信息, 还是要靠统计来做, 特别是信息不对称, 只知道自己手牌不知道对方手牌的时候, 这个时候是需要统计的;但是,其实有些情况是可以推测的, 所以对于函数的拟合,也就是学习,还是很有用的, 光靠统计是只有超多资源的时候才可以.



![image-20190118114803477](/img/RL_Introduction.assets/image-20190118114803477.png)

eligibility vector ： $\nabla  \ln \pi(A_t\vert S_t,\theta) $

作为一个随机梯度方法，REINFORCE法有一个**良好的理论收敛 good theoretical convergence**性质。通过构造可以使得期望更新的方向和评估函数梯度方向一直，这就保证了对于足够小的参数α 算法一定能够收敛到一个**局部最优**。但是MC形式的REINFORCE方法会带来较大的方差和较慢的学习速度。

可能是收敛到一个统计上的局部最优



#### REINFORCE with Baseline

可以增加一个action value的baseline（基准）

只要梯度是一样的,则方向就是一样的, 所以不会有问题

$$
\nabla J(\theta)  \propto   \sum_s \mu(s)  \sum_a \Big( q_\pi(s,a) - b(s) \Big) \nabla \pi(a|s,\theta)
$$

其中，b(s)可以是任意函数，甚至是随机变量，**只要与a无关都行**，可以与$\theta$ 有关

The baseline can be any function, even a random variable, as long as it does not vary with a; the equation remains valid because the subtracted quantity is zero:
$$
\sum_a b(s)\nabla\pi(a|s,\theta) = b(s)\nabla\sum_a \pi(a|s,\theta) = b(s)\nabla 1 = 0
$$

$$
\theta_{t+1} \doteq \theta_t + \alpha \Big (G_t - b(S_t) \Big ) \frac{\nabla \pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}
$$

由于baseline的均值为0，则上式是REINFORCE的扩展，虽然baseline的期望值对更新无影响，但**对更新的方差影响很大**。  显然，对某个状态s，a，得分必须比基准高才往那个方向倾



自然的选择每个状态值函数的估计值$\widehat{v}(S_{t},w)$作为算法的baseline ； 选所有G的均值也蛮好的

![image-20190118182435510](/img/RL_Introduction.assets/image-20190118182435510.png)

其中对于值函数估计中的系数$\alpha^{w}$有一个简单的设定策略，但是对于策略系数 $\alpha^{\theta}$并没有什么清楚的设定方式。



#### Actor–Critic Methods

尽管带baseline的REINFORCE算法同时使用了策略函数和状态值函数，但是我们不认为它是一个actor-critic算法因为它的值函数只作为一个baseline而没有作为一个critic。That is, it is not used for bootstrapping, but only as a baseline for the state.

[^_^]: 感觉真正核心的还是 价值网络,  策略网络只是基于之上的调度; 而且对很多时候, 值函数拟合的不好容易造成很多问题

引入了bootstrap, 也就引入了偏差, 以及对值函数拟合的依赖。但这通常是有益的，一般会减少方差并且加速收敛。带baseline的REINFORCE算法是无偏的并且会渐进收敛到**局部最小值**，但是MC形式的算法一般会学习的很慢(有估值方差很高的问题), 而且不方便应用到online问题和continuing问题。使用TD方法可以解决这些弊端, 而且通过multi-step的部署可以灵活地选择 bootstrap的程度(degree).  下面介绍 actor–critic methods with a bootstrapping critic.

One-step actor–critic methods:   one-step return 替换了REINFORCE里面的G (full return), 并且use a learned state-value function as the baseline.  **注意, 这里的baseline,用的当前已经学到的 $v_\theta(s)$**  
$$
\begin{aligned}
\theta_{t+1} & \doteq   \theta_t + \alpha \Big (G_{t:t+1} - \hat v(S_t,\mathbf w) \Big ) \frac{\nabla \pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}
\\ & = \theta_t + \alpha \Big (R_{t+1} + \gamma \hat v(S_{t+1}, \mathbf w) - \hat v(S_t,\mathbf w) \Big ) \frac{\nabla \pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}
\\ & = \theta_t + \alpha \delta_t \frac{\nabla \pi(A_t|S_t,\theta_t)}{\pi(A_t|S_t,\theta_t)}
\end{aligned}
$$



 v(s)的拟合可以用  semi-gradient  TD(0). 

![image-20190118213627593](/img/RL_Introduction.assets/image-20190118213627593.png)

将这个算法延伸到forward view的n步算法以及λ-return形式的算法是很直接的。对于拓展到使用资格迹的backward view形式的λ−return 形式的actor-critic算法也很简单

![image-20190118213912404](/img/RL_Introduction.assets/image-20190118213912404.png)



#### Policy Gradient for Continuing Problems

$$
\begin{aligned}
J(\theta) \doteq r(\pi) & \doteq  \lim_{h \to \infty} \frac{1}{h} \sum_{t=1}^h \mathbb E[R_t |S_0,A_{0:t-1} \sim \pi ]
\\ & = \lim_{t \to \infty}   \mathbb E[R_t |S_0,A_{0:t-1} \sim \pi ]
\\ & = \sum_s \mu(s) \sum_a \pi(a|s) \sum_{s',r} p(s',r|s,a)r
\end{aligned}
$$

μ是policy π 下的steady-state distribution ，$\mu(s) = lim_{t_\to\infty} Pr \lbrace S_t =s|A_{0:t}\sim\pi\rbrace$
 假设其存在且依赖于S0（ergodicity assumption）。注意这是一个稳态分布下的值，当在policy π 下选择action之后，会保持在同一个distribution：

$$
\sum_s \mu(s) \sum_a \pi(a|s,\theta) p(s'|s,a) = \mu(s'), \quad \forall s' \in \mathcal S
$$

![image-20190118220335634](/img/RL_Introduction.assets/image-20190118220335634.png)

 

定义连续的 return

$$
G_t \doteq R_{t+1} - r(\pi) +R_{t+2} - r(\pi) +R_{t+3} - r(\pi) + \dots
$$


##### Proof of the Policy Gradient Theorem (continuing case)

$$
\begin{aligned}
\nabla v_\pi(s) & = \nabla \Big[ \sum_a \pi(a|s)q_\pi(s,a)  \Big]
\\ & = \sum_a \Big[ \nabla \pi(a|s)q_\pi(s,a)  + \pi(a|s) \nabla q_\pi(s,a)  \Big]
\\ & = \sum_a \Big[ \nabla \pi(a|s)q_\pi(s,a)  + \pi(a|s) \nabla \sum_{s',r}p(s',r|s,a) \Big (r -r(\theta) + v_\pi(s') \Big)   \Big]
\\ & = \sum_a \Big[ \nabla \pi(a|s)q_\pi(s,a)  + \pi(a|s) \big[ -\nabla r(\theta) +  \sum_{s'}p(s'|s,a)\nabla v_\pi(s') \big]    \Big]
\end{aligned}
$$

After re-arranging terms, we obtain

$$
\nabla r(\theta) = \sum_a \Big[\nabla\pi(a|s)q_\pi(s,a)+\pi(a|s)\sum_{s'}p(s'|s,a)\nabla v_\pi(s') \Big] - \nabla v_\pi(s)
$$

 左边与具体s无关，所以右边也无关 ， 所以把 右边的在所有状态s上去加权求和

$$
\begin{aligned}
\nabla J(\theta) & = \sum_s \mu(s) \bigg( \sum_a \Big[\nabla\pi(a|s)q_\pi(s,a)+\pi(a|s)\sum_{s'}p(s'|s,a)\nabla v_\pi(s') \Big] - \nabla v_\pi(s) \bigg) 
\\ & = \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(s,a)
\\ &\quad \quad \quad + \sum_s \mu(s)\sum_a\pi(a|s)\sum_{s'}p(s'|s,a)\nabla v_\pi(s')
- \sum_s \mu(s) \nabla v_\pi(s)
\\ & = \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(s,a)
\\ &\quad \quad \quad + \sum_{s'} \sum_s \mu(s)\sum_a\pi(a|s) p(s'|s,a)\nabla v_\pi(s')
- \sum_s \mu(s) \nabla v_\pi(s)
\\ & = \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(s,a) + \sum_{s'}\mu(s')\nabla v_\pi(s')
-\sum_s \mu(s) \nabla v_\pi(s)
\\ & = \sum_s \mu(s) \sum_a \nabla \pi(a|s)q_\pi(s,a) \quad \text{Q.E.D}
\end{aligned}
$$




#### Policy Parameterization for Continuous Actions

之前 s 是连续的， 现在 a 是连续的

基于策略的方法提供了以一个简单的方式来解决大规模动作空间的问题，甚至可以解决一个有无限个动作的连续空间也可以。我们不再学习每个动作的概率，而是学习采取动作的概率分布。

为了构造一个参数化的策略，策略可以被定义为一个分布在实数标量值的动作上的高斯分布。其中的均值和标准差是依赖于动作和给定参数的近似函数。

$$
\pi(a|s,\theta) \doteq \frac{1}{\sigma(s,\theta) \sqrt {2 \pi}} \exp \Big( - \frac{(a-\mu(s,\theta))^2}{2 \sigma(s,\theta)^2} \Big)
$$

其中的均值和方差被参数化为

$$
\mu(s,\theta) \doteq \theta_\mu^\top \mathbf x_\mu(s) \quad and \quad \sigma(s,\theta) \doteq \exp \Big(\theta_\sigma^\top \mathbf x_\sigma(s) \Big)
$$

这种拟合也是一种近似吧，  如果在a的空间上，离得很远的两个a有最好效果， 感觉这个正态分布应该搞不定

 













# Looking Deeper



## Psychology 心理学



## Neuroscience 神经科学



## Applications and Case Studies

本章描述了强化学习的几个案例研究。主要想表现在实际应用问题中的一些权衡与问题。比如我们很看重专业知识移植到问题的构建和解决方法中的方式。同样我们也很关注对于成功的应用都很重要的问题表达。



#### TD-Gammon

强化学习中最惊人的一个应用之一是Gerald Tesauro做的backgammon游戏的算法。 TD-Gammon，不需要游戏背景知识但是仍然能够接近这个游戏的世界大师级别。TD-Gammon中使用的学习算法是TD(λ)与非线性函数逼近的直接结合。非线性函数逼近使用的是通过反向传播TD error来训练的多层ANN网络。

TD-Gammon使用的是非线性TD(λ)。评估值$\widehat{v}(s,w)$ 来表示每个状态s下获胜的概率。为了得到这个概率，奖励值设定为每一步都为0，除了最后获胜的那步。





<img src="/img/RL_Introduction.assets/image-20190120153514223.png" width="50%">





#### Samuel’s Checkers Player





#### Personalized Web Services

A/B testing



比较了两种算法。第一种是叫做greedy optimization，最大化即时点击率。另一种算法是基于MDP结构的强化学习算法，为了提高每个用户多次访问网页的点击率。后一个算法叫做life-time-value(LTV)optimization。两种算法都要面临奖励稀疏的问题，会带来反馈值很大的方差。

银行系统的数据集被用来训练和测试这些算法。greedy optimization算法基于用户特征来做一个预测点击的映射函数。这个映射通过监督学习来训练，使用的是随机森林算法(RF)。这个算法被广泛用在工业中，因为很有效，且不易过拟合而对outlier不敏感。

LTV optimization算法使用一个batch模式的强化学习算法叫做fitted QIteration(FQI)。这是fitted value iteration算法的一个q-learning算法变体。batch模式意味着这些数据从一开始就是都可以获得的，和本书之前讲的在线形式的需要等待学习算法执行动作生成数据的算法不同。



结果显示greedy optimization算法在CTR评估时表现得更好，而LTV optimization在使用LTV评估时更好。另外off-policy评估方法的高可信程度保证了LTV optimization算法能够得到对于实施的策略改进。



