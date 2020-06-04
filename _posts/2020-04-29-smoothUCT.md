---
layout:     post
title:      SmoothUCT
subtitle:   
date:       2020-04-01 12:00:00
author:     "tengshiquan"
header-img: "img/post-bg-dice.jpg"
catalog: true
tags:
    - AI
    - Imperfect Information
    - Game Theory 
    - UCT

---



# SmoothUCT 算法

**SmoothUCT = 平均策略引导Smooth UCB + UCT** 

可以收敛,  但收敛不到完美均衡.



因为UCT在perfect-information博弈中的成功,  并且 extensive-form博弈本质上是树，所以可以考虑将UCT应用到imperfect-information extensive-form games.  理论上首先要解决的是, UCT能否学到近似纳什均衡.

fictitious play 的目标是找到针对其他人平均策略的 best response(BR). 



### Monte Carlo Tree Search

<img src="/img/2020-04-29-smoothUCT.assets/image-20200528114032814.png" alt="image-20200528114032814" style="zoom: 33%;" />



MCTS 是基于仿真的搜索算法**simulation-based search algorithms** ，其中许多算法将Monte Carlo强化学习应用于局部搜索树. 

MCTS适用于大规模环境基于以下的特性:  **planning online** 在线规划，它可以将搜索工作集中在一个相关的（局部）状态空间的子集上。而不是像Minimax搜索中那样，对局部子树进行全遍历(full-width)的评估，MCTS有选择地对博弈的轨迹进行采样，从而减轻了curse of dimensionality。这些模拟是由一个动作选择策略指导的，它探索状态空间中最有潜力的区域。搜索树在被探索的状态中逐步展开，从而产生一个紧凑的、不对称的搜索树。

components:  

1. black box simulator , samples a successor state and reward
2. RL algorithm,  使用模拟得到的trajectories和收益 来更新树上被访问节点的统计数据; 
3. **tree** policy,  chooses actions based on a node’s statistics
4. **rollout** policy, 确定超出搜索树范围的状态的默认行为。



node *u* 的 action value 评估公式
$$
Q(u, a)=\frac{1}{N(u, a)} \sum_{k=1}^{N(u, a)} G(u, k)
$$
UCT :
$$
\pi_{\text {tree}}(s)=\underset{a}{\arg \max } Q(s, a)+c \sqrt{\frac{\log N(s)}{N(s, a)}}
$$
exploration bonus parameter *c*  用于调整exploration与exploitation的平衡。 对于合适的c，UCT会在Markovian env中收敛到最优策略。 合适的c的大小取决于奖励的大小。



##### 其他一些MCTS方法总结:

1. Silver和Veness（2010）将MCTS 应用到POMDP，并证明了给定真实初始信念状态的收敛性。given a true initial belief state
2. Auger(2011)和Cowling等人(2012)将MCTS扩展到不完全信息博弈中，并使用 regret-minimising bandit方法EXP3(Auer等人，1995)进行行动选择。
3. Lisy等(2013)证明了MCTS在同步动作simultaneous-move 博弈中的收敛性，对于后悔最小化树策略，包括EXP3变体。
4. Cowling等（2015）提出了在多人不完全信息博弈中推理的MCTS方法。
5. Lisy ́等（2015）提出了Online Outcome Sampling，这是一种在双人零和不完全信息博弈中保证收敛的在线MCTS变体。
6. Ponsen等人（2011）比较了Outcome Sampling和UCT在扑克游戏中的全局搜索性能。他们的结论是，UCT很快就找到了一个良好的但次优策略，而结果采样最初的学习速度较慢，但随着时间的推移会收敛到最优策略。





### MCTS in Extensive-Form Games

在extensive-form game with imperfect information中，玩家信息的不对称性不允许使用单一的搜索树。 为了将 single-agent Partially Observable Monte-Carlo Planning (POMCP)  扩展到 extensive-form game, 我们对每个玩家使用单独的搜索树. 

具体做法, 每个玩家 $i$ ,  information states  infoset $\mathscr{S}^{i} $ ,  对应的树为 $T^{i}$ ; 某个 infostate 对应的 node  $T^{i}\left(s^{i}\right)$  ; 对perfect recall博弈, 玩家 $i$ 先前的信息状态和行动的序列 $s_{1}^{i}, a_{1}^{i}, s_{2}^{i}, a_{2}^{i}, \ldots, s_{k}^{i}$ , 被纳入信息状态 $s_{k}^{i}$ 中 , 因此$T^{i}$ 是一棵特有树  proper tree.  
如果是 imperfect-recall, 则会产生recombining trees.  即一个节点可能有多个父节点. 



![image-20200528145224463](/img/2020-04-29-smoothUCT.assets/image-20200528145224463.png)




<img src="/img/2020-04-29-smoothUCT.assets/image-20200528152242115.png" alt="image-20200528152242115" style="zoom:50%;" />

 

算法1描述了一种用于 多人不完全信息extensive博弈 的通用MCTS方法。 




### Extensive-Form UCT

Extensive-Form UCT使用UCB来选择和更新算法1中的tree policy 。Algorithm 2 可以看作是部分可观察的UCT（PO-UCT）的多代理版本（Silver和Veness，2010）。PO-UCT在POMDP中搜索一个代理的观察和行动的历史树，而extensive-form的UCT在信息状态和行动上搜索多个代理的树。在一个perfect-recall 博弈中，一个信息状态意味着对前一个信息状态和行动序列的知识，相当于一个完整的历史。



<img src="/img/2020-04-29-smoothUCT.assets/image-20200528155058874.png" alt="image-20200528155058874" style="zoom:50%;" />





### Smooth UCT

Smooth UCT = MCTS + Smooth UCB . 

Smooth UCB是UCB的启发式修改。Fictitious players 根据其他人的平均策略学习BR。通过累加每一个动作的次数，UCB已经有了一个平均策略。$\pi_{t}(a)=\frac{N_{t}(a)}{N_{t}}$  .   然而，UCB并没有使用这一策略，因此在搜索中可能会忽略一些有用的信息。

Smooth UCB的基本思想是在选择行动时**混入平均策略**，以诱导其他代理对其作出反应。这类似于fictitious play的想法，代理对平均策略做出BR。平均策略可能还有其他有益的特性。首先，它是一种随机策略，而且随着时间的推移，它的变化越来越慢。这可以降低玩家行为之间的相关性，从而有助于稳定自博弈过程。此外，一个更平稳变化的策略可以被其他代理所依赖，而且比起像UCB这样反复无常地变化的贪婪策略更容易适应。

Smooth UCB 需要的信息与UCB相同，但通过动作计数明确地利用了平均策略。特别是，它混合了UCB和平均策略.  用了一个概率参数$\eta_k$, 该参数是衰减的. 有个复杂的公式.   使代理可以将经验集中在与对手的平均策略的对弈上，以获得BR。另一方面，衰减也会减缓对平均策略的探索和更新。



<img src="/img/2020-04-29-smoothUCT.assets/image-20200529003844704.png" alt="image-20200529003844704" style="zoom:50%;" />





### Experiments

![image-20200529004209672](/img/2020-04-29-smoothUCT.assets/image-20200529004209672.png)

上图可见 Smooth UCT收敛, UCT发散 . 



<img src="/img/2020-04-29-smoothUCT.assets/image-20200529004746985.png" alt="image-20200529004746985" style="zoom:50%;" />





<img src="/img/2020-04-29-smoothUCT.assets/image-20200529004952661.png" alt="image-20200529004952661" style="zoom:50%;" />

Smooth UCT的长期性能, 上图显示很难收敛到完美纳什均衡. 









### Conclusion

在小型扑克游戏中， Smooth UCT的学习速度不亚于UCT，能收敛到（近似）纳什均衡，而UCT则出现了发散.







## Reference

Reinforcement Learning from Self-Play in Imperfect-Information Games

















