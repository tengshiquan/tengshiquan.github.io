---
layout:     post
title:      Approximately Optimal Approximate Reinforcement Learning
subtitle:   Note on "Approximately Optimal Approximate Reinforcement Learning" (2002)
date:       2020-01-01 12:00:00
author:     "tengshiquan"
header-img: "img/about-bg.jpg"
catalog: true
tags:
    - AI
    - Reinforcement Learning

---

# Note on "Approximately Optimal Approximate Reinforcement Learning"

论文笔记.  这个标题有点绕.  近乎最优的 近似强化学习  2002



#### Abstract

为了解决现实的强化学习问题，使用近似算法(approximate algorithms)至关重要。 本文提出了一种保守的策略迭代算法(**conservative policy iteration, CPI**),  配合 重启分布 **restart distribution**（从一个特定的分布中取next state）和 近似贪婪策略选择器(**approximate greedy policy chooser**), 该算法可以寻找 “近似”最优策略。**greedy policy chooser** 输出 一个新策略: 在当前策略下,通常选择有最大action-value的那些actions , 即，它输出“近似”贪婪策略。 greedy policy chooser 可以用值函数近似(value funtion approximation)技术来实现.  

**conservative policy iteration**算法: 1. 保证改进 2. 保证停止 3. 返回一个"近似"最优策略.  
2,3取决于greedy policy chooser的质量, 并不显式地(explicitly)取决于状态空间的大小。



#### 1 Introduction

强化学习领域, 两个已经成功的常见方法: 贪婪动态规划(**greedy dynamic programming**) 和 策略梯度(**policy gradient**).  然而, 两个方法都可能无法有效改进策略.  

- 对greedy dynamic programming,  **近似值函数(Approximate value function)方法**缺少有力的理论上的性能保证.
- PG算法需要太多的sample来准确计算梯度, 因为PG将 **exploration** 和 **exploitation** 交替在一起. 

本论文中,  考虑一个环境setting,  假定我们的算法拥有 **restart distribution** 和 **greedy policy chooser**。

- **Restart distribution** 允许agent从自己设计的一个固定的分布里获取next state(应该是指 restart state).  通过分布更平均的restart distribution, agent可以获取平时不一定访问的状态的信息.  保证探索性
- **Greedy policy chooser** 是一个黑盒, 输出一个新策略, 大体上(on average)选择相较当前策略有大幅优势的动作, 即, 产出一个近似贪婪策略**"approximate" greedy policy**(不是贪婪固定地选择比当前有优势的动作, 还保留了探索性, 所以是近似贪婪) .  Greedy policy chooser 可以用基于value function的回归算法来实现.

作者基于常见算法的优点, 提出 **conservative policy iteration** 算法.  关键要素: 
1. 在更加均匀uniform的状态空间(state space)上改进策略 
2. 执行更保守**conservative**的策略更新，新策略是当前策略和贪婪策略的混合.  (两种策略的线性组合, 不是全部更新, 所以是保守的, 这种手段可能只适合某些问题)

换言之, 1. 体现 探索性  2. 避免greedy dynamic programming的缺陷,  greedy dynamic programming直接使用近似贪婪策略**"approximate" greedy policy**, 可能使得策略**退化(degradation)**

作者证明该算法在"很少"的步数内就可以收敛, 并返回一个近似最优策略**"approximately" optimal policy**. 该策略的性能,并且不显式地取决于状态空间大小.



#### 2 Preliminaries

- $D$  : starting state distribution
- $\mathcal R$ : reward funtion ,   $\mathcal R : S \times A \to [0, R]$
- $\pi(a;s)$ :  在state s 下选取 action a 的probability,  ; 分号用于区分 参数 与 概率分布的随机变量



**Definition 2.1.**   **$\mu$ restart distribution** : 从 distribution $\mu$ 里选取 next state.(应该是指 restart state)

- 下面假定已经拥有**restart distribution**. **restart distribution** 是 **生成模型**(**generative model**)的一个弱化版. 这两者都比 对环境完全掌握 full transition model 要弱. 而这些又都比 "irreversible" experience 要强.  在"irreversible" experience 中, agent只能遍历一整个trajectory, 无法随时reset进入另外一个trajectory.   
- **generative model**, or simulator of the MDP. a “black box” input  state-action pair (*s*,*a*), output (r, s') : a randomly sampled next state and reward from the distributions associated with (*s*, *a*)
- agent完全可控 白盒 >generative model 黑盒 > 起始状态可控 > 不可控

$\mu$ 只要是一个相对均匀的分布,  (不必须与$D$ 一样),  就可以消除对**显式地探索 (explicit exploration)**的需要.

- **Value function**:  discounted average reward

$$
V_{\pi}(s) \equiv(1-\gamma) E\left[\sum_{t=0}^{\infty} \gamma^{t} \mathcal{R}\left(s_{t}, a_{t}\right) | \pi, s\right]
$$

   这里V使用了normalized 值, 前面乘以了 $1- \gamma$, 标准化后 $V_\pi(s) \in [0,R]$ .   比较少见

- **action-value**:  标准化的Q值的定义
$$
Q_\pi(s,a) \equiv (1 - \gamma)\mathcal R(s,a) + \gamma E_{s' \sim P(s';s,a)}[V_\pi(s')]
$$

- **advantage**:  s下, 选择某个特定a比按照$\pi$本来的策略要好多少.
$$
A_\pi(s,a) \equiv  Q_\pi(s,a) - V_\pi(s)
$$

- 由于标准化normalization,  $Q\in [0, R]$ $A(s,a)\in [-R, R]$
- **$\gamma$-discounted future state distribution** for  a <u>starting distribution</u> $\mu$ :  策略$\pi$从$\mu$开始的discount的未来状态分布(都是timestep长度上的占比, 即所有s的d加起来等于时间步长度N)

$$
d_{\pi, \mu}(s) \equiv(1-\gamma) \sum_{t=0}^{\infty} \gamma^{t} \operatorname{Pr}\left(s_{t}=s ; \pi, \mu\right) \tag{2.1}
$$

  $1-\gamma$ 是normalization所必须的.  上式表示, 起始状态的选取符合$\mu$ 分布,   按照$\pi$来执行, 之后所有state可能为s的 discounted 几率和, 折扣后占比.  这里, $\gamma$ 主要是为了乘以R用的.

-  $d_{\pi,s}$ , 表示从状态s开始的 折扣未来状态分布 discounted future state distribution.
-  $(a',s') \sim \pi d_{\pi, s}$ 表示, s之后各个状态s'的时间长度上的占比与在状态s'下遵循策略$\pi$选择a'的几率相乘 ;  	
$$
V_{\pi}(s)  = E_{(a',s') \sim \pi d_{\pi, s}} [\mathcal{R} (s', a')]
$$

- 当$\gamma \to 1$ , $d_{\pi, s}$ 趋向于 非折扣环境undiscounted setting下 所有状态的平稳分布 stationary distribution, 占比

**goal**:  maximize discounted reward from  start distribution $D$ :   即规定了出发状态的 $V_\pi$
$$
\eta_D(\pi) \equiv E_{s \sim D} [V_\pi(s)]
$$

- 同时有 $\eta_D(\pi)  = E_{(a,s) \sim \pi d_{\pi, D}} [\mathcal{R} (s, a)]$ ,  E的下标表示, $s_0 \sim D$, 之后时长占比$s \sim d_{ \pi, D}$ , s下选择a服从$\pi$
- 众所周知, 存在一个同时使所有状态的$V_\pi(s)$最大化的策略 = 最优策略



#### 3 The Problems with Current Methods

三个问题:
1. 有什么性能指标(Performance measure)是保证每一步都有进步 ？ 
2. 验证特定更新是否改进了此性能指标的 困难程度？
3. 策略更新次数后，将获得什么性能级别？



##### 3.1 Approximate Value Function Methods

- **精确值函数方法(exact value function methods)**, 如 policy iteration,  PI对$\pi$, 计算$Q_\pi(s, a)$, 然后创建新的deterministic policy $\pi'$ such that   $\pi'(а;ѕ)=1$ іff $\arg \max_aQ(s,a)$ . 重复该过程直到state-action values收敛到最佳optimal values.  
- Exact value function 有很强的边界，表明值收敛到最佳值的速度有多快(参考7)

基于**近似值函数方法(approximate value function methods)**的策略的性能缺乏理论研究结果, 这会导致该方法对三个问题的答案都很弱。

- 考虑某个近似函数function approximator $\tilde{V}(s)$ 的误差 $l_{\infty}$ 范数, $l_{\infty}$-error :    $\varepsilon$表示最大的近似误差
$$
\varepsilon=\max _{s}\left|\tilde{V}(s)-V_{\pi}(s)\right|
$$

- $\pi$ 是某个策略, $\pi^{\prime}$ 是该策略的近似基础上的贪婪策略,  a greedy policy based on this approximation.  由参考3 , 对所有状态s, 有下式成立:

$$
{V_{\pi^{\prime}}(s) \geq V_{\pi}(s)-\frac{2 \gamma \varepsilon}{1-\gamma}}  \tag{3.1}
$$

- 也就是是, 性能绝对不会下降超过 $\frac{2 \gamma \varepsilon}{1-\gamma}$ .  不能回答问题2, 因为不保证改进, 并且没有定义怎么性能度量 performance measure.  
- 对approximate方法, 花费多少时间能达到一定的性能level 也没有很好的搞清楚.



##### 3.2 Policy Gradient Methods

- PG算法尝试通过沿着未来reward的梯度, 在限定的一类策略中 找到一个好策略. 
- 根据参考8, 下式计算梯度

$$
\nabla \eta_{D}=\sum_{s, a} d_{\pi, D}(s) \nabla \pi(a ; s) Q_{\pi}(s, a) \tag{3.2}
$$

- PG算法对问题1有个不错的答案, 因为在梯度提升的情况下, 性能会保证改进.   
- 对问题2, 要判定梯度的方向是困难的.  我们发现,梯度方法缺乏探索性(lack of exploration)意味着需要大量样本才能准确估计梯度方向。即梯度需要是真值才行,抽样来的并不准确.

<img src="/img/2020-01-01-OptimalRL.assets/image-20200104205347255.png" alt="image-20200104205347255" style="zoom: 25%;" />

一个例子, 上图中的MDP, 一个agent, 三个action, 两个往左,一个往右. 只有到达state1才有reward 1, 在三个action几率一样的情况下, 从state n出发, 走到最左边state1的 期望时间是 $3\left(2^{n}-n-1\right),$ 当 $n=50,$ 大约是 $10^{15}$.  

- 这个MDP属于一类MDPs, 随机操作更有可能增加到目标状态的距离,越来越远。对这类问题, 使用无方向的探索,即随机游走, 到达目标的预期时间是状态空间大小的指数级别. 因此, 任何 "**on-policy**" 方法都必须必须走这么长,才能找到改进的点.  lack of exploration.   

- > *想法: 因为计算机数据结构的原因, 造成了只能一步步探索, 是短视的, 能否利用程序或者结构, 引入类似视野这样一个东西作为整体来考虑?? 例如cnn就是把9个像素看成一个整体, 然后一层层网络就把图片看成一个整体*

- 在没有达到目标状态的情况下，对梯度的任何合理的估计都将为0，并且获得非0估计值需要使用“on-policy”样本的指数级时间。 **Importance sampling** 方法对于此类问题而言并不可行,  因为如果agent可以遵循一些“off-policy”的轨迹在合理的时间内达到目标状态，则重要性权重必须是指数级别的数值大小。

- 梯度的0值估计在数量级上还是比较准确的 .  但0没有提供方向这个关键信息.  参考[2]表明, 只需要一个相对较小的样本大小即可准确估计数量级，但如果梯度较小则意味着方向不准确。 不幸的是，当政策远未达到最优时，梯度的大小可能非常小。

<img src="/img/2020-01-01-OptimalRL.assets/image-20200104205436283.png" alt="image-20200104205436283" style="zoom: 33%;" />

另外一个例子, 也说明使用PG算法到达收敛需要指数级的时间, 虽然只有两个状态. 然后梯度也可能在某些时候, 起到负面效果. 

上图左边是个两个state的MDP,  目标是最大化平均回报. 显然最佳策略就是在i执行a2到j,然后一直a1.    策略使用 Gibbs table-lookup distributions, $\{\pi_{\theta}: \pi(a ; s) \propto \exp \left(\theta_{s a}\right) \} $ , 在i处增加自循环的机会会降低j的平稳概率，从而妨碍在j状态下的学习. 
初始化一个策略,  使得状态的平稳分布为 :$\rho(i)=.8$,  $\rho(j)=.2$, (策略  $\pi(a_1;i) =.8$ , $\pi(a_1;j) =.2$ 论文里面这里是0.9, 有点问题? )  在i处学习必然会影响在j处学习, 造成一个很平坦的高原. 从下图看出, 按照刚才的初始化, $\rho(j)$ 会降到 $10^{-7}$, 非常严重.   就像在例1中一样, 要获得非0的梯度估值必须访问 j.  如果上图左边状态更多一些, 则问题更加严重. 

<img src="/img/2020-01-01-OptimalRL.assets/image-20200104205515790.png" alt="image-20200104205515790" style="zoom: 33%;" />

虽然可能会渐进地 asymptotically 找到一个好的策略，但这些结果对于问题3的答案不是好兆头，问题3涉及到找到这样一个策略的速度。这些结果表明，在任何合理数量的步骤中，梯度方法可能最终被困在平台上，其中估计梯度方向具有不合理的大样本复杂度。回答问题3对于理解梯度方法的性能至关重要，但目前没相关知识。 

> 即梯度限制了状态的切换, 很难从一个局部最小点出来.  能否像游戏地图的LOD技术一样, 先粗略看loss平面, 然后再慢慢细化, 即随机抽样, 然后再慢慢细化 . 



#### 4 Approximately Optimal RL

- PG算法的问题是，对不太可能发生的状态 (unlikely states)下的政策改进不敏感，因为采样少, 尽管这些不可能状态下的政策改进可能对寻求最优策略是必要的。
- 我们希望有一个替代性的性能指标，不会降低unlikely states 或者 actions 的权重。性能度量的一个候选者是**更均匀**地衡量来自所有状态的改进(不是$\mathcal D$). 	 $ \eta_\mu(\pi) \equiv E_{s \sim \mu}[V_\pi(s)] $   	这里, $\mu$ 是一个**'exploratory' restart distribution**.     
- 初始状态从更加uniform的$\mu$中选取,之前一些访问不到的状态都可以作为起始状态, 相当于强制visit. 

[^_^]: 能否根据熵来判断这些state的权重.  或者统计数据来决定, 或者某些state是否明显跟其他state不一样, 权重不一样

- 下面的问题是, 能否有最优策略在最大化 $\eta_\mu$ 的同时, 保证在$\eta_D$ 下也是一个好的策略.  毕竟出发点的分布改变了.
- 任何最优策略都能同时最大化  $\eta_\mu$ 和 $\eta_D$.   但最佳策略不是那么容易找到的, 也不容易判定.
- 但是, 能最大化$\eta_\mu$ 的那类策略, 可能在$\eta_D$上表现不好. 所以必须确保最大化$\eta_\mu$ 的策略在$\eta_D$上也是好的策略.  什么情况会造成这样?

**Greedy policy iteration** updates the policy to some $\pi'$ based on some approximate state-action values.
conservative update rule: 
$$
\pi_{new}(a;s) = (1 - \alpha)\pi(a;s) + \alpha \pi'(a;s) \tag{4.1}
$$

- 当$\alpha = 1$, 为了保证改进, $\pi'$ 必须在每个状态都选择更好的action, 否则可能像公式3.1那样造成性能下降.
- 下面讨论 $\alpha < 1$ 的情况
  - 4.1 可以证明, $\eta_\mu$ 可以在经常(但不是每次必须)选择贪婪action的情况下, 改进策略
  - 4.2 假定拥有 greedy policy chooser, 输出 'approximately' greedy policy $\pi'$  ,  然后根据greedy policy chooser的质量来 bound 我们算法发现的策略的性能。



#### 4.1 Policy Improvement

- 一个合理的情况, 取$\alpha > 0$, 我们可以在大多数状态下(不是所有states)通过策略$\pi'$ 获取更好的action来改进策略.
- 定义优势函数 ,  advantage of $\pi'$   ,   定义式, 可用于核量算法改进了多少

$$
\mathbb{A}_{\pi, \mu}\left(\pi^{\prime}\right) \equiv E_{s \sim d_{\pi, \mu}}\left[E_{a \sim \pi^{\prime}(a ; s)}\left[A_{\pi}(s, a)\right]\right]
$$

- 公式含义, 里面一层的期望是 在s的时候, 按$\pi'$能比$\pi$好多少,  外面的期望是, 起始状态  $s \sim \mu$, 执行$\pi$策略,经历的trajectory的期望.  该优势函数可以衡量$\pi ^ {\prime}$选择actions具有的优势程度.

- 显然, $$\mathbb{A}_{\pi, \mu}\left(\pi \right)= 0 $$,  所以 CPI的 $$A_\pi(\pi_\text{new}) = \alpha A_\pi(\pi')$$

- Note that a policy found by one step of policy improvement maximizes the policy advantage.  一步PI产出的策略最大化Advantage

- $$
  \frac{\partial \eta_{\mu}}{\partial \alpha}=\sum_{s, a} d_{\pi_{new}, \mu}(s) \frac{\partial{\pi_{new}} }{\partial \alpha} A_{\pi_{new}}(s, a) \\
  =\frac{1}{1-\gamma} E_{s \sim d_{\pi_{new}, \mu}} \sum_a [ (\pi'(a;s) -  \pi(a;s)) A_{\pi_{new}}(s, a)] \\
  \alpha =  0 \to \pi_{new} = \pi
  \\
  \frac{\partial \eta_{\mu}}{\partial \alpha}\vert_{\alpha=0} =\frac{1}{1-\gamma} E_{s \sim d_{\pi, \mu}} \sum_a [  \pi'(a;s) A_{\pi}(s, a)] \\
  = \frac{1}{1-\gamma} \mathbb{A}_{\pi, \mu}(\pi')
  $$

  

所以 $\eta_{\mu}$ 上的改变就是:

$$
\quad \Delta \eta_{\mu}=\frac{\alpha}{1-\gamma} \mathbb{A}_{\pi, \mu} (\pi')+O (\alpha ^{2}) \tag{4.2}
$$

- 对足够小的$\alpha$, 如policy advantage 是正的, 则会改进. 
- 如果极端情况$\alpha =1 $, 则可能造成退化. 
- 将这两者结合起来, 来决定策略改进多少是可能的.

**重要定理 Theorem 4.1.**  策略改进的下限:  用 $\mathbb{A}$ 表示  $\pi^{\prime}$ 的关于$\pi , \mu$ 的 policy advantage : 
let $\varepsilon= \max_{s} \vert E_{a \sim \pi'(a ; s)} [A_{\pi}(s, a) ] \vert$   ;   for all $\alpha \in[0,1]$ :    这里 $\varepsilon$ 表示按新策略能得到的最大的A优势

$$
\eta_{\mu}\left(\pi_{n e w}\right)-\eta_{\mu}(\pi) \geq \frac{\alpha}{1-\gamma}\left(\mathbb{A}-\frac{2 \alpha \gamma \varepsilon}{1-\gamma(1-\alpha)}\right)
$$

- 可以构造一个具有两个状态的例子,  论文里没举例, 来说明这个边界对所有的$\alpha$都是紧的. 
- 第一项类似于公式4.2里面的一阶增长, 第二项是惩罚项.  
- 如果 $\alpha=1,$ 则下边界:   下式右边, 第二项可能大于第一项, 所以可能是性能退化

$$
\eta_{\mu}\left(\pi_{\mathrm{new}}\right)-\eta_{\mu}(\pi) \geq \frac{\mathbb{A}}{1-\gamma}-\frac{2 \gamma \varepsilon}{1-\gamma}
$$

- 惩罚项的形式与 greedy dynamic programming 里的一样 ,  $\varepsilon$ 与公式3.1定义的 $l_{\infty}$ 类似 .

下面的推论显示,  policy advantage 越大, 性能单调的改进越大.  

**Corollary 4.2.** 令最大可能的reward为 $R$,  **如果** $\mathbb{A} \geq 0,$  然后令 $\alpha=\frac{(1-\gamma) \mathbb{A}}{4 R}$ 可以保证按下式改进策略:
$$
\eta_{\mu}\left(\pi_{n e w}\right)-\eta_{\mu}(\pi) \geq \frac{\mathbb{A}^{2}}{8 R}
$$

Proof. 基于定理4.1 , 右边可以继续放松推导  $\geq \frac{\alpha}{1-\gamma}\left(\mathbb{A}-\alpha \frac{2 R}{1-\gamma}\right)$ ,  然后$\alpha$取能使边界最大的值, 即导数为0极值点

其实$A_\pi(\pi') \geq 0$也不是那么容易满足的.   上面讨论了怎么选步长,   下面讨论怎么选择 $\pi'$. 



#### Answering question 3

- 要解决问题3, 首先解决 收敛到某个策略的速度，然后限定该策略的质量。
- 我们期望能够获得具有较大优势的政策，从而影响改进的速度和最终政策的质量。 
- 文章里并不直接选择Advantage比较大的策略代替老的策略, 而是使用**policy chooser**.  Instead of explicitly suggesting algorithms that find policies with large policy advantages, we assume access to an **$\varepsilon$-greedy policy chooser** that solves this problem.  叫作**$ \varepsilon$-good algorithm** $G_{\varepsilon}(\pi, \mu)$, 定义如下:

**Definition  4.3.**   **$\varepsilon$ -greedy policy chooser**,  $G_{\varepsilon}(\pi, \mu)$,  是 $\pi$ 和 $\mu$ 的函数, 返回一个策略 $\pi^{\prime}$ , 使得 $ \mathbb A_{\pi, \mu} (\pi') \geq \text{OPT} ( \mathbb A_{\pi, \mu} ) -\varepsilon$   , 其中 $\operatorname{OPT} (\mathbb A_{\pi, \mu} ) \equiv \max_{\pi^{\prime}} \mathbb A_{\pi, \mu} (\pi')$  ; OPT是最优策略, 这个chooser是返回比最佳差 $\varepsilon$的策略

- 下面讨论, 使用一个回归算法, 以平均误差$\frac{\varepsilon}{2}$来拟合 advantage,  足够构造这样的 $G_{\varepsilon} .$
- "**break point**":  当 greedy policy chooser 无法保证返回一个 正数的advantage的策略的时候, 即当 $\mathrm{OPT}\left(\mathrm{A}_{\pi, \mu}\right)<\varepsilon$时, 无法保证改进. 

**Conservative Policy Iteration** 算法:
1. Call $G_{\varepsilon}(\pi, \mu)$ to obtain some $\pi'$
2. Estimate the policy advantage $\mathbb{A}_{\pi, \mu}\left(\pi^{\prime}\right)$
3. If the policy advantage is small (less than $\varepsilon$ ) STOP and return $\pi$.
4. Else, update the policy and go to (1).

- 为了简化问题, 下面假设 $\varepsilon$ 已知.  
- 该算法在获得策略优势 $ \leq \varepsilon$ 时候停止   
- 根据greedy policy chooser的定义,  可以得出 $\pi$ 的 最大 optimal policy advantage小于 $2 \varepsilon .$ 

下面的定理表明, 在多项式的时间内, 完整的算法会找到一个策略, 接近"break point" .  

**Theorem 4.4.** 至少以概率 $1 - \delta$ , **conservative policy iteration**算法会 

1. 每次policy update都改进 $\eta_{\mu}$ 

2. 在最多调用 $72 \frac{R^{2}}{\varepsilon^{2}}$ 次 $G_{\varepsilon}(\pi, \mu)$ 前停止 

3. 返回策略 $\pi$ 使得 $O P T\left(\mathbb{A}_{\pi, \mu}\right)<2 \varepsilon$

  证明在附录

- 为了完整回答问题3, 需要解决此算法输出的策略的质量。
- 注意，由于 $\text{OPT}\left(\mathbb A_{\pi, \mu}\right)<2 \varepsilon$ , 产出的策略$\pi$的性能确实取决于$\mu$，但算法停止的时间的界限并不取决于重启分布$\mu$ 
- 大致上, 一个策略要有接近最优的性能，那么所有的优势都必须是小的.  显然
- 不幸的是，如果$d_{\pi, \mu}$非常不均匀，那么一个小的最优策略优势并不一定意味着所有的优势都很小。

以下推论（定理6.2）限定了算法所找到的策略的性能 $\eta_{D}$ 。 使用 $l_{\infty}$ -norm, $\|f\|_{\infty} \equiv \max _{s} f(s)$

**Corollary  4.5.**  假设有 $\pi$ 满足 $O P T\left(A_{\pi, \mu}\right)<\varepsilon $ 那么, $\pi^*$是一个最优策略

$$
\begin{aligned}
\eta_D \left(\pi^* \right)-\eta_D (\pi) & \leq \frac{\varepsilon}{(1-\gamma)}\left \Vert\frac{d_{ \pi^*, D}}{d_{\pi, \mu}}\right \Vert_\infty \\
   & \leq \frac{\varepsilon}{(1-\gamma)^2}\left \Vert \frac{d_{ \pi^*, D}}{\mu}\right \Vert_\infty
   \end{aligned}
$$

- 其中 $\left \Vert \frac{ d_{ \pi^*, D } }{ d_{\pi, \mu} } \right \Vert_\infty$ 表示当前策略的状态分布于最优策略的状态分布的不匹配 , 并说明了 使用给定 start-state 分布 $D$ 而不是一个更均匀的分布. 
- 从本质上讲，更均匀的 $d_{\pi, \mu}$可以确保在最优策略执行中遇到的状态（由$d_{ \pi^*, D}$ 决定）的优势很小。
- 第二个不等式从$d_{\pi, \mu}(s) \geq(1-\gamma) \mu(s)$ 开始，它表明一个统一的度量可以防止这种不匹配变得任意大。下面开始证明。



#### Conservative Policy Iteration

为了简单起见，假设已经知道$\varepsilon$,  保守策略迭代算法为：

1. Call $G_{\varepsilon}(\pi, \mu)$ to obtain some $\pi^{\prime}$

2. Use $O\left(\frac{R^{2}}{\varepsilon^{2}} \log \frac{R^{2}}{\delta \varepsilon^{2}}\right)$ $\mu$ -restarts to obtain an $\frac{\varepsilon}{3}$-accurate estimate $\hat{\mathbb{A}}$ of $\mathbb{A}_{\pi, \mu}\left(\pi^{\prime}\right)$

3. If $\hat{\mathbb{A}}<\frac{2 \varepsilon}{3},$ STOP and return $\pi$

4. If $\hat{\mathbb{A}} \geq \frac{2 \varepsilon}{3},$ then update policy $\pi$ according to equation 4.1 using $\frac{(1-\gamma)\left(\hat{\mathbb{A}}-\frac{\varepsilon}{3}\right)}{4 R}$ and return to step 1.

   

- 其中，$\delta$ 是算法的失败概率。注意，步骤（2）的估计过程允许我们设置学习速率α。
- 我们现在指定步骤（2）的估计过程，以获得 $\frac{\varepsilon}{3}$ - $\mathbb{A}_{\pi}\left(\pi^{\prime}\right)$的精确估计值.
- policy advantage 可以记作: $\mathbb A_{\pi, \mu} (\pi' )=E_{s \sim d_{ \pi, \mu} }\left[\sum_a \left(\pi'(a ; s)-\pi(a ; s)\right) Q_{\pi}(s, a)\right]$
- 通过调用一次$\mu$ -restart distribution, 可以得到$\mathbb A_{\pi, \mu} (\pi')$ 的近乎无偏估计 $x_i$.  
- 为了从 $d_{\pi, \mu}$获取一个采样$s$, 我们得到一个trajectory, 从$s_{0} \sim \mu$ 开始, 并以概率 $(1-\gamma)$接受当前状态$s_{\tau}$ .  
- 然后从均匀分布选择一个动作a, 从状态s继续这个trajectory, 得到一个$Q_{\pi}(s, a)$ 近似无偏估计 $ \hat Q_\pi (s, a)$ . 
- 利用重要性采样, 从第$i$ -th个采样得出的策略优势的近似无偏估计为 $x_i = n_a \hat Q_i (s, a) ( \pi' (a ; s)- \pi (a ; s))$ .  假定每个轨迹运行的时间足够长, 所以 $x_{i}$的偏差小于$\frac{\varepsilon}{6}$

由于 $\hat Q_i \in[0, R]$,  我们的样本满足 $x_i \in [-n_a R, n_a R ]$ . 利用k独立同分布随机变量的Hoeffding不等式，我们得到：

$$
\operatorname{Pr}(|\mathbb{A}-\hat{\mathbb{A}}|>\Delta) \leq 2 e^{-\frac{k \Delta^{2}}{2 n_{a}^{2} R^{2}}}  \tag{5.1}
$$

这里 $\hat{\mathbb{A}}=\frac{1}{k} \sum_{i=1}^{k} x_{i}$   , $\mathbb A$是 $\frac{\varepsilon} {6} $ biased.  要获得一个$\Delta$准确的采样,带固定的错误率, 需要的轨迹数量是$O \left( \frac{n^2_a R^2}{\Delta^2} \right)$



#### 6 How Good is The Policy Found?

- 算法停止速度的界限并不取决于所使用的重启分布restart distribution。 
- 相比之下，现在表明，最终策略的质量可能在很大程度上取决于这种分布。 

引理 **Lemma 6.1.**   对所有策略, 所有starting state distribution $\mu$ , 有

$$
\eta_{\mu}(\tilde{\pi})-\eta_{\mu}(\pi)=\frac{1}{1-\gamma} E_{(a, s) \sim \tilde{\pi} d_{\tilde{\pi}, \mu}}\left[A_{\pi}(s, a)\right]
$$

Proof.  令 $P_{t}\left(s^{\prime}\right) \equiv P\left(s_{t}=s^{\prime} ; \tilde{\pi}, s_{0}=s\right)$ ,  起始状态是s 
由 $V_{\tilde{\pi}}(s)$ 的定义, 

$$
\begin{aligned}
V_{\tilde{\pi}}(s) =&(1-\gamma) \sum_{t=0}^{\infty} \gamma^{t} E_{\left(a_{t}, s_{t}\right) \sim \tilde{\pi} P_{t}}\left[\mathcal{R}\left(s_{t}, a_{t}\right)\right] \\
=& \sum_{t=0}^{\infty} \gamma^{t} E_{\left(a_{t}, s_{t}\right) \sim \tilde{\pi} P_{t}}\left[(1-\gamma) \mathcal{R}\left(s_{t}, a_{t}\right) +V_\pi(s_t)-V_\pi(s_t) \right] \\
=& \sum_{t=0}^{\infty} \gamma^{t} E_{\left(a_{t}, s_{t}\right) \sim \tilde{\pi} P_{t}  P_(s_{t+1} ; s_{t}, a_{t} )}  [(1-\gamma) \mathcal{R} (s_{t}, a_{t}) + \gamma V_\pi(s_{t+1}) -V_\pi(s_{t} ) ] + V_\pi(s) \\
=& V_{\pi}(s)+\sum_{t=0}^{\infty} \gamma^{t} E_{\left(a_{t}, s_{t}\right) \sim \tilde{\pi} P_{t}}\left[A_{\pi}\left(s_{t}, a_{t}\right)\right] \\
=& V_{\pi}(s)+\frac{1}{1-\gamma} E_{\left(a, s^{\prime}\right) \sim \tilde{\pi} d_{\tilde{\pi}, s}}\left[A_{\pi}\left(s^{\prime}, a\right)\right]
\end{aligned}
$$

- 推导过程中有个累加然后消去, 只留下最开始的$V_\pi(s)$ ;
- 该式说明, 两个策略的优劣, 从同一个状态出发, V_new - V_old 等于按照new来走, 遇到的每个状态下, 策略new来选择的action比策略old的action优势之和. 
- 该引理说明了一个基本的 **measure mismatch**.  
- 当$\alpha$比较小时, 性能度量 $\eta_{D}(\pi)$的改变 同比于 $\mathbb A_{\pi, D}  (\pi')$ , 对状态分布$d_{\pi, D}$的策略优势.  
- 然而, 对最优策略, $$\eta_{D}\left(\pi^{*}\right)$$ 与$$\eta_{D}(\pi)$$ 同步于 对状态分布 $$d_{\pi^*, D}$$ 的策略优势. 因此, 就算 对$\pi$ 和 $D$ 的最优策略优势很小, 对$$d_{\pi^*, D}$$ 的策略优势未必很小.   这促使使用更均匀的分布 $\mu$
- 算法停止后, 会返回一个对 $\mu$ 有较小策略优势的 策略 $\pi$ .  
- 下面量化, 对任意测量指标$\tilde{\mu}$, 该策略离最优策略有多远.   

**Theorem 6.2.**   假定对 $\pi$, 有 $O P T (\mathbb A_{\pi, \mu})<\varepsilon .$  对任意状态分布 $\tilde{\mu}$  

$$
\begin{aligned}
\eta_{\tilde{\mu}}\left(\pi^{*}\right)-\eta_{\tilde{\mu}}(\pi) & \leq \frac{\varepsilon}{(1-\gamma)}\left \Vert \frac{d_{\pi ^ * , \tilde \mu }}{d_{\pi, \mu}}\right \Vert_{\infty} \\
& \leq \frac{\varepsilon}{(1-\gamma)^{2}}\left  \Vert \frac{d_{\pi^* , \tilde \mu}}{\mu}\right  \Vert_{\infty}
\end{aligned}
$$

Proof. 	最优策略优势  $$ \text{OPT}  (\mathbb A_{\pi, \mu} )= \sum_s d_{\pi, \mu}(s) \max _a A_\pi (s, a)$$ , 因此


$$
\begin{aligned} 
\varepsilon & > \sum_{s} \frac{d_{\pi, \mu}(s)}{d_{\pi^{*}, \tilde{\mu}}(s)} d_{\pi^{*}, \tilde{\mu}}(s) \max_{a} A_{\pi}(s, a) \\
& \geq {\min {s}\left(\frac{d_{\pi, \mu}(s)}{d_{\pi^{*}, \tilde{\mu}}(s)}\right) \sum_{s} d_{\pi^{*}, \tilde{\mu}}(s) \max_{a} A_{\pi}(s, a)} \\
& \geq {\left \Vert \frac{d_{\pi^{*}, \tilde{\mu}}}{d_{\pi, \mu}}\right \Vert_{\infty}^{-1} \sum_{s, a} d_{\pi^{*}, \tilde{\mu}}(s) \pi^{*}(a ; s) A_{\pi}(s, a)} \\
& = {(1-\gamma)\left \Vert \frac{d_{\pi^{*}, \tilde{\mu}}}{d_{\pi, \mu}}\right \Vert_{\infty}^{-1}\left(\eta_{\tilde{\mu}}\left(\pi^{*}\right)-\eta_{\tilde{\mu}}(\pi)\right)}
\end{aligned}
$$



最后一步由引理 6.1.    第二个不等于 由  $d_{\pi, \mu}(s) \leq(1-\gamma) \mu(s)$

注意,   $\left \Vert \frac{ d_{ \pi^*, \tilde \mu } }{   \mu  } \right \Vert_\infty$  是基于 $\mu$ 来 度量 不匹配的, 而不是最优策略的 future-state distribution.

关于 $\frac{1}{1-\gamma}$ 的解释

1. 最优策略与$\pi$的性能差是   $\frac{1}{1-\gamma}$ 乘以 对 $d_{\pi^{*}, \tilde{\mu}}(s)$ 的平均优势 
2. $d_{\pi, \mu}$ 本身的不均一性  ( 因为$d_{\pi, \mu}(s) \leq(1-\gamma) \mu(s) $)



#### Discussion

前面提出了一个算法, 找到一个 "approximately" optimal solution that is polynomial in the approximation parameter $\varepsilon,$ but not in the size of the state space. 

##### The Greedy Policy Chooser

找到一个具有大的策略优势的策略可以认为是一个回归问题, 但是没有解决该问题的复杂度. 

$$
E_{s \sim d_{\pi, \mu}} \max _{a}\left|A_{\pi}(s, a)-f_{\pi}(s, a)\right|
$$

loss 是状态空间上的平均误差 (是个在所有动作上的 $\infty$ -loss ). 如果可以使该误差小于 $\frac{\varepsilon}{2},$ 那么可以构建一个$\varepsilon$-greedy policy chooser , 通过选择 基于近似函数 $f_{\pi}$的贪婪策略.  这个回归问题的 $l_1$ 条件要弱, 与最小化在所有状态空间上的 $l_{\infty}$ -error 相比, 是greedy dynamic programming的相对误差. 

Direct policy search 也可以被用于 greedy  policy chooser.



##### What about improving $\eta_D $?

"Can we improve the роlісу ассоrdіng tо bоth $\eta_D$ аnd $\eta_\mu$ аt еасh uрdаtе?"
In general the answer is "no", but consider improving the performance under $\tilde \mu= (1- \beta)\mu + \beta D$ instead of just $\mu$. This metric only slightly changes the quality of the asymptotic policy. However by giving weight to D, the possibility of improving $\eta_D$ is allowed if the optimal policy has large advantages under D, though we do not formalize this here. <u>The only situation where joint improvement with $\eta_D$  is not possible is when $\text{OPT}(\mathbb A_{\pi,D}) $ is small.</u> However, this is the problematic case where, under D, the large advantages are not at states visited frequently.  

##### Implications of the mismatch

The bounds we have presented directly show the importance of ensuring the agent <u>starts in states where</u>
<u>the optimal policy tends to visit</u>. It also suggests that certain optimal policies are easier to learn in large state spaces-namely those optimal policies which tend to visit a significant fraction of the state space. An interesting suggestion for how to choose u, is to <u>use prior knowledge</u> of which states an optimal policy tends to visit. 能用先验知识就用. 



#### Appendix: Proofs

定理 4.1 的直观解释:  $\alpha$ 决定从 $\pi^{\prime}$ 选择 action 的概率. 如果当前状态分布是 $d_{\pi, \mu}$, 当从 $\pi^{\prime}$ 选择一个action, 那么性能的改进与策略优势成正比. proportional to the policy advantage.   当从 $\pi^{\prime}$ 选择一个action, 由于状态分布不完全等同于$d_{\pi, \mu}$ , 所以证明还涉及限制性能下降的下界. 

Proof. Throughout the proof, the $\mu$ dependence is not explicitly stated. For any state $s$. 对一个状态, 换一个策略,能得到的性能提升: 

$$
\begin{aligned}
& \sum_{a} \pi_{\mathrm{new}}(s ; a) A_{\pi}(s, a) \\
=& \sum_{a}\left((1-\alpha) \pi(a ; s)+\alpha \pi^{\prime}(a ; s)\right) A_{\pi}(s, a) \\
=& \alpha \sum \pi^{\prime}(a ; s) A_{\pi}(s, a)
\end{aligned}
$$

where we have used $\sum_{a} \pi(a ; s) A_{\pi}(s, a)=0 $  ;   按照之前的策略, 提升为0, 显然以及重要

- 关键点: 新策略若在某些情况下没选择a,则该路径等同于旧策略. 
- 由于换了策略,  (s,a)的分布将会改变;  这里从action的角度进行sum, 将action的选取看成两部分, 从旧策略以及从新策略; 从旧策略的提升为0直接不看; 从新策略里面选择, 分为该action在时间t之前未被选的几率以及选择过的几率; 如果t 时刻在s下, 刨去之前策略选择a的几率, 新策略碰巧都没有选择a, 则在a上面, 跟旧策略一样;   
- 对任意时间步 t, 有从  $\pi^{\prime}$ 中选择某个action的几率是 $\alpha .$ 令 $c_{t}$ 表示在 t 之前从 $\pi^{\prime}$ 中选择action的次数, 因此 $\operatorname{Pr}\left(c_{t}=0\right)=(1-\alpha)^{t} .$ 令  $\rho_{t} \equiv \operatorname{Pr}\left(c_{t} \geq 1\right)=1-(1-\alpha)^{t}$ , $P\left(s_{t} ; \pi\right)$ 是在t时刻的状态分布, 初始状态服从 $s \sim \mu,$ 则有

$$
\begin{aligned}
& E_{s \sim P\left(s_{t} ; \pi_{\text {new }}\right)}\left[\sum_{a} \pi_{\text {new }}(a ; s) A_{\pi}(s, a)\right] \\
=& \alpha E_{s \sim P\left(s_{t} ; \pi_{\text {new }}\right)}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}(s, a)\right] \\
=& \alpha\left(1-\rho_{t}\right) E_{s \sim P\left(s_{t} | c_{t}=0 ; \pi_{\text {new }}\right)}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}(s, a)\right] \\
& + \alpha \rho_{t}  E_{s \sim P\left(s_{t} | c_{t}\ge 1 ; \pi_{\text {new }}\right)}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}(s, a)\right] \\
\ge & \alpha E_{s \sim P\left(s_{t} | c_{t}=0 ; \pi_{\text {new }}\right)}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}(s, a)\right] -2 \alpha \rho_{t} \varepsilon \\
= & \alpha E_{s \sim P\left(s_{t} ; \pi\right)}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}(s, a)\right]-2 \alpha \rho_{t} \varepsilon
\end{aligned}
$$

where we have used the definition of $$\varepsilon$ and $P\left(s_{t} | c_{t}=0 ; \pi_{\mathrm{new}}\right)=P\left(s_{t} ; \pi\right)$$  
By substitution and lemma 6.1, we have:
$$
\begin{aligned}
& \eta_{\mu}\left(\pi_{\text {new }}\right)-\eta_{\mu}(\pi) \\
=& \sum_{t=0}^{\infty} \gamma^{t} E_{s \sim P\left(s_{t} ; \pi_{\text {new }}\right)}\left[\sum_{a} \pi_{\text {new }}(a ; s) A_{\pi}(s, a)\right] \\
\geq & \alpha \sum_{t=0}^{\infty} \gamma^{t} E_{s \sim P \left(s_{t} ; \pi\right)}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}(s, a)\right]  
  - 2 \alpha \varepsilon \sum_{t=0}^{\infty} \gamma^t(1-(1-\alpha)^t)  \\
=& \frac{\alpha}{1-\gamma} E_{ s \sim d_{\pi}}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}\left(s_{t}, a\right)\right]  
  - 2 \alpha \varepsilon (\frac{1}{1-\gamma}-\frac{1}{1-\gamma(1-\alpha)} ) 
\end{aligned}
$$



定理 4.4 证明:
Proof. 

- 在4.3定义的步骤 $(2),$ 需要足够多的样本,使得每次循环满足 $|\mathbb A -\hat{\mathbb A}|<\frac{\varepsilon}{3}$ , 并且下面证明,需要最多 $\frac{72 R^{2}}{\varepsilon^{2}}$ 次循环. 
- If we demand that the probability of failure is less than $\delta,$ then by the union bound and inequality $5.1,$ we have for $k$ trajectories $P(\text { failure })<\frac{72 R^{2}}{\varepsilon^{2}} 2 e^{-\frac{k \varepsilon^{2}}{36 n_{a}^{2} R^{2}}}<\delta$ , where we have taken $\Delta=\frac{\varepsilon}{6}$ since the bias in our estimates is at most $\frac{\varepsilon}{6} .$ 
- Thus, we require $O\left(\frac{R^{2}}{\varepsilon^{2}} \log \frac{R^{2}}{\delta \varepsilon^{2}}\right)$ trajectories.
- Thus, if $\hat{\mathbb A} \geq \frac{2 \varepsilon}{3},$ then $\mathbb A \geq \frac{\varepsilon}{3}>0 .$ 
- By corollary 4.2, step 4 guarantees improvement of $\eta_{\mu}$ by at least $\frac{\left(\hat{\mathbb A}-\frac{\epsilon}{3}\right)^{2}}{8 R} \geq \frac{\varepsilon^{2}}{72 R}$ using $\alpha=\frac{(1-\gamma)\left(\hat{\mathbb A}-\frac{\varepsilon}{3}\right)}{4 R},$ which proves $i$.
- since $0 \leq \eta_{\mu} \leq R,$ there are at most $\frac{72 R^{2}}{\varepsilon^{2}}$ steps before $\eta_{\mu}$ becomes larger than $R,$ so the algorithm must cease in this time, which proves $ii$. 
- In order to cease and return $\pi,$ on the penultimate loop, the $G_{\varepsilon}$ must have returned some $\pi^{\prime}$ such that $\hat {\mathbb A} <\frac{2 \varepsilon}{3},$ which implies $\mathbb A_{\pi}\left(\pi^{\prime}\right)<\varepsilon $.   By definition of $G_{\varepsilon},$ it follows that $\mathrm{OPT}\left(\mathbb A_{\pi}\right)<2 \varepsilon$ which proves $iii$.  

















