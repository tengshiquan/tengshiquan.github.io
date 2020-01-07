---
layout:     post
title:      Approximately Optimal Approximate Reinforcement Learning
subtitle:   Note on "Approximately Optimal Approximate Reinforcement Learning"
date:       2020-01-01 12:00:00
author:     "tengshiquan"
header-img: "img/about-bg.jpg"
catalog: true
tags:
    - AI
    - Reinforcement Learning

---



# Note on "Approximately Optimal Approximate Reinforcement Learning"

论文笔记.  这个标题有点绕.  近似最优的 近似强化学习



#### Abstract

为了解决现实的强化学习问题，使用近似算法(approximate algorithms)至关重要。 本文提出了一种保守的策略迭代算法(**conservative policy iteration, CPI**),  配合 重启分布 restart distribution（从一个特定的分布中取next state）和 近似贪婪策略选择器(approximate greedy policy chooser), 该算法可以寻找 “近似”最优策略。**greedy policy chooser** 输出 一个新策略: 在当前策略下,通常选择有最大action-value的那些actions , 即，它输出“近似”贪婪策略。 greedy policy chooser 可以用常规的值函数近似(value funtion approximation)技术来实现.  

CPI算法: 1. 保证改进 2. 保证停止 3. 返回一个"近似"最优策略.  2,3取决于greedy policy chooser的质量, 不显式(explicitly)取决于状态空间的大小。



#### Introduction

强化学习领域, 两个已经成功的常见方法: 贪婪动态规划(**greedy dynamic programming**) 和 策略梯度(**policy gradient**).  然而, 两个方法都可能无法有效改进策略.   论文里,  greedy dynamic programming 和 approximate value function都是指,  先value estimating 然后再迭代策略的方法. 

对GDP,  近似值函数(Approximate value function)方法缺少有力的理论上的性能保证. 

PG算法需要太多的sample来准确计算梯度, 因为PG将 exploration 和 exploitation 交替在一起. 

本论文中,  考虑一个设置setting,  设定我们的算法拥有 restart distribution 和 greedy policy chooser。

Restart distribution 允许agent从自己设计的一个固定的分布里获取next state.  通过分布更平均的restart distribution, agent可以获取平时不一定访问的状态的信息.

Greedy policy chooser是一个黑盒, 输出一个新策略, 一般(on average)选择较当前策略有大幅优势的动作, 即, 产出一个近似贪婪策略.  Greedy policy chooser 可以用 对value function的回归算法来实现.

作者基于常见算法的优点, 提出 conservative policy iteration 算法.  关键要素: 

1. 在更加均匀的状态空间(state space)上, 改进策略 
2. 执行更保守的策略更新，新策略是当前策略和贪婪策略的混合.   

换言之, 1. 体现 探索性  2. 避免greedy dynamic programming的缺陷,  greedy dynamic programming直接使用近似贪婪策略, 可能使得策略退化(degradation)

作者证明该算法在"很少"的步数内就可以收敛, 并返回一个近似最优策略. 该策略的性能,并且不显式取决于状态空间大小.



#### Preliminaries

$D$  : starting state distribution

$\mathcal R$ : reward funtion ,   $\mathcal R : S \times A \to [0, R]$

$\pi(a;s)$ :  在state s 下选取 action a 的probability,  ; 分号用于区分 参数 与 概率分布的随机变量

**Definition 2.1.**   **$\mu$ restart distribution** : 从 distribution $\mu$ 里选取 next state.(应该是指 restart state)

restart distribution 是 生成模型(**generative model**, 参考5)的一个弱化版. 这两者都比掌握 full transition model 要弱. 而这些又都比 "irreversible" experience 要强.  在"irreversible" experience 中, agent只能遍历一整个trajectory, 无法随时reset进入另外一个trajectory. 

$\mu$ 只要是一个相对均匀的分布,  (不必须与$D$ 一样),  就可以避免显式地去探索 (explicit exploration).

**Value function**:  discounted average reward
$$
V_{\pi}(s) \equiv(1-\gamma) E\left[\sum_{t=0}^{\infty} \gamma^{t} \mathcal{R}\left(s_{t}, a_{t}\right) | \pi, s\right]
$$

where $s_t$ and $a_t$ are random variables  at t upon executing $\pi$ from starting state s.  这里V使用了normalized 值, $V_\pi(s) \in [0,R]$ .

**action-value**:
$$
Q_\pi(s,a) \equiv (1 - \gamma)\mathcal R(s,a) + \gamma E_{s' \sim P(s';s,a)}[V_\pi(s')]
$$

**advantage**:  s下, 选择某个特定a比按照$\pi$本来的策略要好多少.
$$
A_\pi(s,a) \equiv  Q_\pi(s,a) - V_\pi(s)
$$

由于归一化normalization,  $Q\in [0, R]$ $A(s,a)\in [-R, R]$

**$\gamma$-discounted future state distribution** for  a <u>starting distribution</u> $\mu$ :
$$
d_{\pi, \mu}(s) \equiv(1-\gamma) \sum_{t=0}^{\infty} \gamma^{t} \operatorname{Pr}\left(s_{t}=s ; \pi, \mu\right)
$$

$1-\gamma$ 是normalization所必须的.  上式表示, 起始状态的选取符合$\mu$ 分布,   按照$\pi$来执行, 之后所有state可能为s的 discounted 几率和, 折扣后占比.

$d_{\pi,s}$ 表示, 从状态s开始的 discounted future state distribution.  则有
$$
V_{\pi}(s)  = E_{(a',s') \sim \pi d_{\pi, s}} [\mathcal{R} (s', a')]
$$

当$\gamma \to 1$ , $d_{\pi, s}$ 趋向于 非折扣环境undiscounted setting下 所有状态的平稳分布 stationary distribution, 占比

**goal**:  maximize discounted reward from  start distribution $D$
$$
\eta_D(\pi) \equiv E_{s \sim D} [V_\pi(s)]
$$

$$
\eta_D(\pi)  = E_{(a,s) \sim \pi d_{\pi, D}} [\mathcal{R} (s, a)]
$$

众所周知的结果是，存在一个同时使所有状态的$V_\pi(s)$最大化的策略。



#### The Problems with Current Methods

三个问题:

1. 有什么性能指标是保证每一步都有进步？

2. 验证特定更新是否改进了此性能指标的 困难程度？
3. 策略更新次数后，将获得什么性能级别？



##### Approximate Value Function Methods

精确值函数方法(exact value function, 指 tabular之类方法, 不是用函数来拟合value estimating), 如 policy iteration,  PI对$\pi$, 计算$Q_\pi(s, a)$, 然后创建新的deterministic policy $\pi'$ such that   $\pi'(а;ѕ)=1$ іff $\arg \max_aQ(s,a)$
重复该过程直到 state-action values 收敛到最佳值optimal values.  Exact value function 有很强的边界，表明值收敛到最佳值的速度有多快(参考7)



基于近似值函数的策略的性能缺乏理论研究结果, 这会导致该方法对三个问题的答案都很弱。

考虑一个近似函数 $\tilde{V}(s)$ 误差的 $l_{\infty}$ 范数, $l_{\infty}$-error :

$$
\varepsilon=\max _{s}\left|\tilde{V}(s)-V_{\pi}(s)\right|
$$

$\pi$ 是任意策略, $\pi^{\prime}$ 是该策略的近似基础上的greedy policy.  由参考3 , 对所有状态s, 有下式成立:

$$
{V_{\pi^{\prime}}(s) \geq V_{\pi}(s)-\frac{2 \gamma \varepsilon}{1-\gamma}}  \tag{3.1}
$$

也就是是, 性能绝对不会下降超过 $\frac{2 \gamma \varepsilon}{1-\gamma}$ .  不能回答问题2, 因为不保证改进, 并且没有定义怎么度量.

对approximate方法, 花费多少time能达到一定的性能level 也没有很好的搞清楚.



##### Policy Gradient Methods

PG算法尝试通过沿着未来reward的梯度, 在限定的一类策略中 找到一个好策略. 

根据参考8 sutton, 下式计算梯度
$$
\quad \nabla \eta_{D}=\sum_{s, a} d_{\pi, D}(s) \nabla \pi(a ; s) Q_{\pi}(s, a) \tag{3.2}
$$

PG算法对问题1有个不错的答案, 因为在梯度提升的情况下, 性能会保证改进.   对问题2, 要判定梯度的方向是困难的.  我们发现,梯度方法缺乏探索性(lack of exploration)意味着需要大量样本才能准确估计梯度方向。

<img src="/img/2020-01-01-OptimalRL.assets/image-20200104205347255.png" alt="image-20200104205347255" style="zoom: 25%;" />

一个agent, 两个action, 左右.  在两个action几率一样的情况下, 从最左边到达目标最右边, 期望时间是 $3\left(2^{n}-n-1\right),$ 当 $n=50,$ 大约是 $10^{15}$.  这个MDP属于一类MDPs, 随机操作更有可能增加到目标状态的距离,越来越远。对这类问题, 使用无方向的探索,即随机游走, 到达目标的预期时间是状态空间大小的指数级别. 因此, 任何 "on-policy" 方法都必须必须走这么长,才能找到改进的点.  lack of exploration.

在没有达到目标状态的情况下，对梯度的任何合理的估计都将为0，并且获得非0估计值需要使用“on-policy”样本的指数级时间。 Importance sampling 方法对于此类问题而言并不可行。 因为如果agent可以遵循一些“off-policy”的轨迹在合理的时间内达到目标状态，则重要性权重必须是指数级别。

0估计是一个相当准确的梯度估计,在数量级上.  但0没有提供方向这个关键信息.  参考[2]表明, 只需要一个相对较小的样本大小即可准确估计数量级，但如果梯度较小则意味着方向不准确。 不幸的是，当政策远未达到最优时，梯度的大小可能非常小。

<img src="/img/2020-01-01-OptimalRL.assets/image-20200104205436283.png" alt="image-20200104205436283" style="zoom: 33%;" />

上图左边是个MDP,  使用 Gibbs table-lookup distributions, $\{\pi_{\theta}: \pi(a ; s) \propto \exp \left(\theta_{s a}\right) \} $ 在i处增加自循环的机会会降低j的平稳概率，从而妨碍在j状态下的学习. 
初始化一个策略,  $\rho(i)=.8$,  $\rho(j)=.2$,  $\pi(a_1;i) =.8$ , $\pi(a_1;j) =.9$ ;  在i处学习必然会影响在j处学习, 造成一个很平坦的高原. 从下图看出, 按照刚才的初始化, $\rho(j)$ 会降到 $10^{-7}$, 非常严重.   就像在例1中一样, 要获得非0的梯度估值必须访问 j.  如果上图左边状态更多一些, 则问题更加严重. 

<img src="/img/2020-01-01-OptimalRL.assets/image-20200104205515790.png" alt="image-20200104205515790" style="zoom: 33%;" />

虽然可能会渐进地 asymptotically 找到一个好的策略，但这些结果对于问题3的答案不是好兆头，问题3涉及到找到这样一个策略的速度。这些结果表明，在任何合理数量的步骤中，梯度方法可能最终被困在平台上，其中估计梯度方向具有不合理的大样本复杂度。回答问题3对于理解梯度方法的性能至关重要，但目前没相关知识。



#### Approximately Optimal RL

PG算法的问题是，对不太可能发生的状态 (unlikely states)下的政策改进不敏感，尽管这些不可能状态下的政策改进可能对寻求最优策略是必要的。我们希望有一个替代性的性能指标，不会降低unlikely states 或者 actions 的权重。性能度量的一个候选者是更均匀地衡量来自所有状态的改进(不是$\mathcal D$). 

$$
\eta_\mu(\pi) \equiv E_{s \sim \mu}[V_\pi(s)]
$$

这里, $\mu$ 是一个'exploratory' restart distribution.     
初始状态从更加uniform的$\mu$中选取,之前一些访问不到的状态都可以作为起始状态, 相当于强制visit

[^_^]: 能否根据熵来判断这些state的权重. 

下面的问题是, 能否有最优策略在最大化 $\eta_\mu$ 的同时, 保证在$\eta_D$ 下也是一个好的策略.  毕竟出发点的分布改变了.

任何最优策略都能同时最大化  $\eta_\mu$ 和 $\eta_D$.  但是, 能最大化$\eta_\mu$ 的有限类型的策略, 可能在$\eta_D$上表现不好. 所以必须确保最大化$\eta_\mu$ 的策略在$\eta_D$上也是好的策略.

**Greedy policy iteration** updates the policy to some $\pi'$ based on some approximate state-action values.
conservative update rule: 
$$
\pi_{new}(a;s) = (1 - \alpha)\pi(a;s) + \alpha \pi'(a;s) \tag{4.1}
$$

当$\alpha = 1$, 为了保证改进, $\pi'$ 必须在每个状态都选择更好的action, 否则可能像公式3.1那样造成性能下降.



#### Policy Improvement

一个更合理的情况是,  取$\alpha > 0$, 我们可以通过一个可以某些状态取更好action策略$\pi'$ (不要求在所有states)来改进策略

定义优势函数 ,  advantage of $\pi'$ 

$$
\mathbb{A}_{\pi, \mu}\left(\pi^{\prime}\right) \equiv E_{s \sim d_{\pi, \mu}}\left[E_{a \sim \pi^{\prime}(a ; s)}\left[A_{\pi}(s, a)\right]\right]
$$

公式含义, 里面的期望是 在s的时候, 按$\pi'$能比$\pi$好多少,  外面的期望是, s的分布按照$\pi$ 来一遍.

该优势函数可以衡量$\pi ^ {\prime}$选择actions具有的优势程度，起始状态  $s \sim \mu$ , with respect to  the set of states visited under $\pi$.  注意，如果一步 policy improvement 能使得 policy advantage 最大, 则找到了一个更好的策略.

利用公式3.2 可得  $\frac{\partial \eta_{\mu}}{\partial \alpha} \vert_{\alpha=0}= \frac{1}{1-\gamma} \mathbb{A}_{\pi, \mu}$ ,   ???? 

所以 $\eta_{\mu}$ 上的改变就是:

$$
\quad \Delta \eta_{\mu}=\frac{\alpha}{1-\gamma} \mathbb{A}_{\pi, \mu} (\pi')+O (\alpha ^{2}) \tag{4.2}
$$

因此, 对足够小的$\alpha$, 如果policy advantage 是正的, 则会 policy improvement. 如果极端情况$\alpha =1 $, 则可能造成退化. 将这两者结合起来, 来决定策略改进多少是可能的.

**Theorem 4.1.**  用$\mathbb{A}$表示  $\pi^{\prime}$ 的关于$\pi , \mu$ 的 policy advantage. 
$\varepsilon= \max_{s} \vert E_{a \sim \pi'(a ; s)} [A_{\pi}(s, a) ] \vert$  

for all $\alpha \in[0,1]:$

$$
\eta_{\mu}\left(\pi_{n e w}\right)-\eta_{\mu}(\pi) \geq \frac{\alpha}{1-\gamma}\left(\mathbb{A}-\frac{2 \alpha \gamma \varepsilon}{1-\gamma(1-\alpha)}\right)
$$

可以构造一个具有两个状态的例子, 来说明这个边界对所有的$\alpha$都是紧的. 

第一项类似于公式4.2里面的一阶增长, 第二项是惩罚项.  

如果 $\alpha=1,$ 边界减少到:
$$
\eta_{\mu}\left(\pi_{\mathrm{new}}\right)-\eta_{\mu}(\pi) \geq \frac{\mathbb{A}}{1-\gamma}-\frac{2 \gamma \varepsilon}{1-\gamma}
$$
惩罚项的形式与 greedy dynamic programming 里的一样 ,  $\varepsilon$ 与公式3.1定义的 $l_{\infty}$ 类似 .

下面的结论显示,  policy advantage 越大, 保证的性能改进就越大. 

**Corollary 4.2.** 令最大可能的reward为 $R$,  如果 $\mathbb{A} \geq 0,$  然后令 $\alpha=\frac{(1-\gamma) \mathbb{A}}{4 R}$ 可以保证按下式改进策略:
$$
\eta_{\mu}\left(\pi_{n e w}\right)-\eta_{\mu}(\pi) \geq \frac{\mathbb{A}^{2}}{8 R}
$$

Proof. 基于之前的结论, 改变的大小被限定在 $\frac{\alpha}{1-\gamma}\left(\mathbb{A}-\alpha \frac{2 R}{1-\gamma}\right)$ , 然后$\alpha$取能使边界最大的值.

##### Answering question 3

要解决问题3, 首先解决 收敛到某个策略的速度，然后限定该策略的质量。我们期望能够获得具有较大优势的政策，从而影响改进的速度和最终政策的质量。 Instead of explicitly suggesting algorithms that find policies with large policy advantages, we assume access to an $\varepsilon$-greedy policy chooser that solves this problem.  叫作$ \varepsilon$-good algorithm $G_{\varepsilon}(\pi, \mu)$, 定义如下:

**Definition  4.3.**   **$\varepsilon$ -greedy policy chooser**, $G_{\varepsilon}(\pi, \mu),$ 是 $\pi$ 和 $\mu$ 的函数, 返回一个策略 $\pi^{\prime}$ , 

使得 $ \mathbb A_{\pi, \mu} (\pi') \geq \text{OPT} ( \mathbb A_{\pi, \mu} ) -\varepsilon$   , 其中 $\operatorname{OPT} (\mathbb A_{\pi, \mu} ) \equiv \max_{\pi^{\prime}} \mathbb A_{\pi, \mu} (\pi')$

作者证明, 使用一个回归算法, 以平均误差$\frac{\varepsilon}{2}$来拟合 advantage 足够构造这样的 $G_{\varepsilon} .$
"break point":  当 greedy policy chooser 无法保证返回一个 正数的advantage的策略的时候, 即当 $\mathrm{OPT}\left(\mathrm{A}_{\pi, \mu}\right)<\varepsilon$时, 无法保证改进. 

**Conservative Policy Iteration** 算法:

1. Call $G_{\varepsilon}(\pi, \mu)$ to obtain some $\pi'$
2. Estimate the policy advantage $\mathbb{A}_{\pi, \mu}\left(\pi^{\prime}\right)$
3. If the policy advantage is small (less than $\varepsilon$ ) STOP and return $\pi$.
4. Else, update the policy and go to (1).



为了简化问题, 下面假设 $\varepsilon$ 已知.  根据greedy policy chooser的定义,  可以得出 $\pi$ 的optimal policy advantage小于 $2 \varepsilon .$ 

下面的定理表明, 在多项式的时间内, 完整的算法会找到一个策略, 接近"break point" .  

**Theorem 4.4.** 至少以概率 $1-\delta$ conservative policy iteration: 

1. 每次policy update都改进 $\eta_{\mu}$ 

2. 在最多调用 $72 \frac{R^{2}}{\varepsilon^{2}}$ 次 $G_{\varepsilon}(\pi, \mu)$ 前停止 

3. 返回策略 $\pi$ 使得 $O P T\left(\mathbb{A}_{\pi, \mu}\right)<2 \varepsilon$

   

为了完整回答问题3, 需要解决此算法发现的策略的质量。请注意，尽管我们发现的策略$\pi$的性能确实取决于$\mu$，但算法停止的时间的界限并不取决于重启分布$\mu$，因为$\text{OPT}\left(\mathbb A_{\pi, \mu}\right)<2 \varepsilon$ .  大致上, 一个策略要有接近最优的性能，那么所有的优势都必须是小的。不幸的是，如果$d_{\pi, \mu}$非常不均匀，那么一个小的最优策略优势并不一定意味着所有的优势都很小。

以下推论（定理6.2）限定了算法所找到的策略的性能 $\eta_{D}$ 。 使用 $l_{\infty}$ -norm, $\|f\|_{\infty} \equiv \max _{s} f(s)$

**Corollary  4.5.**  假设有 $\pi$ 满足 $O P T\left(A_{\pi, \mu}\right)<\varepsilon $ 那么, 

$$
\begin{aligned}
\eta_D \left(\pi^* \right)-\eta_D (\pi) & \leq \frac{\varepsilon}{(1-\gamma)}\left \Vert\frac{d_{ \pi^*, D}}{d_{\pi, \mu}}\right \Vert_\infty \\
   & \leq \frac{\varepsilon}{(1-\gamma)^2}\left \Vert \frac{d_{ \pi^*, D}}{\mu}\right \Vert_\infty
   \end{aligned}
$$

其中 $\left \Vert \frac{ d_{ \pi^*, D } }{ d_{\pi, \mu} } \right \Vert_\infty$ 表示当前策略的状态分布于最优策略不匹配, 

并说明了 使用给定 start-state 分布 $D$ 而不是一个更均匀的分布的问题. 从本质上讲，更均匀的 $d_{\pi, \mu}$可以确保在最优策略访问的状态下（由$d_{ \pi^*, D}$ 决定）的优势很小。第二个不等式从$d_{\pi, \mu}(s) \geq(1-\gamma) \mu(s)$ 开始，它表明一个统一的度量可以防止这种不匹配变得任意大。下面开始证明。



#### Conservative Policy Iteration

为了简单起见，假设已经知道$\varepsilon$,  保守策略迭代算法为：

1. Call $G_{\varepsilon}(\pi, \mu)$ to obtain some $\pi^{\prime}$

2. Use $O\left(\frac{R^{2}}{\varepsilon^{2}} \log \frac{R^{2}}{\delta \varepsilon^{2}}\right)$ $\mu$ -restarts to obtain an $\frac{\varepsilon}{3}$-accurate estimate $\hat{\mathbb{A}}$ of $\mathbb{A}_{\pi, \mu}\left(\pi^{\prime}\right)$

3. If $\hat{\mathbb{A}}<\frac{2 \varepsilon}{3},$ STOP and return $\pi$

4. If $\hat{\mathbb{A}} \geq \frac{2 \varepsilon}{3},$ then update policy $\pi$ according to equation 4.1 using $\frac{(1-\gamma)\left(\hat{\mathbb{A}}-\frac{\varepsilon}{3}\right)}{4 R}$ and return to step 1.

   

其中，$\delta$ 是算法的失败概率。注意，步骤（2）的估计过程允许我们设置学习速率α。我们现在指定步骤（2）的估计过程，以获得 $\frac{\varepsilon}{3}$ - $\mathbb{A}_{\pi}\left(\pi^{\prime}\right)$的精确估计值.

policy advantage 可以记作: $\mathbb A_{\pi, \mu} (\pi' )=E_{s \sim d_{ \pi, \mu} }\left[\sum_a \left(\pi'(a ; s)-\pi(a ; s)\right) Q_{\pi}(s, a)\right]$

通过调用一次$\mu$ -restart distribution(定义2.1), 可以得到$\mathbb A_{\pi, \mu} (\pi')$ 的近乎无偏估计 $x_i$.  为了从 $d_{\pi, \mu}$获取一个采样$s$, 我们得到一个trajectory, 从$s_{0} \sim \mu$ 开始, 并以概率 $(1-\gamma)$接受当前状态$s_{\tau}$ .  然后从均匀分布选择一个动作a, 从状态s继续这个trajectory, 得到一个$Q_{\pi}(s, a)$ 近似无偏估计 $ \hat Q_\pi (s, a)$ . 利用重要性采样, 从第$i$ -th个采样得出的策略优势的近似无偏估计为 $x_i = n_a \hat Q_i (s, a) ( \pi' (a ; s)- \pi (a ; s))$ .  假定每个轨迹运行的时间足够长, 所以 $x_{i}$的偏差小于$\frac{\varepsilon}{6}$

由于 $\hat Q_i \in[0, R]$,  我们的样本满足 $x_i \in [-n_a R, n_a R ]$ . 利用k独立同分布随机变量的Hoeffding不等式，我们得到：

$$
\operatorname{Pr}(|\mathbb{A}-\hat{\mathbb{A}}|>\Delta) \leq 2 e^{-\frac{k \Delta^{2}}{2 n_{a}^{2} R^{2}}}  \tag{5.1}
$$

这里 $\hat{\mathbb{A}}=\frac{1}{k} \sum_{i=1}^{k} x_{i}$   , $\mathbb A$是 $\frac{\varepsilon} {6} $ biased.  要获得一个$\Delta$准确的采样,带固定的错误率, 需要的轨迹数量是$O \left( \frac{n^2_a R^2}{\Delta^2} \right)$



#### How Good is The Policy Found?

回想一下，我们的算法停止速度的界限并不取决于所使用的重启分布restart distribution。 相比之下，现在表明，最终策略的质量可能在很大程度上取决于这种分布。 引理

**Lemma 6.1.**   对所有策略, 所有starting state distribution $\mu$ , 有

$$
\eta_{\mu}(\tilde{\pi})-\eta_{\mu}(\pi)=\frac{1}{1-\gamma} E_{(a, s) \sim \tilde{\pi} d_{\tilde{\pi}, \mu}}\left[A_{\pi}(s, a)\right]
$$

Proof.  令 $P_{t}\left(s^{\prime}\right) \equiv P\left(s_{t}=s^{\prime} ; \tilde{\pi}, s_{0}=s\right)$ 
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



该引理说明了一个基本的 不匹配的度量.  当$\alpha$比较小时, 性能度量 $\eta_{D}(\pi)$的改变 同比于 $\mathbb A_{\pi, D}  (\pi')$ , 对状态分布$d_{\pi, D}$的策略优势.  然而, 对最优策略, $\eta_{D}\left(\pi^{*}\right)$ 与$\eta_{D}(\pi)$ 同步于 对状态分布 $d_{\pi^*, D}$ 的策略优势. 因此, 就算 对$\pi$ 和 $D$ 的最优策略优势很小, 对$d_{\pi^*, D}$ 的策略优势未必很小.   这促使使用 跟均匀的分布 $\mu$

算法停止后, 会返回一个对 $\mu$ 有小的策略优势的 策略 $\pi$ .  下面量化, 对任意测量指标$\tilde{\mu}$, 该策略离最优策略有多远.   

**Theorem 6.2.**   假定对 $\pi$, 有 $O P T (\mathbb A_{\pi, \mu})<\varepsilon .$  对任意状态分布 $\tilde{\mu}$  

$$
\begin{aligned}
\eta_{\tilde{\mu}}\left(\pi^{*}\right)-\eta_{\tilde{\mu}}(\pi) & \leq \frac{\varepsilon}{(1-\gamma)}\left \Vert \frac{d_{\pi ^ * , \tilde \mu }}{d_{\pi, \mu}}\right \Vert_{\infty} \\
& \leq \frac{\varepsilon}{(1-\gamma)^{2}}\left  \Vert \frac{d_{\pi^* , \tilde \mu}}{\mu}\right  \Vert_{\infty}
\end{aligned}
$$


Proof. 	最优策略优势  $ \text{OPT}  (\mathbb A_{\pi, \mu} )= \sum_s d_{\pi, \mu}(s) \max _a A_\pi (s, a)$ , 因此


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

前面提出了一个算法, finds "approximately" optimal solution that is polynomial in the approximation parameter $\varepsilon,$ but not in the size of the state space. 

##### The Greedy Policy Chooser

找到一个具有大的策略优势的策略可以认为是一个回归问题, 但是没有解决该问题的复杂度. 

$$
E_{s \sim d_{\pi, \mu}} \max _{a}\left|A_{\pi}(s, a)-f_{\pi}(s, a)\right|
$$

loss 是状态空间上的平均误差 (是个在所有动作上的 $\infty$ -loss ). 如果可以是该误差小于 $\frac{\varepsilon}{2},$ 那么可以构建一个$\varepsilon$-greedy policy chooser , 通过选择 基于近似函数 $f_{\pi}$的贪婪策略.  这个回归问题的 $l_1$ 条件要弱, 与最小化在所有状态空间上的 $l_{\infty}$ -error 相比, 是greedy dynamic programming的相对误差. ?

Direct policy search 也可以被用于 greedy  policy chooser.



##### What about improving $\eta_D $?

"Can we improve the роlісу ассоrdіng tо bоth $eta_D$ аnd $eta_\mu$ аt еасh uрdаtе?"
In general the answer is "no", but consider improving the performance under $\tilde \mu= (1- \beta)\mu + \beta D$ instead of just $\mu$. This metric only slightly changes the quality of the asymptotic policy. However by giving weight to D,
the possibility of improving $\eta_D$ is allowed if the optimal policy has large advantages under D, though we do not formalize this here. The only situation where joint improvement with $\eta_D$  is not possible is when $\text{OPT}(\mathbb A_{\pi,D}) $ is small. However, this is the problematic case where, under D, the large advantages are not at states visited frequently.

##### Implications of the mismatch

The bounds we have presented directly show the importance of ensuring the agent starts in states where
the optimal policy tends to visit. It also suggests that certain optimal policies are easier to learn in large state spaces- namely those optimal policies which tend to visit a significant fraction of the state space. An interesting suggestion for how to choose u, is to use prior knowledge of which states an optimal policy tends to visit.



#### Appendix: Proofs

The intuition for the proof of theorem 4.1 is that $\alpha$ determines the probability of choosing an action from $\pi^{\prime}$ If the current state distribution is $d_{\pi, \mu}$ when an action from $\pi^{\prime}$ is chosen, then the performance improvement is proportional to the policy advantage. The proof involves bounding the performance decrease due to the state distribution not being exactly $d_{\pi, \mu},$ when an action from $\pi^{\prime}$ is chosen.

Proof. Throughout the proof, the $\mu$ dependence is not explicitly stated. For any state $s$

$$
\begin{aligned}
& \sum{a} \pi_{\mathrm{new}}(s ; a) A_{\pi}(s, a) \\
=& \sum{a}\left((1-\alpha) \pi(a ; s)+\alpha \pi^{\prime}(a ; s)\right) A_{\pi}(s, a) \\
=& \alpha \sum \pi^{\prime}(a ; s) A_{\pi}(s, a)
\end{aligned}
$$

where we have used $\sum_{a} \pi(a ; s) A_{\pi}(s, a)=0$
For any timestep, the probability that we choose an action according to $\pi^{\prime}$ is $\alpha .$ Let $c_{t}$ be the random variable indicating the number of actions chosen from $\pi^{\prime}$ before time $t .$ Hence, $\operatorname{Pr}\left(c_{t}=0\right)=(1-\alpha)^{t} .$ Defining  $\rho_{t} \equiv \operatorname{Pr}\left(c_{t} \geq 1\right)=1-(1-\alpha)^{t}$ and $P\left(s_{t} ; \pi\right)$ to be distribution over states at time $t$ while following $\pi$ starting from $s \sim \mu,$ it follows that

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

where we have used the definition of $\varepsilon$ and $P\left(s_{t} | c_{t}=\right.$ $\left.0 ; \pi_{\mathrm{new}}\right)=P\left(s_{t} ; \pi\right)$
By substitution and lemma 6.1, we have:

$$
\begin{aligned}
& \eta_{\mu}\left(\pi_{\text {new }}\right)-\eta_{\mu}(\pi) \\
=& \sum_{t=0}^{\infty} \gamma^{t} E_{s \sim P\left(s_{t} ; \pi_{\text {new }}\right)}\left[\sum_{a} \pi_{\text {new }}(a ; s) A_{\pi}(s, a)\right] \\
\geq & \alpha \sum_{t=0}^{\infty} \gamma^{t} E_{s \sim P \left(s_{t} ; \pi\right)}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}(s, a)\right] \\
& - 2 \alpha \varepsilon \sum_{t=0}^{\infty} \gamma^t(1-(1-\alpha)^t)  \\
=& \frac{\alpha}{1-\gamma} E_{ s \sim d_{\pi}}\left[\sum_{a} \pi^{\prime}(a ; s) A_{\pi}\left(s_{t}, a\right)\right] \\
& - 2 \alpha \varepsilon (\frac{1}{1-\gamma}-\frac{1}{1-\gamma(1-\alpha)} ) 
\end{aligned}
$$

The result follows from simple algebra. 



The proof of theorem 4.4 follows.
Proof. During step $(2),$ we need enough samples such that $|\mathbb A -\hat{\mathbb A}|<\frac{\varepsilon}{3}$ for every loop of the algorithm and, as proved below, we need to consider at most $\frac{72 R^{2}}{\varepsilon^{2}}$ loops. If we demand that the probability of failure is less than $\delta,$ then by the union bound and inequality $5.1,$ we have for $k$ trajectories $P(\text { failure })<\frac{72 R^{2}}{\varepsilon^{2}} 2 e^{-\frac{k \varepsilon^{2}}{36 n_{a}^{2} R^{2}}}<\delta$
where we have taken $\Delta=\frac{\varepsilon}{6}$ since the bias in our estimates is at most $\frac{\varepsilon}{6} .$ Thus, we require $O\left(\frac{R^{2}}{\varepsilon^{2}} \log \frac{R^{2}}{\delta \varepsilon^{2}}\right)$
trajectories.

Thus, if $\hat{\mathbb A} \geq \frac{2 \varepsilon}{3},$ then $\mathbb A \geq \frac{\varepsilon}{3}>0 .$ By corollary 4.2, step 4 guarantees improvement of $\eta_{\mu}$ by at least $\frac{\left(\hat{\mathbb A}-\frac{\epsilon}{3}\right)^{2}}{8 R} \geq \frac{\varepsilon^{2}}{72 R}$ using $\alpha=\frac{(1-\gamma)\left(\hat{\mathbb A}-\frac{\varepsilon}{3}\right)}{4 R},$ which proves $i$.
since $0 \leq \eta_{\mu} \leq R,$ there are at most $\frac{72 R^{2}}{\varepsilon^{2}}$ steps before $\eta_{\mu}$ becomes larger than $R,$ so the algorithm must cease in this time, which proves ii. In order to cease and return $\pi,$ on the penultimate loop, the $G_{\varepsilon}$ must have returned some $\pi^{\prime}$ such that $\hat {\mathbb A} <\frac{2 \varepsilon}{3},$ which implies $\mathbb A_{\pi}\left(\pi^{\prime}\right)<\varepsilon $.   By definition of $G_{\varepsilon},$ it follows that
$\mathrm{OPT}\left(\mathbb A_{\pi}\right)<2 \varepsilon$ which proves iii.  

















