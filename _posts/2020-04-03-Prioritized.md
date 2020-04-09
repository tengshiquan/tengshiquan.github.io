---
layout:     post
title:      Prioritized Experience Replay
subtitle:   DQN from Deepmind
date:       2020-04-01 12:00:00
author:     "tengshiquan"
header-img: "img/post-atari.jpg"
catalog: true
tags:
    - AI
    - DeepMind
    - Reinforcement Learning
    - Experience Replay
    - DQN

---

 

# Prioritized Experience Replay

带优先级的经验回放, 加权经验回放

论文里面讨论以及延伸部分信息量很大. 但Prioritized memory的提升不是特别大.



### INTRODUCTION

**Experience replay** (Lin, 1992) **(s, a, r, s')**  解决两个问题: 

1. correlated updates that break the i.i.d.
2. data efficiency

DQN used a large **sliding window replay memory**, 随机均匀采样, 平均每个transition访问8次.  

一般来说，Experience Replay可以减少学习所需的transition数量，代价是计算量和内存--这些资源往往比RL agent与env的交互成本更低。

本文研究了如何引入 prioritizing, 使得比 在Experience replay 上均匀采样更有效率. 核心思想是, agent能从某些transition上学得更高效. 哪些transition更加surprising, redundant, or task-relevant. 有些transition可能现在对agent没用,但之后agent能力提升以后, 变得有用. 

作者建议, more frequently replay transitions with high expected learning progress, 学习进度比较大的transition, 值得更多地去访问. 怎么衡量学习程度, 用 **temporal-difference(TD) error**.   
TD-error作为优先级会造成loss of diversity,  使用stochastic prioritization来缓解; 也会引入bias, 要使用importance sampling 来修正.   



### BACKGROUND

先是神经学方面的启发.

已证明, planning 算法如 value iteration  可以通过按适当的顺序进行更新来提高效率.    
**Prioritized sweeping** (Moore & Atkeson, 1993; Andre et al., 1998) 选择下一个要更新的状态时，根据值的变化大小进行优先级排序.  
TD-error提供了衡量这些优先级的一种方法（van Seijen & Sutton，2013）  
作者的方法使用了类似的prioritization方法，但是针对model-free RL，而不是model-based planning。  
此外，作者使用了更稳健的stochastic prioritization在从样本中学习function approximation。

TD-errors 也被用作确定资源(resources)的优先级, 如何时在哪explore (White et al., 2014) , 或者选择哪些特征 features (Geramifard et al., 2011; Sun et al., 2011).

在监督学习中，当类特征已知时，有许多处理不平衡数据集(imbalanced datasets)的技术，包括重采样、下采样和过采样技术 re-sampling, under-sampling and over-sampling，可能结合ensemble方法. 最近一篇论文, 提出了一种 re-sampling 结合RL中的experience replay (Narasimhan et al., 2015)  
Hinton（2007）引入了一种基于误差的非均匀抽样形式，通过importance sampling correction，使得MNIST的数字分类速度提高了3倍。



### PRIORITIZED REPLAY

使用replay memory有两个考虑要点

1. 存哪些exp
2. 取哪些exp去replay, 以及怎么取

本文主要研究当有个buffer以后,怎么高效利用



#### A MOTIVATING EXAMPLE

‘Blind Cliffwalk’ , 该例子体现了在奖励稀少时, 探索的困难.   
n states, 只能左右,  env 需要 指数级的随机步数，才能找到第一个非零奖励. 准确地说，一个随机的行动序列能收到奖励的概率是 $2^{-n}$.  两个agent都使用Q-learning从同一个replay memory上学习.

![image-20200409015241740](/img/2020-04-01-DQN.assets/image-20200409015241740.png)

用这个例子来说明, 两种agent选择 exp方式的不同效果.  

1. uniformly at random
2. oracle, 定制规则,  greedily 选择能最大程度减少当前状态下 global loss 的transition

如图, 右边, 可以看出 需要学习的步数的中位数是一个smaples size的函数;   相比均匀随机, 可以指数级别的提升速度.  



#### PRIORITIZING WITH TD-ERROR

prioritized replay 的关键是找到 transition优先级的衡量标准.   
一个理想的方案, 用能学到多少东西衡量. expected learning progress;    
因没法直接计算, 替代方案: TD error δ, 表示 遇到这个transition有多意外‘surprising’.  
即,  how far the value is from its next-step bootstrap estimate  
这特别适用于incremental, online RL ，如SARSA或Q-learning，这些算法已经计算出了TD-error.

为了说明TD-error有效,提出 **greedy TD-error prioritization**,  下面在Blind Cliffwalk 上用 uniform and oracle 做baseline看效果.   
该算法将上次遇到的TD error与transition 一起存储在buffer里,  然后取TDerror绝对值最大的出来replay.   
新的transition没有计算TD-error, 则给与最大优先级, 来保证最少被访问一次.

实现算法: size N, 使用 binary heap data structure 去实现 优先级队列.    
采样时, 找到最大优先级是O(1), 更新优先级是 O(log N )



#### STOCHASTIC PRIORITIZATION

greedy TD-error prioritization 的几个问题:

1. 为了避免昂贵的对整个buffer进行sweep,  只更新 最近replay的那些 transition. 一个后果就是, 如果一个transition的TDerror比较低,则很可能很长时间得不到update. 实际上,buffer都是sliding window, 所以该transition很可能就不会再被访问. 
2. 对噪声尖峰(noise spikes)很敏感.  例如，当奖励是随机的. 噪声可能因为bootstrap而加剧; 而approximation的误差可以成为另外一个噪声源. 
3. 因为greedy, 所以focos在整个buffer上面的一个小的子集里面, 到时error缩减的很慢; 特别是用function approximation的时候, 初始的高error的transition会频繁的被replay. 缺乏diversity,容易造成过拟合.

所以提出 **stochastic sampling method** ,  在 greedy prioritization 与 uniform random 之间折中. 确保被选中的概率与优先级是单调的,同时最低优先级的也有几率选到. 定义选中概率为:

$$
P(i)=\frac{p_{i}^{\alpha}}{\sum_{k} p_{k}^{\alpha}}  \tag{1}
$$

$p_{i}>0$ 是 transition i 的优先级. 指数 $\alpha$ 决定优先级被使用的程度, $\alpha=0$ 等于 uniform. 

1. **proportional prioritization** .  $p_{i}=\vert \delta_{i} \vert+\epsilon$ ,    $\epsilon$  小的常量, 保证δ=0的也能被访问. 
2. **rank-based prioritization** .  $p_{i}=\frac{1}{\text { rank }(i)},$   rank( $i$ ) 是 transition放入buffer的时候, 根据 $\vert \delta_{i}\vert$排序的rank.   这种情况下, $P$  是幂律分布, power-law distribution with exponent $\alpha$ 

这两种都是 $|\delta|$ 单调的. 但方案2更健壮, 对离群值outliers 不敏感.   
这两种都比 uniform baseline 要好.

![image-20200409041726548](/img/2020-04-01-DQN.assets/image-20200409041726548.png)

实现算法:  
为了高效地从buffer里面sample, complexity 不能是 N.   
对rank-based, 可以使用k分段线性函数来近似密度函数P. 线段边界可以预先计算（只在N或α变化时才会改变,所以后期就是固定的）.  运行时，先选一个分段，然后对其uniform采样。 这与基于minibatch的算法结合效果特别好: 选k为minibatch的大小, 然后从每个分段里面random选一个transition.这是一种分层采样的形式，它的额外优点是平衡了minibatch（batch中总有error值高的, 也有低的,分布的比较均匀).   
对proportional, 也有一个高效的实现,  基于"**sum-tree**"的数据结构. (其中每个节点是其子节点的总和，优先级作为叶子节点)



![image-20200409042327316](/img/2020-04-01-DQN.assets/image-20200409042327316.png)



#### ANNEALING THE BIAS

使用 stochastic updates 对期望值的估计依赖于这些更新有与期望值相同的分布。Prioritized replay 会引入偏差因为改变了这种分布，因此改变了估计值将收敛到的解（即使策略和状态分布是固定的). 可以通过**importance-sampling (IS)** 权重来修正. 

$$
w_{i}=\left(\frac{1}{N} \cdot \frac{1}{P(i)}\right)^{\beta}
$$

完全补偿 non-uniform probabilities $P(i)$ if $\beta=1$.    
该权重可以合并到Q-learning的更新里, 使用$w_{i} \delta_{i}$代替 $\delta_{i}$.   
为了稳定性, 总是归一化权重,  $\frac{1}{\max_i w_i}$ , 使得更新的比例降低.



在典型的强化学习场景中，更新的无偏向性在训练快结束时接近收敛时是最重要的，由于策略、状态分布和 bootstrap target的变化，这个过程是高度非稳态的.  我们假设这种情况下, 可以忽略一个小的bias. 显然, 就是说早期不怎么用IS修正也问题不大, 后期快收敛了要严格修正.    
因此，我们利用IS修正量随时间退火annealing 的灵活性，定义一个在学习结束时达到1的指数$\beta$上的schedule. 实操中, 线性退火: $\beta_0 \to 1$ .  注意, $\beta$ 的选择会影响到 prioritization 指数 $\alpha$ 的选择. 同时增加这两个参数, 会使采样的优先级更强，同时对其进行更强的修正。

在non-linear function approximation（如DNN）的情况下，重要性采样与优先级重放相结合时，还有另一个好处：在这里，大的步长可能是非常有破坏性的，因为梯度的一阶近似只在局部可靠，必须用较小的全局步长来。相反，在我们的方法中，prioritization 确保了high-error的transition被多次访问，而IS校正减少了梯度幅度（从而减少了参数空间中的有效步长大小），允许算法遵循 非线性优化的曲率，因为泰勒展开公式是不断地重新逼近。

在Double DQN算法的基础上，结合prioritized replay。主要修改是用 stochastic prioritization 和 IS 取代了Double DQN使用的uniform random。





### ATARI EXPERIMENTS

sliding window memory size: $10^6$.   minibatches : 32.    
一个minibatch updage 会让4个新的transition进入buffer, 所有的experience平均会被replay 8次.   
reward 以及 TD-error 会被 clip 在[-1, 1], 为了稳定

实验与之前double DQN架构一样, 只是从 replay memory 中的采样机制不一样.   
要进行一次超参数调整。考虑到prioritized replay会更频繁地选择high-error transition，造成梯度幅度更大，所以将步长$\eta$比 DQN设置减少了4倍.   
对超参 $\alpha$ $\beta_0$ , 做了一个比较粗的grid search, 在8个游戏上验证.   sweet spot 是 对rank-based  $\alpha=0.7,  \beta_0=0.5$ ; 对proportional  $\alpha=0.6,  \beta_0=0.4$ ;  这些都是在权衡aggressiveness 与 robustness.   
可以通过减少$\alpha$ 或者增加$\beta$ 很容易退化到 baseline. 

一个主要的衡量指标: quality of the best policy, 游戏开始状态是从人类trajectory里面采样. 

<img src="/img/2020-04-01-DQN.assets/image-20200409150601156.png" alt="image-20200409150601156" style="zoom:50%;" />

可以看出有提升,但幅度不是很大.

![image-20200409140204800](/img/2020-04-01-DQN.assets/image-20200409140204800.png)

实验结果如图, 表明有不少游戏有进步,但少数还不如原来的. 

另外一个衡量标准是: learning speed. 

![image-20200409192513057](/img/2020-04-01-DQN.assets/image-20200409192513057.png)

如图, 训练速度缩短到了一半, 只有原来1/3 多. 



### DISCUSSION

- 在rank-based 与 proportional表现优秀的头部数据的对比中,   rank-based更加 robust, 因为它不受离群值和误差大小的影响.   
  rank-based的  heavy-tail 重尾属性也保证了样本的多样性，从不同误差的分区中分层采样，可以使总的minibatch梯度在整个训练过程中保持在一个稳定的幅度。
- 另一方面，rank使得该算法忽视了 相对误差尺度，当误差分布有结构化的分布可以利用时，比如在稀疏的奖励场景中，这可能会导致性能下降。
- 令人惊讶的是，这两个变体在实践中的表现都很相似；我们怀疑这是由于DQN算法中大量使用了clip（reward和TD-error），从而消除了离群值。
- 监测了一些游戏的TD-error作为时间的函数的分布（图10），发现随着学习的进行，它变得接近于重尾分布，而在不同的游戏中仍有很大的差异；这在经验上验证了公式1的形式。图11 显示了这个分布是如何与公式1相互作用而产生Effective replay probability。

![image-20200409235648965](/img/2020-04-01-DQN.assets/image-20200409235648965.png)

图10, Visualization of the last-seen absolute TD-error , sorted.  横坐标是数量百分比, 纵坐标是归一化的错误; 一开始肯定都集中在最左边, 慢慢扩散到右边, 变成heavy-tailed分布.  显然带优先级的比uniform的扩散的快.



![image-20200409235717493](/img/2020-04-01-DQN.assets/image-20200409235717493.png)

图11, **Effective replay probability**.  红色线是公式1, 取$\alpha = 0.7$ . 与灰色 uniform 对比.  可以看出, 总体上td-error大的被选中的概率大. 

- 还有另外一个现象,  有一部分被访问过的transition直到移出buffer都从未replay，而更多的transition是在它们被进入buffer以后很久才第一次被replay.   
  同时，uniform 也隐含着对过时out-of-date的transition的偏向 . uniform sampling is implicitly biased toward out-of-date transitions that were generated by a policy that has typically seen hundreds of thousands of updates since.   
  Prioritized replay 直接解决了第一个问题, 并有倾向解决第二个问题. 因为较新的transition往往有较大的误差--这是因为旧的transition有更多的机会被纠正，而且新的数据往往较少被值函数预测。
- NN与prioritized replay交互的另一个角度:  当我们区分学习给定表示（即顶层）的值和学习改进的表示（即底层）的值时;  那么对于表示良好的transition，会迅速降低其误差，然后被replayed的次数就会减少很多，从而增加学习重点在其他表示较差的地方，从而将更多的资源投入到区分不一致的状态上. When we distinguish learning the value given a representation (i.e., the top layers) from learning an improved representation (i.e., the bottom layers), then transitions for which the representation is good will quickly reduce their error and then be replayed much less, increasing the learning focus on others where the representation is poor, thus putting more resources into distinguishing aliased states. 



### EXTENSIONS

##### Prioritized Supervised Learning

监督学习中, 也有类似的做法.  从数据集中进行非均匀地采样，每个样本使用一个基于最后看到的错误的优先级。这可以帮助将学习集中在那些仍然可以学习的样本上, 将额外的资源投入到 (hard) boundary 案例中. 有点类似于 bosting (Galar et al., 2012). 

如果数据集是不平衡的，我们假设来自稀有类别的样本将被不成比例地采样，因为它们的误差收缩得不快，而从common类别中选择的样本将是最接近决策边界的样本，从而导致类似于hard negative mining的效果（Felzenszwalb等人，2008）

为了检验这些直觉是否成立，我们对经典的MNIST数字分类问题的做了一个class-imbalanced的版本,进行了初步实验，在训练集中，我们删除了训练集中0、1、2、3、4号数字的99%的样本，而测试/验证集则不做任何处理(即保留了类平衡) . 我们比较了两种情况: 在知情(informed)的情况下，我们人为地对训练数据贫乏的类别的误差进行了重新加权（系数100），而在未知情(uninformed)的情况下，我们没有提供任何提示，即测试分布与训练分布不同。Prioritized sampling（uninformed情况下，with $\alpha$= 1，$\beta$= 0）优于uninformed情况下的uniform baseline，在泛化方面性能接近于informed uniform baseline；同样，prioritized 训练在学习速度也更快。

![image-20200410013441515](/img/2020-04-01-DQN.assets/image-20200410013441515.png)

图示, 纵轴是错误数以及loss, 所以越低越好. 可以看出 prioritized 一开始很低, 说明学的快.  右边可以看,过拟合的情况. 



##### Off-policy Replay

off-policy RL 两种方式: 拒绝采样(rejection sampling) 和  IS ratio $\rho$  来修正 on-policy的 transition.   
我们这里的方法 replay probability P and  IS-correction w  与之类似.    很自然可以将其应用到off-policy. 

我们的算法 proportional 方案,可以变成 weighted IS , 令 $w = \rho, \alpha = 0, \beta =1$ ;  变成 rejection sampling,  令$w = min(1,\rho), \alpha = 1, \beta =0$ . 实验表明，中间版本，possibly with annealing or ranking，在实践中可能更有用.



##### Feedback for Exploration

一个有趣的副作用是，一个transition最终会被replay的总次数$M_i$差异很大.   这个信息就可以反馈给正在exploration的策略. 

例如, 在episode 开始的时候, 可以基于$M_i$, 更新一个类似于exploration的参数, 使得策略能产生更多有价值的experience.

或者, 如Gorila 的agent, 可以引导资源分配.

思路很好.



##### Prioritized Memories

决定是否要replay一个transition的因素也与 什么时候存储或者删除该transition有关. 显然, 用处不大的transition可以早早删掉, 或者不加入, 节省内存. 个人觉得这个里面可以发掘的东西比较多. 



### CONCLUSION

prioritized replay speeds up learning by a factor 2 and leads to a new state-of-the-art of performance on the Atari benchmark.



### A PRIORITIZATION VARIANTS

讨论了一下 priority 的一些变体. 主要是衡量标准 TD-error的其他方案. 

1. TD-error 能捕捉到潜在的改进规模，但忽略了奖励或transition中固有的随机性，以及部分可观察性或FA能力的可能限制. 换句话说, 它是有问题的, 当面对一些 unlearnable transitions. 在这种情况下，它的**导数derivative**--可以通过当前 $\vert \delta \vert$和上次replay时的$\vert \delta \vert$之间的差值来近似--可能更有用.  在初步实验中，我们发现它的表现并不优于$\vert \delta \vert$
2. 一个正交orthogonal的变体是考虑replay transition所引起的**权重变化weight-change的法线** --如果底层优化器采用了自适应的阶梯大小，可以减少高噪声方向的梯度，那么这种方法是有效的(Schaul等人，2013; Kingma & Ba, 2014)，从而将区分可学习和不可学习过渡的负担放在优化器上。
3. 可以通过不把正的TD错误与负的TD错误一视同仁来调节优先级. 安娜-卡列尼娜原则:有许多种方式可以使transition不如预期的好，但只有一种方式可以更好. 直白的说, 就是好的action的比例肯定是少的. 引入**不对称性asymmetry**，把正的TD错误优先于同等量级的负的TD错误，因为前者更有可能是信息量大的。 
4. 来自神经科学的证据表明，基于 **episodic return** 而非 expected learning progress 的优先级可能也是有用的。在这种情况下，我们可以提高整个episodes的replay概率，而不是transitions，或者通过观察到的return-to-go（甚至是他们的值估计）来提高单个transitions。
5. 对于保留足够的多样性问题(防止过拟合、过早收敛 或impoverished representations)，我们要引入随机性有其他的解决方案. 例如，优先级可以通过观察空间中的novelty新奇度量来调节。可以设计一个 hybrid 混合的方法，每个minibatch的其中一部分元素根据一个优先级采样，其余的根据另一个优先级采样，引入额外的多样性。一个正交的想法是通过引入一个显式的**滞后性奖励staleness bonus**来增加一段时间内没有被replay的transitions的优先级，这个奖励保证每一个transition都会不时地被replay，而且这个机会会随着最后一次看到的TD-error变得滞后而增加。在这种奖励随时间线性增长的简单情况下，可以通过从任何更新的新优先级中减去一个与全局步数成比例的数量来实现，而不需要额外的成本。
6. 在值函数bootstrapping的 RL的特殊情况下，可以利用 replay memory 的顺序结构：一个导致大量学习（关于它的流出状态）的transition有可能改变所有导致进入该状态的transition的bootstrapping target，因此有更多的东西要学习这些东西。当然，我们至少知道了其中的一个，即历史上的**前任predecessor transition**，因此提升它的优先级使得它更有可能很快被重新审视。与资格迹eligibility traces相似，这让信息从未来的结果中回溯到导致这个结果的行动和状态的价值估计。在实践中，我们会将当前transition的$\vert \delta \vert$加入到predecessor transition的优先级中。这个想法与与最近的prioritized sweeping的延伸有关（van Seijen & Sutton，2013）。






## Reference

Deep Reinforcement Learning with Double Q-learning  https://arxiv.org/abs/1509.06461

























