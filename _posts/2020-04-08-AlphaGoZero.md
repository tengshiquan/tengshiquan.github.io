---
layout:     post
title:      AlphaGo Zero
subtitle:   AlphaGo from Deepmind
date:       2020-04-08 12:00:00
author:     "tengshiquan"
header-img: "img/post-AlphaGo.jpg"
catalog: true
tags:
    - AI
    - DeepMind
    - Reinforcement Learning
    - AlphaGo

---

# AlphaGo Zero

总体思路:

Policy + Value NN(ResNet)   guide MCTS,   MCTS self-play  gen data ($\pi , z$). 

MCTS 可以看成 **PI算子** ,   利用$f_{\theta}$ search出来的结果比$f_{\theta}$原来的要好;  之前的PI, 是通过PG来实现的. 



![image-20200617233849218](/img/2020-04-08-AlphaGo.assets/image-20200617233849218.png)







##### AlphaGo Zero, 与AlphaGo Fan和AlphaGo Lee的差异

1. No human data, trained by self-play RL, starting from random play; 没有专家棋谱, 完全从zero开始.
2. No hand-crafted features, 只使用棋盘盘面作为输入特征 48 -> 17层
3. 只使用一个ResNet, 输出 Value, Policy.
4. 更简单的 tree search,  没有Monte Carlo rollouts, 依靠上面的NN来评估v和p







<img src="https://lh3.googleusercontent.com/atTt6Okt1LQIjIAF225ptfDdCtndp-OD4ZNPAmxTEAESk-sRvQg0cHbULyxh4wci7QH_TD3jIMGWMraOZHPW-C9UU6ZUx9jN4yms3g=w1440" alt="img" style="zoom:50%;" />





## Mastering the game of Go without human knowledge

#### Reinforcement Learning in AlphaGo Zero

- input : raw board representation s of the position and its history
- output : move probabilities and  value $(\mathbf{p}, v)=f_{\theta}(s)$
  - $p_{a}=\operatorname{Pr}(a \vert s)$ 
  - v , **probability of the current player winning** from position s.
- **policy** and **value** **在一个网络架构中**
- 网络由许多卷积层的residual blocks组成，使用batch normalisation和ReLU



##### Self­-play with search

- MCTS(s) =  $\pi$
  - input, position $s$ ;  $f_{\theta}$ 引导 MCTS search ;  **output** :  每个move的概率向量  $\boldsymbol{\pi}$
  - $\boldsymbol{\pi}$  通常强于  $f_{\theta}(s)$ 的 $\mathbf{p}$  ,   **improve**
  - **MCTS 可以看成一个强大的 policy improvement operator**.  **MCTS : PI算子**
- Self-play with search
  - **improved MCTS­-based policy** to select each move
  - game winner $z$ as a **sample** of the value,  用于 $f_{\theta}$ 拟合,  **evaluate**
  - Self-play with search 可以被看成是强大的 policy evaluation operator. **Self-play with search : PE算子**
- policy iteration :  PI, PE, PI, PE,...
- $(\mathbf{p}, v)=f_{\theta}(s)$ 越来越匹配  target  $(\boldsymbol{\pi}, z)$ ,  $f_{\theta}$ 越强,  MCTS search 也越强



##### Self-­play training pipeline

Provide training data

![image-20200414175303634](/img/2020-04-07-AlphaGo.assets/image-20200414175303634.png)

Guide simulation

![image-20200414175401546](/img/2020-04-07-AlphaGo.assets/image-20200414175401546.png)





##### MCTS

![image-20200414182439143](/img/2020-04-07-AlphaGo.assets/image-20200414182439143.png)

使用 net $f_\theta$  引导 simulation.   这部分比之前简单了, 因为少了rollout的部分. 引导的作用, 主要体现在 对叶子节点的评估, 然后 net 输出的策略P, 也作为expand的一个指标.

1. select , simulation 从root开始,  一直选 最大 **upper confidence bound** $Q(s, a)+U(s, a)$ , $U(s, a) \propto P(s, a) /(1+ N(s, a))$ 的边 ,  直到叶子节点 $s^{\prime}$ . 
2. Expand and evaluate,  叶子节点, 展开, 并评估 $\left(P\left(s^{\prime}, \cdot\right), V\left(s^{\prime}\right)\right)= {f_{\theta}\left(s^{\prime}\right)}$ 
3. backup : simulation中遍历到的边, 都更新 $N(s, a)$ 和  $Q(s, a)=1 / N(s, a) \sum_{s^{\prime}\vert s, a \rightarrow s^{\prime} } V\left(s^{\prime}\right)$ 
4. play,  search完成后, 返回 $\pi$, $  \propto N ^{1 / \tau}$ ,  MCTS 最后选择action, 是按照访问次数的指数



MCTS 可以看做 **self-play 算法**.  给定 $\theta$ ,  input: 根节点$s$,  output: action序列$\boldsymbol{\pi}=\alpha_{\theta}(s)$   
action被选中的几率正比于 $\pi_{a} \propto N(s, a)^{1 / \tau},$   $\tau$ : temperature parameter. 



不一样的点:    
对于新创建的子节点，需要评估该节点所代表的状态的价值。evaluate的时候,  alphago采用混合机制对状态价值进行估计： $V\left(s_{L}\right)=(1-\lambda) v_{\theta}\left(s_{L}\right)+\lambda z_{L}$ , 其中，第一部分是value net，第二部分是从该节点状态开始使用快速rollout策略（fast rollout policy）走出的胜负结果。  
**zero版本只使用网络的价值输出作为拓展节点的价值估计**。所以zero版本中不需要rollouts





下面与之前不一样的, self-play的时候, 就用MCTS了, 用MCTS的输出$\pi$ 作为target

train NN :  self-play RL , use MCTS play each move.

下面1,2两个流程可以是并行的, work in parallel

- NN initialized random weights $\theta_{0}$ 
   1. (Fig. 1a) , at iteration $i \geq 1$ ,  games of self-play are generated , plays a game $s_{1}, \ldots, s_{T}$ against itself
      1. at $t$ ,  **MCTS** search $\pi_{t}=\alpha_{\theta_{i-1}}\left(s_{t}\right)$ , using $f_{\theta_{i-1}}$,   play  $a_t$  ~  sampling $\boldsymbol{\pi}_{t}$ 
      2. terminates at step $T$  ,  final reward of $r_{T} \in\{-1,+1\}$ 
      3. data for $t$  stored as $$\left(s_{t}, \boldsymbol{\pi}_{t}, z_{t}\right)$$ ,   $$z_{t}=\pm r_{T}$$ , 当前player角度
   2. (Fig. 1b) , in parallel,  train NN $\theta_{i}$ 
      1. data: 均匀采样 $(s, \boldsymbol{\pi}, z)$  
      2. $(\mathbf{p}, v)=f_{\theta}(s)$ ,  target  $(\boldsymbol{\pi}, z)$ 
      3. loss function  $l=(z-v)^{2}-\boldsymbol{\pi}^{\mathrm{T}} \log \boldsymbol{p}+c\|\theta\|^{2}$ ,  mse+CrossEntropy+L2



这里Loss函数跟之前的不一样,  一个loss包含了value以及policy两个输出的loss之和; 加了L2正则化;  L2可以让参数均匀分布,防止过拟合. 





#### Empirical analysis of AlphaGo Zero training

training: 

- 4.9 million games of self­play 
- 1,600 simulations for each MCTS, 0.4 s thinking time per move. 
- Parameters updated :  700,000 个 mini­batches(2,048)
- NN 20 residual blocks 


![image-20200415033353331](/img/2020-04-07-AlphaGo.assets/image-20200415033353331.png)

 $c$, Mean-squared error (MSE) of human professional game outcomes. MSE is between the actual outcome $z \in\{-1,+1\}$ and the NN value $v,$ scaled by $\frac{1}{4}$ to the range of $0-1$  .

训练36小时之后，AlphaGo Zero就超过了AlphaGo Lee，AlphaGo Lee训练了几个月。  
训练72小时后，AlphaGo Zero使用具有4个TPU的单机，而AlphaGo Lee则是分布在许多机器上，并且使用48个TPU。AlphaGo Zero以100比0击败AlphaGo Lee。

为了评估self­play RL相对于使用人类棋谱进行学习的优势，我们训练了第二个神经网络（使用相同的架构）来预测在KGS服务器数据，取得了与以前的工作相比更准确的预测准确度。监督学习SL取得了较好的初始性能，并且更好地预测了人类棋手的动作（图3）。值得注意的是，尽管SL获得了更高的棋步预测精度，但self­play的棋手总体上表现更好，在训练的前24 h内就击败了用人类数据进行训练的程序。这表明，AlphaGo Zero可能学习到一种与人类下棋有质的区别的策略。

SL里面也应该有过拟合, 并且从人类data出发,或造成探索的空间受限. 

![image-20200415040715087](/img/2020-04-07-AlphaGo.assets/image-20200415040715087.png)
separate (sep) or combined policy and value (dual) network

 

**P和V都放到一个net里面效果好, 无论resnet还是CNN.**

为了将结构architecture和算法algorithm的贡献分离，我们将AlphaGo Zero使用的NN architecture的性能与AlphaGo Lee进行了比较（见图4）。  
创建了四个NN，就像在AlphaGo Lee中那样，使用独立的策略网络和价值网络；或者使用AlphaGo Lee使用的卷积网络架构或AlphaGo Zero使用的残差网络架构。训练网络时都最大限度地减少相同的损失函数，使用的数据集是AlphaGo Zero在72小时的自我博弈训练后产生的固定数据集。  
利用**残差网络更准确**，使AlphaGo 达到较低的错误率和性能的改进，达到了超过600Elo。  
将策略和价值合成一个单一的网络会轻微地降低落子prediction accuracy，但降低了value error，并且使AlphaGo的性能提高大约600Elo。这部分由于提高了计算效率，但更重要的是 , **dual objective regularizes NN to a common representation that supports multiple use cases**.   一个网络两个输出, 就会使得网络学到更加一般性的东西. 





#### Knowledge learned by AlphaGo Zero

AlphaGo Zero在自我博弈训练过程中达到了围棋的新高度。  棋理方面.



#### Final performance of AlphaGo Zero

![image-20200415043546455](/img/2020-04-07-AlphaGo.assets/image-20200415043546455.png)

训练更强的 AlphaGo Zero的第二个实例。训练又从完全随机的行为开始，持续了大约40天。  
在训练过程中，产生了2900万场自我博弈。310万个minibatch(2048)。  
神经网络包含40个残差块.

通过内部比赛对AlphaGo Zero进行了评估，AlphaGo Master是基于本文所介绍的算法和架构，但使用了人类数据和特征的程序--在2017年1月的在线比赛中，我们以60-0击败了最强的人类职业棋手。  
评估中，所有的程序都被允许每一步棋的思考时间为5s；AlphaGo Zero和AlphaGo Master分别在4个TPU的单机上下棋；AlphaGo Fan和AlphaGo Lee分别分布在176个GPU和48个TPU上。还包括了一个完全基于AlphaGo Zero的原始神经网络的棋手；这个棋手只是简单地选择了概率最大的棋子。

100 games, AlphaGo Zero  89 ,   11 AlphaGo Master



#### Conclusion

我们的结果全面证明了纯强化学习方法是完全可行的，在没有人类指导的情况下，就可以训练到超人的水平。此外，与基于人类专家数据的训练相比，纯粹的强化学习方法只需要几个小时的训练时间，就可以获得更好的渐近性能。使用这种方法，AlphaGo Zero以很大的优势击败了之前最强的AlphaGo版本。
人类积累的围棋知识，在短短几天的时间里，AlphaGo Zero就重新发现了其中的大部分围棋知识，以及提供了新策略。





### Supplementary

#### Methods

##### Reinforcement learning

Policy iteration  理论部分,  主要讨论MCTS的意义;  **MCTS 首先是一个基于统计的policy**,  其次, MCTS也可以引入**启发式策略**来改进. 而NN本身做为一个策略, 也可以作为启发式的部分结合到MCTS里面去. 

基于分类的RL算法, 通过 **Monte Carlo search**来improve.  **Classification­-based reinforcement learning**   improves the policy using a simple **Monte Carlo search**. Many **rollouts** are executed for each action; the action with the **maximum mean value** provides a positive training example, while all other actions provide negative training examples; **a policy is then trained to classify actions as positive or negative, and used in subsequent rollouts.** This may be viewed as a precursor to the policy component of AlphaGo Zero’s training algorithm when $\tau \to 0$.    
这里,  在rollout中如何选择action没有提及. 这里的rollout 扮演了 生成sample,  然后policy evaluation的角色. 如果policy本身 select action 时, 选了max,  所以相当于一个 greedy improve; 下面网络拟合Fit ,  即是evaluate也是improve的过程;  
 **MC rollout + NN** , 所以对Fit , data improve, net improve

较新的的一个例子, **classification­ based modified policy iteration (CBMPI)**, also performs **policy evaluation** by regressing a value function towards truncated rollout values, similar to the value component of AlphaGo Zero; this achieved state­-of­-the­-art results in the game of Tetris. However, this previous work was limited to **simple rollouts** and **linear function approximation** using **hand­ crafted features**.  三个短板

The AlphaGo Zero **self­play** algorithm can similarly be understood as an **approximate policy iteration** scheme in which **MCTS** is used for **both policy improvement and policy evaluation**.   
**Policy improvement** starts with a neural network policy, executes an MCTS based on that policy’s recommendations, and then projects the (much stronger) search policy back into the function space of the neural network.  
**Policy evaluation** is applied to the (much stronger) search policy : the outcomes of self­play games are also projected back into the function space of the neural network. These projection steps are achieved by training the neural network parameters to match the search probabilities and self­play game outcome respectively. 即target $(\boldsymbol{\pi}, z)$  
AlphaGo Zero 的selfplay 也可以看做近似 策略迭代GPI, 在其中MCTS 算子 :PE+PI 两个角色.    
Policy improve:  NN Policy $(\mathbf{p}, v)=f_{\theta}(s)$  作为  MCTS 的启发式部分  =  更强的策略  $(\boldsymbol{\pi}, z)$   
Policy evaluation:  selfplay 产出的 data $(\boldsymbol{\pi}, z)$ ,  NN去fit  

Guo的论文中, 将MCTS的产出映射的到一个NN, 通过基于action的分类或者V值的回归.   MCTS产出 $(\boldsymbol{\pi}, z)$ , NN用来fit.  但这里, MCTS是固定算法, 纯粹的基于统计, 没有利用NN来选择action. Guo et al.7 also project the output of MCTS into a neural network, either by regressing a value network towards the search value, or by classifying the action selected by MCTS. This approach was used to train a neural network for playing Atari games; however, the MCTS was fixed—there was no policy iteration—and did not make any use of the trained networks.   



#### Self-play reinforcement learning in games

我们的方法最直接适用于完美信息的零和博弈。
Self-play强化学习之前已经被应用到围棋游戏中。NeuroGo 使用了一个神经网络来表示值函数，通过时差学习训练，预测对弈中的领地。RLGO，用一个线性特征组合来表示值函数，枚举了所有3×3的棋子模式；通过时差学习训练,来预测对弈中的赢家。NeuroGo和RLGO都取得了较弱的业余水平。

MCTS也可以被看作是一种self­play强化学习的形式。搜索树的节点包含了搜索过程中遇到的位置的值函数；这些值被更新，以预测模拟的自我博弈的赢家。MCTS程序以前在围棋中取得了很强的业余水平，但使用了大量的领域知识：基于手工特征的fast rollout policy，通过运行模拟直到棋局结束来评估位置；以及同样基于手工特征的tree policy，在搜索树内选择moves。  
Self-play强化学习方法已经在其他游戏中取得了很高的成功率：象棋..   在所有这些例子中，通过回归或时差学习，从Self-play产生的训练数据中训练出一个值函数。训练好的值函数被用作evaluation函数, 使用在alpha–beta 搜索、简单的蒙特卡洛搜索 或 counterfactual regret minimization。然而，这些方法使用手工的输入特征 或手工的特征模板。此外，学习过程中使用监督学习来初始化权重，手工选择棋子值的权重，对动作空间进行手工限制，或者使用已有的程序作为训练对手，或者生成游戏记录。 

许多最成功、最广泛使用的强化学习方法都是在零和游戏的背景下首次引入的：时差学习首次被引入到跳棋程序中，而MCTS被引入到围棋游戏中。然而，非常类似的算法后来被证明在视频游戏、机器人、工业控制和在线推荐系统中非常有效。



#### AlphaGo versions

- AlphaGo Fan:  176个GPU,  分布式
- AlphaGo Lee: 48个TPU, 分布式
- AlphaGo Master:  使用了与Lee 相同的手工特征和rollouts
- AlphaGo Zero:  no rollouts  , 4 TPU



#### Domain knowledge

1. AlphaGo Zero被提供了完美的博弈规则知识  
   These are used during MCTS, to simulate the positions resulting from a sequence of moves, and to score any simulations that reach a terminal state. Games terminate when both players pass or after 19 × 19 × 2 = 722 moves. In addition, the player is provided with the set of legal moves in each position.
2. AlphaGo Zero uses Tromp–Taylor scoring during MCTS simulations and selfplay training.
3. The input features describing the position are structured as a 19 × 19 image
4. 围棋的规则在旋转和reflection下是不变的；这一知识在AlphaGo Zero中得到了很好的应用，既可以在训练时对数据集进行扩容，将每个位置的旋转和reflection都包括在内，也可以在MCTS过程中对位置的随机旋转或反射进行采样(见搜索算法)

MCTS也没有使用任何其他启发式或特定领域的规则来增强。没有任何合法的棋步被排除----即使是那些填入棋手自己的eyes的棋步（这是以前所有程序中使用的标准启发式）。

MCTS search parameters were selected by Gaussian process optimization, so as to optimize selfplay performance of AlphaGo Zero using a neural network trained in a preliminary run. For the larger run (40 blocks, 40 days), MCTS search param eters were reoptimized using the neural network trained in the smaller run (20 blocks, 3 days). The training algorithm was executed autonomously without human intervention.



#### Self-play training pipeline

三个主要部分组成，所有这些都是**异步并行**执行的。

1. 神经网络参数$θ_i$会从最近的self-play数据中不断优化
2. AlphaGo Zero的棋手$α_{θ_i}$会不断被**评估**
3. 而到目前为止表现最好的棋手$α_{θ_*}$会被用来生成新的self-play数据。



##### Optimization  

关于优化部分的一些超参.

Each neural network $f_{\theta_{i}}$ is optimized on the Google Cloud using TensorFlow, with 64 GPU workers and 19 CPU parameter servers. The batch-size is 32 per worker, for a total mini-batch size of 2048. Each mini-batch of data is sampled uniformly at random from all positions of the most recent 500,000 games of self-play. Neural network parameters are optimized by stochastic gradient descent with momentum and learning rate annealing, using the loss in equation (1). The learning rate is annealed according to the standard schedule in Extended Data Table 3. The momentum parameter is set to 0.9. The cross-entropy and MSE losses are weighted equally (this is reasonable because rewards are unit scaled, $r \in\{-1,+1\})$ and the $L 2$ regularization parameter is set to $c=10^{-4} $ .   
The optimization process produces a new checkpoint every 1,000 training steps. This checkpoint is evaluated by the evaluator and it may be used for generating the next batch of self-play games, as we explain next. 

##### Evaluator

为了确保总是生成最佳质量的数据，我们在将每个新的神经网络checkpoint与当前的最佳网络$f_{\theta *}$进行比较，然后再将其用于数据生成。通过MCTS search $$\alpha_{\theta_{i}}$$的性能来评估神经网络$$f_{\theta_{i}}$$，MCTS使用$f_{\theta_{i}}$来评估叶子节点的局面以及各个action的概率 。每次评估包括400局，使用一个有1600个模拟的MCTS来选择每一步棋，使用一个无限小的温度$\tau \rightarrow 0$（也就是说，**选择访问次数最大的acton**，以给出可能的最强下法）。如果新的player以>55%的幅度取胜（为了避免噪声干扰），那么它就成为最佳player $$\alpha_{\theta_*}$$，随后被用于self­play的生成，也成为后续比较的baseline。    访问次数最多, 对应着  博弈论里面的 平均策略.  

##### Self-play

在每一次迭代中，$$\alpha_{\theta_*}$$进行25,000次self­play，使用1,600次模拟MCTS来选择每一步棋（这需要每次搜索约0.4s）。对于每局棋的前30步棋，温度被设置为$\tau=1$；这将根据MCTS中的访问次数按比例选择move，并确保能访问多样性的局面。在棋局的剩下部分中，使用的是无限小的温度，$\tau \rightarrow 0$。通过在根节点$s_0$的先验概率中加入Dirichlet噪声来实现额外的探索，具体来说，$P(s, a)= (1-\varepsilon) p_{a}+\varepsilon \eta_{a}$ ,where $\boldsymbol{\eta} \sim \operatorname{Dir}(0.03)$ and $\varepsilon=0.25$；这个噪声保证了所有的move都可能被尝试，但搜索仍然可能会否决坏棋。   
为了节省计算量，显然输掉的棋局都会被放弃。resignation阈值$v_\text{resign}$是自动选择的，以保持假阳性（如果AlphaGo没有弃权，可能会赢的棋子）低于5%。为了测量假阳性率，我们在10%的自选对局中禁用弃权，直到终止。



##### Supervised learning

为了比较，我们还通过监督学习训练了神经网络参数$\theta_{\mathrm{SL}}$。神经网络的架构与AlphaGo Zero相同。我们从KGS数据集中随机抽取了Mini-batche $(s, \pi, z)$，对人类专家棋谱中的 move a设置 $\pi_{a}=1$ 。学习率退火, 带动量的SGD来优化参数.  动量参数被设置为0.9，L2正则化参数被设置为$c=10^{-4}$ .  loss公式跟上面一样, MSE component by a factor of 0.01. 
通过使用策略和值网络的组合，并通过使用低权重的value component，可以避免对values的过度拟合。在72 h后，move预测的准确率超过了之前的工作，在KGS测试集上达到了60.4%；value预测误差也比之前的有很大的提高



#### Search algorithm

AlphaGo Zero使用了AlphaGo Fan和AlphaGo Lee中使用的异步asynchronous policy and value MCTS algorithm (APVMCTS) 的一个更简单的变体。

搜索树中的每个节点s包含所有合法行为$a \in \mathcal{A}(s)$的边$(s, a)$。每个边存储了一组统计信息。
$$
\{N(s, a), W(s, a), Q(s, a), P(s, a)\}
$$
where $N(s, a)$ is the visit count, $W(s, a)$ is the total action value, $Q(s, a)$ is the mean action value and $P(s, a)$ is the prior probability of selecting that edge. 

多个模拟在不同的搜索线程上并行执行。该算法通过三个阶段的迭代进行，然后选择下一步棋。 注意下面都是异步执行的.

##### Select (Fig. 2a)

每个模拟的第一个树内阶段从搜索树的根节点$s_0$开始，当模拟到达一个叶子节点 $s_{L}$ 的时间步数L时结束。在每个time-steps, $t<L$, 按照 $a_{t}=\arg \max \left(Q\left(s_{t}, a\right)+U\left(s_{t}, a\right)\right)$  选择action.  使用 **PUCT** 算法的变体.
$$
U(s, a)=c_{\text {puca }} P(s, a) \frac{\sqrt{\Sigma_{b} N(s, b)}}{1+N(s, a)}
$$
其中$c_{\text {puct}}$ 是一个决定探索水平的常数；这种搜索控制策略最初倾向于高先验概率和低访问量的行动，但逐渐地倾向于高行动值的行动。

##### Expand and evaluate (Fig. 2b​)

叶节点 $s_{L}$ 被添加到神经网络的工作队列中进行评估 $\left(d_{i}(\boldsymbol{p}), v\right)=f_{\theta}\left(d_{i}\left(s_{L}\right)\right)$ , where $d_{i}$ is a dihedral reflection or rotation selected uniformly at random from $i$ in $[1..8]$  .   
队列中的Positions由神经网络进行评估, mini-batch:8；搜索线程被锁定，直到评估完成。

叶子节点被展开, 每条边$\left(s_{L}, a\right)$  被初始化为 $$\left\{N\left(s_{L}, a\right)=0, W\left(s_{L}, a\right)=0, Q\left(s_{L}, a\right)=0, {P}\left(s_{L}, a\right)=p_{a}\right\} $$;  然后value $v$ is backed up.

##### Backup (Fig. 2c)

向上更新所有的值. The edge statistics are updated in a backward pass through each step $t \leq L$.   
The visit counts are incremented, $N\left(s_{t}, a_{t}\right)=N\left(s_{t}, a_{t}\right)+1,$ and the action value is updated to the mean value, $$W\left(s_{t}, a_{t}\right)=W\left(s_{t}, a_{t}\right)+v, Q\left(s_{t}, a_{t}\right)=\frac{W\left(s_{t}, a_{t}\right)}{N\left(s_{t}, a_{t}\right)}$$
We use virtual loss to ensure each thread evaluates different nodes . 

##### Play (Fig. 2d)

在搜索结束时，AlphaGo Zero会选择在根部位置$s_0$ 中下一步棋，与它的指数化访问次数成正比$\pi\left(a | s_{0}\right)=N\left(s_{0}, a\right)^{1 / \tau} / \sum_{h} N\left(s_{0}, b\right)^{1 / \tau} $ ,    $\tau$用于控制探索水平。搜索树在随后的时间步中被重复使用:  
与下棋动作相对应的子节点成为新的根节点；在这个子节点下面的子树连同它的所有统计数据一起被保留，而树的其余部分被丢弃。AlphaGo Zero 认输, 如果其根节点的值和最佳子节点的值低于阈值 $v_{\text {resign: }}$

与AlphaGo Fan和AlphaGo Lee中的MCTS相比，主要区别在于：AlphaGo Zero不使用任何rollouts；它使用单一的神经网络，而不是单独的策略和值网络；叶子节点总是扩展，而不是使用动态扩展；每个搜索线程只需等待神经网络评估，而不是异步执行评估和backup；没有tree policy。

在AlphaGo Zero的大型（40块，40天）实例中也使用了transposition table。 



#### Neural network architecture

这次的输入也非常的干净.   
19×19×17 features.  The last feature is the color of the stone to play now.  



<img src="/img/2020-04-07-AlphaGo.assets/image-20200417025045654.png" alt="image-20200417025045654" style="zoom: 50%;" />



![img](/img/2020-04-07-AlphaGo.assets/1*WqwAVtTzYNQnlDEf2iJnsg.png)



<img src="/img/2020-04-07-AlphaGo.assets/alphago-zero-sketch.jpeg" alt="img" style="zoom: 33%;" />

其他也有用 resnet 训练围棋的, 不过只是 SL, 输出只有policy.



#### Neural network architecture comparison 网络结构比较

几个alphago版本的网络机构的异同:

- **dual–res**: contains a 20 block residual tower, followed by both a policy head and a value head.  used in **AlphaGo Zero**.
- **sep–res**: two 20 block residual towers. The first tower is followed by a policy head and the second tower is followed by a value head.
- **dual–conv**: a nonresidual tower of 12 convolutional blocks, followed by both a policy head and a value head.
- **sep–conv**: the network contains two nonresidual towers of 12 convolutional blocks. The first tower is followed by a policy head and the second tower is followed by a value head. used in **AlphaGo Lee**.

都是用的AlphaGo Zero selfplay产生的数据训练. 超参用的SL里面的: Each network was trained on a fixed dataset containing the final 2 million games of selfplay data generated by a previous run of AlphaGo Zero, using stochastic gradient descent with the annealing rate, momentum and regularization hyperparameters described for the supervised learning experiment; however, crossentropy and MSE components were weighted equally, since more data was available.



#### Evaluation 评估标准

Elo 评分标准.



**extended data table 1** | **Move prediction accuracy**

$$
\begin{array}{llll}
\hline & K G S \text { train } & K G S \text { test } & \text {GoKifu validation} \\
\hline \text { Supervised learning (20 block) } & 62.0 & 60.4 & 54.3 \\
\text { Supervised learning (12 layer }^{12} \text { ) } & 59.1 & 55.9 & - \\
\text { Reinforcement learning (20 block) } & - & - & 49.0 \\
\text { Reinforcement learning (40 block) } & - & - & 51.3 \\
\hline
\end{array}
$$

通过强化学习（即AlphaGo Zero）或监督学习训练的神经网络的下法预测准确率百分比。对于有监督学习，在KGS数据上训练了3天的神经网络（业余比赛）；比较结果也来自参考文献12。对于强化学习，20-block 网络被训练了3天，40-block 网络被训练了40天。网络也在基于GoKifu数据集中的职业比赛的验证集上进行了评估。

**extended data table 2** | **Game outcome prediction error**
$$
\begin{array}{llll}
\hline & K G S \text { train } & K G S \text { test } & \text {GoKifu validation} \\
\hline \text { Supervised learning (20 block) } & 0.177 & 0.185 & 0.207 \\
\text { Supervised learning (12 layer }^{12} \text { ) } & 0.19 & 0.37 & - \\
\text { Reinforcement learning (20 block) } & - & - & 0.177 \\
\text { Reinforcement learning (40 block) } & - & - & 0.180 \\
\hline
\end{array}
$$

**extended data table 3** | **Learning rate schedule**

$$
\begin{array}{ccc}
\hline \text { Thousands of steps } & \text { Reinforcement learning } & \text { Supervised learning } \\
\hline 0-200 & 10^{-2} & 10^{-1} \\
200-400 & 10^{-2} & 10^{-2} \\
400-600 & 10^{-3} & 10^{-3} \\
600-700 & 10^{-4} & 10^{-4} \\
700-800 & 10^{-4} & 10^{-5} \\
>800 & 10^{-4} & - \\
\hline
\end{array}
$$











## Reference

Mastering the game of Go without human knowledge  http://augmentingcognition.com/assets/Silver2017a.pdf



AlphaGo Zero — a game changer.  https://medium.com/@jonathan_hui/alphago-zero-a-game-changer-14ef6e45eba5



AlphaGo Zero Explained In One Diagram https://medium.com/applied-data-science/alphago-zero-explained-in-one-diagram-365f5abf67e0



network pic from https://blog.acolyer.org/2017/11/17/mastering-the-game-of-go-without-human-knowledge/