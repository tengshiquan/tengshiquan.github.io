---
layout:     post
title:      CS 285. Exploration
subtitle:   CS 285. Deep Reinforcement Learning, Decision Making, and Control
date:       2020-03-16 12:00:00
author:     "tengshiquan"
header-img: "img/post-bg-berkeley.jpg"
catalog: true
tags:
    - AI
    - Reinforcement Learning

---

 

## Exploration

Exploration and exploitation 问题

最后一部分通过demonstration来解决exploration问题.

 

#### Lecture

1. What is exploration? Why is it a problem?
2. **Multi-armed bandits** and theoretically grounded exploration
3. Optimism-based exploration
4. Posterior matching exploration
5. Information-theoretic exploration

##### Goals:

- Understand what the exploration is
- Understand how theoretically grounded exploration methods can be derived 
- Understand how we can do exploration in deep RL in practice



#### What’s the problem?

<img src="/img/CS285.assets/image-20200328014229859.png" alt="image-20200328014229859" style="zoom: 33%;" />





##### Montezuma’s revenge

- Getting key = reward
- Opening door = reward
- Getting killed by skull = nothing (is it good? bad?)
- Finishing the game only weakly correlates with rewarding events
- We know what to do because we **understand** what these sprites mean!

感觉只有一种学习网络结构是不够的,  CNN配DNN 这种也不够, 应该是更复杂的某种形式. 



##### Put yourself in the algorithm’s shoes

扑克牌 Mao ,  主持人设置隐藏规则,  然后玩家通过打牌来猜.  这个难度过大. 



### Exploration and exploitation

- Two potential definitions of exploration problem
  - How can an agent discover high-reward strategies that require a temporally extended sequence of complex behaviors that, individually, are not rewarding?  长序列中间0reward
  - How can an agent decide whether to attempt new behaviors (to discover ones with higher reward) or continue to do the best thing it knows so far? 探索还是当前高回报

- Actually the same problem:
  - **Exploitation**: doing what you *know* will yield highest reward
  - **Exploration**: doing things you haven’t done before, in the hopes of getting even higher reward



#### Exploration is hard

Can we derive an **optimal** exploration strategy?

what does optimal even mean?

regret vs. Bayes-optimal strategy? more on this later...



<img src="/img/CS285.assets/image-20200328015425865.png" alt="image-20200328015425865" style="zoom:50%;" />



#### What makes an exploration problem tractable?

- multi-arm bandits 
  - can formalize exploration as POMDP identification
- contextual bandits
  - policy learning is trivial even with POMDP
- small, finite MDPs 
  - can frame as Bayesian model identification, reason explicitly about value of information
- large or infinite MDPs
  - optimal methods don’t work...but can take inspiration from optimal methods in smaller settings use hacks



#### Bandits

<img src="/img/CS285.assets/image-20200328015746845.png" alt="image-20200328015746845" style="zoom:50%;" />

单台

$$
\mathcal{A}=\{\mathrm{pull} \mathrm{arm}\}  \\    
      r(\text { pull arm })=?  
$$

多台机器

$$
\mathcal{A}=\left\{\mathrm{pull}_{1}, \mathrm{pull}_{2}, \ldots, \mathrm{pull}_{n}\right\}\\
 r\left(a_{n}\right)=?\\
 \text { assume } r\left(a_{n}\right) \sim p\left(r  \vert  a_{n}\right)
$$

unkown per-action reward distribution.



#### How can we define the bandit?

- assume $$r\left(a_{i}\right) \sim p_{\theta_{i}}\left(r_{i}\right)$$   每台的r服从一个分布
- e.g., $$p\left(r_{i}=1\right)=\theta_{i}$$ and $$p\left(r_{i}=0\right)=1-\theta_{i}$$
- $$\theta_{i} \sim p(\theta)$$,  but otherwise unknown   关键就是这个每台机器的参数未知
- this defines a POMDP with $$\mathbf{s}=\left[\theta_{1}, \ldots, \theta_{n}\right]$$
  **belief state** is $$\hat{p}\left(\theta_{1}, \ldots, \theta_{n}\right)$$



- solving the POMDP yields the optimal exploration strategy
- but that’s overkill: **belief state is huge**!
- we can do very well with much simpler strategies



how do we measure goodness of exploration algorithm?
**regret**: difference from optimal policy at time step $$T$$ :  用于理论分析
$$
\operatorname{Reg}(T)=T E\left[r\left(a^{\star}\right)\right]-\sum_{t=1}^{T} r\left(a_{t}\right)
$$

- r* : expected reward of best action   (the best we can hope for in expectation) 现实中可能无法得知
- r: actual reward of action actually taken



- Variety of relatively simple strategies
- Often can provide theoretical guarantees on regret
  - Variety of optimal algorithms (up to a constant factor)
  - But empirical performance may vary...
- **Exploration strategies** for more complex MDP domains will be inspired by these strategies



#### Optimistic exploration

探索优先, UCB

- keep track of average reward $$\hat{\mu}_{a}$$ for each action $$a$$

- **exploitation**: pick $$a=\arg \max \hat{\mu}_{a}$$   只做到这, 会陷入局部最优

- **optimistic estimate**: $$a=\arg \max \hat{\mu}_{a}+C \sigma_{a}$$
  
  - $$\sigma_{a}$$: some sort of variance estimate , 代表对a的uncertainty,  可以设计
  - C代表exploration的权重
  
- **intuition: try each arm until you are sure it's not great**

- example: UCB (Auer et al. Finite-time analysis of the multiarmed bandit problem):

  $$
  a=\arg \max \hat{\mu}_{a}+\sqrt{\frac{2 \ln T}{N(a)}}
  $$
  
  $$\operatorname{Reg}(T)$$ is $$O(\log T),$$ provably as good as any algorithm



#### Probability matching/posterior sampling

概率匹配, 后验抽样  ,  基于假设的模型来采样, 然后修正模型

- assume $$r\left(a_{i}\right) \sim p_{\theta_{i}}\left(r_{i}\right)$$
- this defines a POMDP with $$\mathbf{s}=\left[\theta_{1}, \ldots, \theta_{n}\right]$$
- **belief state** is $$\hat{p}\left(\theta_{1}, \ldots, \theta_{n}\right)$$ -- this is a model of our bandit



1. idea: sample $$\theta_{1}, \ldots, \theta_{n} \sim \hat{p}\left(\theta_{1}, \ldots, \theta_{n}\right)$$
2. pretend the model $$\theta_{1}, \ldots, \theta_{n}$$ is correct
3. take the optimal action
4. update the model , goto 1



- This is called **posterior sampling** or **Thompson sampling**
- Harder to analyze theoretically
- Can **work very well empirically** ,  larger MDP problem 可以用

See: Chapelle & Li, “An Empirical Evaluation of Thompson Sampling.”



#### Information gain

- Bayesian experimental design:

  - say we want to determine some latent variable $$z$$ (e.g., $$z$$ might be the optimal action, or its value) 
  - which action do we take?



因为现在不知道z的具体值, 所以对z有一个估计: $$\hat p(z=?)$$这个分布

- let $$\mathcal{H}(\hat{p}(z))$$ be the current entropy of our $$z$$ estimate , 如果很明确知道z这个值, 则entropy很小. 

- let $$\mathcal{H}(\hat{p}(z)  \vert  y)$$ be the entropy of our $$z$$ estimate after observation $$y$$ , 

  - e.g. , $$y$$ might be $$r(a)$$ 如果y很接近z, 并且之前z不太对, 那entropy就会大幅变小; 如果y与z基本不相关, 则变化幅度不大.

- **the lower the entropy, the more precisely we know $$z$$**   
$$
  \mathrm{IG}(z, y)=E_{y}[\mathcal{H}(\hat{p}(z))-\mathcal{H}(\hat{p}(z)  \vert  y)]
$$

  

typically depends on action, so we have  $$\mathrm{IG}(z, y  \vert  a)$$



##### Information gain example

how much we learn about $$z$$ from action $$a,$$ given current beliefs

Example bandit algorithm: Russo \& Van Roy "Learning to Optimize via Information-Directed Sampling"

- $$y=r_{a}, z=\theta_{a}\left(\text { parameters of model } p\left(r_{a}\right)\right)$$
- $$g(a)=\mathrm{IG}\left(\theta_{a}, r_{a}  \vert  a\right)-$$ information gain of $$a$$
- $$\Delta(a)=E\left[r\left(a^{\star}\right)-r(a)\right]-$$ expected suboptimality of $$a$$ ,  current belief , 这个值越小, a越接近optimal
- choose a according to $$\arg \min _{a} \frac{\Delta(a)^{2}}{g(a) }$$ 
  - don’t take actions that you’re sure are suboptimal
  - don’t bother taking actions if you won’t learn anything



#### General themes

- UCB:

  $$
  a=\arg \max \hat{\mu}_{a}+\sqrt{\frac{2 \ln T}{N(a)}}
  $$

- Thompson sampling:

  $$
  \theta_{1}, \ldots, \theta_{n} \sim \hat{p}\left(\theta_{1}, \ldots, \theta_{n}\right) \\
  a=\arg \max_{a}  { E_{\theta_a} [r(a)]   }
  $$

- Info gain:

  $$
  \mathrm{IG}(z, y  \vert  a)
  $$
  





- Most exploration strategies require some kind of uncertainty estimation (even if it’s naïve)

- Usually assumes some value to new information
  - Assume unknown = good (optimism)
  - Assume sample = truth
  - Assume information gain = good



#### Why should we care?

- Bandits are easier to analyze and understand
- Can derive foundations for exploration methods
- Then apply these methods to more complex MDPs
- Not covered here:
  - **Contextual bandits** (bandits with state, essentially 1-step MDPs)
  - Optimal exploration in small MDPs
  - Bayesian model-based reinforcement learning (similar to information gain)
  - **Probably approximately correct (PAC)** exploration





### Classes of exploration methods in deep RL

- **Optimistic exploration**:
  - new state = good state
  - requires estimating state visitation frequencies or novelty
  - typically realized by means of exploration bonuses
- **Thompson sampling** style algorithms:
  - learn distribution over Q-functions or policies
  - sample and act according to sample
- **Information gain** style algorithms
  - reason about information gain from visiting new states



#### Optimistic exploration in RL

- UCB:

  $$
  a=\arg \max \hat{\mu}_{a}+ \underbrace{\sqrt{\frac{2 \ln T}{N(a)}}}_\text{“exploration bonus”}
  $$

  - lots of functions work, so long as they decrease with $$N(a)$$

- can we use this idea with MDPs?

- **count-based** exploration: use $$N(\mathbf{s}, \mathbf{a})$$ or $$N(\mathbf{s})$$ to add exploration bonus

- use $$r^{+}(\mathbf{s}, \mathbf{a})=r(\mathbf{s}, \mathbf{a})+\mathcal{B}(N(\mathbf{s}))$$

  - bonus that decreases with $$N(\mathrm{s})$$

- use $$r^{+}(\mathbf{s}, \mathbf{a})$$ instead of $$r(\mathbf{s}, \mathbf{a})$$ with any model-free algorithm
  - \+ simple addition to any RL algorithm
  - \- need to tune bonus weight



##### The trouble with counts

- what’s a count?

- Uh oh... we never see the same thing twice!   有些MDP s很难复现

- But some states are more similar than others



#### Fitting generative models

- idea: fit a density model $$p_{\theta}(\mathbf{s})$$  (or  $$p_{\theta}(\mathbf{s}, \mathbf{a})$$ )
- $$p_{\theta}(\mathbf{s})$$ might be high even for a new $$\mathbf{s}$$ if s is similar to previously seen states
- can we use $$p_{\theta}(\mathbf{s})$$ to get a "pseudo-count"?
- if we have small MDPs the true probability is: probability/density 
  

$$
P(\mathbf{s})=\frac{N(\mathbf{s})}{n}
$$
- after we see $$\mathbf{s},$$ we have:

$$
P^{\prime}(\mathbf{s})=\frac{N(\mathbf{s})+1}{n+1}
$$

- can we get $$p_{\theta}(\mathbf{s})$$ and $$p_{\theta^{\prime}}(\mathbf{s})$$ to obey these equations?





##### Exploring with pseudo-counts

1. fit model $$p_{\theta}(\mathbf{s})$$ to all states $$\mathcal{D}$$ seen so far
2. take a step $$i$$ and observe $$\mathbf{s}_{i}$$
3. fit new model $$p_{\theta^{\prime}}(\mathbf{s})$$ to $$\mathcal{D} \cup \mathbf{s}_{i}$$
4. use $$p_{\theta}\left(\mathbf{s}_{i}\right)$$ and $$p_{\theta^{\prime}}\left(\mathbf{s}_{i}\right)$$ to estimate $$\hat{N}(\mathbf{s})$$
5. $$\operatorname{set} r_{i}^{+}=r_{i}+\mathcal{B}(\hat{N}(\mathrm{s})) \longleftarrow$$  "pseudo-count" ,   goto 1



- how to get $$\hat{N}(\mathrm{s}) ?$$ use the equations
  
  $$
  p_{\theta}\left(\mathbf{s}_{i}\right)=\frac{\hat{N}\left(\mathbf{s}_{i}\right)}{\hat{n}} \quad \quad p_{\theta^{\prime}}\left(\mathbf{s}_{i}\right)=\frac{\hat{N}\left(\mathbf{s}_{i}\right)+1}{\hat{n}+1}
  $$

- two equations and two unknowns!
  
  $$
  \hat{N}\left(\mathbf{s}_{i}\right)=\hat{n} p_{\theta}\left(\mathbf{s}_{i}\right) \quad \quad \hat{n}=\frac{1-p_{\theta^{\prime}}\left(\mathbf{s}_{i}\right)}{p_{\theta^{\prime}}\left(\mathbf{s}_{i}\right)-p_{\theta}\left(\mathbf{s}_{i}\right)} p_{\theta}\left(\mathbf{s}_{i}\right)
  $$



Bellemare et al. “Unifying Count-Based Exploration...”





#### What kind of bonus to use?

Lots of functions in the literature, inspired by optimal methods for bandits or small MDPs

- UCB:
  
  $$
  \mathcal{B}(N(\mathbf{s})) = \sqrt{\frac{2 \ln T}{N(\mathbf{s})}}
  $$
  
- MBIE-EB (Strehl & Littman, 2008):  this is the one used by Bellemare et al. ‘16
  
  $$
  \mathcal{B}(N(\mathbf{s})) = \sqrt{\frac{1}{N(\mathbf{s})}}
  $$
  
- BEB (Kolter & Ng, 2009):
  
  $$
  \mathcal{B}(N(\mathbf{s})) = \frac{1}{N(\mathbf{s})}
  $$
  
  
  
  

#### Does it work?

<img src="/img/CS285.assets/image-20200328161128647.png" alt="image-20200328161128647" style="zoom:50%;" />

<img src="/img/CS285.assets/image-20200328161201767.png" alt="image-20200328161201767" style="zoom:50%;" />

#### What kind of model to use?

$$
p_{\theta}\left(\mathbf{s}\right)
$$

- need to be able to **output densities**, but doesn’t necessarily need to produce great samples

- opposite considerations from many popular generative models in the literature (e.g., GANs)

- Bellemare et al.: “CTS” model: condition each pixel on its top-left neighborhood

  <img src="/img/CS285.assets/image-20200328161652782.png" alt="image-20200328161652782" style="zoom:33%;" />

- Other models: stochastic neural networks, compression length, EX2



#### Counting with hashes

**What if we still count states, but in a different space?**

- idea: compress s into a $$k$$ -bit code via $$\phi(\mathbf{s}),$$ then count $$N(\phi(\mathbf{s}))$$
- shorter codes $$=$$ more hash collisions
- similar states get the same hash? **maybe**
- improve the odds by learning a compression:

autoencoder

<img src="/img/CS285.assets/image-20200328161926398.png" alt="image-20200328161926398" style="zoom:50%;" />

Work 的不错

<img src="/img/CS285.assets/image-20200328161859552.png" alt="image-20200328161859552" style="zoom:50%;" />

Tang et al. “#Exploration: A Study of Count-Based Exploration”



#### Implicit density modeling with exemplar models

Fu et al. “EX2: Exploration with Exemplar Models...”

- $$p_{\theta}\left(\mathbf{s}\right)$$ need to be able to output densities, but doesn’t necessarily need to produce great samples
- Can we explicitly **compare** the new state to past states?
- Intuition: the state is **novel** if it is **easy** to distinguish from all previous seen states by a classifier

for each observed state $$\mathbf{s}$$, fit a classifier to classify that state against all past states $$\mathcal{D}$$, use classifier error to obtain density
$$
p_{\theta}(\mathbf{s})=\frac{1-D_{\mathbf{s}}(\mathbf{s})}{D_{\mathbf{s}}(\mathbf{s})}
$$
 assigns that $$s$$ is "positive" : $$\{ \mathbf s\}$$ ,  negatives: $$\mathcal{D}$$ 



- hang on... aren't we just checking if $$\mathbf{s}=\mathbf{s} ?$$ 

- if $$\mathbf s \in \mathcal D$$, then the optimal $$D_s(\mathbf s) \neq 1$$ .

- in fact: 
  $$
  \quad D_{\mathrm{S}}^{\star}(\mathrm{s})=\frac{1}{1+p(\mathrm{s})} \longrightarrow p_{\theta}(\mathbf{s})=\frac{1-D_{\mathbf{s}}(\mathbf{s})}{D_{\mathbf{s}}(\mathbf{s})}
  $$
  
- in reality, each state is unique, so we regularize the classifier

- isn't one classifier per state a bit much?

- train one amortized model: single network that takes in exemplar as input!

<img src="/img/CS285.assets/image-20200328163102930.png" alt="image-20200328163102930" style="zoom:50%;" />

<img src="/img/CS285.assets/image-20200328163154249.png" alt="image-20200328163154249" style="zoom:50%;" />





#### Heuristic estimation of counts via errors

$$p_{\theta}\left(\mathbf{s}\right)$$ need to be able to output densities, but doesn’t necessarily need to produce great samples

...and doesn’t even need to output great densities ...just need to tell if state is **novel** or not!



- let's say we have some target function $$f^{\star}(\mathbf{s}, \mathbf{a})$$
- given our buffer $$\mathcal{D}=\left\{\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)\right\},$$ fit $$\hat{f}_{\theta}(\mathbf{s}, \mathbf{a})$$
- use $$\mathcal{E}(\mathbf{s}, \mathbf{a})=\left \Vert \hat{f}_{\theta}(\mathbf{s}, \mathbf{a})-f^{\star}(\mathbf{s}, \mathbf{a})\right \Vert ^{2}$$ as bonus



<img src="/img/CS285.assets/image-20200328163343321.png" alt="image-20200328163343321" style="zoom:33%;" />

##### what should we use for $$f^{\star}(\mathbf{s}, \mathbf{a}) ?$$

- one common choice: set $$f^{\star}(\mathbf{s}, \mathbf{a})=\mathbf{s}^{\prime}$$  -- i.e., next state prediction

- also related to information gain, which we'll discuss next time!
- even simpler: $$f^{\star}(\mathbf{s}, \mathbf{a})=f_{\phi}(\mathbf{s}, \mathbf{a}),$$ where $$\phi$$ is a random parameter vector



#### Posterior sampling in deep RL

What do we sample?  How do we represent the distribution?

bandit setting: $$\hat{p}\left(\theta_{1}, \ldots, \theta_{n}\right)$$ is distribution over rewards

MDP analog is the $$Q$$ -function!

1. sample Q-function $$Q$$ from $$p(Q)$$
2. act according to $$Q$$ for one episode 
3. update $$p(Q)$$  .   goto 1.

对3.  since Q-learning is off-policy, we don't care which Q-function was used to collect data

how can we represent a distribution over functions?



#### Bootstrap

- given a dataset $$\mathcal{D},$$ resample with replacement $$N$$ times to get $$\mathcal{D}_{1}, \ldots, \mathcal{D}_{N}$$
- train each model $$f_{\theta_{i}}$$ on $$\mathcal{D}_{i}$$
- to sample from $$p(\theta),$$ sample $$i \in[1, \ldots, N]$$ and use $$f_{\theta_{i}}$$

training N big neural nets is expensive , can we avoid it ?

<img src="/img/CS285.assets/image-20200328164546212.png" alt="image-20200328164546212" style="zoom: 33%;" />

Osband et al. “Deep Exploration via Bootstrapped DQN”



##### Why does this work?

Exploring with random actions (e.g., epsilon-greedy): oscillate back and forth, might not go to a coherent or interesting place

Exploring with random Q-functions: commit to a randomized but internally consistent strategy for an entire episode

<img src="/img/CS285.assets/image-20200328164722430.png" alt="image-20200328164722430" style="zoom:33%;" />

\+ no change to original reward function

\- very good bonuses often do better





### Reasoning about information gain (approximately)

- Info gain:

  $$
  \mathrm{IG}(z, y  \vert  a)
  $$

- information gain about what? 
- information gain about reward $$r(\mathbf{s}, \mathbf{a}) ?$$       not very useful if reward is sparse
- state density $$p(\mathbf{s}) ?$$                       a bit strange, but somewhat makes sense! 
- information gain about dynamics $$p\left(\mathbf{s}^{\prime}  \vert  \mathbf{s}, \mathbf{a}\right) ? \quad$$ good proxy for learning the MDP, though still heuristic



**Generally intractable to use exactly, regardless of what is being estimated!**

A few approximations:

prediction gain: $$\log p_{\theta'}(\mathbf s) - \log p_{\theta}(\mathbf s)$$     (Schmidhuber ‘91, Bellemare ‘16)

intuition: if density changed a lot, the state was novel



variational inference:                  (Houthooft et al. "VIME") 

- IG can be equivalently written as $$D_{\mathrm{KL}}(p(z  \vert  y)  \Vert  p(z))$$ 
- learn about transitions $$p_{\theta}\left(s_{t+1}  \vert  s_{t}, a_{t}\right): z=\theta$$
- $$y=\left(s_{t}, a_{t}, s_{t+1}\right)$$ . 

$$
D_{\mathrm{KL}}\left(p\left(\theta  \vert  h, s_{t}, a_{t}, s_{t+1}\right)  \Vert  p(\theta  \vert  h)\right)
$$

- $$\theta$$ : model parameters for $$p_{\theta}\left(s_{t+1}  \vert  s_{t}, a_{t}\right)$$
- $$h$$ :  history of all prior transitions
- $$s_{t}, a_{t}, s_{t+1}$$ : newly observed transition



- intuition: a transition is more informative if it causes belief over $$\theta$$ to change 
- idea: use variational inference to estimate $$q(\theta  \vert  \phi) \approx p(\theta  \vert  h)$$ 
- given new transition $$\left(s, a, s^{\prime}\right),$$ update $$\phi$$ to get $$\phi^{\prime}$$



VIME implementation:

IG can be equivalently written as $$D_{\mathrm{KL}}\left(p\left(\theta  \vert  h, s_{t}, a_{t}, s_{t+1}\right)  \Vert  p(\theta  \vert  h)\right)$$

- $$q(\theta  \vert  \phi) \approx p(\theta  \vert  h)$$   specifically, optimize variational lower bound $$D_{\mathrm{KL}}(q(\theta  \vert  \phi)  \Vert  p(h  \vert  \theta) p(\theta))$$

- represent $$q(\theta  \vert  \phi)$$ as product of independent Gaussian parameter distributions with mean $$\phi$$
  (see Blundell et al. "Weight uncertainty in neural networks").  贝叶斯网络
  $$
  p(\theta  \vert  \mathcal{D})=\prod p\left(\theta_{i}  \vert  \mathcal{D}\right)\\
  p(\theta_i  \vert  \mathcal{D})= \mathcal N \left(\mu_{i} , \sigma_i\right)    \\
  \\           \quad\quad\quad\quad\quad \nwarrow \nearrow
  \\   \quad\quad\quad  \quad\quad  \phi
  $$
  
- given new transition $$\left(s, a, s^{\prime}\right),$$ update $$\phi$$ to get $$\phi^{\prime}$$
- this corresponds to updating the network weight means and variances
- use $$D_{\mathrm{KL}}\left(q\left(\theta  \vert  \phi^{\prime}\right)  \Vert  q(\theta  \vert  \phi)\right)$$ as approximate bonus

<img src="/img/CS285.assets/image-20200328192902641.png" alt="image-20200328192902641" style="zoom:50%;" />

Approximate IG:

- \+ appealing mathematical formalism
- \- models are more complex, generally harder to use effectively



if we forget about IG, there are many other ways to measure this



Stadie et al. 2015:

- encode image observations using auto-encoder
- build predictive model on auto-encoder latent states
- use model error as exploration bonus

high novelty

low novelty

Schmidhuber et al. (see, e.g. “Formal Theory of Creativity, Fun, and Intrinsic Motivation):

- exploration bonus for model error
- exploration bonus for model gradient
- many other variations





#### Suggested readings

- Schmidhuber. (1992). **A Possibility for Implementing Curiosity and Boredom in Model-Building Neural Controllers.**
- Stadie, Levine, Abbeel (2015). **Incentivizing Exploration in Reinforcement Learning with Deep Predictive Models.**

- Osband, Blundell, Pritzel, Van Roy. (2016). **Deep Exploration via Bootstrapped DQN.** 
- Houthooft, Chen, Duan, Schulman, De Turck, Abbeel. (2016). **VIME: Variational Information Maximizing Exploration.** 
- Bellemare, Srinivasan, Ostroviski, Schaul, Saxton, Munos. (2016). **Unifying Count-Based Exploration and Intrinsic Motivation.** 
- Tang, Houthooft, Foote, Stooke, Chen, Duan, Schulman, De Turck, Abbeel. (2016). **#Exploration: A Study of Count-Based Exploration for Deep Reinforcement Learning. **
- Fu, Co-Reyes, Levine. (2017). **EX2: Exploration with Exemplar Models for Deep Reinforcement Learning.**

 



## Imitation vs. Reinforcement Learning

- imitation learning
  - Requires demonstrations
  - Must address **distributional shift**
  - **Simple, stable** supervised learning 
  - Only **as good as the demo**

- reinforcement learning
  - Requires reward function
  - Must address exploration
  - Potentially non-convergent RL 
  - Can become arbitrarily good



Can we get the best of both?
 e.g., what if we have demonstrations *and* rewards?

结合模仿学习与强化学习



#### Addressing distributional shift with RL?

**IRL *already* addresses distributional shift via RL**

<img src="/img/CS285.assets/image-20200328213202486.png" alt="image-20200328213202486" style="zoom:50%;" />

But it doesn’t use a known reward function!



#### Simplest combination: pretrain & finetune

- **Demonstrations can overcome exploration**: show us how to do the task
- **Reinforcement learning** can improve **beyond** performance of the demonstrator
- Idea: initialize with imitation learning, then finetune with reinforcement learning!

用专家经验来初始化网络, 然后再 RL finetune.

1. collected demonstration data $$\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)$$
2. initialize $$\pi_{\theta}$$ as $$\max _{\theta} \sum_{i} \log \pi_{\theta}\left(\mathbf{a}_{i}  \vert  \mathbf{s}_{i}\right)$$
3. run $$\pi_{\theta}$$ to collect experience
4. improve $$\pi_{\theta}$$ with any $$\mathrm{RL}$$ algorithm  , goto 3





#### Simplest combination: pretrain & finetune

Muelling et al. ‘13



结合 DAgger

1. train $$\pi_{\theta}\left(\mathbf{a}_{t} \vert \mathbf{o}_{t}\right)$$ from human data $$\mathcal{D}=\left\{\mathbf{o}_{1}, \mathbf{a}_{1}, \ldots, \mathbf{o}_{N}, \mathbf{a}_{N}\right\}$$
2. run $$\pi_{\theta}\left(\mathbf{a}_{t} \vert \mathbf{o}_{t}\right)$$ to get dataset $$\mathcal{D}_{\pi}=\left\{\mathbf{o}_{1}, \ldots, \mathbf{o}_{M}\right\}$$
3. Ask human to label $$\mathcal{D}_{\pi}$$ with actions  $$a_{t}$$
4. **Aggregate**: $$\mathcal{D} \leftarrow \mathcal{D} \cup \mathcal{D}_{\pi}$$ , goto 1.



#### What’s the problem?

**Pretrain & finetune**

1. collected demonstration data $$\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)$$
2. initialize $$\pi_{\theta}$$ as $$\max _{\theta} \sum_{i} \log \pi_{\theta}\left(\mathbf{a}_{i}  \vert  \mathbf{s}_{i}\right)$$
3. run $$\pi_{\theta}$$ to collect experience   <=== can be very bad (due to distribution shift)
4. improve $$\pi_{\theta}$$ with any $$\mathrm{RL}$$ algorithm  , goto 3     <=== first batch of (very) bad data can destroy initialization

Can we avoid **forgetting the demonstrations**?



#### Off-policy reinforcement learning

- Off-policy RL can use any data
- If we let it use demonstrations as off-policy samples, can that mitigate the exploration challenges?
  - **Since demonstrations are provided as data in every iteration, they are never forgotten**
  - But the policy can still become *better* than the demos, since it is not forced to mimic them

off-policy policy gradient (with importance sampling) 

off-policy Q-learning



#### Policy gradient with demonstrations

$$
\nabla_{\theta} J(\theta)=\sum_{\tau \in \mathcal{D}}\left[\sum_{t=1}^{T} \nabla_{\theta} \log \pi_{\theta}\left(\mathbf{a}_{t}  \vert  \mathbf{s}_{t}\right)\left(\prod_{t^{\prime}=1}^{t} \frac{\pi_{\theta}\left(\mathbf{a}_{t^{\prime}}  \vert  \mathbf{s}_{t^{\prime}}\right)}{q\left(\mathbf{a}_{t^{\prime}}  \vert  \mathbf{s}_{t^{\prime}}\right)}\right)\left(\sum_{t^{\prime}=t}^{T} r\left(\mathbf{s}_{t^{\prime}}, \mathbf{a}_{t^{\prime}}\right)\right)\right]
$$

  $$\mathcal D$$ : includes demonstrations *and* experience 

Why is this a good idea? Don’t we want on-policy samples?

**optimal importance sampling**

- say we want $$E_{p(x)}[f(x)]$$ ,  $$E_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i} \frac{p\left(x_{i}\right)}{q\left(x_{i}\right)} f\left(x_{i}\right)$$ , which $$q(x)$$ gives lowest variance?
- answer: $$q(x) \propto p(x) \vert f(x) \vert $$

best sampling distribution should have high reward!

<img src="/img/CS285.assets/image-20200328232246300.png" alt="image-20200328232246300" style="zoom:33%;" />



##### How do we construct the sampling distribution?

- standard IS         $$E_{p(x)}[f(x)] \approx \frac{1}{N} \sum_{i} \frac{p\left(x_{i}\right)}{q\left(x_{i}\right)} f\left(x_{i}\right)$$  
- Self-normalized IS   $$E_{p(x)}[f(x)] \approx \frac{1}{\sum_{j} \frac{p\left(x_{j}\right)}{q\left(x_{j}\right)}} \sum_{i} \frac{p\left(x_{i}\right)}{q\left(x_{i}\right)} f\left(x_{i}\right)$$ 

problem 1: which distribution did the demonstrations come from? 

- option 1: use supervised behavior cloning to approximate  
- option 2: assume Diract delta: $$\pi_{\text {demo }}(\tau)=\frac{1}{N} \delta(\tau \in \mathcal{D})$$ , this works best with self-normalized importance sampling

problem 2: what to do if we have multiple distributions?

- fusion distribution: $$q(x)=\frac{1}{M} \sum_{i} q_{i}(x)$$





#### Q-learning with demonstrations

- Q-learning is *already* off-policy, no need to bother with importance weights!
- Simple solution: drop demonstrations into the **replay buffer**

  

**full Q-learning with replay buffe**r:

1. initialize $$\mathcal{B}$$ to contain the demonstration data
2. collect dataset $$\left\{\left(\mathbf{s}_{i}, \mathbf{a}_{i}, \mathbf{s}_{i}^{\prime}, r_{i}\right)\right\}$$ using some policy, add it to $$\mathcal{B}$$ 
3.  sample a batch $$\left(\mathbf{s}_{i}, \mathbf{a}_{i}, \mathbf{s}_{i}^{\prime}, r_{i}\right)$$ from $$\mathcal{B}$$
4. $$\phi \leftarrow \phi-\alpha \sum_{i} \frac{d Q_{\phi}}{d \phi}\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)\left(Q_{\phi}\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)-\left[r\left(\mathbf{s}_{i}, \mathbf{a}_{i}\right)+\gamma \max _{\mathbf{a}^{\prime}} Q_{\phi}\left(\mathbf{s}_{i}^{\prime}, \mathbf{a}_{i}^{\prime}\right)\right]\right)$$ , goto 3, or goto 2 every K steps 



Vecerik et al., ‘17, “Leveraging Demonstrations for Deep Reinforcement Learning...”

<img src="/img/CS285.assets/image-20200328234110361.png" alt="image-20200328234110361" style="zoom:50%;" />



##### What’s the problem?

Importance sampling: recipe for getting stuck 

<img src="/img/CS285.assets/image-20200328234158563.png" alt="image-20200328234158563" style="zoom:33%;" />



Q-learning: just good data is not enough

<img src="/img/CS285.assets/image-20200328234219088.png" alt="image-20200328234219088" style="zoom:50%;" />



##### More problems with Q learning

$$Q(\mathbf{s}, \mathbf{a}) \leftarrow r(\mathbf{s}, \mathbf{a})+\max _{\mathbf{a}^{\prime}} Q\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)$$ ,  $$Q(\mathrm{s}, \mathrm{a})$$ is trained on $$(\mathbf{s}, \mathbf{a}) \sim \beta(\mathbf{s}, \mathbf{a})$$

what action will this pick?

<img src="/img/CS285.assets/image-20200328234723535.png" alt="image-20200328234723535" style="zoom: 33%;" />

if $$\mathbf{a}^{\star}=\arg \max _{\mathbf{a}} Q(\mathbf{s}, \mathbf{a})$$ makes $$\beta\left(\mathbf{s}, \mathbf{a}^{\star}\right)$$ small,  we end up training on garbage!

See, e.g. Riedmiller, Neural Fitted Q-Iteration ‘05 Ernst et al., Tree-Based Batch Mode RL ‘05



$$Q(\mathbf{s}, \mathbf{a}) \leftarrow r(\mathbf{s}, \mathbf{a})+E_{\mathbf{a}^{\prime} \sim \pi_{\text {new }}}\left[Q\left(\mathbf{s}^{\prime}, \mathbf{a}^{\prime}\right)\right]$$   how to pick $$\pi_{\text {new }}(\mathbf{a}  \vert  \mathbf{s}) ?$$

- option 1: stay close to $$\beta$$ ,   e.g.  $$D_{\mathrm{KL}}\left(\pi_{\text {new }}(\cdot  \vert  \mathbf{s})  \Vert  \beta(\cdot  \vert  \mathbf{s})\right) \leq \epsilon$$

  - issue 1: we don't know $$\beta$$
  - issue 2: this is way too conservative

  key idea: constrain to support of $$\beta$$ 

  

$$
\max _{\pi \in \Delta_{ \vert S \vert }} \mathbb{E}_{a \sim \pi(\cdot  \vert  s)}\left[\hat{Q}_{k}(s, a)\right]-\lambda \sqrt{\operatorname{var}_{\mathbf{k}} \mathbb{E}_{a \sim \pi(\cdot  \vert  s)}\left[\hat{Q}_{k}(s, a)\right]} \quad \text{pessimistic w.r.t. epistemic uncertainty}
\\
\text{s.t. } \mathbb{E}_{s \sim \mathcal{D}}[\operatorname{MMD}(\mathcal{D}(s), \pi(\cdot  \vert  s))] \leq \varepsilon  \quad \text{support constraint}
$$
<img src="/img/CS285.assets/image-20200328235326838.png" alt="image-20200328235326838" style="zoom:50%;" />



<img src="/img/CS285.assets/image-20200328235356130.png" alt="image-20200328235356130" style="zoom:50%;" />

See: Kumar, Fu, Tucker, Levine. **Stabilizing Off-Policy Q-Learning via Bootstrapping Error Reduction.** See also: Fujimoto, Meger, Precup**. Off-Policy Deep Reinforcement Learning without Exploration.**





Can we strike a compromise? A little bit of supervised, a little bit of RL?

#### Imitation as an auxiliary loss function

- imitation objective: $$\sum_{(\mathbf{s}, \mathbf{a}) \in \mathcal{D}_{\text {demo }}} \log \pi_{\theta}(\mathbf{a}  \vert  \mathbf{s}) $$   (or some variant of this)
- RL objective: $$E_{\pi_{\theta}}[r(\mathbf{s}, \mathbf{a})]$$                                      (or some variant of this)
- hybrid objective: $$E_{\pi_{\theta}}[r(\mathbf{s}, \mathbf{a})]+\lambda \sum_{(\mathbf{s}, \mathbf{a}) \in \mathcal{D}_{\text {deme }}} \log \pi_{\theta}(\mathbf{a}  \vert  \mathbf{s})$$  need to be careful in choosing this weight



##### Example: hybrid policy gradient

Rajeswaran et al., ‘17, “Learning Complex Dexterous Manipulation...”

<img src="/img/CS285.assets/image-20200329000035815.png" alt="image-20200329000035815" style="zoom:50%;" />


$$
\begin{aligned}
g_{a u g}=&\sum_{(s, a) \in \rho_{\pi}} \nabla_{\theta} \ln \pi_{\theta}(a  \vert  s) A^{\pi}(s, a)  \quad  \quad    \text{standard policy gradient}
\\
+ &\sum_{\left(s, a^{*}\right) \in \rho_{D}} \nabla_{\theta} \ln \pi_{\theta}\left(a^{*}  \vert  s\right) w\left(s, a^{*}\right) \quad  \quad    \text{increase demo likelihood}
\end{aligned}
$$




##### Example: hybrid Q-learning

Hester et al., ‘17, “Learning from Demonstrations...”
$$
J(Q) =  \mathop{\underline{J_{DQ}(Q)}}_\text{Q-learning loss} + \mathop{\underline{\lambda_1 J_{n}(Q)}}_\text{n-step Q-learning loss} + \mathop{\underline{\lambda_2 J_{E}(Q)}} +
\mathop{\underline{\lambda_3 J_{L2}(Q)}}_\text{regularization loss because why not...}  
$$

$$J_{E}(Q)=\max _{a \in A}\left[Q(s, a)+l\left(a_{E}, a\right)\right]-Q\left(s, a_{E}\right)$$  margin-based loss on example action


<img src="/img/CS285.assets/image-20200329002145663.png" alt="image-20200329002145663" style="zoom:50%;" />



##### What’s the problem?

hybrid objective: $$E_{\pi_{\theta}}[r(\mathbf{s}, \mathbf{a})]+\lambda \sum_{(\mathbf{s}, \mathbf{a}) \in \mathcal{D}_{\text {deme }}} \log \pi_{\theta}(\mathbf{a}  \vert  \mathbf{s})$$ 

- Need to tune the weight
- The design of the objective, esp. for imitation, takes a lot of care 
- Algorithm becomes problem-dependent





- Pure imitation learning
  - [x] Easy and stable supervised learning
  - **Distributional shift** 
  - No chance to get better than the demonstrations

- Pure reinforcement learning
  - [x] Unbiased reinforcement learning, can get arbitrarily good
  - Challenging exploration and optimization problem

- Initialize & finetune
  - [x] Almost the best of both worlds
  - ...but can forget demo initialization due to distributional shift

- Pure reinforcement learning, with demos as off-policy data 
  - [x] Unbiased reinforcement learning, can get arbitrarily good
  - Demonstrations don’t always help

- Hybrid objective, imitation as an “auxiliary loss”
   - [x] Like initialization & finetuning, almost the best of both worlds
   - [x] Noforgetting
   - But no longer pure RL, may be biased, may require lots of tuning









