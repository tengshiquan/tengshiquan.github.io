

# A Course in Game Theory

1994 , Martin J. Osborne

一些笔记, 待整理

公式太多, 体系比较全.



### 总体结构

<img src="C:/img/2020-04-18-Game.assets/image-20200501182914681.png" alt="image-20200501182914681" style="zoom:50%;" />



## Introduction

#### 1.1	Game Theory 博弈论

博弈论的基本假设 : 假设对手是理性的 rational



#### 1.2	Games and Solutions

##### Noncooperative and Cooperative Games   合作, 非合作博弈



##### Strategic Games and Extensive Games  策略博弈和扩展博弈

策略博弈是这样一种情形的模型:每个参与人一劳永逸地选择一次行动计划plan, 并且所有参与人的决策是同时做出的(也就是说,在选择行动计划时每个参与人并不知道其他参与人的行动计划)。与此相反，扩展博弈模型则规定了事件的可能顺序:每个参与人不仅可以在博弈开始时考虑自己的行动计划,并且每当他必须做出决定时，也可以考虑自己的行动计划。



##### Games with Perfect and Imperfect Information  完全与不完全信息博弈

完全信息就是对其他人的行动都了解;  不完全信息就是不清楚其他人的行动. 



#### 1.3	Game Theory and the Theory of Competitive Equilibrium  博弈论与竞争均衡理论

为了近一步弄清博弈论的本质,现在将它与经济学中的**竟争均衡理论**作比较。博弈论要考虑决策主体在做出决策前企图获得其他参与人的**行为信息**,而竞争理论给出的假定是:每个参与人只对某些环境参数感兴趣(例如价格),即使这些参数是被全体参与人的行为所决定的。
我们通过考虑下面一种情形来说明这两个理论的差异:在该情形中,每个参与人的某种行动(如钓鱼)的水准依赖于污染的程度,反过来污染程度又依赖于全体参与人的活动。若用竞争理论分析,我们便会去寻找一个与全体参与人行动相一致的污染程度,此时每个参与人都认为这个程度是给定的;若用博弈理论分析,我们则要求每个参与人的行动均为**最优optimal**,此时所有参与人一起造成的污染预期是给定的。



#### 1.4	Rational Behavior 理性行为

我们研究模型时, **假设**每个决策主体都是“**理性的 rational**”, 即决策主体知道他的所做的选择, 对未知的因素形成预期,具有明确的**偏好preferences**, 并在经过一定的优化过程后 特意地选择他的行动。

在不存在**不确定性uncertainty**的情况下，以下因素构成了**理性选择模型model of rational choice**：

- set $A$ of **actions** 行为集
- set $C$ of possible **consequences**  上述行为的可能**结果集合**
- **consequence function** $g: A \rightarrow C$   ;  **结果函数** ,  每个action对于一个consequence;   **reward**
- **preference relation** (a complete transitive reflexive binary relation 完全的,可传递的, 自反的 二元关系) $\succsim$ on the set $C$      ;  结果集合上的  **偏好关系**;   就是策略本身
- **utility function**   $U: C \rightarrow \mathbb{R}$ ,  有时用 **效用函数** 来表示偏好.   用奖励的函数表示策略  
  效用函数定义了一个偏好关系:   $x \succsim y$ 当且仅当 $U(x) \geq U(y)$   
  给定一个集合$B \subseteq A$ ,  在某特別情形下是可行feasible的, 一个理性的决策者选择一个可行的最优行动 $$a^*$$ (  $$a^* \in B$$  ) , 当对所有$a \in B$ 满足  $g\left(a^*\right) \succsim g(a)$ .    
  或者说, 他解决了问题 $\max _{a \in B} U(g(a))$ .   
  值的注意的是, 使用这个决策模型需要假设每个决策主体在不同的集合$B$上使用同一个偏好关系. 



下面讨论包含不确定性的决策建模. 

为了对不确定性下的决策进行建模，几乎所有的博弈论都使用了冯-诺依曼(1944)和萨维奇(1972)的理论。  
也就是说，如果结果函数是随机的，并且对决策者来说是已知的（即对每一个$a \in A$ ，结果 $g(a)$ 是 $C$上的随机（概率分布）），那么决策者就被假定为, 他的行动 为了最大化 **von Neumann-Morgenstern 效用函数**的期望 而去行动.     
如果行动与结果间的随机映射关系未给定，那么决策者的行为就被假定为他按照自己心中（主观的）概率分布去行动。在这种情况下，决策者的行为就好像他心中有一个 "**状态空间state space**" $\Omega$ ，一个$\Omega$上的概率测度，一个函数 $g: A \times \Omega \rightarrow C$，和一个效用函数$u: C \rightarrow \mathbb{R}$ ;  他被假设为按照概率测度来选择行动 $a$，使$u(g(a, \omega))$ 的期望值最大化。



#### 1.5	The Steady State and Deductive Interpretations 稳态 和 推论

对于策略博弈和扩展博弈的解 有两种相互冲突的解释。**稳态steady state**(或如Binmore(1987/88)所称的**演化evolutive**)解释与经济学中的标准解释密切相关。博弈论和其他科学一样，处理的是规律性问题。正如卡纳普（1966年，第3页）所写的那样，"我们在日常生活中进行的观察以及科学的更系统化的观察都揭示了世界上的某些重复性或规律性......科学的规律不过是尽可能精确地表达这些规律性的陈述。" 稳态解释将一个博弈视为一个模型，旨在解释在相似情况下观察到的一些规律。每个参与者都凭借从长期的经验中获得的知识, 从而 "知道 "均衡并测试其行为的最优性。相形之下，**推论deductive**（或者说，Binmore所说的**演绎式eductive**）解释，将一个游戏孤立地看成是一个 "一次性的 "事件，并试图推断出理性对结果的限制；它假设每个博弈者仅从理性原则推断出其他博弈者的行为方式。我们试图避免博弈论中经常出现的两种解释之间的混淆。



#### 1.6	Bounded Rationality 有限理性

当在现实中谈论博弈的时候，会注意个体之间的能力的不对称。这些在生活中非常关键的差异，在目前的博弈论中是缺失的。
为了说明这一事实，考虑国际象棋。棋手们可能在对合法棋步的认识和分析能力上存在差异。然而，当用博弈论对国际象棋进行建模时，假设棋手对游戏规则的知识是完美的，分析能力是理想的。我们在第2章和第6章中证明的结果（定理22.2和99.2）意味着，对于 "理性 "的棋手来说，国际象棋是一个 平庸博弈：存在一种可以用来 "解决 "这个游戏的算法。这个算法定义了一对策略，每一个棋手都有一对策略，它导致了一个 "平衡 "的结果，具有这样的属性，即一个遵循他的策略的棋手可以肯定，无论对方使用什么策略，结果至少和平衡结果一样好。这种策略的存在表明，国际象棋是没有意思的，因为它只有一种可能的结果。尽管如此，国际象棋仍然是一个非常受欢迎和有趣的游戏。它的均衡结果还有待计算；目前还无法用算法来计算。比如说，即使有一天白棋被证明有一个获胜的策略，人类也不可能执行这个策略。因此，虽然抽象的国际象棋模型可以让我们推导出一个重要的事实，但同时它也忽略了决定实际下棋结果的最重要的因素：棋手的 "能力"。
对不同的棋手在能力和对局势的认知上的不对称性进行建模，是未来研究的一个引人入胜的挑战，而 "**有限理性**"的模型已经开始解决这个问题。



#### 1.7	Terminology and Notation  术语 标记

本书使用 deductive reasoning 演绎推理. 

函数$f$ 是 **凹函数 concave** :  if $f\left(\alpha x+(1-\alpha) x^{\prime}\right) \geq \alpha f(x)+(1-\alpha) f\left(x^{\prime}\right)$ for all $x \in \mathbb{R}$ , all $x^{\prime} \in \mathbb{R},$ and all $\alpha \in[0,1]$  ;  
$\arg \max _{x \in X} f(x)$ 表示函数 $f: X \rightarrow \mathbb{R}$ 的最大值集合;   
对任何 $Y \subseteq X$ , 用 $f(Y)$ 表示集合 $\{f(x): x \in Y\}$  

$N$ :  玩家集合.   

- 将某个变量的值的集合(每个玩家都对应一个)作为一个**配置profile** : $$x = \left(x_{i}\right)_{i \in N}$$  
- 为了简单, 如果明确有$i \in N$,  配置简单记为 $\left(x_{i}\right) $ .   
- $x_{-i}$表示除玩家$i$以外的所有人的配置.   
- 给定列表 $$x_{-i} = \left(x_{j}\right)_{j \in N \backslash\{i\}}$$  和元素  $$x_{i}$$ , 可以用 $$\left(x_{-i}, x_{i}\right)$$ 表示配置 $$\left(x_{i}\right)_{i \in N}$$  
- 若对每个 $i \in N$ ,  $X_{i}$ 是一个集合, 则可以用 $X_{-i}$ 表示集合 $\times_{j \in N \backslash\{i\}} X_{j}$

对于集合$A$ 上的 **二元关系 binary relation** $\succsim$ :  

- if $a \succsim b$ or $b \succsim a$ for every $a \in A$ and $b \in A$ , 则是 **完备的complete**
- if $a \succsim a$ for every $a \in A$ , 则是 **自反的reflexive** 
- if $a \succsim c$ whenever $a \succsim b$ and $b \succsim c$ ,  则是 **传递的transitive**

偏好关系是 complete reflexive transitive 的二元关系.  

- if $a \succsim b$ , not $b \succsim a$ , 记为 $a \succ b$ 
- if $a \succsim b$  and $b \succsim a$ ,  记为 $a \sim b$

集合$A$ 上的偏好关系  $\succsim$  是**连续的continuous** :  if $a \succsim b$ whenever there are sequences $$\left(a^{k}\right)_{k}$$ and $$\left(b^{k}\right)_{k}$$  in $A$ that converge to $a$ and $b$ respectively for which $$a^{k} \succsim b^{k}$$ for all $k$  .   
A preference relation $\succsim$ on $$\mathbb{R}^{n}$$ is **quasi-concave 拟凹的** if for every $b \in \mathbb{R}^{n}$ the set $$\left\{a \in \mathbb{R}^{n}: a \succsim b\right\}$$ is convex; it is **strictly quasi-concave 严格拟凹的** if every such set is strictly convex.



$|X|$ : 集合元素的个数.       
$X$ 的**分割partition** 是$X$的 **非连通子集disjoint subsets** 的一个集合. 非连通子集的和为 $X$   
Let $N$ be a finite set and let $X \subseteq \mathbb{R}^{N}$ be a set. Then $x \in X$ is **帕累托有效 Pareto efficient** if there is no $y \in X$ for which $y_{i}>x_{i}$ for all $i \in N ; x \in X$ is **strongly Pareto efficient** if there is no $y \in X$ for which $y_{i} \geq x_{i}$ for all $i \in N$ and $y_{i}>x_{i}$ for some $i \in N$. 

一个有限(或可数)集合$X$ 上的 **概率测度probability measure** $\mu$  是一个可加函数.  associates a nonnegative real number with every subset of $X$  (that is, $\mu(B \cup C)=\mu(B)+\mu(C)$   whenever  $B$  and  $C$ are disjoint) and satisfies $\mu(X)=1 .$ 



## Strategic Games 策略博弈

在这一部分中，研究一种被称为策略博弈的策略互动模型，或者用冯-诺依曼的术语来说，是一种 "**通常形式的博弈 game in normal form**"。这个模型为每个玩家指定了一组可能的行动集合，并在可能的行动集合上的偏好顺序。



### 2	Nash Equilibrium 纳什均衡

纳什均衡是博弈论中最基本的概念之一



#### 2.1	Strategic Games 策略博弈

##### 2.1.1	Definition

策略博弈是一种相互作用的决策模型，在这个模型中，每个决策者都会仅仅选择一次自己的行动计划，而且这些选择是同时进行的。 即一次性给出策略就完事了.

我们称一个行动配置$a= (a_j)_{j\in N}$为结果(outcome),并用$A$表示结果集合$\times_{j \in N} A_{j}$ 

这里要求将每个参与人$i$的偏好定义在$A$ 而不是$A_i$上,这是将策略博弈从决策向题中区分出来的特征所在,即每个
参与人不仅要考虑自己的行动,还要考虑其他参与人采取的行动。



DEFINITION 11.1 A strategic game consists of

- a finite set $N$ (the set of players)
- for each player $i \in N$ a nonempty set $A_{i}$ (the set of actions available to player $i)$ 
- for each player $i \in N$ a preference relation $\succsim_{i}$ on $A=\times_{j \in N} A_{j}$ (the preference relation of player $i$ ).

If the set $A_{i}$ of actions of every player $i$ is finite then the game is **finite有限的**.



这个模型过于抽象, 在具体问题上必须更加具体化才能得到好的结果. 

在某些情况下，行为者的**偏好preferences** 最自然地不是根据**行动配置action profiles**而是根据其**结果consequences**来定义。例如，在建立寡头垄断的模型时，我们可以把参与者的集合看成是一组公司，把每家公司的行动集合看成是价格集合；但我们可能希望建立模型的假设是，每家公司只关心自己的利润，而不关心产生该利润的价格配置。  
为此,  引入**结果函数** :   $g: A \rightarrow C $ 以及  **结果consequences集合**$C$ 上的偏好关系配置 $$\left(\succsim_{i}^{*}\right)$$ ;   
那么策略博弈中, 每个玩家的偏好关系可以定义为:  $$a \succsim_{i} b$$ if and only if $$g(a) \succsim_{i}^{*} g(b)$$ 

有时，我们希望对一种情形建模，即行动配置的结果受到外来的一个**随机变量**的影响, 而玩家们事先并不了解随机变量是怎么实现的. 一个动作曲线的后果会受到一个外生随机变量的影响，而这个外生随机变量在玩家采取行动之前是不知道的。   
也可以通过策略博弈来建模.  引入**概率空间 probability space** $\Omega$ , 随机结果函数: $g: A \times \Omega \rightarrow C$ , 该函数的输出,   $g(a, \omega)$ 是结果consequence .   一个行动配置相当于造成了结果集合$C$ 上的一个**随机lottery**. 对每个玩家的偏好关系$$\succsim_{i}^{*}$$, 必须在所有随机的集合上具体指定.  玩家 $i$ 的偏好关系并定义为:  $$a \succsim_{i} b$$ if and only if the lottery over $C$ induced by $g(a, \cdot)$ is at least as good according to $$\succsim_{i}^{*}$$ as the lottery induced by $g(b, \cdot)$ .



在广泛的情况下，策略博弈中玩家$i$的偏好关系$\succsim_{i}$可以用一个**报酬函数(效用函数)payoff function** $u_{i}: A \rightarrow \mathbb{R}$  来表示。只要 $a \succsim_{i}$ b , 就有 $u_{i}(a) \geq u_{i}(b)$  .  通常我们**通过报酬函数来表示玩家的偏好关系**。在这种情况下，我们用  $\left\langle N,\left(A_{i}\right),\left(u_{i}\right)\right\rangle$ 而不是 $\left\langle N,\left(A_{i}\right),\left(\succsim_{i}\right)\right\rangle$ .   用策略结果来表示策略. 





<img src="C:/img/2020-04-18-Game.assets/image-20200503002729772.png" alt="image-20200503002729772" style="zoom:50%;" />

两个玩家 玩家1,玩家2的有限策略博弈可以用图13.1中的表格来方便地描述。一个玩家的行动用行来表示，另一个玩家的行动用列来表示。 表格里面的两个数分别是两个玩家的payoff 报酬.  具体见下面例子.



##### 2.1.2	Comments on Interpretation 关于如何解释

策略博弈的一种常见解释是，它是一个**事件只发生一次**的模型；每个玩家都知道博弈的细节和所有玩家都是 "理性的"，玩家同时独立地选择自己的行动。在这种解释下，每个人在选择自己的行动时，并不知道其他人做的选择；没有任何信息（除了模型的基本元素）可以作为对其他玩家行为的预期的基础。

本书采用了另一种解释，即玩家可以根据过去的博弈获取的信息形成对其他人的行为的预期。当一个game被连续玩了很多次, 每次play之间没有联系, 才可以用 策略博弈来建模.  即, 一个人玩了很多次，必须只关注他的瞬时回报，而忽略了他当前的行为对其他玩家未来行为的影响。因此，在这种解释中，只有在互动的发生之间缺乏跨时空的策略联系的情况下，才适合将一个情境建模为策略博弈。(第8章讨论的重复博弈模型涉及的是一系列的策略互动，在这些互动中确实存在着这种时间上的联系)。   即不考虑当前行为会被其他人记住学习.





#### 2.2	Nash Equilibrium

博弈论中最常用的解的概念是纳什均衡。这个概念抓住了策略博弈中的**稳态 steady state**，在这个状态下，每个博弈者都对其他博弈者的行为抱有正确的预期，并理性地采取行动。它并不试图研究达到稳定状态的过程。

DEFINITION 14.1 A Nash equilibrium of a strategic game $$\left\langle N,\left(A_{i}\right)   \left(\succsim_{i}\right)\right\rangle$$ is a **profile** $$a^{*} \in A$$ of actions with the property that for every player $i \in N$ we have

$$
\left(a_{-i}^{*}, a_{i}^{*}\right) \succsim_{i}\left(a_{-i}^{*}, a_{i}\right) \text { for all } a_{i} \in A_{i}
$$

因此，若 $$a^{*}$$是纳什均衡，必须是没有一个玩家$i$的行动产生的结果比他选择$a^∗_i$时产生的结果更好，当其他玩家$j$选择了他的均衡行动$a^∗_j$。简而言之，考虑到其他玩家的行动，没有一个玩家可以偏离纳什均衡来获利。

下面是对定义的重新表述。对于任何  $a_{-i} \in A_{-i}$ 定义 $B_{i}\left(a_{-i}\right)$ 为玩家 $i$ 在给定 $a_{-i}$ 条件下的最优行动集合:

$$
B_{i}\left(a_{-i}\right)=\left\{a_{i} \in A_{i}:\left(a_{-i}, a_{i}\right) \succsim_{i}\left(a_{-i}, a_{i}^{\prime}\right) \text { for all } a_{i}^{\prime} \in A_{i}\right\}
$$

称集合值函数 $B_{i}$ 为玩家$i$ 的 **最佳响应函数best-response function** .   
则纳什均衡作为一个行动配置满足 $$a^{*}$$ : 

$$
a_{i}^{*} \in B_{i}\left(a_{-i}^{*}\right) \text { for all } i \in N
$$

该定义的另一种表述方式指出了一种寻找纳什均衡的方法(不一定有效):  首先计算出每个玩家的最佳响应函数, 然后再寻找一个行动配置  $$a^{*}$$ 使得 $$a_{i}^{*} \in B_{i}\left(a_{-i}^{*}\right)$$ for all $i \in N$   
若函数 $B_{i}$ 是单值的singleton-valued, 则第二步就是求解有 $|N|$ 个未知 $\left(a_{i}^{*}\right)_{i \in N}$ 的 $|N|$ 个方程. 







#### 2.3	Examples

先看一些经典例子.  都是只有两个参与者, 两个动作



##### Example 15.3 Bach or Stravinsky? (BoS)

两个人想一起出去听一场巴赫或斯特拉文斯基的音乐会。他们最关心的是一起出去玩，但一个人更喜欢巴赫，另一个人更喜欢斯特拉文斯基。 payoff function 报酬函数 表示为: 

<img src="C:/img/2020-04-18-Game.assets/image-20200502173936136.png" alt="image-20200502173936136" style="zoom:50%;" />

BoS模拟的是一个玩家希望协调他们的行为，但又有利益冲突的情况。这个博弈有两个纳什均衡，分别是（巴赫，巴赫）和（斯特拉文斯基，巴赫）。(巴赫，巴赫)和(斯特拉文斯基，斯特拉文斯基)。也就是说，有两个稳定状态：一个是两个玩家总是选择巴赫，一个是两个玩家总是选择斯特拉文斯基。



##### Example 16.1  A coordination game 合作博弈

就像在BoS中，两个人希望一起出去玩，但在这种情况下，他们就更理想的演唱会达成一致。图16.2给出了一个符合这种情况的博弈。
和BoS一样，这个博弈也有两个纳什均衡。(Mozart，Mozart)和(Mahler，Mahler)。与BoS不同的是，博弈者对达到其中一个均衡状态有共同的利益，即(Mozart,Mozart)；然而，纳什均衡的概念并不排除有一个稳定状态(Mahler,Mahler)，在这个稳定状态果较差。

<img src="C:/img/2020-04-18-Game.assets/image-20200502195324831.png" alt="image-20200502195324831" style="zoom: 33%;" />

##### Example 16.2 (The Prisoner’s Dilemma)   囚徒困境

两名犯罪嫌疑人被分别关进不同的牢房。如果他们都认罪，每人将被判处三年监禁。如果他们中只有一人认罪，他将被释放，并作为证人对另一人不利，后者将被判处四年徒刑。如果两人都不认罪，都会被认定为轻罪，都会被判处一年有期徒刑。选择一个方便的payoff 报酬表示偏好，我们得到了图17.1中的博弈。
在这个博弈中，合作是有收益的--对博弈者来说，最好的结果是双方都不认罪--但每个博弈者都有成为 "自由人 "的动机。无论一个玩家做什么，另一个玩家都会选择 "坦白 "而不是 "不坦白"，所以这个博弈有一个独特的纳什均衡（坦白，坦白）。

<img src="C:/img/2020-04-18-Game.assets/image-20200502195756402.png" alt="image-20200502195756402" style="zoom: 33%;" />



##### Example 16.3 (Hawk–Dove)  老鹰鸽子

两种动物在争夺一些猎物。各自可以像鸽子一样，也可以像鹰一样。对每一种动物来说，最好的结果是它的行为像鹰，而另一种动物的行为像鸽子；最坏的结果是两种动物的行为都像鹰。这个博弈有两个纳什均衡，（鸽子，鹰）和（鹰，鸽子），分别对应着两个不同的约定。

<img src="C:/img/2020-04-18-Game.assets/image-20200502200640483.png" alt="image-20200502200640483" style="zoom: 33%;" />

##### Example 17.1 (Matching Pennies)  猜硬币

两个人各自选择正面或反面。如果选择不同，第1人付给第2人一元钱；如果选择相同，第2人付给第1人一元钱。每个人只关心自己得到的钱的多少。一个模拟这种情况的博弈如图17.3所示。这样的游戏，在这种游戏中，参与者的利益是截然相反的，这种游戏被称为 "严格竞争 strictly competitive"。**这个博弈没有纳什均衡**。

<img src="C:/img/2020-04-18-Game.assets/image-20200502201222686.png" alt="image-20200502201222686" style="zoom: 33%;" />

策略博弈的概念包含了比前五个例子中描述的情况要复杂得多的情况。以下是已被广泛研究过的三个博弈的代表：拍卖、时间博弈和位置博弈。

##### Example 18.1 (An auction)  拍卖

n个人盲拍(sealed-bid) auction, 同时出价, 标的物被给予出价最高的买家中id最低的那位.  这n个人集合$$\{1,2,\dots, n \}$$ ,  每个人对标的物的估值为$v_i$ , 且有  $v_1>v_2>\dots>v_n>0$ . 

在**第一价格first price 拍卖**中，获胜者的付款是他的出价。

习题18.2 将第一价格拍卖作为一个策略博弈，并分析其纳什均衡。特别是，表明在所有均衡状态下，玩家1获得了目标。

在**第二价格 second price**拍卖中，获胜者的付款是由没有获胜的选手提交的最高价（这样，如果只有一个选手提交最高价，那么支付的价格就是第二高价）。

练习18.3 证明在第二价格拍卖中，任何玩家i的出价vi是一个弱主导行为：玩家i出价vi时的报酬至少和出价其他值时的报酬一样高，而不考虑其他玩家的行为。证明尽管如此，仍有（"低效"）均衡状态，其中赢家不是玩家1。



##### Example 18.4 (A war of attrition) 消耗战

两个玩家在一个物体上发生了争夺，对玩家i来说，物体的价值是vi > 0。时间被建模为一个连续的变量，从0开始到无穷。每个玩家选择何时让步给另一个玩家；如果第一个让步的玩家在时间t的时候让步，那么另一个玩家在那个时候获得该物体。如果两个玩家同时让步，则物品被平分，玩家i得到的回报为vi/2。时间是有价值的：在第一个让步之前，每个玩家每损失一个单位时间的回报。

练习18.5 把这种情况表述为一个策略博弈，并表明在所有纳什均衡状态下，其中一方立即认输。



##### Example 18.6 (A location game) 位置游戏

每一个人都会选择是否成为政治候选人，如果是的话，则选择哪个位置。
有一个公民的连续体，每个人都有一个最喜欢的位置；位置的分布由[0，1]上的密度函数f给出，对于所有x∈[0，1]，f（x）>0。一个候选人如果位置比任何其他候选人的位置更接近一个公民, 则可以赢取该公民的选票；如果有k个候选人选择了相同的位置，那么每个人都能得到该位置选票的1/k。在竞争中，得票最多的候选人为获胜者。每个人宁愿成为唯一的胜出者，也不愿意与第一名并列，宁愿与第一名并列，也不愿意出局，宁愿不参加比赛，也不愿意输掉比赛。

习题19.1 将此情境设为策略博弈，找到当n=2时的纳什均衡集，并证明当n=3时不存在纳什均衡。



#### 2.4	Existence of a Nash Equilibrium 纳什均衡的存在性

不是每个strategic game 都有 Nash equilibrium. 



Proposition 20.3 The strategic game $$\left\langle N,\left(A_{i}\right),\left(\succsim_{i}\right)\right\rangle$$ has a Nash equilibrium if for all $i \in N$ 

- the set $$A_{i}$$ of actions of player i is a nonempty compact convex subset of a Euclidian space

and the preference relation $$\succsim_{i}$$ is

- continuous
- quasi-concave on $$A_{i}$$



注意，这个结果保证一个策略博弈在满足一定条件下至少有一个纳什均衡；正如我们所看到的，一个博弈可以有一个以上的均衡。(我们没有讨论一个博弈有唯一的纳什均衡的条件)。还注意到定理20.3并不适用于任何一个博弈中的玩家有无限多行动的博弈，因为这样的博弈违反了每个玩家的行动集是凸的条件。



#### 2.5	Strictly Competitive Games 严格竞争博弈, 零和博弈



### 3	Mixed, Correlated, and Evolutionary Equilibrium 混合,相关与演进均衡

#### 3.1 Mixed Strategy Nash Equilibrium  混合策略纳什均衡

##### 3.1.1 Definitions

混合策略纳什均衡的概念是为了模拟一个稳定状态的博弈，在这个博弈中，参与者的选择不是确定性的，而是受概率规则调节。 

之前将策略博弈定义为三元组$\langle N,  \left(A_{i}\right),\left(\succsim_{i}\right) \rangle$ ,  每个人i的偏好关系 $\succsim_{i}$被定义在行动组集合$A=\times_{i \in N} A_{i}$上. 本章允许玩家的选择是非确定性的, 需要增加在不确定性上的偏好.  遵从现代博弈论的习惯, 假定偏好关系满足assumptions of von Neumann and Morgenstern, 所以偏好可以表示为$u_{i}: A \rightarrow \mathbb{R}$ 函数的期望值.   本章关于策略相互作用的模型是 $\left\langle N,\left(A_{i}\right),\left(u_{i}\right)\right\rangle$ .

 令 $G=\left\langle N,\left(A_{i}\right),\left(u_{i}\right)\right\rangle$ 是一个 **strategic game**. 我们用 $\Delta\left(A_{i}\right)$ 表示 $A_{i}$ 上的概率分布集合,  $\Delta\left(A_{i}\right)$ 的一个元素为 玩家$i$的一个 **混合策略mixed strategy** ; 假定每个玩家的混合策略是独立随机化的independent randomizations. 为明确起见, 我们称 $A_{i}$ 的一个元素为**纯策略pure strategy**. 对有限集$X$ and $\delta \in \Delta(X)$ , 用$\delta(x)$ 表示 $\delta$ 赋予 $x \in X$ 的概率, 将 $\delta$ 的**支撑集support** 定义为 $x \in X$ 的集合, 其中 $\delta(x)>0 .$ 混合策略的一个配置 $$\left(\alpha_{j}\right)_{j \in N}$$ 产生了 $A$ 上的一个概率分布; 例如, 如果每个 $A_{j}$ 是有限集, 则在独立随机化条件下, 行动配置 $a=\left(a_{j}\right)_{j \in N}$ 的概率是 $$\Pi_{j \in N} \alpha_{j}\left(a_{j}\right)$$,  所以玩家 $i$对 $$\left(\alpha_{j}\right)_{j \in N}$$的估值为$$\sum_{a \in A}\left(\Pi_{j \in N} \alpha_{j}\left(a_{j}\right)\right) u_{i}(a)$$ . 



**DEFINITION 32.1** The **mixed extension** of the strategic game $$\langle N,\left(A_{i}\right) 
\left(u_{i}\right) \rangle$$ is the strategic game $\left\langle N,\left(\Delta\left(A_{i}\right)\right),\left(U_{i}\right)\right\rangle$ in which $\Delta\left(A_{i}\right)$ is the set of probability distributions over $A_{i},$ and $U_{i}: \times_{j \in N} \Delta\left(A_{j}\right) \rightarrow \mathbb{R}$ assigns to each $\alpha \in \times_{j \in N} \Delta\left(A_{j}\right)$ the expected value under $u_{i}$ of the lottery over $A$ that is induced by $\alpha$ (so that  $U_{i}(\alpha)=\sum_{a \in A}\left(\Pi_{j \in N} \alpha_{j}\left(a_{j}\right)\right) u_{i}(a)$ if $A$ is finite).

Note that each function $U_{i}$ is multilinear. That is, for any mixed strategy profile $\alpha,$ any mixed strategies $\beta_{i}$ and $\gamma_{i}$ of player $i,$ and any number $\lambda \in[0,1],$ we have $U_{i}\left(\alpha_{-i}, \lambda \beta_{i}+(1-\lambda) \gamma_{i}\right)=\lambda U_{i}\left(\alpha_{-i}, \beta_{i}\right)+ (1-\lambda) U_{i}\left(\alpha_{-i}, \gamma_{i}\right) $. Note also that when each $A_{i}$ is finite we have 
$$
U_{i}(\alpha)=\sum_{a_{i} \in A_{i}} \alpha_{i}\left(a_{i}\right) U_{i}\left(\alpha_{-i}, e\left(a_{i}\right)\right)
$$
for any mixed strategy profile $\alpha$, where $e\left(a_{i}\right)$ is the degenerate mixed strategy of player $i$ that attaches probability one to $a_{i} \in A_{i}$. 

**DEFINITION 32.3**  A **mixed strategy Nash equilibrium of a strategic game** is a Nash equilibrium of its mixed extension.

Suppose that $$\alpha^{*} \in \times_{j \in N} \Delta\left(A_{j}\right)$$ is a mixed strategy Nash equilibrium of $$G=\left\langle N,\left(A_{i}\right),\left(u_{i}\right)\right\rangle$$ in which each player $$i$$ 's mixed strategy $$\alpha_{i}^{*}$$ is degenerate in the sense that it assigns probability one to a single member - say $$a_{i}^{*}-$$ of $$A_{i} .$$ Then, since $$A_{i}$$ can be identified with a subset of $$\Delta\left(A_{i}\right),$$ the action profile $$a^{*}$$ is a Nash equilibrium of $$G .$$ Conversely, suppose that $$a^{*}$$ is a Nash equilibrium of $$G .$$ Then by the linearity of $$U_{i}$$ in $$\alpha_{i}$$ no probability distribution over actions in $$A_{i}$$ yields player $$i$$ a payoff higher than that generated by $$e\left(a_{i}^{*}\right),$$ and thus the profile $$\left(e\left(a_{i}^{*}\right)\right)$$ is a mixed strategy Nash equilibrium of $$G$$ .

我们刚刚论证了一个策略博弈的纳什均衡集是其混合策略纳什均衡集的子集。在第二章中我们看到，有些博弈的纳什均衡集是空的。也有混合策略纳什均衡集合为空的博弈。然而，每个博弈中的每个玩家都有**有限多行动**的博弈至少有一个混合策略纳什均衡。

**Proposition 33.1** Every finite strategic game has a mixed strategy Nash equilibrium.

**LEMMA 33.2** Let $G=\left\langle N,\left(A_{i}\right),\left(u_{i}\right)\right\rangle$ be a finite strategic game. Then $\alpha^{*} \in \times_{i \in N} \Delta\left(A_{i}\right)$ is a mixed strategy Nash equilibrium of $G$ if and only if for every player $i \in N$ every pure strategy in the support of $\alpha_{i}^{*}$ is $a$ best response to $\alpha_{-i}^{*}$

由此可见，支持任何玩家的均衡混合策略的每一个行动都会给该玩家带来相同的报酬。



##### 3.1.2 	Examples

**EXAMPLE 34.1(BoS)**   与第二章的例子一样, 不过这里从混合策略均衡的角度, 解释为von Neumann-Morgenstern(VNM) utilities. 





#### 3.2 Interpretations of Mixed Strategy Nash Equilibrium   混合策略纳什均衡的解读

一些大佬的各自解读。

##### 3.2.1 Mixed Strategies as Objects of Choice

> 显然，很多人是无意识的引入随机

天真地看，混合策略意味着**玩家故意**决定将随机性引入他的行为中：选择混合策略的玩家将自己承诺给一个随机装置，该装置概率地选择他的行动集的成员。在所有玩家都如此承诺后，装置被启动，一个行动profile被实现。因此每个玩家$i$选择$\Delta\left(A_{i}\right)$的一个成员，就像他在第2章讨论的策略博弈中选择$A_{i}$的一个成员一样。当然，也有一些情况下，玩家会在他们的行为中引入随机性。例如，玩家在扑克牌中随机地 "虚张声势"，政府随机地审核纳税人，一些商店随机地提供折扣。

AR： 然而，策略博弈中的混合策略均衡的概念并没有抓住玩家在行为中引入随机性的**动机**。通常一个玩家故意随机化以影响其他玩家的行为。例如，考虑儿童版的 "匹配硬币"（例17.1），玩家选择显示奇数或偶数的手指。这个游戏被经典地用来说明混合策略均衡的动机，但随机化是对玩家在游戏中的刻意策略的一种奇怪描述。一个玩家的行动是他对其他玩家选择的猜测的反应，猜测是一种心理操作，是非常刻意的，而不是随机的。另外，考虑另一个经常被举出来说明混合策略均衡的例子，即税务机关和纳税人之间的关系。税务机关的目的是阻止纳税人逃税；对成本的考虑导致他们只进行随机审计。他们希望纳税人知道他们的策略，并且在对纳税人进行审计的策略和不进行审计的策略之间不会无动于衷，这也是混合策略均衡中的要求。这种情况应该被模拟成一种博弈，在这种博弈中，当局首先选择审计的概率，然后，纳税人在被告知这一概率后，采取一种行动。在这样的模型中，可能的随机化集合就是纯策略的集合。

 MJO： 将一个玩家的均衡混合策略解释为一种刻意的选择，主要问题在于，在混合策略均衡中，每个玩家在所有混合策略之间是无所谓的，这些策略的支持是她的均衡策略的子集：她的均衡策略只是许多策略中的一种，在其他玩家的均衡行为下，这些策略会给她带来相同的预期报酬。然而，这个问题并不限于混合策略均衡。例如，它困扰着许多连续游戏（包括所有重复游戏）中的均衡，在这些游戏中，一个玩家在她的均衡策略和许多非均衡策略之间是无所谓的。此外，在某些游戏中，可能还有其他原因选择均衡混合策略。例如，在严格竞争的博弈中，我们已经看到，均衡混合策略可能严格地使玩家能够保证的报酬最大化。(例如，在Matching Pennies中就是如此。)最后，Harsanyi(1973)的巧妙论证(下面在3.2.4节中考虑)为均衡混合策略的这一特征提供了一些缓解。

MJO 似乎很有可能，Matching Pennies的混合策略均衡提供了一个很好的描述，即对**随机选择的对手**反复进行游戏的玩家的稳态行为。在这样的情况下，玩家无法猜测对手在任何一次特定的交锋中的行动，她采取能够保证报酬最大化的策略是合理的。如果两个玩家反复互动，那么猜测的心理学可能会对他们的行为提供启示，不过即使在这种情况下，博弈的混合策略均衡也可以对他们的行为提供很好的描述。税务稽查的情况同样可以被模拟成一个策略博弈，其中玩家的选择是同时进行的。在这种博弈中，当局选择的均衡审计概率与当局先动的博弈中的概率是一样的；考虑到纳税人的行为，当局在审计与不审计之间是无所谓的。



##### 3.2.2 Mixed Strategy Nash Equilibrium as a Steady State

在第2章中，我们将纳什均衡解释为一个环境的**稳定状态**，在这个环境中，玩家重复行动，忽略了游戏之间可能存在的任何战略联系。我们可以将混合策略纳什均衡类似地解释为**随机稳态**。玩家拥有关于过去采取行动的频率的信息（在这个游戏中，扮演玩家1的玩家在80%的时间里采取了行动a1，20%的时间里采取了行动b1"）；每个玩家利用这些频率形成他对其他玩家未来行为的信念，从而制定他的行动。在均衡状态下，这些频率随着时间的推移保持不变，并且是稳定的，即在**稳态信念**的前提下，玩家采取的任何具有正概率的行动都是最优的。

混合策略均衡预测了博弈的结果是随机的，因此对于博弈的单次博弈，它的预测不如纯策略均衡的预测精确。但正如我们在1.5节中所论述的，理论的作用是解释规律性；混合策略均衡的概念抓住了随机规律性。

这种解释的一个变体是基于对$n$玩家博弈的解释，即把$n$个大群体的互动模型。每次博弈的发生都是在$n$玩家被随机抽取后发生的，每个群体中抽取一个。玩家i的平衡混合策略中的概率被解释为$A_{i}$成员在第$i$个种群中使用的稳态频率。在这种解释中，博弈是一个模型的简化形式，其中的种群是明确描述的。

稳态解释的一个假设是，没有玩家发现其他玩家的行为之间或其他玩家的行为和他自己的行为之间有任何**相关性**。去掉这个假设就会产生**相关均衡**的概念，我们在第3.3节讨论。	



##### 3.2.3 Mixed Strategies as Pure Strategies in an Extended Game

在选择他的行动之前，玩家可能会收到**随机的私人信息**，从其他玩家的角度来看是无关紧要的，他的行动可能取决于此。玩家可能不会有意识地选择他的行动和他的私人信息的实现之间的联系；它可能只是碰巧这两者之间有一个相关性，导致他的行动从另一个玩家或外部观察者的角度来看是 "随机 "的。在将玩家的行为建模为随机时，混合策略纳什均衡捕捉到了行为对玩家认为不相关因素的依赖性。另外，玩家可能意识到外部因素决定了他的对手的行为，但可能发现不可能或非常昂贵地确定这种关系。出于同样的原因，我们将抛硬币的结果建模为随机的，而不是将其描述为其起始位置和速度、风速和其他因素相互作用的结果）。概括地说，从这个角度看，混合策略纳什均衡是对系统稳定状态的描述，它反映了博弈原始描述中缺少的元素。

更具体地说，考虑博弈（例34.1 ）。正如我们所看到的，这个游戏有一个混合策略纳什均衡 $\left(\left(\frac{2}{3},\frac{1}{3}\right),\left(\frac{1}{3},\frac{2}{3}\right)\right)$ 现在假设每个玩家有三种可能的 "情绪"，由他不理解的因素决定。每个玩家有三分之一的时间处于这三种情绪中，与其他玩家的情绪无关；他的情绪对他的收益没有影响。假设玩家1在情绪1或2时选择巴赫，在情绪3时选择斯特拉文斯基，玩家2在情绪1时选择巴赫，在情绪2或3时选择斯特拉文斯基。将这种情况看作是一个贝叶斯博弈，其中每个玩家的三种类型对应于他可能的情绪，这种行为定义了一个纯策略均衡，正好对应于原始博弈的混合策略纳什均衡。需要注意的是，这种对混合策略均衡的解释并不依赖于每个玩家拥有三种同样可能且独立的情绪，我们只需要玩家的私人信息足够丰富，以至于他们能够创造出相应的随机变量。尽管如此，这种信息结构存在的要求限制了解释。

AR 对这种解释有三个批评。首先，很难接受一个玩家的刻意行为取决于对其报酬没有影响的因素。人们通常会为他们的选择给出理由；在任何特定情况下，一个希望应用混合策略均衡概念的建模者应该指出那些与报酬无关的理由，并解释玩家的私人信息和他的选择之间所需的依赖性。

MJO 在混合策略均衡中，每个游戏者在支持她的均衡策略的所有行动之间都是无所谓的，因此，选择的行动取决于被建模者视为 "不相关 "的因素并不是不可信的。当被问及为什么从成员具有同等吸引力的集合中选择某个行动时，人们经常会给出 "我不知道--我只是觉得喜欢 "这样的答案。

AR 其次，这种解释下的均衡所预测的行为是非常脆弱的。如果管理者的行为是由他所吃的早餐类型决定的，那么模型之外的因素，如他的饮食习惯或鸡蛋价格的变化，可能会改变他选择行为的频率，从而引起其他参与者信念的变化，造成不稳定。

MJO 对于随机事件的每一种结构，都有一种行为模式，导致相同的均衡。例如，如果在鸡蛋涨价前，有一个均衡，即经理在吃鸡蛋当早餐的日子里，在早上7：30之前起床时提供折扣，那么在涨价后，可能会出现一个均衡，即她在吃鸡蛋时，在早上8点之前起床时提供折扣，在价格变化后，她的旧的行为模式不再是对其他玩家策略的最佳反应；系统是否会稳定地调整到新的均衡，取决于调整的过程。混合策略纳什均衡是脆弱的，因为玩家没有正向激励来坚持他们的均衡行为模式（因为**均衡策略不是唯一最优的**）；除此之外，这种解释下的均衡并不比任何其他解释下的均衡更脆弱。(而且，这又是下一节讨论的Harsanyi模型所要解决的问题)。

AR 第三，为了以这种方式解释一个特定问题的均衡，我们需要指出玩家行为所依据的 "现实生活 "外生变量。例如，为了解释价格竞争模型中的混合策略纳什均衡，人们既要指明作为企业定价政策基础的未建模因素，又要证明信息结构足够丰富，足以跨越所有混合策略纳什均衡的集合。应用混合策略均衡概念的人很少这样做。

MJO 一个世界上的玩家可以获得大量的随机变量，她的**行动可能取决于这些随机变量**：她早上起床的时间，她的 "心情"，她的报纸被送来的时间，....。这些随机变量的结构是如此丰富，以至于在理论的每一次应用中都没有必要把它们说出来。把混合策略解释为大博弈中的纯策略，就很好地抓住了玩家所选择的行动可能取决于模型之外的因素这一观点。



##### 3.2.4 Mixed Strategies as Pure Strategies in a Perturbed Game

现在我们提出一个由于Harsanyi（1973）的混合策略均衡的理由。一个博弈被看作是一种经常发生的情况，在这种情况下，玩家的偏好受到小的随机变化的影响。因此，如同上一节的论证一样，引入了随机因素，但**这里的随机因素是与报酬相关的**）。在每次发生的情况下，每个玩家都知道自己的偏好，但不知道其他玩家的偏好。混合策略均衡是对玩家在一段时间内选择行动的频率的总结。

。。。

因此，Harsanyi关于混合策略均衡的理由是，即使没有任何玩家努力以所需的概率使用他的纯策略，报酬函数的随机变化也会诱使每个玩家以正确的频率选择他的纯策略。其他玩家的均衡行为是这样的，一个为他的报酬函数的每一个实现选择唯一最优的纯策略的玩家以他的均衡混合策略所要求的频率选择他的行动。

MJO Harsanyi的结果是对以下说法的一个优雅的回应，即一个玩家没有理由选择她的均衡混合策略，因为她在所有具有相同支持的策略之间是无所谓的。我在上文中指出，对于一些游戏，包括严格的竞争性游戏，这种批评是微弱的，因为玩家还有其他理由选择他们的均衡混合策略。Harsanyi的结果表明，在几乎任何博弈中，批评的力量是有限的，因为几乎任何混合策略纳什均衡都接近于博弈的任何扰动的严格的纯策略均衡，其中玩家的报酬受到小的随机变化。



##### 3.2.5 Mixed Strategies as Beliefs

在另一种解释下，我们在第5.4节中详细说明，混合策略纳什均衡是一个信念的配置$\beta$，其中$\beta_{i}$是所有其他玩家对玩家i的行动的共同信念，其属性是对于每个玩家$i$来说，在$\beta_{i}$支持下的每个行动都是最优的，给定$\beta_{-i} .$在这种解释下，每个玩家选择的是单一行动而不是混合策略。一个均衡是玩家信念的稳定状态，而不是他们的行动。这些信念需要满足两个属性：它们是所有玩家共同的，并且与每个玩家都是期望效用最大化的假设相一致。

。。。

但请注意，当我们以这种方式解释混合策略均衡时，均衡的预测内容很小：它只预测每个玩家使用的行动是均衡信念的best response 。这种best response 的集合包括support 玩家的均衡混合策略的任何行动，甚至可以包括该策略support 之外的行动。





### 3.3 Correlated Equilibrium

在第3.2.3节中，我们讨论了对混合策略纳什均衡的解释，即每个玩家的行动取决于他从 "自然 "接收的信号的稳定状态。在这个解释中，信号是私有的和独立的。

如果**信号**不是私有和独立的，会发生什么呢？例如，假设在BoS中（见图35.1），两个游戏者**都观察到一个随机变量**，这个随机变量以概率$\frac{1}{2}$取$x$和$y$中的每一个，那么就会出现一个新的均衡，在这个均衡中，如果实现的是$x$，则两个游戏者都选择巴赫，如果实现的是$y$，则选择斯特拉文斯基。给定每个玩家的信息，他的行动是最优的：如果实现是$x$，那么他知道另一个玩家选择巴赫，所以他选择巴赫是最优的，如果实现是$y$，则对称地选择巴赫。

在这个例子中，玩家观察的是**同一个**随机变量。更一般地，他们的信息可能是**不完全相关**的。例如，假设有一个随机变量，它有三个值$x，y，$和$z$，玩家1只知道它的实现是$x$或者是$\{y，z\}$的成员，而玩家2只知道它是$\{x，y\}$的成员或者是$z$。也就是说，玩家1的信息分区是$\{\{x\}, \{y,z\}\}$，玩家$2$ 是 $\{\{x, y\},\{z\}\} $。在这些假设下，玩家1的策略由两个行动组成：一个是当她知道实现是$x$时使用的行动，一个是当她知道实现是${{y，z}}$的成员时使用的行动。同样，玩家2的策略由两个行动组成，一个是$\{x, y\}$，一个是$z$。如果给定另一个玩家的策略，一个玩家的策略是最优的：对于他的信息的任何实现，他都不能通过选择一个不同于他的策略的行动来做得更好。

为了说明玩家如何利用他的信息来选择最优行动，假设$y$和$z$的概率是$\eta$和$\zeta$，玩家2的策略是，如果他知道实现是$\{x, y\}$，则采取$a_{2}$的行动，如果他知道实现是$z$，则采取$b_{2}$的行动。那么如果玩家1被告知$y$或$z$已经发生，他选择的行动是最优的，因为玩家2选择$a_{2}$的概率是$\eta /(\eta+\zeta)$,   ( $y$的概率条件是$\{y, z\}$），而$b_{2}$的概率是$\zeta /(\eta+\zeta)$。这些例子使我们得出了以下的平衡概念。

​	

一个策略的相关均衡：

DEFINITION 45.1 A correlated equilibrium of a strategic game $\left\langle N,\left(A_{i}\right),\left(u_{i}\right)\right\rangle$ consists of

- a finite probability space $(\Omega, \pi)(\Omega$ is a set of states and $\pi$ is a probability measure on $\Omega$ )
  for each player $i \in N$ a partition $\mathcal{P}_{i}$ of $\Omega$ (player $i$ 's information partition)
- for each player $i \in N$ a function $\sigma_{i}: \Omega \rightarrow A_{i}$ with $\sigma_{i}(\omega)=\sigma_{i}\left(\omega^{\prime}\right)$
  whenever $\omega \in P_{i}$ and $\omega^{\prime} \in P_{i}$ for some $P_{i} \in \mathcal{P}_{i}\left(\sigma_{i}\right.$ is player i's strategy)
  such that for every $i \in N$ and every function $\tau_{i}: \Omega \rightarrow A_{i}$ for which $\tau_{i}(\omega)=\tau_{i}\left(\omega^{\prime}\right)$ whenever $\omega \in P_{i}$ and $\omega^{\prime} \in P_{i}$ for some $P_{i} \in \mathcal{P}_{i}$ (i.e. for
  every strategy of player $i$ ) we have

$$
\sum_{\omega \in \Omega} \pi(\omega) u_{i}\left(\sigma_{-i}(\omega), \sigma_{i}(\omega)\right) \geq \sum_{\omega \in \Omega} \pi(\omega) u_{i}\left(\sigma_{-i}(\omega), \tau_{i}(\omega)\right)
$$

Note that the probability space and information partition are not exogenous but are part of the equilibrium. Note also that (45.2) is equivalent to the requirement that for every state $\omega$ that occurs with positive probability the action $\sigma_{i}(\omega)$ is optimal given the other players' strategies and player $i$ 's knowledge about $\omega$. (This equivalence depends on the assumption that the players' preferences obey expected utility theory.



我们首先表明，相关均衡集包含混合策略纳什均衡集。

PROPOSITION 45.3 For every mixed strategy Nash equilibrium $\alpha$ of a finite strategic game $\left\langle N,\left(A_{i}\right),\left(u_{i}\right)\right\rangle$ there is a correlated equilibrium $\left\langle(\Omega, \pi),\left(\mathcal{P}_{i}\right),\left(\sigma_{i}\right)\right\rangle$ in which for each player $i \in N$ the distribution on $A_{i}$ induced by $\sigma_{i}$ is $\alpha_{i} .$ 



。。。

我们可以将本证明中构建的相关均衡解释为：首先由公共随机装置决定K个相关均衡中的哪一个，然后实现第k个相关均衡对应的随机变量。









#### 3.4 Evolutionary Equilibrium

在这一节中，我们将描述纳什均衡概念的一个变体--进化均衡--背后的基本思想。这个概念是为了模拟参与者的行动由进化的力量决定的情况。我们将讨论限制在一个简单的情况下，其中一个生物种群的成员（动物，人类，植物，......）彼此成对地交互。在每场比赛中，每个生物体都从一个集合$B$ 中选择一个动作，生物体并不自觉地选择动作，而是从它们的祖先那里继承了行为模式，或者通过突变分配给它们。我们假设有一个函数$u$来衡量每个生物体的生存能力：如果一个生物体在面对其潜在对手的行动分布$\beta$时采取了行动$a$，那么它的生存能力就由$u(a, b)$在$\beta $下的期望值来衡量。 这个描述对应于一个两人对称的策略游戏 $\left\langle\{1,2\},(B, B),\left(u_{i}\right)\right\rangle$，其中 $u_{1}(a, b)=u(a, b)$和$u_{2}(a, b)=u(b, a)$.
进化平衡的候选者是$B$中的一个行动。平衡的概念是为了捕捉一个稳定的状态，在这个状态下，所有的生物都会采取这种行动，而且没有突变体可以入侵种群。更准确地说，这个概念是指对于每一个可能的行动$b \in B$中，进化过程偶尔会将种群中的一小部分转化为遵循$b$的突变体。在一个平衡中，任何这样的突变体都必须获得比平衡行动更低的预期报酬，因此它将被淘汰。现在，如果种群中$\epsilon>0$ 的部分由采取$b$行动的突变体组成，而所有其他生物都采取$b^{*}$行动，那么突变体的平均报酬为$(1-\epsilon) u\left(b, b^{*}\right)+\epsilon u(b, b)$（因为在概率为 $1-\epsilon$的情况下，它遇到了一个非突变体，而在概率为$\epsilon$的情况下，它遇到了另一个突变体），而一个非突变体的平均报酬是$(1-\epsilon) u\left(b^{*}, b^{*}\right)+\epsilon u\left(b^{*}, b\right)$ 。 因此，要使$b^{*}$成为一个演化均衡，我们要求
$$
(1-\epsilon) u\left(b, b^{*}\right)+\epsilon u(b, b)<(1-\epsilon) u\left(b^{*}, b^{*}\right)+\epsilon u\left(b^{*}, b\right)
$$
for all values of $\epsilon$ sufficiently small. This inequality is satisfied if and only if for every $b \neq b^{*}$ either $u\left(b, b^{*}\right)<u\left(b^{*}, b^{*}\right),$ or $u\left(b, b^{*}\right)=u\left(b^{*}, b^{*}\right)$ and $u(b, b)<u\left(b^{*}, b\right),$ so that we can define an evolutionary equilibrium as follows.

DEFINITION 49.1 Let $G=\left\langle\{1,2\},(B, B),\left(u_{i}\right)\right\rangle$ be a symmetric strategic game, where $u_{1}(a, b)=u_{2}(b, a)=u(a, b)$ for some function $u$. An evolutionarily stable strategy (ESS) of $G$ is an action $b^{*} \in B$ for which $\left(b^{*}, b^{*}\right)$ is a Nash equilibrium of $G$ and $u(b, b)<u\left(b^{*}, b\right)$ for every best response $b \in B$ to $b^{*}$ with $b \neq b^{*}$.





## Extensive Games with Perfect Information 完全信息扩展博弈

**扩展extensive**博弈明确描述了 玩家在战略情况下遇到的决策问题的顺序结构。该模型使我们能够研究每个玩家不仅在博弈开始时可以考虑他的行动计划，而且在他必须做出决策的任何时间点也可以考虑他的行动计划。



### 6	Extensive Games with Perfect Information 完全信息扩展博弈

#### 6.1	Extensive Games with Perfect Information

##### 6.1.1	Definition

**扩展博弈**是对战略情境中玩家所遇到的决策问题的**顺序结构**的详细描述。在这样的博弈中，如果每个博弈者在做出任何决策时，**都能完美地了解到之前发生的所有事件**，那么在这样的博弈中就有完美的信息。为了简单起见，我们最初将注意力限制在没有两个玩家同时做出决策的博弈中，并且所有相关的行动都是由玩家做出的（没有随机性介入）。我们在第6.3节中取消了这两个限制）。

**Definition 89.1** An **extensive game with perfect information 完全信息扩展博弈** has the following components.

- A finite set $$N($$ the set of **players**)   **参与者集合**
- A set $$H$$ of **sequences** (finite or infinite) that satisfies the following three properties.  **动作序列集合**
  - The empty sequence $$\varnothing$$ is a member of $$H$$  **空**
  - If $$\left(a^{k}\right)_{k=1, \ldots, K} \in H$$ (where $$K$$ may be infinite) and $$L<K$$ then $$\left(a^{k}\right)_{k=1, \ldots, L} \in H$$  **前序**
  - If an infinite sequence $$\left(a^{k}\right)_{k=1}^{\infty}$$ satisfies $$\left(a^{k}\right)_{k=1, \ldots, L} \in H$$ for every positive integer $$L$$ then $$\left(a^{k}\right)_{k=1}^{\infty} \in H$$     **无限长度**   
    (Each member of $$H$$ is a **history 动作历史**; each component of a history is an **action** taken by a player.)  A history $$\left(a^{k}\right)_{k=1, \ldots, K} \in H$$ is **terminal** if it is infinite or if there is no $$a^{K+1}$$ such that  $$\left(a^{k}\right)_{k=1, \ldots, K+1} \in H$$ .  The **set of terminal histories** is denoted $$Z$$ .
- A function $$P$$ that assigns to each nonterminal history (each member of $$H \backslash Z$$ ) a member of $$N$$ .($$P$$  is the **player function**, $$P(h)$$ being the player who takes an action after the history $$h$$.)  **玩家函数**
- For each player $$i \in N$$ a preference relation $$\succsim_{i}$$ on $$Z$$ (the **preference relation** of player $$i$$ )  **偏好关系**

当不要确定玩家偏好的时候, 用三元组表示. Sometimes it is convenient to specify the structure of an extensive game without specifying the players' preferences. We refer to a triple 三元组 $$\langle N, H, P\rangle$$ whose components satisfy the first three conditions in the definition as an **extensive game form with perfect information**.

若历史有限, 则成为博弈**有限**. If the set $$H$$ of possible histories is finite then the game is **finite**. If the longest history is finite then the game has a **finite horizon**. Let $$h$$ be a history of length $$k ;$$ we denote by $$(h, a)$$ the history of length $$k+1$$ consisting of $$h$$ followed by $$a$$

After any nonterminal history $$h$$ player $$P(h)$$ chooses an action from the set   **可用动作集合**

$$
A(h)=\{a:(h, a) \in H\}
$$

初始状态.  The empty history is the **starting point** of the game; we sometimes refer to it as the **initial history**. At this point player $$P(\varnothing)$$ chooses a member of $$A(\varnothing) .$$ For each possible choice $$a^{0}$$ from this set player $$P\left(a^{0}\right)$$ subsequently chooses a member of the set $$A\left(a^{0}\right) ;$$ this choice determines the next player to move, and so on.  







## EXTENSIVE GAMES WITH IMPERFECT INFORMATION

### 11.1	Extensive Games with Imperfect Information



#### 11.1.2	Definitions

下面的定义是对具有完美信息的扩展博弈（89.1）的推广，允许玩家在采取行动时**对过去的事件并不完全地了解**。它还允许外部的不确定性：有些行为可能是由 "**偶然性 chance** "决定的（见6.3.1节）。它并不包含我们在第6.3节中讨论过的另一个扩展博弈的定义，即在这个定义中，不止一个人可以在任何历史事件之后行动（见例202.1之后的讨论）。

**DEFINITION 200.1**  An **extensive game** has the following components.

- A finite set $$N($$ the set of **players**)   参与者集合
- A set $$H$$ of **sequences** (finite or infinite) that satisfies the following three properties.  动作序列集合
  - The empty sequence $$\varnothing$$ is a member of $$H$$
  - If $$\left(a^{k}\right)_{k=1, \ldots, K} \in H$$ (where $$K$$ may be infinite) and $$L<K$$ then $$\left(a^{k}\right)_{k=1, \ldots, L} \in H$$
  - If an infinite sequence $$\left(a^{k}\right)_{k=1}^{\infty}$$ satisfies $$\left(a^{k}\right)_{k=1, \ldots, L} \in H$$ for every positive integer $$L$$ then $$\left(a^{k}\right)_{k=1}^{\infty} \in H$$    
    (Each member of $$H$$ is a **history**; each component of a history is an **action** taken by a player.) A history $$\left(a^{k}\right)_{k=1, \ldots, K} \in H$$ is terminal if it is infinite or if there is no $$a^{K+1}$$ such that  $$\left(a^{k}\right)_{k=1, \ldots, K+1} \in H$$ .  The set of actions available after the nonterminal history $$h$$ is denoted $$A(h)=\{a:(h, a) \in H\}$$ and the set of terminal histories is denoted $$Z$$ . 
- A function $$P$$ that assigns to each nonterminal history (each member of $$H \backslash Z)$$ a member of $$N \cup\{c\} $$.($$P$$   is the **player function**,  $$P(h)$$ being the player who takes an action after the history $$h .$$ If $$P(h)=c$$ then **chance** determines the action taken after the history $$h .)$$
- 可以理解为强化学习里面$\pi$ 策略函数:  A function $$f_{c}$$ that associates with every history $$h$$ for which $$P(h)=c$$ a **probability measure** $$f_{c}(\cdot \vert h)$$ on $$A(h),$$ where each such probability measure is independent of every other such measure. $$(f_{c}(a \vert h)$$ is the probability that $a$ occurs after the history  $h$ )   
- For each player $$i \in N$$ a partition $$\mathcal{I}_{i}$$ of $$\{h \in H: P(h)=i\}$$ with the property that $$A(h)=A\left(h^{\prime}\right)$$ whenever $$h$$ and $$h^{\prime}$$ are in the same member of the partition 即对玩家来说不可分辨. For $$I_{i} \in \mathcal{I}_{i}$$ we denote by $$A\left(I_{i}\right)$$ the set $$A(h)$$ and by $$P\left(I_{i}\right)$$ the player $$P(h)$$ for any $$h \in I_{i}$$ . ($$\mathcal{I}_{i}$$ is the **information partition 信息分割** of player $$i ;$$ a set $$I_{i} \in \mathcal{I}_{i}$$ is an **information set 信息集合** of player $$i .$$ ) 
- 使用收益来反应策略偏好.  For each player $$i \in N$$ a preference relation $$\succsim_{i}$$ on lotteries over $$Z$$ (the **preference relation** of player $$i$$ ) that can be represented as the expected value of a payoff function defined on $$Z$$

We refer to a tuple $$\left\langle N, H, P, f_{c},\left(\mathcal{I}_{i}\right)_{i \in N}\right\rangle$$ (which excludes the players' preferences whose components satisfy the conditions in the definition as an **extensive game form**.

新加入信息集 Relative to the definition of an extensive game with perfect information and chance moves (see Section 6.3.1 ), the new element is the collection $$\left(\mathcal{I}_{i}\right)_{i \in N}$$ of **information partitions**. We interpret the histories in any given member of $$\mathcal{I}_{i}$$ to be **indistinguishable 不可分辨** to player $$i .$$ Thus the game models a situation in which after any history $$h \in I_{i} \in \mathcal{I}_{i}$$ player $$i$$ is informed that some history in $$I_{i}$$ has occurred but is not informed that the history $$h$$ has occurred. The condition that $$A(h)=A\left(h^{\prime}\right)$$ whenever $$h$$ and $$h^{\prime}$$ are in the same member of $$\mathcal{I}_{i}$$ captures the idea that if $$A(h) \neq A\left(h^{\prime}\right)$$ then player $$i$$ could deduce, when he faced $$A(h),$$ that the history was not $$h^{\prime},$$ contrary to our interpretation of $$\mathcal{I}_{i} .$$

