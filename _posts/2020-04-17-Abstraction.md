---
layout:     post
title:      PsOpti算法, 伪最优
subtitle:   Approximating Game-Theoretic Optimal Strategies for Full-scale Poker
date:       2020-04-01 12:00:00
author:     "tengshiquan"
header-img: "img/post-bg-dice.jpg"
catalog: true
tags:
    - AI
    - Game Theory 
    - Imperfect Information
    - Texas

---



# Approximating Game-Theoretic Optimal Strategies for Full-scale Poker

2003 

**abstraction** 这种技术感觉更多是**简化**的意思.  本文针对德州这个游戏的特性,对各个阶段的玩法进行简化来计算策略. 



### Abstract

第一个近似博弈论最优策略的全规模德州算法。   

几种 abstraction 抽象技术结合在一起，来表示双人德州扑克$O(10^{18})$，相关模型$O(10^{17})$。  
尽管规模缩小了1000亿倍，但所得到的模型仍保留了真实游戏的关键属性和结构。  
抽象的游戏的线性规划解决方案被用于创建大幅改进的扑克游戏程序，能够击败强大的人类玩家，并与世界级的对手竞争。



### Introduction

数学博弈论是由约翰-冯-诺依曼在20世纪40年代提出的，此后成为现代经济学的基础之一[冯-诺依曼和Morgenstern，1944]。冯-诺依曼将扑克牌游戏作为2人**零和zero-sum**对抗性博弈的基本模型，并证明了第一个基本结果，即著名的**极大极小值定理*minimax theorem***。几年后，约翰-纳什(John Nash)补充了N人非合作博弈的结果，他后来因此获得了诺贝尔奖[Nash, 1950]。许多决策问题都可以用博弈论来建模，近年来，博弈论被广泛地应用于各个领域。

大家感兴趣的是**最优解*optimal solutions***的存在，即**纳什均衡*Nash equilibria***。一个最优解提供了一个**随机混合策略randomized mixed strategy**，基本上就是在各种可能的情况下如何进行博弈的配方。使用这种策略可以确保代理至少获得博弈的博弈理论值 game-theoretic value，而不管对手的策略如何。遗憾的是，寻找确切的最优解仅限于相对较小的问题规模，对于大多数实际领域来说并不实用。

本文探讨了使用高度抽象的数学模型，这些模型抓住了真实领域中最基本的属性，因此，对较小问题的精确解可以为真实领域的最优策略提供一个有用的近似值。所使用的应用领域是扑克游戏，特别是德州扑克。

由于计算上的限制，过去只解决了简化的扑克变体（例如[Kuhn, 1950; Sakaguchi and Sakai, 1992]）。虽然这些方法具有理论意义，但同样的方法对于真实的游戏来说是不可行的，因为真实的游戏大了很多数量级([Koller and Pfeffer, 1997])。
Shi和Littman，2001]研究了**抽象abstraction**技术，利用扑克牌的简化变体，研究了抽象技术来降低问题的巨大搜索空间和复杂度。Takusagawa, 2000]为三种特定的Hold’em flops和下注序列创建了近乎最佳的策略。Selby, 1999]计算出了一个关于*preflop Hold’em*游戏的最优解。

利用新的抽象技术，我们为2人德州扑克游戏制作了可行的 "伪最佳 "策略。由此产生的扑克游戏程序在性能上有了巨大的改进。以前最好的扑克程序可以轻易地被任何有能力的人类玩家打败，而新的扑克程序能够打败非常强大的玩家，并且能够在世界一流的对手面前保持自己的地位。
虽然一些特定领域的知识是创建精确的缩小规模的模型的财富，但类似的方法可以开发出许多其他imperfect information领域和广义的游戏树。我们描述了一种一般的问题重构方法，通过估计每次计算所需的条件概率作为输入，来允许子树的独立解决。

本文两个贡献：

1. 抽象技术，可以将扑克牌搜索空间$O(10^{18})$，缩小到$O(10^{17})$一个可管理的空间, 而不失去游戏中最重要的属性。
2. 一个扑克牌AI，相比前作有了很大的进步，能够与世界一流的对局者同台竞技。



### Game Theory 博弈论

博弈论涵盖了两个或多个**代理agent**之间的所有形式的竞争。与国际象棋或跳棋不同，扑克牌是一种不完美信息和**随机结果*chance outcomes***的博弈。它可以用**不完美的信息博弈树*imperfect information game tree*** 来表示，它有**随机chance节点**和**决策decision节点**，这些节点被归纳为**信息集 information sets**。

由于这个树中的节点不是独立的，所以计算子树的**分治法divide- and-conquer**（如**α-β算法**）并**不适用**。关于不完全信息博弈树结构的更详细的描述，请参见[Koller和Megiddo，1992]。

**策略*strategy***是在树的每个决策节点上选择一个动作的规则集。一般情况下，这将是一个**随机混合策略*randomized mixed strategy***，它是各种可选方案的概率分布。**一个玩家必须在同一信息集中的所有节点上使用相同的策略**，因为从该玩家的角度来看，它们是无法区分的。

解决这类问题的传统方法是将描述性表达或**展开型*extensive form***转换为线性方程系统，然后由**线性规划（LP）**系统（如**Simplex算法**）求解。同时计算出所有玩家的最优解，确保每个玩家都能得到best worst-case结果。

传统上，转换为***normal form 标准型 , 即静态博弈***,  规模是原问题大小的指数，这意味着在实践中只能解决非常小的问题。?? [Koller *et al.*, 1994] 提出了另一种称为**序列形式*sequence form***的LP表示法，它利用了**完美回忆*perfect recall***的共同属性（即所有的玩家都知道之前的游戏历史），得到了一个方程和未知数系统，而这个系统的规模和游戏树的大小上是线性的。这种指数级的缩减表示方式重新为许多领域的博弈论分析提供了可能性。然而，由于博弈树本身可以非常大，LP求解方法仍然限于中等大小的问题（通常小于10亿个节点）。



### Texas Hold’em 德州规则

52张, 没大小王. 

一场(*hand*)德州扑克游戏包括四个阶段，每个阶段之后是一轮下注。 玩家的核心策略就是各种押注以及放弃.

- **Preflop** : 每位玩家都要面朝下发两张私人底牌（*hole cards*）
- **Flop**：三张公牌 *community cards*
- **Turn** : 发一张公牌
- **River** ：最后一张公牌

下注后，所有active玩家都会亮出他们的hole cards ,  摊牌 *showdown* ; 每个人两张底牌和五张公牌组成的最好的五张扑克牌five-card poker hand的玩家赢取所有的赌注（有可能是平手）;

游戏开始时，玩家会把两个强制下注（盲注 *blinds*）放入*pot 底池*中。当轮到玩家行动的时候，他们必须下注 bet/加注raise ，让牌check/跟注call（与对手的下注或加注的金额相匹配），或者fold 收牌（退出并交出所有的钱到pot）。



<img src="2020-04-17-Abstraction.assets/image-20200511184409487.png" alt="image-20200511184409487" style="zoom:50%;" />



### Abstractions

德州扑克有一个容易识别的结构，在机会节点和投注回合之间交替进行，分为四个不同的阶段。图1是游戏树的抽象视图。

Hold'em可以被重新设计成类似但更小的游戏。其目的是在不严重改变游戏的基本结构或由此产生的最优策略的情况下，缩小问题的规模。

一些最精确的抽象方法包括 *suit equivalence isomorphisms*（最多减少4!=24倍）、*rank equivalence*（只有在一定条件下）和 *rank near-equivalence* 。这些抽象的问题的最优解要么完全相同，要么有一个小的约束误差，我们称之为 **近似最优*near-optimal*** 解。但是, 并不能很大地缩减，解决不了完整规模的扑克游戏问题。

控制游戏规模的一个常见方法是减少牌组 *deck reduction*。使用少于标准的52张牌的牌组，可以大大降低概率节点的分支系数。其他方法包括减少玩家的手牌数(2-card hand to a 1-card hand)，以及减少board cards的数量（例如1张翻牌) 。[Koller and Pfeffer, 1997] 用这样的参数生成了各种各样的可解决的游戏。

我们使用了一些小型和中等规模的游戏，从8张牌（2个suits，4个ranks）到24张牌（3个suits，8个ranks），目的是研究抽象方法，将结果与已知的精确或接近最优解进行比较。然而，这些较小的游戏并不适合作为德州扑克的近似方法，因为游戏的底层结构不同。为了产生好的全局游戏策略，我们要寻找不改变这种基本结构的真实游戏的抽象方法。

实践中使用的抽象技术在缩小问题的大小方面是很强大的，也是对前面提到的那些技术的归纳。但是，由于它们也比较粗糙，所以我们称它们为***pseudo-optimal* 伪最优解**，以强调**不能保证得到的近似值是准确的，甚至是合理的**。有些将是低风险的命题，而另一些则需要通过经验检验来确定它们是否有价值。



#### Betting round reduction  减少押注次数

limit Hold’em 的标准规则允许每个玩家每轮最多下注4次。 因此，在2人极限扑克中，有19个可能的下注顺序，其中2个在实践中没有出现。 在剩下的17个顺序中，8个以fold结束（导致游戏树中的terminal节点），9个以call跟注结束（转到下一个chance节点）。 使用 k=check, b =bet, f =fold, c =call, r =raise, 大写表示第二个玩家, 则所有可能bet序列为: 

kK kBf kBc kBrF kBrC kBrRf kBrRc kBrRrF kBrRrC   
bF bC bRf bRc bRrF bRrC bRrRf bRrRc

我们把这个局部的决策节点集合称为*betting tree*，用一个三角形来表示。  
通过*betting round reduction*，每个玩家每轮最多允许下注三次，从而消除了上面每行的最后两个。投注树的有效分支系数从9个减少到7个。这似乎并没有对玩法，或对每个玩家的预期值（EV）产生实质性的影响。这一观察已经得到了实验验证。相比之下，我们计算了相应的 postflop模型，每轮每个玩家最多有两个下注的情况下，我们发现最优策略发生了根本性的变化，强烈地表明这种级别的abstraction并不安全。



#### Elimination of betting rounds 去掉押注轮次

扑克game tree的大小可以通过去掉下注回合来大幅减少。 有几种方法可以做到这一点，而且一般都会对游戏的性质产生重大影响。 

首先，游戏可以通过消除最后一轮或几轮来缩短游戏。 在Hold'em游戏中，忽略最后一张牌和最后一轮的下注，就会产生一个实际4-round game的3-round model; 2+4张。 3-round mode的解决方案失去了真正的最优策略中的一些微妙之处，但这种**退化degradation**主要适用于turn的高级战术。对flop策略的影响较小，第一轮下注的策略可能没有明显的变化，因为它包含了未来两轮下注的所有结果情况。我们用这个特殊的abstraction 来定义第一轮下注的合适策略，因此我们称其为**preflop model**.  即flop之前的都还尽量还原.

通过使用**期望值叶子节点*expected value leaf nodes*** 可以减少截断的影响。我们不是突然结束游戏, 将底池给那个时刻的最强手，而是在所有可能的概率结果中计算出一个平均结论。对于一个3-round model, 在turn轮后结束 , 2+4 ，我们将所有52-8=44张可能的river cards都推出来*roll-out* , 假设没有进一步下注，（或者，假设最后一回合每个玩家都有一个下注）。每个玩家都会获得一部分的底池，对应于他们赢牌的概率。在2-round preflop model中 2+3，剩下52-7=45.  我们roll-out所有 turn and river的990个 2张牌组合。 这个是有点道理的, 就是怎么评判之前的策略的优劣.就是对剩下的情况都求概率. 

最极端的截断形式的结果是1-round model，没有对未来的投注回合进行预测。由于未来的每一轮都是对近似的细化refinement，这将不能反映出真实游戏的正确策略。特别是，有些bet plan是超过一个回合，比如手持强牌, 但后面回合再raise，就会完全丧失。然而，即使如此,  这些简单化的模型与预期值叶子节点结合起来也是有用的。

**思路, 减少 bet 回合,  公牌数 , bet轮次数 , 用来减少gametree 规模, 然后配上  剩余牌的概率分布.** 

Alex Selby计算出了一个最优解，用于*preflop* Hold'em的游戏，它只包括第一轮下注，然后是五张board cards的EV roll-out，来决定谁赢。尽管基于1-round model的策略有一些严重的局限性，我们将Selby的preflop系统纳入到我们的程序中，*PsOpti1* . 

与截断truncating rounds相比，我们可以绕过某些早期阶段的游戏。我们经常使用的是**postflop** model，即忽略了preflop betting round，而使用三张牌的单一固定flop（见图1）。**忽略第一轮下注**

很自然地，我们会考虑到 独立投注回合*independent betting rounds*的想法，即把每一阶段的游戏都孤立地对待。不幸的是，前几轮的投注历史几乎都会包含对做出适当决策至关重要的背景信息。每个玩家的手牌的概率分布强烈地依赖于导致该决策点的路径，因此，如果要避免相当大的信息损失的风险，就不能忽视它。然而，在某些情况下，naive的独立性假设是可行的，我们在设计PsOpti1时确实隐含地使用了这一假设，以弥补1-round preflop model和3-round postflop model之间的差距。 

我们探索的另一个可能的抽象是将两个或更多的回合合并*merging*为一个回合，比如创建一个组合 2-card turn/river。然而，目前还不清楚这种复合回合的适当下注大小应该是多少。无论如何，这些模型的解决方案（在整个可能的下注大小范围内），结果都与3-round的同类模型有很大的差异，因此该方法被否决了。



#### Composition of preflop and postflop models 

核心, 串联两个 简化的模型

虽然imperfect information game tree的节点在一般情况下不是独立的，但也可以进行一些分解 decomposition 。例如，由不同的preflop betting sequences 产生的子树不能再有属于同一信息集的节点。 只要给出适当的先决条件作为输入，我们的postflop models的子树就可以独立计算。不幸的是，要知道正确的条件概率通常需要解出整个游戏，所以分解没有任何好处。 即, 第一轮什么样的bet , 对应后面什么样的变化, 这个对应的概率是无法得到的.

<img src="2020-04-17-Abstraction.assets/image-20200512023830930.png" alt="image-20200512023830930" style="zoom:50%;" />

对于简单的postflop models,，我们放弃了先验概率prior probabilities。对于PsOpti0和PsOpti1中使用的postflop models，我们简单地忽略了第一轮下注 的影响，假设每个玩家的所有可能的hands都是统一分布的。我们计算了postflop 的解, 对2、4、6和8的初始底池大小（对应于0、1、2、3 raises的preflop sequences，但忽略了是哪个玩家首先加注）。在PsOpti1中，四种postflop solutions被简单地附加到Selby preflop strategy中（图2）。虽然这些简化的假设在技术上是错误的，但最终的玩法仍然是令人惊讶的有效。

一个更好的方法是利用preflop model的解来估计条件概率 , 然后来组合postflop models。有了一个可解的preflop model，我们就有了在root估计适当策略的方法，从而确定随之而来的概率分布。

对*PsOpti2*,  3-round preflop model. 求得 preflop的 pseudo-optimal strategy 用来决定每个玩家的手牌分布 (该策略与Selby 的策略有明显差异) . 这为七种 第一轮下注序列中的每一种提供了必要的输入参数，这些参数都会延续到flop阶段。 由于每一个 postflop models 都有了full game 的 （近似的）perfect recall knowledge，所以它们是完全兼容的，可以整合在preflop model下面（图2）。理论上，这相当于计算出了更大的game tree，但受制于preflop betting model的准确性和适当性。 



#### Abstraction by bucketing

下面对于两个分段的简化模型, 怎么计算伪最优策略. 

对于我们的伪pseudo-optimal strategies的计算，最重要的abstraction方法叫做 "bucketing"。这是一个自然而直观的概念的延伸，所有可能的hand的集合被划分为等价类（也称为*buckets*或*bins*）。一个多对一映射函数决定了哪些手将被分组。理想的情况下，这些手牌应该根据策略相似度*strategic similarity*来分组，这意味着它们都可以以相似的方式出牌，而不会有太大的EV损失。

如果每个hand都是用一种特定的纯策略来玩，那么一个完美的映射函数将把所有遵循相同计划的手牌进行分组，给每个玩家17个分类就足以满足每一轮的下注。然而，由于在某些情况下，混合策略可能会被表示为最优的下注策略，因此我们希望**将那些在行动计划上具有相似概率分布的手牌进行分组**。

一个很明显但相当粗糙的bucketing函数是根据strength（即相对于所有可能的手牌的*rank排名*，或当前处于领先的概率）对所有手牌进行分组。这可以通过考虑roll-out所有未来牌的出牌情况来改进，给出一个（非加权）估计的胜率。

然而，这只是一个hand类型的one-dimensional view，在可以被认为是一个n维的策略空间中，有大量不同的分类方法。一个更好的方法是将所有hands的集合投射到一个二维空间中，这个空间由(roll-out) hand **strength**和hand **potential潜力**组成。所得的散点图中的Clusters表明，合理的groups of hands可以得到相似地对待。

我们用了一个简单的折中方案。 对于 n 个可用buckets , 我们分配 n-1 到roll-out hand strength。每个分类中的hand types的数量并不统一，强strength hand的分类比一般hand和弱hand的要小，这样可以更好地区分出应该raise或re-raise的小部分hands。

有一个特殊的bucket被指定给那些strength低但潜力大*high potential*的手牌，比如同花或顺子的好牌。这在判断当前手牌是否适用于**诈唬 bluffing**方面起着重要作用。对postflop, 将使用6个strength buckets的方案与使用5个strength加1个high-potential bucke的方案进行比较，我们看到后者的大多数唬牌都是从high-potential bucket中拿出来的，有时就像 strongest bucket 一样。这证实了我们的预期，即high-potential bucket可以改善各种下注策略，提高整体的EV。

 3-round model 中可以配合使用的buckets数非常少，通常每个棋手的buckets数是6到7个（即36或49pairs of bucket ）。显然，这导致的结果是一个非常粗糙的抽象游戏，但这可能与一般人类玩家可能做出的区分数量没有本质上的区别。不管怎么说，鉴于这种方法的计算限制，这是目前我们能做的最好的方法。

最后需要将抽象的游戏与真实game tree 区分开来的是转移概率*transition probabilities*。在flop和turn之间的概率节点代表着从剩余的45张牌库中拿出一张特定的牌。在抽象博弈中，没有牌，只有buckets。在抽象博弈中，turn card的作用是决定了从flop时的one pair of buckets的概率到 turn的时候 任意pair of buckets的概率。因此，博弈树中的概率节点的集合由一个 n * n 到 n * n的过渡网络表示，如图3所示。对于postflop model，这可以通过走遍整个树，枚举少量characteristic flop的所有transitions来估计。对于preflop model，完整的枚举是比较昂贵的(枚举所有可能的flop, 48选3=17296个flop)，所以可以通过取样或(并行)枚举截断tree来估计。

<img src="2020-04-17-Abstraction.assets/image-20200512031908381.png" alt="image-20200512031908381" style="zoom:50%;" />



对于一个3-round postflop model, 我们可以很舒服地解决抽象的游戏，每轮每个玩家最多有7个buckets。改变buckets的分布，比如flop是6个，turn是7个，river是8个，似乎并不会对解法的质量产生明显的影响，不管是好是坏。

最后的线性规划方案为抽象游戏中的每一个可触及的情况产生一个大的混合策略表（fold, call, or raise的概率）。为了使用这一点，扑克游戏程序根据相同的手牌强度和potential measures寻找相应的情况，并从混合策略中随机选择一个动作。

大型的LP计算通常需要不到一天的时间（使用CPLEX, barrier法），2G RAM。较大的问题将超过可用内存，这对于大型LP系统来说是常见的。某些LP技术，如约束生成*constraint generation* 等可能会大大扩展可解实例的范围，但这可能只允许每个玩家使用一到两个额外的bucket。





### Experiments

#### Testing against computer players

Win rates are measured in small bets per hand (sb/h) .   
每个对比, 2000games,  有些100000.



必须明白，博弈论的最优的玩家原则上不是为了胜利而设计的。它的目的是为了不输。

扑克牌比较复杂，理论上，最优玩家可以赢，但前提是对手犯了支配性错误 dominated errors。任何时候，当玩家做出的任何选择都是某种博弈论最优策略的随机混合策略的一部分，那么这个action就不是dominated。换句话说，可以以高度次优sub-optimal 的方式下棋，但仍然可以与最优玩家打成平手，因为这些选择并不是 strictly dominated。 

由于伪最佳策略不做对手建模，所以不能保证它们在面对非常糟糕或高度可预测的棋手时特别有效。它们必须只依靠这些基本的策略错误，结果胜率可能会相对不高。

<img src="2020-04-17-Abstraction.assets/image-20200512193146739.png" alt="image-20200512193146739" style="zoom:50%;" />

所有的 pseudo-optima玩家都比之前的任何计算机程序都要玩得好得多。即使是PsOpti0，它的设计并不是为了玩完整的游戏，但它从postflop betting rounds中赚取了足够的利润，以抵消preflop round的EV损失（它从不加注好牌，也不fold坏牌）。

值得怀疑的是，PsOpti1的表现优于PsOpti2，从原则上讲，PsOpti2应该是一个比较好的近似。随后对PsOpti2的进行分析，发现了一些编程上的错误，同时也表明preflop模型的bucket分配存在缺陷。



要想制作出超越所有人类棋手的程序，几乎必然需要对手建模。



### Conclusions and Future Work

第一个人类水平的2人完整德州AI.

有用的 abstractions 包括 betting tree reductions, truncation of betting rounds combined with EV leaf nodes, and bypassing betting rounds.  

most powerful abstractions  ,  base on bucketing

最后，拥有合理的最优策略的近似值并不会降低良好的对手建模的重要性。在随机博弈中对自适应的对手进行学习是一个具有挑战性的问题，在将两种不同形式的信息结合起来的过程中，会有很多想法需要探索。这将可能是一个能与高手竞争的程序和一个超越所有人类玩家的程序之间的关键区别。