---
layout:     post
title:      DeepStack
subtitle:   DeepStack:Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker
date:       2020-04-01 12:00:00
author:     "tengshiquan"
header-img: "img/post-bg-dice.jpg"
catalog: true
tags:
    - AI
    - Game Theory 
    - Imperfect Information
    - Alberta
    - Texas

---



# DeepStack: Expert-Level Artificial Intelligence in Heads-Up No-Limit Poker

2017  from Alberta



### Abstract

近些年来，人工智能领域出现了很多突破，其中游戏往往被用作重要的里程碑。过去实现那些成功的游戏的一个常见的特征是它们都具有完美信息（perfect information）的性质。扑克是一个典型的不完美信息（imperfect information）游戏，而且其一直以来都是人工智能领域内的一个难题。在这篇论文中，我们介绍了 DeepStack，这是一种用于扑克这样的不完美信息环境的新算法。它结合了回归推理（recursive reasoning）来处理信息不对称性，还结合了分解（decomposition）来将计算集中到相关的决策上，以及一种形式的直觉（intuition）——该直觉可以使用深度学习进行self-play而自动学习到。在一项 44000手比赛的扑克研究中，DeepStack 在一对一无限押注德州扑克（heads-up no-limit Texas hold'em）上击败了职业扑克玩家。这种方法在理论上是可靠的，并且在实践中也能得出比之前的方法更难以被利用的策略。



### Introduction

perfect information是很多游戏成功关键 , 例如可以 play的时候 local search 

asymmetric information 不对称信息

**Heads-up no-limit Texas hold’em (HUNL)**   一对一无限注德州扑克 ;  之前的AI是固定下注, $10^{14}$ 个decision points.  围棋, $10^{170}$ 个决策点.  information game HUNL与围棋相当, 有$10^{160}$ . 

reasoning 推理的重要性:  不完全信息博弈比同样大小的完全信息博弈需要更复杂的推理。在某一特定时刻的正确决策取决于对手所掌握的私人信息的概率分布，而这些信息是通过对手过去的行动所揭示的。然而，我们的对手的行动如何揭示出这些信息，取决于他们对我们的私人信息的了解，以及我们的行动如何揭示出这些信息。**这种递归推理是不能轻易对博弈情境进行孤立推理的原因，这也是完美信息博弈的启发式搜索方法的核心。**不完全信息博弈中的竞争性人工智能方法通常会对整个博弈进行推理，并在博弈前产生一个完整的策略（14-16）。**虚拟遗憾最小化（CFR）**（14,17,18）就是这样一种技术，它通过在迭代中对自己的策略进行递归推理，利用自我博弈来做递归推理。如果博弈太大，无法直接解决，常见的反应是解决一个较小的、抽象的博弈。要想玩好原始博弈，就要**把原始博弈**中的情境和动作**转换为抽象博弈**。

尽管这种方法使程序在HUNL这样的游戏中进行推理是可行的，但它是通过将HUNL的$10^{160}$种情况压缩到$10^{14}$种抽象情况的顺序来实现的。可能是由于这种信息丢失的结果，这样的程序在专家级的人类游戏中落后于专家级的人类。在2015年，计算机程序Claudico以91 mbb/g的优势输给了一个职业扑克玩家团队，这是一个 "巨大的胜率"(20)。此外，最近有研究表明，在年度计算机扑克大赛中，**基于抽象概念的程序存在巨大的缺陷**(21)。我们使用一种局部最佳反应技术对四种这样的程序（包括2016年比赛中的顶级程序）进行了评估，该技术能得到关于一个策略能输多少的近似下限。这四种基于抽象的程序都能以超过3000mbb/g 被击败，这是每场比赛简单folding的四倍。

DeepStack采取了一种根本不同的方法。它继续使用CFR的递归推理来处理信息不对称。然而，它不计算和存储一个完整的策略，因此**不需要显式抽象**。相反，它考虑的是每一个特定的情况，因为它是在博弈过程中出现的，但不是孤立的。它通过用一个**快速的近似估计**来代替超过一定深度的计算，避免了对整个游戏的剩余部分进行推理。这种估计可以被认为是DeepStack的**直觉**：在任何可能的牌局中，对任何可能的私人牌的价值的直觉。最后，DeepStack的直觉，就像人类的直觉一样，需要进行训练。我们用深度学习（22）训练它，使用从随机扑克牌情境中生成的例子。我们表明，DeepStack在理论上是健全的，产生的策略比基于抽象的技术更难利用，并且在统计学意义上击败了HUNL的职业扑克玩家。



### DeepStack

DeepStack是一个用于一大类 不完全信息序列博弈sequential imperfect information games 的通用算法。为了清楚起见，我们将在HUNL游戏中描述它的原理。扑克游戏的状态可以分为玩家的私人信息(两张牌面朝下发的手牌)和公开状态(由牌面朝上的牌和玩家的下注动作序列组成)。游戏中可能的公开状态序列形成了一个**公开树 public tree**，每个公开状态都有一个相关的公开子树（图1）。

![Fig. 1 – Poker Game Tree](2020-04-16-DeepStack.assets/DeepStack-F1.jpg)



图1 , HUNL中的公开树的一部分。节点代表公开状态，而边代表行动：红色和蓝绿色表示玩家的下注行动，绿色代表随机发的公开牌。游戏在终点节点结束，显示为一个相关收益的筹码。

玩家的策略定义了每个决策点的有效行动的概率分布，一个决策点是公开状态和行动玩家的手牌的组合。给出一个玩家的策略，对于任何公开状态，我们可以计算出玩家的**范围range**:玩家可能的手牌的概率分布。

在固定了双方的策略后，在游戏结束的终点公开状态下，特定玩家在终点公开状态下的收益是由游戏规则确定的报酬矩阵决定的玩家范围的双线性函数。玩家在任何其他公开状态（包括初始状态）下的预期收益是在玩家的固定策略下，在可达到的终点状态概率分布上的预期收益。最佳反应策略是指玩家的预期收益最大化的策略。在双人零和博弈中，如HUNL，当与最佳反应对手策略对弈时，纳什均衡策略(23)使预期收益最大化。策略的**可利用性exploitability**是指在对阵最佳反应对手时的预期收益与纳什均衡下的预期收益之差。

DeepStack 的目标: low-exploitability strategy ,  solve for an approximate Nash equilibrium

DeepStack 三部分组成:  

1. a sound local strategy computation for the current public state
2. depth-limited lookahead using a learned value func- tion to avoid reasoning to the end of the game
3. a restricted set of lookahead actions

从概念层次看, 这3个都属于heuristic search.

在DeepStack之前，在不完美信息游戏中，还没有任何理论上合理的启发式搜索的应用.启发式搜索的核心是 "**持续再搜索 continual re-searching**"，每当代理必须采取行动时，就会执行一个靠谱的局部搜索，而不保留任何关于它是如何或为什么达到当前状态的记忆。

DeepStack的核心是 **持续再解决continual re-solving**，这是一个完善的局部策略计算，只需要最小的记忆力就可以知道它是如何行动的，为什么要达到目前的公开状态。



![Fig. 2 – DeepStack Operation](2020-04-16-DeepStack.assets/DeepStack-F2.jpg)

(A) DeepStack在公开树中的推理, 输出 在公开状态中它可能持有的所有牌组对应 行动概率。它维护两个向量：自己的范围和对手的counterfactual值。随着游戏的进行，执行一个action后, 通过贝叶斯法则计算出的行动概率, 更新自己的范围。对手的counterfactual值会在 "持续再求解 "中讨论过的那样更新。当它必须采取行动时，为了计算出行动概率，它使用自己的范围和对手的counterfactual值进行重解。为了使重解更容易，它限制了玩家的可操作性，并且lookahead被限制到回合结束。在重解过程中，超出其 lookahead 的公开状态的counterfactual值会使用 DeepStack 的学习评价函数进行近似。  
(B) 评估函数用神经网络表示，该神经网络以迭代中的公开状态和范围作为输入，并输出两个玩家的counterfactual值（图3）。  
(C) 神经网络, 在play前通过生成随机的游戏状态(奖池大小、牌型和范围)并求解以产生训练实例来训练神经网络。完整的伪代码可以在Algorithm S1中找到。




#### Continual re-solving

imperfect recall

假设我们按照特定的策略采取了行动，但在某些公开状态下，我们忘记了这个策略。我们是否可以在不需要重新解决整个游戏的情况下重建一个子树的求解策略？可以, 通过Continual re-solving. 

我们既需要知道我们在公开状态下的范围，也需要知道对手在前一solution下对每个对手手牌的预期值向量（24）。有了这些值，我们就可以重构出一个只适用于剩余游戏的策略，这并不会增加我们的整体可利用性。对手向量中的每个值都是一个counterfactual值，这是一个条件性的 "what-if "值，它给出了如果对手以特定的手牌达到公开状态时的预期值。CFR算法也是使用counterfactual值，如果我们使用CFR作为我们的求解器，那么很容易计算出对手在任何公开状态下的counterfactual值向量。

然而，re-solving是从一个策略开始的，而我们的目标是避免在整个游戏中维持一个策略。我们通过Continual re-solving来绕过这一点：每次需要行动的时候，我们都要通过re-solving来重构策略；在下一次行动之后，永远不使用这个策略。为了能够在任何公开状态下进行re-solving，我们只需要跟踪自己的范围和对手的counterfactual值的合适向量就可以了。这些值必须是对手在当前公开状态下每一手牌能达到的值的上界，同时不大于对手在达到公开状态下偏离了公开状态后能达到的值。这是对re-solving中通常使用的counterfactual值的一个重要的放宽。

在游戏开始时，我们的范围是统一的，对手的counterfactual值被初始化为每张私下牌的值。当轮到我们行动的时候，我们使用存储的范围和对手值在当前的公开状态下重新解决子树，并根据计算出的策略行动，在再次行动之前丢弃策略。每次行动后，无论是玩家还是chance发牌，我们都会根据以下规则更新自己的范围值和对手的counterfactual值。(i) 自己的行动：用我们所选行动的re-solve策略,再更新对手的counterfactual值。再更新我们自己的范围。(ii) Chance行动：将对手的counterfactual值替换为上次re-solve策略中为这个chance行动计算的值。通过给定新的公开牌，将不可能的范围内的手牌归零，更新自己的范围。(iii) 对手行动：不需要改变我们的范围或对手的值。

这些更新确保了对手的counterfactual值满足我们的充分条件，整个过程产生了任意接近纳什均衡的近似值（见定理1）。请注意，Continual re-solving从不跟踪对手的范围，而只跟踪对手的counterfactual值。此外，它从不需要知道对手的行动来更新这些值，这也是与传统re-solving的一个重要区别。这两点将被证明是使该算法高效的关键，并避免了动作抽象方法所需的转换步骤。

Continual re-solving在理论上是合理的，但本身是不好实践的。虽然它永远不会保持一个完整的策略，但除了在游戏接近尾声的时候，re-solving本身是不切实际的。为了使Continual re-solving具有实用性，我们需要限制re-solving的深度和广度。



#### Limited depth lookahead via intuition

就像在完美信息博弈的启发式搜索中，我们希望限制子树的深度，在重新求解时，我们必须对子树进行推理。无论如何，在不完全信息博弈中，我们不能简单地用启发式或预计算值来代替子树。在公开状态下的counterfactual值不是固定的，而是取决于玩家如何play达到公开状态，也就是玩家的范围(17)。当使用迭代算法（如CFR）重新求解时，这些范围在求解器的每一次迭代中都会发生变化。
DeepStack克服了这一挑战，它用一个学习的counterfactual值函数取代了超过一定深度的子树，该函数可以近似于如果用当前迭代的范围来求解公开状态的结果值。这个函数的输入是两个玩家的范围，以及底池大小和公开牌，这些都足以指定公开状态。输出是每个玩家的向量，包含了在该情况下每个玩家持有的counterfactual值。换句话说，输入本身就是对扑克游戏的描述：被发牌的概率分布，游戏的赌注，以及已发出的公开牌；输出是在当前游戏中，持有特定牌组的价值的估计。价值函数是一种直觉，是对在任意牌局中发现自己的价值的快速估计。由于深度限制为4个动作，这种方法将游戏中的重解规模从游戏开始时的$10^{160}$个决策点减少到不超过$10^{17}$个决策点。DeepStack使用深度神经网络作为它的学习值函数.



#### Sound reasoning

DeepStack的深度限制的持续重解是合理的。如果DeepStack的直觉是 "好的"，并且在每个重解步骤中都使用了 "足够的 "计算，那么DeepStack扮演的是一个任意接近纳什均衡的近似值。



**Theorem 1** If the values returned by the value function used when the depth limit is reached have error less than $\epsilon,$ and $T$ iterations of CFR are used to re-solve, then the resulting strategy's
exploitability is less than $k_{1} \epsilon+k_{2} / \sqrt{T},$ where $k_{1}$ and $k_{2}$ are game-specific constants. 

即收敛.



#### Sparse lookahead trees

DeepStack的最后一个要素是减少了考虑的行动数量，从而构建了一个稀疏的 lookahead 树。DeepStack只使用动作fold（如果有效）、跟注、2或3个下注动作和all-in来构建lookahead树。这一步使定理1的稳健性失效，但它允许DeepStack以传统的人类速度下注。通过稀疏和深度限制的lookahead树，重新解决的游戏大约有$10^7$个决策点，使用单块NVIDIA GeForce GTX 1080显卡在5秒内解决。我们还使用稀疏和深度受限的 lookahead 求解器在游戏开始时就计算出对手的反事实值，用于初始化 DeepStack 的持续重解。



#### Relationship to heuristic search in perfect information games

DeepStack要在不完美的信息游戏中融入启发式搜索思想，需要克服三个关键挑战。首先，如果不知道博弈者是如何以及为什么采取行动达到公开状态的，就无法实现对公开状态的合理再解。相反，必须保持两个额外的向量，即代理的范围和对手的counterfactual值，才能用于再解。其次，再求解是一个迭代过程，需要多次遍历 lookahead 树，而不是只遍历一次。每一次迭代都需要对超出深度限制的每一个公开状态以及不同的range, 再次调用评价函数。第三，当达到深度极限时所需要的评估函数在概念上比完美信息更复杂。counterfactual值函数需要返回的不是给定一个单一状态的值，而是给定公开状态和玩家的范围的值向量。由于这种复杂性，为了学习这样的值函数，我们使用深度学习，在完美信息博弈中，深度学习在学习复杂的评价函数方面也取得了成功.



#### Relationship to abstraction-based approaches

虽然DeepStack使用了抽象的思想，但它与基于抽象的方法有本质的区别。DeepStack限制了其 lookahead 树中的动作数量，这很像动作抽象（25，26）。然而，DeepStack中的每次重解都是从实际的公开状态开始的，所以它总是完美地理解当前的情况。该算法也从不需要使用对手的实际行动来获得正确的范围或对手的counterfactual值，从而避免了对手下注的转换。我们使用手牌聚类作为counterfactual值函数的输入，很像显式扑克抽象方法（27，28）。然而，我们的聚类是用来在lookahead树的末尾估计counterfactual值，而不是限制玩家在行动时对其牌的信息进行限制。我们后来表明，这些差异导致了策略更难被对手利用。





### Deep Counterfactual Value Networks

两个独立的网络被训练：一个是在前三张公共牌发完后估计counterfactual值（flop网络），另一个是在第四张公共牌发完后估计counterfactual值（turn网络）。在任何公共牌出牌前的辅助网络用于加速早期actions的re-solving。

![Fig. 3 – DeepStack Neural Networks](2020-04-16-DeepStack.assets/DeepStack-F3.jpg) 

output: post-processed to guarantee the values satisfy the **zero-sum constraint**, and then mapped back into a vector of counterfactual values. 



##### Architecture

DeepStack使用了一个标准的前馈网络，该网络有七个全连接的隐藏层，每个隐藏层有500个节点，输出为Relu。这种架构被嵌入到一个外部网络中，迫使counterfactual值满足零和属性。外部计算取估计的counterfactual值，并利用两个玩家的输入范围计算出一个加权总和，从而产生单独的博弈值估计。这两个值应该相加为零，但可能不是。然后从两个棋手的估计counterfactual值中减去实际总和的一半。这整个计算是可微分的，可以用梯度下降法训练。该网络的输入是作为玩家总筹码的一部分的pot大小和作为公共牌的函数的玩家范围的编码。像传统的抽象方法(27, 28, 33)一样，通过将手数聚类到1000个bucket中进行编码，并将其输入为bucket上的概率向量。该网络的输出是每个玩家和手牌的counterfactual值向量，解释为赌注大小的分数。



##### Training

turn网络是通过1000万个随机生成的turn游戏来训练的。这些turn游戏使用了随机生成的范围、公共牌和随机pot大小。每个训练游戏的目标counterfactual值都是通过求解游戏产生的，玩家的操作被限制在fold, call, a pot-sized bet, and an all-in bet，但没有牌的抽象。flop网络也是用100万个随机生成的flop游戏进行了类似的训练。然而，目标counterfactual值是用我们的深度限制求解程序和我们训练的turn网络计算出来的。这些网络是用Adam梯度下降优化程序(34)和Huber损失(35)训练的。





### Evaluating DeepStack



![Fig. 4 – Poker Match Results](2020-04-16-DeepStack.assets/DeepStack-F4.jpg)



### Exploitability





### Discussion

DeepStack在HUNL上击败了职业扑克玩家，这个游戏的规模类似围棋，但在信息不完善的情况下，增加了复杂性。它用很少的领域知识.  DeepStack代表了大型不完全信息序列博弈的近似解的范式转变。近20年来，抽象和对完整策略的离线计算一直是主流方法（33，40，41）。DeepStack允许计算集中于决策时出现的特定情况，并使用训练的值函数进行计算。这是推动了完美信息游戏成功的两个核心原则，尽管在这些场景中实现起来在概念上更简单。因此，大型的完美信息博弈和不完美信息博弈之间的鸿沟基本已经被缩小。





### Local Best Response of DeepStack

DeepStack的目标是近似纳什均衡，即产生一个低可利用性的策略。HUNL的大小使得显式的最佳响应计算变得困难重重，因此无法测量精确的可利用性。一个常见的替代方案是将两种策略相互对弈。然而，在不完全信息博弈中，一对一的性能已经多次被证明是对均衡近似质量的不好估计。例如，在 "石头-纸-剪刀 "博弈中，一个确切的纳什均衡策略与一个几乎都在玩 "石头 "的策略对弈。结果是平手，但它们的博弈优势在可利用性方面却大相径庭。

引入局部最佳反应（LBR）作为一种寻找策略可利用性下限的技术，证明了HUNL中存在同样的问题。LBR是一种简单而又强大的技术，可以在HUNL中产生一个策略可利用性的下限(21)。它探索了一组固定的选项，以找到一个针对策略的 "局部 "好的行动。虽然更多的选项似乎是很自然的，但事实并非总是如此。更多的选项可能会导致它找到一个局部好的行动，从而错过了未来利用对手更大的缺陷的机会。事实上，LBR有时会导致在早期回合不考虑任何下注的情况下，会导致更大的下注量，从而增加pot的大小，从而增加策略未来失误的幅度。最近，LBR被用来表明，**基于抽象的代理有显著的可利用性**（见表3）。前三个策略是最近的年度计算机扑克比赛中提交的。它们都同时使用了牌和动作抽象，在所有测试的情况下，都发现它们比简单地fold每一局棋的可利用性更强。

至于DeepStack，在LBR可用的所有测试设置下，它没有发现任何可利用的漏洞。



### DeepStack Implementation Details

#### Deep Counterfactual Value Networks

DeepStack使用了两个反事实值网络，一个用于flop，一个用于turn，还有一个辅助网络，在flop前结束时给出反事实值。为了训练这些网络，我们在flop和turn开始时生成了随机的扑克情况。每个扑克情况都由pot大小、双方玩家的范围和发牌的公共牌来定义。完整的下注历史是不需要的，因为彩池和范围是一个足够的表示。网络的输出是反事实值的向量，每个玩家一个。输出的值被解释为pot大小的分数，以提高对state的概括性。







![image-20200517042533506](2020-04-16-DeepStack.assets/image-20200517042533506.png)



![image-20200517042546367](2020-04-16-DeepStack.assets/image-20200517042546367.png)















