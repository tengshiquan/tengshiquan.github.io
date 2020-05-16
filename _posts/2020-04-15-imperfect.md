---
layout:     post
title:      完美信息与完全信息
subtitle:   Imperfect Information VS Incomplete Information
date:       2020-04-01 12:00:00
author:     "tengshiquan"
header-img: "img/post-bg-dice.jpg"
catalog: true
tags:
    - AI
    - Game Theory 
    - Imperfect Information

---



# Imperfect Information VS Incomplete Information

博弈论里面很容易混淆的两个概念.    按照下面的说法, 可以说 imperfect 是 incomplete 的子集. 

- imperfect information: 玩家只是不知道其他玩家的具体动作action. 但知道其他玩家的剩余信息.   
  例子。两位玩家玩一个配对便士的游戏，这样，每个玩家都有一个便士，必须偷偷地转到人头或人尾，然后揭晓。如果两个便士都匹配，则玩家1获胜，如果便士不匹配，则玩家2获胜。这是一个信息imperfect的游戏，双方都不知道对方的行动，但都知道其他的一切。
- Incomplete information:  一些信息是一个或多个玩家不知道的。这可能是他们的类型、策略、偏好，types, strategies, preferences,  也可能是一种或多种信息的组合.   
  举例,  与上述游戏相同，但现在其中一个玩家可以选择使用有偏向的硬币和无偏向的硬币。在这种情况下，两个玩家都不知道另一个玩家所采取的行动，而其中一个玩家不知道另一个玩家的类型（用有偏见的硬币或无偏见的硬币进行游戏）。



另一个回答 以及 wiki 则说不是子集.  

例子   "我跟她是朋友么？"

1. **Imperfect and incomplete information**: 你不知道她打算做什么，也不知道她是怎么告诉男生她喜欢他们的
2. **Perfect and incomplete information**: 她晚上约你喝咖啡，但你不知道她是怎么告诉男生她喜欢他们的
3. **Imperfect and complete information**: 你不知道她打算做什么，但你知道她只找她喜欢的男生喝咖啡
4. **Perfect and complete information**: 她晚上约你喝咖啡，你知道她只约她喜欢的男人喝咖啡

可以看出，完美信息是指你知道对方参与 "游戏 "的所有行为，而完全信息是指你知道对方的策略+喜好+付出 strategies+preferences+payoffs 是什么。



综上, 重点知道,  imperfect 强调的是不知道 其他对手的具体 **action**  ; 对博弈论来说, 更关注 imperfect. 



<img src="/img/2020-04-15-imperfect.assets/image-20200511173928005.png" alt="image-20200511173928005" style="zoom:50%;" />



- **Incomplete information** (不完全信息): a player does not know another player’s characteristics (in particular, **preferences**);   **不知道收益函数**

- **imperfect information** (不完美信息): a player does not know what **actions** another player has taken.   **不知道动作历史**

a **dynamic game of perfect information**, each player is perfectly informed of the **history** of what has happened so far, up to the point where it is her turn to move.

**Harsanyi Transformation** : Following Harsanyi (1967), we can change a dynamic game of **incomplete** information **into** a dynamic game of **imperfect** information, by making **nature** as a mover in the game. In such a game, **nature chooses player i’s type**, but another player j is not perfectly informed about this choice.

加入随机选择玩家1类型的action, 然后玩家2不知道, 所以就是**imperfect**



#### A dynamic game of complete but imperfect information

博弈里的经典例子变体, 进入博弈.  是complete但imperfect

An entry game: the challenger may stay out, prepare for combat and enter (ready), or enter without preparation (unready).  Each player’s **preferences** are **common knowledge**.

<img src="/img/2020-04-15-imperfect.assets/image-20200511001711729.png" alt="image-20200511001711729" style="zoom:50%;" />

This is a game of **incomplete** information. But we can change it into a game of **imperfect** information by letting nature have the initial move of choosing the type of the challenger:

<img src="/img/2020-04-15-imperfect.assets/image-20200511013045812.png" alt="image-20200511013045812" style="zoom:50%;" />







## Reference

https://www.quora.com/What-is-the-difference-between-incomplete-and-imperfect-information

https://en.wikipedia.org/wiki/Complete_information



















