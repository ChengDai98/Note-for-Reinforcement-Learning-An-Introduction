# Note for Reinforcement Learning: An Introduction
## Chapter One: Introduction
### **What is RL**
指的 一类问题和解决问题的方式，也是一种研究领域。
将状态映射到动作上，通过策略行动，来获得最大的回报。动作会影响奖励，需要通过尝试各种动作来决定行动(trail-and-error)。之前的动作会对之后的行为状态有所影响，~~完全没法子动态规划么(dynamic programming)！~~
### **Methods**
动力系统理论(Dynamical systems theory) </br>
马尔科夫决策过程(Markov decision processes) </br>

### **Supervised && Unsupervised && Reinforcement Learning**

#### *Three Learning paradigms(include philosophy and methods)*
#### 三种学习范式
#### **监督学习(Supervised Learning)**
需要标记数据，得到标签(label)，对标签做分类(classification)和回归(regression)问题，来解决不在训练集中的数据的回归，分类问题。
#### **无监督学习(Unsupervised Learning)**
寻找没有标签数据集合的结构
#### **强化学习(Reinforcement Learning)**
*feature*：<br/>
交互(interacting)学习，权衡(trade-off)探索(exploration)和使用(exploitation)，使用、利用之前的经验，探索未知的状态，途中不断的失败，来获得最大收益(reward signal)。<br/>
RL明确的考虑了目标导向(goal-directed)的与不确定环境的交互整个问题(什么意思没搞懂)，在面对子问题时，能够进行有效的规划，对解决某个子问题所产生的影响有所相应，即相互作用。<br/>
强化学习采取一些tasks(怎么讲？任务？)，具有完整的，交互式的寻求目标(goal-seeking)的个体。拥有明确的目标，通过选择动作影响环境，通过权衡探索和利用环境达到决策目的，获得最大收益。能够和大环境，大系统交互(机器人，SC2，围棋etc)。<br/>
~~能解决维数灾难？？？这可太列害了......很多dp问题都可以用RL解决了，太好力。~~

### **About "strong methods" and "weak methods"**
基于特定领域的知识；基于一般性原则

### **Elements of RL**
#### *四个主要的要素：法则或者策略(policy)，奖励(回报)信号(reward signal)，价值函数(value function)，环境模型(model of the enviorment)*
#### *策略法则*
状态到采取行动措施的过程映射，是一个随机过程。
#### *奖励信号*
给出RL的目标，在决策过程中，环境向个体发送奖励信号。智能体的目的是，通过各种决策，在最后，来让个体得到最大化奖励(即最优的)。奖励信号一种在短时间内就能得到的价值，短期利益。
#### *价值函数*
决定长期利益，即考虑遵循某些状态可获得的具有较高价值的长期奖励。由于每次抉择后都会给未来的状态产生一定的影响，如何考虑价值函数成为一个问题。
#### *环境模型*
智能体所在环境的仿真，用来预测下一个状态(环境是如何变化的)，为未来规划。

### **Limitations and Scope**
怎么定义状态，设计状态和行为

### **Tic-Tac-Toe**
感觉结合例子能够刚好的理解四个要素。<br/>
通过编码确定(当前棋盘环境)游戏状态 <br/>
每个状态->胜率估计 <br/>
可以通过搜索树来搞吗 <br/>
如果用价值函数的RL方法来干的话尝试编码 <br/>

# Part I: Tabular Solution Methods
表格解决问题方法：状态和动作空间足够小（同时可以离散化表示？），这种问题通常能够找到最佳的价值函数和策略(policy)。
同时也存在着不能用表格解决的方法，比如：涉及维度爆炸或状态连续。

## Markov Decision Processes
马尔可夫决策过程以及解决其问题的三种方法：动态规划(Dynamic Programming)，蒙特卡罗方法(Monte Carlo)和时序差分(temporal-different)。

## Chapter 2 Multi-armed Bandits
多臂赌博机问题。</br>
强化学习与其他学习方式区别，或者RL的最重要的*特征*，在于其他学习给出了正确的指导（比如给出lable，给出正确的方向，答案），RL是同通过探索和利用（不断地尝试和试错 trial-and-error）来找出最优的答案。其中RL会对做出的动作进行评估，这种东西叫*评价性反馈*(Evaluative Feedback)。</br>
*关联性*(Nonassociative)讨论</br>
相比于能应用于实际上的RL，本章采用了一种更简单的场景：无需考虑每一步行动之间的影响，以及环境对行动的影响。

### 2.1 A K-armed Bandit Problem

#### **k臂老虎机问题背景**

* 你可以选择k种行动(action)中的一种
* 每种动作(action)都对应了一个数值奖励(nurmarical reward)，数值奖励是一个随机变量，符合某一种分布(distribution)，当然agent不知道这个分布。
* 目标是在一定时间内最大化奖励(奖励是一个累计值)。


#### **老虎机问题的数学表述：**

$$ q_*(a) \dot{=} \mathbb{E}[R_t|A_t = a] $$

* $q_*(a)$：是给定动作$a$选择的理论期望值，及对应分布的期望。
* $a$：一个任意的行动。
* $A_t$：是在时间步$t$选择的动作$A_t$。
* $R_t$：选择$A_t$相应的奖励为$R_t$。

在这个问题中，智能体并不知道每一个动作的回报分布，它通过不断地探索和利用来获得最优的评估体系。吧时间步$t$的动作$a$的估计值表示为$Q_t(a)$，我们希望$Q_t(a)\approx q_*(a)$此时出现了两种行为动作，分别是*贪婪*(greedy)和*探索*(exploration)。
*利用*贪婪行为利用了智能体对当前动作价值的了解，并非看重长远利益。
探索行为是选择非常规动作，即选择了一个在当前动作中回报并非那么大的。探索的意义在于找到更有价值的动作和信息。

### 2.2 Action-value Methods
动作价值方法是估算动作的价值，并使用这种估算结果来采取行动来选择策略。
#### **Sample-Average**
这是一种最为简单的方式，即平均奖励法：
$$Q_t(a) \dot{=} {\sum_{i=1}^{t-1}R_i \cdot \mathbb{1}_{A_i = a} \over \sum_{i=1}^{t-1} \mathbb{1}_{A_i = a}}$$
可以看出当样本足够多的时候，根据大数定律，$Q_t(a)$会收敛于$q_*(a)$。
#### **Greedy action**
贪婪动作为有最高估计价值的动作，数学表达为：
$$ A_t \dot{=} argmax_a Q_t(a) $$
可以预见的是，一直选择贪心策略的话，可以获得局部最优解，但是很难得到全局最优解。明显，此时可以通过一定的探索行为来换取长远的利益。

#### **$\epsilon$-greedy action**
即以 $1-\epsilon$ 的概率采取greedy action，以 $\epsilon$ 采取另一个行动 $a$。</br>
* 这个策略能够兼顾探索与利用。
* $\epsilon$ 的取值会影响收益，因此要给出一个合适的 $\epsilon$ 。

### 2.3 The 10-armed Testbed
本章来测试以上策略。

### 2.4 Incremental Implementation
增量问题实现，给出了一个增量形式的动作价值估计算法，基本的数学形式为：
$$ Q_n \dot{=} \frac{\sum_{i = 1}^{n - 1}{R_i}}{n-1} $$
* $R_i$：表示在第$i$次选择动作$a$之后的汇报
* $Q_n$：表示在前$n$次实行动作$a$的经验基础上，对下一次再选到$a$的预测值。

#### Optimization
对算法进行了优化，其数学表示可以变换为：
$$ Q_{n + 1} = Q_n + \frac{1}{n}[R_n - Q_n] $$











