## Chapter 6 Temporal-Di↵erence Learning
时序差分学习是蒙特卡洛方法和DP的结合：
* 与蒙特卡洛方法一样，TD方法可以直接从原始经验中学习，而无需环境动态模型。
* 与DP一样，TD方法部分基于其他学习估计更新估计，而无需等待最终结果（它们是自举）

### 6.1 TD Prediction
TD和蒙特卡罗方法都使用经验来解决预测问题。最简单的TD方式TD(0)：
$$ V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] $$

在TD(0)更新中，括号中的数量是一种误差， 衡量$S(t)$的估计值与更好的估计$R_{t+1}+ \gamma V(S_{t+1})$之间的差异，被称为TD误差。

### 6.2 Advantages of TD Prediction Methods

* TD方法比DP方法的优势在于前者不需要环境模型，其奖励和下一状态概率分布。
* TD方法相对于蒙特卡罗方法的下一个最明显的优势是它们自然地以在线，完全实时的方式实现。

#### guarantee convergence
对于任何固定策略$\pi$，已经证明TD(0)收敛到$v_{\pi}$。

#### effiency comparsion
没有确切的证明，但是通常发现TD方法比constant-$\alpha $MC方法在随机任务上收敛得更快。

### 6.3 Optimality of TD(0)
#### Bash update
批量更新，在处理完每批完整的训练数据后才会进行更新，对于以上两个式子，可以通过批量更新来进行更新操作：
$$ V(S_t) \leftarrow V(S_t) + \alpha[G_t - V(S_t)] $$
$$ V(S_t) \leftarrow V(S_t) + \alpha[R_{t+1} + \gamma V(S_{t+1}) - V(S_t)] $$
* 每一步 t 我们仍按照原来的步骤做计算，求出增量（也就误差值）。
* 先不执行更新，而是将增量（误差值）累积起来。
* 当一整批训练数据都按上述步骤处理完后，再统一将增量更新到目标值上。

#### Maximum-likelihood Estimate 
通常参数的最大似然估计是其生成数据的概率最大的参数值。在这种情况下，最大似然估计是从观察到的事件中以明显方式形成的马尔可夫过程的模型： 
从i到j的估计转移概率是从i到j的观察到的转变的分数，以及相关联的预期奖励是在这些转变中观察到的奖励的平均值。给定此模型，如果模型完全正确，我们可以计算值函数的估计值，该估计值将完全正确。因为它等同于假设潜在过程的估计是确定的而不是近似的。通常，批量TD(0)收敛于确定性等价估计。
尽管确定性等价估计在某种意义上是最优解，但直接计算它几乎是不可行的。 因为某些状态需要$n^2$或更高数量级别的空间和复杂度，难以储存和计算。

### 6.4 Sarsa: On-policy TD Control

TD(0)下状态价值收敛的定理适用于相应的动作价值算法：
$$ Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha[R_{t+1}+\gamma Q(S_{t+1},A_{t+1})−Q(S_t,A_t)$$

回合由一系列状态和状态-动作对组成，此规则使用五元组事件的每个元素$(S_t,A_t,R_{t+1},S_{t+1},A_{t+1})， 它们构成从一个状态-动作对到下一个状态-动作对的转换。这个五元组产生了算法的名称Sarsa

给出基于Sarsa预测方法的在策略控制算法，正如在所有策略方法中一样，我们不断估计行为策略$\pi$的$q_{\pi}$，同时将 $\pi$ 改为 $q_{\pi}$ 的贪婪:
$$ Q(S,A) \leftarrow Q(S,A)+\alpha[R+\gamma Q(S',A')−Q(S,A)] $$

### 6.5 Q-learning: Off-policy TD Control
Q-learning Off-policy TD控制算法，由以下定义:
$$ Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha [R_{t+1}+\gamma max_a Q(S_{t+1},a)−Q(S_t,A_t)] $$
在这种情况下，学习的动作-价值函数Q直接近似$q^*$，即最佳动作-价值函数，与所遵循的策略无关。 

### 6.6 Expected Sarsa
考虑与Q-learning一样的学习算法，区别在于其考虑到当前策略下每个动作的可能性，使用预期值而不是最大化下一个状态-动作对。 也就是说，考虑具有如下更新规则的算法:
$$ Q(S_t,A_t) \leftarrow Q(S_t,A_t)+\alpha [R_{t+1}+\gamma \mathbb{E}_{\pi}[Q(S_{t+1},A_{t+1})|S_{t+1}]−Q(S_t,A_t)] \leftarrow Q(S_t,A_t)+\alpha [R_{t+1}+\gamma \sum_a \pi (a|S_{t+1})Q(S_{t+1},a)−Q(S_t,A_t)] $$

### 6.7 Maximization Bias and Double Learning
#### double learning
* 将样本分划为两个集合，并分别学习出独立的估计，简记作$Q_1(a)$ $Q_2(a)$，两者均是对真实值 $q(a)$ 的估计。
* 用其中一个估计值来决定最优行动 $A^* = argmax_a Q_1(a)$ 。
* 通过另一个估计值来计算最优行动对应的值函数 $Q_2(A^*) = Q_2(argmax_a Q_1(a))$。$Q_2(A^*)$ 是无偏估计，这是因为 $\mathbb{E}[Q_2(A^*)] = q(A^*)$ 。
* 还可以重复一遍上述过程，并替换两个集合，得到另一个无偏估计 $Q_1(argmax_a Q_2(a))$ 。

### 6.8 Games, Afterstates, and Other Special Cases
总是有一些特殊的任务可以通过专门的方式得到更好的处理。
那里学到的函数通常意义上既没有动作价值函数也没有状态价值函数。 传统的状态价值函数评估个体可以选择操作的状态，但是在个体移动之后，井字棋游戏中使用的状态价值函数评估棋盘位置。 让我们称这些为 afterstates，相应得价值函数为 afterstate价值函数。 