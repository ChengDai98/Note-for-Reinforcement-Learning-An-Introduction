## Chapter 5 Monte Carlo Methods
蒙特卡洛方法是通过评估价值函数以及获得最优策略的一种学习方法。蒙特卡洛方法需要的仅仅是经验，即与环境进行真实的或者模拟的交互所得到的状态，动作，奖励的样本序列，所以说蒙特卡洛方法是基于对样本回报求平均的办法来解决强化学习的问题。

### 5.1 Monte Carlo Prediction
两种蒙特卡罗方法：
对于使用策略$\pi$生成的一个片段episode:
$S_0, A_0, R_1, S_1, A_1, R_2, ...,S_{T-1}, A_{T-1} R_T$
#### first-visit MC method
计算所有回合中首次访问状态s的平均回报， 以此作为$v_{\pi}(s)$的估计值，我们称之为 首次访问MC方法。
#### every-visit MC method
计算所有回合中每次访问状态s的平均回报。

根据大数定理，这些估计的平均数会收敛于期望价值。不论是首次访问MC方法还是每次访问MC方法，都会随着访问s的次数趋于无穷而收敛于$v_{\pi}(s)$。

### 5.2 Monte Carlo Estimation of Action Values
可能会有许多状态-动作对从未被访问到。如果$\pi$是一个确定性的策略，那么遵循策略$\pi$,每个状态将会仅仅观察到一个动作的回报。 为了比较所有的可能，我们需要估计每个状态所有可能的动作，而不仅仅是当前选择的动作。

从特定的状态动作对出发，对每种动作都有大于零的概率选择到。这能够保证经历无限个回合后，所有的状态-动作对都会被访问到无限次。我们称这种假设为探索开端（exploration start）。

但是它不具普遍意义，特别是当我们直接从与真实环境的交互中学习时。

### 5.3 Monte Carlo Control
使用蒙特卡洛估计来解决控制问题，即求解近似最优的策略。从一个随机的策略$\pi$开始，以最优策略和最优的动作-价值函数结束：
$$ {\pi}_0 \rightarrow^E q_{π_0} \rightarrow^I {\pi}_1 \rightarrow^E q_{π_1} \rightarrow^I {\pi}_2 \rightarrow^E ... \rightarrow^I {\pi}_∗ \rightarrow^E q_∗ $$
* $ \rightarrow^E $表示一个完整的策略评估，通过以前的回合来估计。
* $ \rightarrow^I $表示一个完整的策略提升，使用贪心策略${\pi}(s) \dot{=} argmax_a q(s,a)$

### 5.4 Monte Carlo Control without Exploring Starts

