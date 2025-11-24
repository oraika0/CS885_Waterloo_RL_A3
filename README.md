# CS885_Waterloo_RL Assignment 3
# Part I：Partially Observable RL

## 1. DRQN
> comparing DQN、DRQN on partially observable cartpole
![](image/)  
![](image/)  

### Explain  
在 partial observable cartpole 中，車輛速度與桿子角速度被從state中去除，使得DQN需要僅憑藉車輛位置與桿子角度就推論出策略，無法得知完整的物理環境資料，這樣的殘缺的狀態使DQN訓練不出有用的策略;DRQN利用LSTM，記憶了先前的狀態並學習補全了原先缺少的部分，最終使DRQN可以學到有用的策略。


# Part II  
## 1. Distributional RL
> comparing DQN、C51 on noisy cartpole  

![](image/)  
![](image/)  

### Explain
noisy cartpole 中加入了摩擦力與隨機的托拽力量，使原先的deterministic env 轉變成了 stochastic env，在這種環境中，C51 能更好的學習到環境中的機率分布，使得最終C51相較DQN有了更好的表現結果。
