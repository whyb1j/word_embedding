## 任务描述
分别基于SVD（Singular Value Decomposition）分解以及基于SGNS（Skip-Gram Negative Sampling）两种方法构建词向量并进行评测。
## 如何运行
1. SVD
在jupyter中直接运行即可，注意
- `save_dir`指定了保存路径
- 运行时，会将过程中的一些变量和最后的结果储存到该路径，以便减少不必要的等待。
2. SGNS
运行main.py
- 需要import wandb库，并且注册该网站的账号
- 如果不满足上述条件，请自行修改main.py中的逻辑，去掉wandb框架即可。

  
