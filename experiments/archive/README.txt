存档说明 — 2026-05-01

以下 2 个实验脚本 + 产出物已归档，原因：

tau_rl_games.py   — 三场 RL 小游戏（网格/上下文/扑克）
                    所有配置结果无差异，clamp 未触发
                    实验设计有缺陷：选项少、噪声小、Q-learning 收敛太快

tau_rl_qlearning.py — Q-Learning 网格世界
                    所有 s^tau 配置结果相同，Softmax T=0.5 最优
                    原因：TD 学习自身已分离 Q 值正负，clamp 无额外收益

结论：s^tau 在标准 tabular Q-learning 环境无结构性优势。
      优势场景：分数含持续噪声 + 选项数量大 + 存在明确负分候选。
      代表性实验：tau_bandit_negative.py（中心化 bandit，优势 35%）。
