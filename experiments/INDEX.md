# Experiments Index

> 每个实验的结论一句话，不用翻代码和聊天记录。

## 共享积木

| 文件 | 说明 |
|---|---|
| `s_tau_lab.py` | **所有实验的基础库**。`stau_norm()` / `softmax_norm()` / `BanditEnv` / 绘图函数 |

## 可视化 (无需运行)

| 文件 | 类型 | 结论 |
|---|---|---|
| `s_tau_viz.html` | 交互网页 | 双击浏览器打开，拖 τ 滑动条看注意力怎么从均分变尖锐 |

## Bandit 实验

| 脚本 | 输出图片 | 结论 |
|---|---|---|
| `tau_bandit.py` | `tau_bandit_results.png` | τ 增大 → regret 单调下降。基础验证。 |
| `tau_bandit_vs_softmax.py` | `duel_static.png` `duel_nonstationary.png` `duel_decomposition.png` | **三部对决**。静态: Softmax 胜 6%（全正分）；非平稳: s^τ 恢复快 10%；分解: s^τ 在烂臂上浪费少 1/3。 |
| `tau_bandit_negative.py` | `duel_negative_scores.png` | **核心实验**。分数减全局均值后一半臂为负分 → s^τ clamp 毙掉烂臂 → **少 35% 后悔**。这是 s^τ 结构优势的最强证据。 |

## 注意力可视化

| 脚本 | 输出 | 结论 |
|---|---|---|
| `tau_attention_gif.py` | `tau_attention.gif` `tau_attention_grid.png` `tau_vs_softmax.png` `tau_entropy.png` | τ 从 0.5 扫到 8.0，热力图从模糊变锐利。s^τ vs softmax 直接对比 + 熵曲线。 |

## 已归档 (archive/)

| 脚本 | 归档原因 |
|---|---|
| `tau_rl_games.py` + 产出 | 三场 RL 小游戏（网格/上下文/扑克）无差异——clamp 条件未触发 |
| `tau_rl_qlearning.py` + 产出 | Q-Learning 网格: Softmax T=0.5 最优。TD 学习自己分开了正负 Q，clamp 无额外收益 |

---

## 外地项目 51 模型关键结论 (F:\gsa-epxa11111\epx-b112)

| 发现 | 证据 |
|---|---|
| **RoPE 推高 τ** (dh16: +140%, dh32/64: +29%) | dh×PE 8 配置扫描 |
| **τ 是配置决定的吸引子** (8 seeds, std=1.97%) | 相同配置不同种子收敛到 τ=3.84±0.076 |
| **PPL < 20 时 τ 无法学习** | char-level(PPL=1.2) vs BPE(PPL≈33) vs 全尺寸(PPL>100) |
| **τ(L) 非单调** (L256 峰值, L1024 谷值) | L 扫描 128~2048 |
| **每层 τ U 型分布** (浅/深层高, 中层低) | per-layer τ 提取 |
