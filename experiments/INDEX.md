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
| `tau_bandit_negative.py` | `duel_negative_scores.png` | **核心实验**。分数减全局均值后一半臂为负分 → s^τ 毙掉烂臂 → **少 35% 后悔**。这是 s^τ 结构优势的最强证据。 |

## 注意力可视化

| 脚本 | 输出 | 结论 |
|---|---|---|
| `tau_attention_gif.py` | `tau_attention.gif` `tau_attention_grid.png` `tau_vs_softmax.png` `tau_entropy.png` | τ 从 0.5 扫到 8.0，热力图从模糊变锐利。s^τ vs softmax 直接对比 + 熵曲线。 |

## 等价性验证

| 脚本 | 结论 |
|---|---|
| `_equiv_experiment.py` | ❌ 已丢失（远端释放）。s^τ ↔ softmax 等价映射，正向/反向误差 < 1e-5。clamp 版路径，已逐步迁移至 softplus。等价性定理的重新证明见 THEORY.md 和 SOLO_CHALLENGE_DRAFT.md §3.1。 |

## 基底函数 φ(s) 对比

| 脚本 | 结论 |
|---|---|
| `tau_base_functions.py` | **17 种 φ(s) 综合对比**。静态安全: 15/17 通过 (mask 正序 + 正值 + 保序)。τ 训练: 全部可学 (Adam+clip)。Top3: **1+tanh/2+ε** (最低 loss)、**softsign+1+ε**、**tanh+1+ε**。softplus+ε 仍排中游但最稳定 (φ 范围最窄)。log(1+\|x\|)+ε 因 mask 不安全排除，exp(-\|x\|)+ε 因保序失败排除。 |
| `../scripts/_vit_phi_trio.py` | **真实数据端到端对比 (2026-05-10)**。ViT from scratch, CIFAR-10 10ep: **tanh 68.80%** > softplus 68.40% > softsign 68.37%。FashionMNIST 5ep: softplus 78.00% > softsign 76.02% > tanh 75.12%。关键发现: tanh 的窄 φ 推高 τ 到 3.82 → 隐式正则化 → E6 后反超 softplus。但差距仅 0.4%，不值得冒险换基底。 |

## 附属分支: 对比注意力 (contrastive/)

> 已移至 `contrastive/` 子目录。详见 `contrastive/BRANCH.md`。

| 脚本 | 结论 |
|---|---|
| `contrastive/tau_contrastive.py` | **Phase 0~0-c 合成验证完成**。正反头在 τ≥4.0 正交 cos=0.017。**Phase 0-c 关键发现**: 联合训练隐式正则化 τ_pos，loss 改善 62-74%。静态融合不如单头（λ\*=1.0）。 |
| `contrastive/tau_contrastive_phase1.py` | **Phase 1 Qwen3 真实模型: 未成功**。τ_neg 符号约束错误（应为负值但 softplus 强制为正）+ gate 卡 0.5。gradcheck 通过。需修复初始化后重跑。 |
| **vs 主线** | 参数 ×3（16τ→48），计算 ×1.5，闭式解可算 τ₊/τ₋ 但 gate 需前传搜索。主线单头 s^τ 已验证有效，对比分支待修复。 |

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

---

## τ 项目近期进展 (v5 softplus → 统计 τ* → Qwen3 生成 → 微调)

| 阶段 | 结论 | 状态 |
|---|---|---|
| **v5 softplus 迁移** | φ(s)=softplus(s)+ε 替代 clamp，梯度天然有限、无 NaN、无爆，因果 mask 安全 | ✅ 完成 |
| **统计 τ* 闭式解** | τ*=Cov(s, log φ(s))/Var(log φ(s))，零搜索定位 softmax 等价点，O(1) 每头 | ✅ 完成 |
| **等价性定理统合** | s^τ 包含 softmax 作为 τ=τ*(s) 切片。无论 φ=clamp/softplus/exp，正函数等价性恒成立 | ✅ 完成 |
| **Qwen3-1.7B 生成质量** | 统计 τ* 零样本注入 → PPL 2.556 vs softmax 2.630（胜出）；网格 τ* 未超越统计 τ* | ✅ 完成 |
| **Token 匹配率 87.9%** | s^τ 在 12% 位置选择了不同 token，PPL 更低。统计 τ* 与网格 τ* 生成完全一致 | ✅ 完成 |
| **微调 2000 步收敛** | 统计 τ* 初始化 → 5 分 24 秒 → 终 PPL 1.127 = softmax | ✅ 完成 |
| **算子性能** | **v10h2f16 (⭐ 主用)**: 原生 fp16 + __half2 向量化，真实 fp16 管线 18% 快于 v9。<br>v11 half4: 4-way 尝试 → 0.95× vs v10 (寄存器压力，已回退)。<br>fp32 路径: v9 float4 保留。 | ✅ v10h2f16 已稳 |
| **Triton 融合注意力** | **s_tau_triton.py**: FlashAttention 分块算法 + s^τ 变换。Forward 5-9× 加速（RTX 3060）。Backward 用 PyTorch eager（cos=1.0 正确性已验证）。完整训练 f+b 暂无加速（backward 物化 L²）。Triton tiled backward 内核已写但有 fp32 精度问题，待修复。 | ✅ forward 已稳 |
| **CUDA C++ 融合 Backward** | **s_tau_fused_bwd.py v7**: 2-kernel backward（dK/dV D-tiling，dQ smem atomicAdd）。cos=1.0 全正确。**性能**: 标量 FMA 无法竞争 Tensor Core，0.09-0.66×。已验证但不用于生产。 | ✅ 正确，❌ 性能不如 eager |
| **torch.compile Backward ⭐** | **s_tau_compiled_bwd.py**: `@torch.compile` 编译 eager backward。Inductor 融合 softplus/sigmoid/chain rule + 优化 matmul 调度。**backward 加速 1.7-2.4×** vs eager。**端到端 (Triton forward + compiled backward): 1.3-3.4× 加速**。需 `triton-windows==3.1.0` 匹配 PyTorch 2.5.1。 | ✅ 生产方案 |
| **待做** | 多架构验证 (LLaMA/Mistral/Gemma)、长序列外推对比、注意力有效秩分析、剪头敏感度实验 | 🟡 推进中 |
| **minimind-3 交叉验证** | 64M 欠训练模型上 stat τ* 失效（PPL 1280 vs softmax 568），grid 搜索替代有效（τ=12/0.1 两极分化），证明闭式解公式前提是充分训练 | ✅ 完成 |
| **causal mask bug 修复** | -∞ 值污染 Cov 计算 → NaN τ，修复方法：mask 前算 τ，mask 后应用 s_tau_norm | ✅ 完成 |
| **SDPA 底层拦截注入** | 通过 `ALL_ATTENTION_FUNCTIONS["sdpa"]` 注册自定义函数，在 Q/K/V 层面应用温度缩放 `scores *= tau`。关键陷阱: 输出需 `.transpose(1,2)` 从 [B,H,L,D] 转 [B,L,H,D]。**Qwen3-0.6B (28/28 FA层): tau=0.7 → PPL -9.31%**。Qwen3.5-0.8B (6/24 FA层): tau=5.0 → PPL -1.60%。tau<1（平滑）> tau>1（尖锐）。τ 扫描的 τ* ≠ PPL 最优 τ。 | ✅ 完成 |
| **Qwen3.5 全架构注入** | SDPA 拦截 6 FA 层 + 逐层 patch 18 GatedDeltaNet 层。**关键发现**: query scaling 只有 -1.15%（Q@K 值太小 0.013），改 **g-gate scaling**（衰减门）→ **PPL -3.52%** (g-tau=2.0)。方向与 softmax 相反: tau>1（更积极遗忘）有效。每层存自己的函数引用 → 必须逐层 patch。bool mask + 手写 softmax = NaN → 必须用原生 SDPA。 | ✅ 完成 |
| **Mamba SSM 注入探索** | Mamba-130M, 测试 dt/B/C/gate 四个注入点。dt/B/C 完全焊死（任何偏离 1.0 都变差）。**gate=1.1 → PPL -1.13%**（唯一改善点，窗口仅 ±10%）。结论: s^τ 效果与架构冗余度正相关——纯 SSM 无冗余，推理注入天花板极低。出路: 训练时融合 s^τ 让模型自学习最优 tau。 | ✅ 完成 |
| **Mamba in_proj 修正 ⭐** | 深度扫描发现 **in_proj (输入投影)** 才是 Mamba 的主注入点，不是 gate。**in_proj=1.05 → PPL -5.78%** (EN+ZH 15文本)，效果 5× gate。联合 in_proj+dt_proj → -6.05%。验证: 注入点选择 >> 注入强度 (与 DeltaNet g-gate 发现一致)。RWKV/Conv 从零训练小模型测试: RWKV -0.15%, Conv -0.15% (PPL≈1.1 太低无法改善)。 | ✅ 完成 |
| **RWKV-4-169M 真实模型 ⭐⭐⭐** | 原生 .pth 权重 + log-space WKV 递推。**time_decay=0.3 → PPL -67.95%**，att.key=5.0 → -67.40%。**联合 decay+key → -85.49%** (PPL 454.78 → 65.97)。RWKV 是 s^τ 响应最强架构，因为 WKV 有独立遗忘+写入旋钮。从零训练小模型 -0.15% vs 真实模型 -85%: 模型必须充分训练。Jamba-tiny-random (随机初始化) 注入无效 (-0.30%) 验证此结论。 | ✅ 完成 |
| **RWKV ≤1B 全参数测试 ⭐⭐⭐** | RWKV-4-430M: baseline PPL 242.98, **decay=0.3+key=2.0 → -70.58%** (PPL 71.48)。RWKV-7-0.4B: baseline PPL 22.61, 最佳单点 key/val/rec=0.9 → -3.21%。**核心发现: s^τ 响应与架构成熟度反相关。** RWKV-4 极度敏感 (-70%), RWKV-7 几乎免疫 (-3%)。原因: RWKV-7 的数据依赖衰减 w0+w1@w2 和 k_k/k_a 调制已内化了 RWKV-4 需要 s^τ 注入才能获得的优化。生成质量: v7 所有配置连贯英文, v4 所有配置乱码。**s^τ 是架构诊断工具。** | ✅ 完成 |
| **RWKV-7 1.5B/2.9B 测试 ⭐⭐** | 目标: 验证 0.4B 发现在更大模型上是否成立。云实例 RTX 5090, 模型已下载。**实现 RWKV-7 前向时发现 6 大 bug**: (1) x_prev 应 per-layer 而非共享, (2) FFN x_prev 应是 TMix+residual 后的 xt, (3) embedding 需 ln0 预处理, (4) 权重加载时需转置 key/value/rec/output/head, (5) H/N 需从原始 r_k shape 提取后再 flatten, (6) 状态结构应为 3*tensor/layer 而非 1。v5 脚本已修复但 quicktest 仍 NaN, 怀疑 w0/w1/w2 shape 问题。实例当前不可达。 | ⏳ 进行中 |
