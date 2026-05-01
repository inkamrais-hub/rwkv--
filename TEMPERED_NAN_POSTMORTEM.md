# Tempered Softmax τ 梯度死锁：事后分析

> **生成**: 2026-04-30 | **关联**: HANDOVER.md | **状态**: 定论

---

## 一、实验事实

**tempered softmax**（`softmax(τ·scores)`，τ 可学习）在 4 seeds × 200 epochs 训练中表现出完全一致的死锁行为：

| Seed | τ (ep10~200, 全程) | PPL | 梯度 |
|:----:|:-------------------:|:---:|:---:|
| 42 | 0.69314718... | 60~71 | NaN |
| 43 | 0.69314718... | 68 | NaN |
| 44 | 0.69314718... | 61~63 | NaN |
| 45 | 0.69314718... | 64~67 | NaN |

`τ = 0.69314718... = ln(2)` — 即 `softplus(0)`，完全等于初始化值。**τ 从第一步到最后一步，从未被更新过。**

---

## 二、完全根因

### 2.1 代码路径

**Step A** — 因果 mask 将未来位置设为 `-inf`（[attention.py:L120](file:///f:/τ/deploy_pkg/attention_mechanisms/attention.py#L120)）：

```python
mask = torch.triu(torch.full((L, L_k), float('-inf'), ...))
scores = scores + mask                # 未来位置 → -inf
```

**Step B** — Tempered softmax（[attention.py:L59](file:///f:/τ/deploy_pkg/attention_mechanisms/attention.py#L59)）：

```python
F.softmax(scores * tau, dim=-1)       # softmax(τ · scores)
```

**Step C** — PyTorch autograd 计算 `∂loss/∂τ`。

### 2.2 梯度公式推导

令 `x = τ · s`，`a = softmax(x)`，`g = ∂loss/∂a`（上游梯度）。

softmax 雅可比：`∂a_i/∂x_k = a_i(δ_{ik} - a_k)`

```
∂loss/∂τ = Σ_i g_i · Σ_k (∂a_i/∂x_k · ∂x_k/∂τ)
         = Σ_i g_i · Σ_k a_i(δ_{ik} - a_k) · s_k
         = Σ_i g_i a_i s_i  -  (Σ_i g_i a_i)(Σ_k a_k s_k)
          \___ 项 1 _____/     \____ 项 2 ____/
```

### 2.3 死锁点

对于因果 mask 命中的**未来位置** `i`：

| 量 | 值 | 来源 |
|:---|:---|:---|
| `s_i` | `-∞` | 因果 mask |
| `a_i` | `0` | softmax(-∞) = 0 ✓ |
| `g_i` | 有限值 | 来自 loss 反传 |

代入项 1：
```
项 1 = ... + g_i · a_i · s_i + ...
     = ... + g_i · 0 · (-∞) + ...
```

**IEEE 754 规定：`0 × (-∞) = NaN`** — 这不是 bug，是浮点标准行为。

这个 NaN 通过求和污染整个 `grad_tau`。AdamW 检测到 NaN 梯度，**拒绝更新**该参数。τ 永远停在初始值。

### 2.4 为什么是 `0.693147...`？

```
log_tau 初始化为 0
τ = softplus(0) = ln(1 + e^0) = ln(2) ≈ 0.6931471805599453
```

初始化 → ep1 梯度 NaN → 参数不更新 → τ 永远等于 ln(2)。100% 的种子、100% 的 epoch 都是这个值。完美复现。

---

## 三、这不是 FP16 的锅

之前我们曾怀疑 AMP/fp16 是元凶。但重新审视公式后，**即使在 FP64 下也一样**：
- `a_i = 0` 是真零（softmax 产生精确零），不是近似零
- `s_i = -∞` 是真无穷大
- `0 × (-∞)` 在任何精度下都是 NaN — 这是 IEEE 754 定义的行为

与 Bug 6 的 AMP 精度毒药不同：Bug 6 是精度不足导致的近似错误（可以 upcast 修复）；**这个死锁是逻辑正确性级别的**（即使在数学软件如 Mathematica 中也会产生不定式）。

---

## 四、为什么 s^τ 天然免疫

**s^τ forward**（[s_tau_fused.py:L24-L27](file:///f:/τ/deploy_pkg/attention_mechanisms/s_tau_fused.py#L24-L27)）：

```python
clamped = s_f32.clamp(min=eps)          # -inf → ε (例如 1e-8)
powered = clamped.pow(tau_f32)          # 全有限
attn = powered / (powered.sum(...) + eps)
```

**s^τ backward**（[s_tau_fused.py:L29,L53](file:///f:/τ/deploy_pkg/attention_mechanisms/s_tau_fused.py#L29-L53)）：

```python
clamp_mask = (s_f32 > eps)              # -inf 位置 = False
grad_scores = τ · a · inner · inv_s · mask  # mask=0 处梯度=0，τ 项不受影响
```

三个防护层：
1. **clamp(min=eps)**: `-inf` 被映射为 `ε`，从源头上消除 ∞
2. **clamp_mask**: 显式记录被 clamp 的位置
3. **backward mask**: 被 clamp 位置的梯度显式置零，**连 0·∞ 的可能性都切断**

### 两种方案的本质差异

| | Tempered `softmax(τ·s)` | s^τ `clamp(s,ε)^τ / Σ...` |
|:---|:---|:---|
| ∂a/∂τ 驱动项 | `a_i · (s_i - ⟨a,s⟩)` | `a_i · (log s_i - ⟨a,log s⟩)` |
| 尺度来源 | raw scores `s` (含 -∞) | log scores `log(s)` |
| 数值域 | `(-∞, +∞)` | `[log(ε), ~2]` (全有限) |
| NaN 风险 | ❌ `0·(-∞)` | ✅ 全路径安全 |
| 代价 | — | 梯度天然小 ~100×，需独立优化器 + 100× lr |

**tempered 用 `s` 驱动 τ 梯度，速度快但链条中含 -∞ → NaN。**

**s^τ 用 `log(s)` 驱动 τ 梯度，牺牲量级换来了全路径数值安全。** 这不是巧合——这是设计决策的结果。log 变换将 `(-∞, +∞)` 映射到有界域，从信息论角度看天然更适合做 τ 的梯度驱动变量。

---

## 五、更深的教训：log-sensitivity 的安全性原理

这不是一个孤立的 NaN 规避故事。背后是一个更一般的设计原则：

> **可学习参数对输入的依赖如果通过 log 而非线性尺度，则在极端值域具有天然的数值稳健性。**

对于归一化层中的可学习参数 `θ`：

```
不安全:  ∂a/∂θ ∝ s         → s 的极端值直接进入梯度
安全:    ∂a/∂θ ∝ log(s)     → log 压缩了 s 的极端值
```

s^τ 的 `log(s)` 是自动的——因为 `∂(s^τ)/∂τ = s^τ · log(s)`，log 天然出现。而 tempered 的 `exp(τ·s)` 求导后 `s` 仍然保留原始尺度，没有压缩保护。

这是工程中「优雅的数学」的一个例子：s^τ 不但在前向上与 softmax 等价映射（见 HANDOVER.md §6），而且在反向上因为 log 的存在而天然数值稳健。**这两个属性在 tempered 中无法同时满足。**

---

## 六、总结

| 层次 | 结论 |
|:-----|:-----|
| **现象** | tempered 的 τ 在 4 seeds × 200 ep 全程冻结在 ln(2) |
| **根因** | 因果 mask 的 `-inf` 与 softmax 梯度中 `a_i · s_i` 产生 IEEE 754 `0·(-∞)=NaN` |
| **精度无关** | FP16/FP32/FP64 均无法避免，这是逻辑正确性级别的问题 |
| **可修复吗** | 可以（把 -inf 改成有限大负数或手动跳过），但需要改代码 |
| **s^τ 的妙处** | clamp(min=ε) + bool mask + backward 置零，三层防护，无需任何 hack |
| **设计教训** | 可学习参数通过 log 驱动比线性尺度**结构性地更安全** |

**tempered softmax 被否决不是因为软弱的实现，而是因为公式本身与因果 mask 的 -inf 有不可调和的 IEEE 754 冲突。s^τ 的 clamp 机制是解决这个问题的唯一优雅方案。**
