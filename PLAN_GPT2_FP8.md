# τ 项目 v3 规划：75M ATTH + TinyStories + FP8 加速

> **生成**: 2026-04-30 | **终版** | **平台**: PRO 6000 / 5090 (单卡, 串行)

---

## 1. 架构 & 参数

```
模型:   ATTHModel (非 GPT-2, 纯 ATTH 架构)
  dim:          768
  n_layers:     10
  n_heads:      12          (head_dim = 64)
  max_seq:      2048        (position embedding)
  训练 ctx:     512
  rope:         false       (标准 learned PE)
  params:       ≈ 72M~75M   (vocab 取决于数据集 ~128~256)

注意力归一化:
  softmax:      F.softmax(scores, dim=-1)
  s^τ:          s_tau_fused.s_tau_norm() (v4 fused operator)
  τ 初始值:     softplus(0)+1 ≈ 1.693

MLP:            SwiGLU (ffn_mult=4)
Norm:           RMSNorm (pre-norm, GPT-2 风格)
```

## 2. 数据: TinyStories

```
数据集:     roneneldan/TinyStories (hf-mirror.com)
编码:       char-level (vocab ~128~256 chars)
样本量:     前 15M 字符 → ~15M tokens
切分:       90% train / 10% val
Batch:      16 seqs × 512 = 8K tokens/step
```

## 3. 训练配置

```
┌───────────┬─────────────┬─────────────┐
│           │   softmax   │    s^τ      │
├───────────┼─────────────┼─────────────┤
│ FP8       │ ❌ (BF16)   │ ✅ (Linear) │
│ fused op  │ ❌          │ ✅ (v4)     │
│ τ opt     │ —           │ τ_lr=1e-2   │
│           │             │ τ_wd=0      │
├───────────┼─────────────┼─────────────┤
│ lr        │ 6e-4        │ 6e-4        │
│ wd        │ 0.01        │ 0.01        │
│ betas     │ (0.9, 0.95) │ (0.9, 0.95) │
│ schedule  │ OneCycle    │ OneCycle    │
│ amp       │ BF16        │ BF16        │
│ grad_clip │ 1.0         │ 1.0         │
│ epochs    │ 30          │ 30          │
│ steps/ep  │ 200         │ 200         │
└───────────┴─────────────┴─────────────┘

执行顺序: 串行 (先 softmax → 后 s^τ)
```

## 4. 时间估算

```
基于 Phase 0-2 基准 (3.2M, L=128, 200ep, s^τ=556s):

73M, L=512, 30ep:
  s^τ:   556 × (73/3.2) × (512/128) × (30/200)
       = 556 × 22.8 × 4 × 0.15 ≈ 7,600s ≈ 2.1h

FP8 ~40% 加速: 2.1 → ~1.3h

softmax (快 ~30%): ~1.5h (无 FP8)

总计 (串行): 1.5h + 1.3h ≈ 2.8h
  - 如果 FP8 不可用: ~3.6h
  - 如果 epochs=15: ~1.4h
```

## 5. 监控

```
本地监控: D:\python\python.exe -u scripts\watch_tiny.py
          SSH 读取 /root/epx/status_tiny.txt, 每5秒刷新
```

## 6. 文件清单

```
deploy_pkg/
├── attention_mechanisms/
│   ├── s_tau_fused.py          ← v4 fused (不变)
│   ├── s_tau_cuda_kernel.py    ← v5 CUDA C++ (不变)
│   ├── s_tau_triton.py         ← Triton (不变)
│   ├── model.py                ← ATTHModel (不变)
│   └── attention.py            ← StandardAttention + apply_attn_norm (不变)
├── model_tiny.py               ← ★ 75M build + opt + τ 工具
├── train_tiny.py               ← ★ 串行训练 (softmax → s^τ)
├── fp8_utils.py                ← ★ FP8 包装 + BF16 自动降级
├── benchmark_norm.py           ← 不变
└── run_parallel.py             ← 不变

scripts/
└── watch_tiny.py               ← ★ 本地监控终端
```

## 7. 远端启动命令

```bash
cd /root/epx

# 默认 30 epochs (~2.8h)
nohup python3 -u train_tiny.py --epochs 30 > tiny_train.log 2>&1 &

# 快速版 15 epochs (~1.4h)
nohup python3 -u train_tiny.py --epochs 15 --steps_per_epoch 100 > tiny_train.log 2>&1 &

# 带 FP8
nohup python3 -u train_tiny.py --epochs 30 --fp8 1 > tiny_train.log 2>&1 &
```
