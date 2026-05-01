# EPX-B112 项目全量交接文档 (v5 — 算子优化 + GPT-2 实验)

> **最后更新: 2026-05-01 20:00**
> **最新文档**: THEORY.md (理论分析) | EXPERIMENT_REPORT.md (实验报告) | [TEMPERED_NAN_POSTMORTEM.md](TEMPERED_NAN_POSTMORTEM.md) (tempered 死锁事后分析) | [S_TAU_INJECTION_REPORT.md](S_TAU_INJECTION_REPORT.md) (s^τ 注入技术报告)
> **项目**: `attention_mechanisms/epx_b112_package`
> **核心问题**: **s^τ 注意力机制 + 算子工程优化 + GPT-2 级验证**
> **当前状态**: ⏸️ 暂停中 — 余额不足, 实例已释放

---

## 目录
1. [项目概述](#1)
2. [算子进化史 (v1→v5)](#2)
3. [Blackwell 兼容性报告](#3)
4. [5090 基准测试结果](#4)
5. [GPT-2 124M 训练实验](#5)
6. [s^τ ↔ softmax 等价性发现](#6)
7. [文件结构与关键脚本](#7)
8. [AutoDL 操作指南 + SSH](#8)
9. [命令速查](#9)
10. [预算与资源](#10)
11. [已知 Bug 与陷阱](#11)
12. [关键开发经验](#12)
13. [后续方向](#13)
14. [可复用组件清单](#12-b)

---

## 1. 项目概述

**核心**: 用 `s^τ` 幂律归一化（τ 可学习）替代 `softmax`，验证 τ 是否可被 SGD 学习、能否提升模型能力。

**两阶段**:
- **Phase 0-2** (已完成): 小模型 Benchmark — 速度/训练/梯度全面验证，τ 正常学习
- **Phase 3** (进行中): GPT-2 124M 级训练 — 真实数据 + 真实 tokenizer + HF 对比

**算子进化**: `v1(autograd)` → `v2(fp32 fix)` → `v3(opt)` → `v4(compiled)` → `v5(CUDA C++手写)`

---

## 2. 算子进化史

### v1 — 初始 autograd.Function
```
代码: s_tau_fused.py
实现: 纯 autograd.Function, forward/backward 全手写解析
问题: AMP 下 fp16 计算 → backward 梯度爆炸 100-500×
状态: 🔴 已废弃
```

### v2 — FP32 修复 + clamp mask
```
代码: s_tau_fused.py (备份: s_tau_fused_v2_backup.py)
修复: forward/backward 核心计算强制 fp32, 输出 cast 回原 dtype
      clamp 位置 backward mask 置零
5090: 0.54ms vs Old 0.89ms → 1.64× 加速
验证: Fused vs Old τ diff=0.017 ✅ MATCH
状态: ✅ 稳定版 (备份保留)
```

### v3 — bool mask + smart cast + fused clamp/log
```
代码: s_tau_fused.py
优化: bool mask 替代 raw_scores (75% 省), 智能 fp32 cast, 共用 safe_s
fp32 模式: 1.42× vs old (vs v2 的 1.64×, 因 v2 实测含 warmup 误差)
验证: v3-v2 FW diff=3.55e-15, tau grad diff=0.00e+00 ✅
状态: ✅ 稳定
```

### v4 — 精简版 (当前默认)
```
代码: s_tau_fused.py
简化: 去掉 raw_save (bool mask 足够), 精简代码
完整模型 5090 Phase 0: 38.46ms vs softmax 32.34ms → 1.19× vs softmax
与 softmax 差距从最初的 2.05× 压到 1.19×
状态: ✅ 当前默认算子
```

### v5 — CUDA C++ 行内编译 (待验证)
```
代码: s_tau_cuda_kernel.py
实现: torch.utils.cpp_extension.load_inline
      forward: 1 个 block-level kernel (clamp+pow+sum+norm)
      backward: 1 个 kernel (score_grad + tau 项行内归约)
特点: 不依赖 Triton, 所有架构兼容, 首次 import 自动编译
预期: 再压 5-8%, 从 1.19× → ~1.12×
状态: ⏳ 代码已写完, 5090 上等待编译验证
```

---

## 3. Blackwell (sm_120) 兼容性报告

| 方案 | RTX 5090 (sm_120) | RTX 4090D (sm_89) |
|:---|:---:|:---:|
| **v4 autograd.Function** | ✅ 完美运行 | ✅ 完美运行 |
| **v5 CUDA C++ inline** | ⏳ 待编译验证 | ✅ 预期正常 |
| **Triton v3.4** | ❌ backward kernel crash | ✅ 正常 |

### Triton 失败详细
```
现象: fwd kernel 正常, bwd kernel 调用 tl.sum → CUDA illegal memory access
根因: Triton 3.4 对 Blackwell sm_120 的 tl.sum 归约兼容性 bug
      forward 用同样的 tl.sum 却正常, 说明是 backward 特有的问题
      (可能是梯度写入和共享内存之间的竞态)
状态: 等待 Triton 更新. 不阻塞项目 — v4 已经 1.19×, Triton 最多再压 7%
```

### 跨架构代码选择策略
```python
# 自动选择最优算子:
try:
    from s_tau_cuda_kernel import s_tau_norm_cuda as norm_fn
except:
    from s_tau_fused import s_tau_norm as norm_fn
```

---

## 4. 5090 基准测试结果

### Phase 0 — 速度 (合成数据 60 类, 4×256 模型)

| 归一化 | ms/step | vs softmax | 内存 |
|:---|:---:|:---:|:---:|
| softmax | 32.34 | 1.00× | 686 MB |
| **s^τ (v4)** | 38.46 | **1.19×** | 781 MB |
| tempered | 29.52 | 0.91× | 727 MB |

### Phase 1 — 训练对比 (200 ep, 4 seeds)

| 归一化 | τ 终点 | PPL | ev@512 |
|:---|:---:|:---:|:---:|
| **s^τ** | **2.28~3.16** | 37.7~38.1 | 40.8~42.5 |
| softmax | — | 38.0~38.2 | 41~43 |
| tempered | 0.693 (NaN) | 60~68 | NaN |

### Phase 2 — 梯度分析 (100 ep)

| | s^τ | tempered | softmax |
|:---|:---:|:---:|:---:|
| ∥∇τ∥ | 0.001~0.003 | NaN | — |
| ∥∇total∥ | ~0.2 | NaN | ~0.2 |

**关键发现**: tempered (`softmax(τ·scores)`) 因因果 mask 的 -inf 产生 IEEE 754 `0·(-∞)=NaN`, τ 永久冻结。详见 [TEMPERED_NAN_POSTMORTEM.md](TEMPERED_NAN_POSTMORTEM.md)。

**Phase 0-2 总耗时: 1064s (17.7min) | 结果已保存至 project_assets/**

---

## 5. GPT-2 124M 训练实验

### 200M ATTH + Qwen2.5 训练实验 (恒源云 PRO 6000, 96GB)

```
模型: ATTH 896d×12L×14H ≈ 253.3M (tie_weights, Qwen2.5-0.5B vocab 151k)
数据: smoltalk-chinese 全量 → 50M tokens (Qwen2 tiktoken, 原生中文)
训练: BF16 autocast, ctx=1024, BS=8, 5 epochs × 500 steps
进度: ✅ softmax 5/5 epoch 完成 (best_ppl=68.27)
      ⏳ s^tau 刚启动 1/5 epoch → 断联
结果: model_softmax_best.pt (已保存远端)
      model_s^tau_best.pt (上轮残留 830MB, 未知)
FP8: ❌ 放弃 — 显存38G vs BF16的29G, 速度2.9 vs 4.7 st/s
经验: FP8六连坑已记录至开发经验 §13, fp8_utils.py 已从 deploy_pkg 删除
```

### 实验 1 (失败): char-level 假数据
```
模型: GPT-2 124M (12层×768×12头) + RoPE + s^τ
数据: char-level Wikitext-2 (ord(c) % 50257, 乱码 token)
结果: loss=0.3 → PPL≈1.35, τ=1.693→1.702 (几乎不动)
根因: 数据量 10MB vs 模型 124M → 严重过拟合, τ 无信号
      问题不在 s^τ, 用 softmax 跑也一样
状态: 🔴 废弃, 实验设计错误
```

### 实验 2 (进行中): 真实 GPT-2 tokenizer + Wikitext-103
```
模型: GPT-2 124M (12层×768×12头) + RoPE + s^τ
数据: wikitext-103 (180M tokens)
分词: HuggingFace GPT-2 tokenizer (50257 词表)
训练: BF16 Amp, 512→1024 curriculum, 50k steps
对照: 训练完自动加载 HF GPT-2 pretrained 跑同组 eval
启动: 2026-04-30 20:00, 仍在下载数据集
命令: cd /root/epx && python3 -u train_gpt2_real.py --norm learned --steps 50000
状态: ⏳ 数据下载中
```

### train_gpt2_real.py 关键参数
```
--norm learned    # 或 softmax (对照)
--lr 3e-4         # 学习率
--ctx 1024        # 最大 ctx (curriculum 512→1024)
--steps 50000     # 总步数
--run_all 1       # 同时跑 learned + softmax + HF 对比
```

### 训练速度估算 (5090, BF16)
```
ctx=512:  ~14 st/s  → 50k steps ≈ 1h
ctx=1024: ~7 st/s   → 40k steps ≈ 1.6h
总计:     ~3h, ¥8.60
```

---

## 6. s^τ ↔ softmax 等价性发现

### 核心定理

> 对于任意 score 分布 s_i 和 τ > 0, 存在一个 softmax 的 score 分布 σ_i 使得注意力输出完全相同:

```
s^τ:         a_i = clamp(s_i, ε)^τ / Σ clamp(s_j, ε)^τ
等价 softmax: a_i = exp(σ_i) / Σ exp(σ_j)
其中:        σ_i = τ · log(clamp(s_i, ε)) + C
```

### 反之亦然

> 对于任意 softmax 模型 (score=σ_i, temperature=T), 存在等价的 s^τ:

```
softmax(T):    a_i = exp(σ_i/T) / Σ exp(σ_j/T)
等价 s^τ:      a_i = s_i^τ / Σ s_j^τ
其中:          s_i = exp(σ_i / (T·τ)), τ 可自由选择
```

### 关键推论

| 方向 | 是否可行 | 自由度变化 |
|:---|:---|:---:|
| **s^τ → softmax** | ✅ 解析映射 | τ 被吸收进 log 变换, 自由度丢失 |
| **softmax → s^τ** | ✅ 且更自由 | **多了一个 τ 可调** |

**实践意义**:
1. 可以用 s^τ 训出模型, 然后转成等价 softmax 推理 (不改变输出)
2. 也可以加载预训练 GPT-2 (softmax), 替换为 s^τ 后多一个 τ 维度调注意力锐度
3. τ 的本质: **动态调节 score 到 attention 的非线性映射函数形式**
   - τ < 1: 压缩 score 差异 → 均匀注意力
   - τ ≈ 1: 近似 L1 归一化
   - τ > 1: 锐化 → 超线性放大

### 待验证 (GPT-2 训练完做)
```
1. 加载 s^τ 模型权重
2. 分别用 s^τ 和等价 softmax 公式推理
3. 验证输出一致 (误差 < 1e-5)
4. 调 τ 观察外推能力变化
```

---

## 7. 文件结构与关键脚本

```
f:\τ\
├── .gitignore                     ← Git 忽略规则
├── HANDOVER.md                    ← 本文档
├── THEORY.md                      ← 理论分析
├── EXPERIMENT_REPORT.md           ← 实验报告
├── TEMPERED_NAN_POSTMORTEM.md     ← tempered 死锁完整事后分析
├── S_TAU_INJECTION_REPORT.md      ← s^τ 注入技术报告
├── project_assets/                ← 训练产出 JSON + checkpoint
│   ├── phase0_speed.json
│   ├── phase1_training.json
│   ├── phase2_gradients.json
│   └── tiny_results/
│
├── deploy_pkg/                    ← 远端部署包
│   ├── attention_mechanisms/
│   │   ├── s_tau_fused.py         ← v4 默认算子 (autograd.Function)
│   │   ├── s_tau_cuda_kernel.py   ← v5 CUDA C++ (待验证)
│   │   ├── model.py               ← 小模型
│   │   ├── attention.py           ← apply_attn_norm() 核心
│   │   ├── attention_complex*.py  ← 复数/等变注意力 (对照)
│   │   └── archive/               ← 已归档算子
│   │       ├── s_tau_fused_v2_backup.py  ← v2 稳定备份
│   │       └── s_tau_triton.py           ← Blackwell 不兼容
│   ├── train_quick.py             ← ★ 一键训练 (魔搭)
│   ├── model_tiny.py              ← ★ 80M 模型构建 + τ 工具
│   ├── train_gpt2_real.py         ← GPT-2 训练 (真实数据)
│   ├── benchmark_norm.py          ← Phase 0-2 基准
│   └── run_parallel.py            ← 并行启动
│
├── scripts/                       ← 运维 + 分析脚本 (已清理)
│   ├── archive/                   ← 一次性调试脚本 (26 个)
│   │   └── README.txt             ← 归档说明
│   ├── check.py / watch.py        ← AutoDL 核心运维
│   ├── harvest.py / nuke.py       ← 收割 + 释放
│   ├── hy_config.py               ← SSH 配置
│   ├── report_final/scan/long.py  ← 实验报告生成
│   ├── _qwen_stau_v2.py           ← ★ Qwen monkey-patch
│   ├── _gpt2_stau.py              ← GPT-2 monkey-patch
│   ├── _ppl_benchmark.py          ← PPL 基准
│   └── _equiv_experiment.py       ← 等价性验证
│
├── experiments/                   ← ★ 实验脚本库
│   ├── INDEX.md                   ← ★ 实验索引 (每个实验一句话结论)
│   ├── s_tau_lab.py               ← ★ 共享积木库
│   ├── s_tau_viz.html             ← 交互可视化
│   ├── tau_bandit_vs_softmax.py   ← s^τ vs softmax 对决
│   ├── tau_bandit_negative.py     ← ★ 负分 bandit (35% 优势)
│   ├── tau_attention_gif.py       ← 热力图动画
│   └── archive/                   ← 已归档 (无差异实验)
│       ├── README.txt             ← 归档说明
│       └── tau_rl_*.py            ← RL 实验 (无差异)
│
└── *.md                           ← 项目文档
```

---

## 8. AutoDL 操作指南 + SSH

### 当前实例 (5090, 有卡模式, 运行中)
```
SSH:   ssh -p 33694 root@connect.westc.seetacloud.com
密码:  TPmz8YFKIW2n
远程:  /root/miniconda3/bin/python3
GPU:   NVIDIA RTX 5090 (34GB)
状态:  ⏳ GPT-2 训练运行中
启动命令: nohup python3 -u /root/epx/train_gpt2_real.py --norm learned --steps 50000 > /root/epx/gpt2_run.log 2>&1 &
```

### hy_config.py (换实例时只改这里)
```python
HOST = 'connect.westc.seetacloud.com'
PORT = 33694
PW = 'TPmz8YFKIW2n'
REMOTE_PYTHON = '/root/miniconda3/bin/python3'
```

### 开新实例流程
```
1. Web 控制台 → 租用新实例 → 选 GPU (4090D ¥1.88/h / 5090 ¥2.88/h)
2. 镜像选: miniconda3-py310-24.03 (PyTorch 2.8+cu128)
3. 开机后把 SSH 复制到 hy_config.py
4. 运行部署脚本或用 ssh 手动上传 deploy_pkg.tar.gz
5. 部署: tar xzf epx.tar.gz && python3 -u train_gpt2_real.py
6. 开监控: python -u scripts/watch_hy.py
7. 用完关实例 (Web 控制台或 API release)
```

### API 操作 (使用开发者 Token)
```
token: eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9...
实例列表: POST /api/v1/dev/instance/pro/list
释放实例: POST /api/v1/dev/instance/pro/release
获取快照: GET /api/v1/dev/instance/pro/snapshot?instance_uuid=xxx
已知 spec UUID: '4090D' (RTX 4090D), 'v-48g' (vGPU-48GB), 'pro6000-p' (PRO 6000)
```

---

## 9. 命令速查

```bash
# ─── 远端操作 (SSH 后) ───

# 检查训练状态
tail -5 /root/epx/gpt2_run.log
ps aux | grep train_gpt2

# GPT-2 训练
cd /root/epx && python3 -u train_gpt2_real.py --norm learned --steps 50000

# 训练 softmax 对照
cd /root/epx && python3 -u train_gpt2_real.py --norm softmax --steps 50000

# 同时跑 learned + softmax + HF 对比
cd /root/epx && python3 -u train_gpt2_real.py --run_all 1 --steps 50000

# 小模型基准 (Phase 0-2)
cd /root/epx && python3 -u benchmark_norm.py

# 查看结果
cat /root/epx/benchmark_results/phase*.json
cat /root/epx/gpt2_results/*.json

# ─── 本地操作 ───

# 本地监控 (80M 实验)
D:\python\python.exe -u scripts\watch_tiny.py

# 本地监控 (旧版)
D:\python\python.exe -u scripts\watch_hy.py

# AutoDL API 操作
D:\python\python.exe scripts\autodl_api.py list       # 列出所有实例
D:\python\python.exe scripts\autodl_api.py snapshot   # 查询实例详情 (SSH/密码)
D:\python\python.exe scripts\autodl_api.py status     # 查询实例运行状态
D:\python\python.exe scripts\autodl_api.py stop       # 关机实例
D:\python\python.exe scripts\autodl_api.py release    # 释放所有实例
D:\python\python.exe scripts\autodl_api.py specs      # GPU 算力规格 ID 一览

# 强制清除
D:\python\python.exe scripts\nuke.py

# 打包部署
tar czf deploy_pkg.tar.gz deploy_pkg/

# 收割结果+释放
D:\python\python.exe scripts\harvest_and_release.py

# 查看实例状态
D:\python\python.exe scripts\check.py
```

---

## 10. 预算与资源

### GPU 规格一览 (AutoDL)

| GPU | VRAM | ¥/h | 半精算力 | spec_uuid | 推荐场景 |
|:---|:---:|:---:|:---:|:---:|:---|
| RTX 5090 | 32GB | 2.88 | 210 TFLOPS | 5090-p | ⭐ 速度 + FP8 |
| RTX 4090D | 24GB | 1.88 | 147 TFLOPS | 4090D | ⭐ 性价比首选 |
| RTX 4090 | 24GB | 1.98 | 165 TFLOPS | 4090 | 速度更快 |
| RTX 3090 | 24GB | 1.32 | 71 TFLOPS | 3090 | ⭐ 省钱首选 |
| RTX 3080Ti | 12GB | 1.08 | 70 TFLOPS | 3080ti | 小模型实验 |
| RTX A4000 | 16GB | 0.92 | 76.7 TFLOPS | a4000 | 轻量级 |
| RTX 3080 | 10GB | 0.88 | 59.5 TFLOPS | 3080 | 入门级 |
| RTX 2080Ti | 11GB | 0.88 | 53.8 TFLOPS | 2080ti | 调试用 |
| PRO 6000 | 96GB | 5.98 | 503.8 TFLOPS | **pro6000-p** | 长序列 8K+ |
| V100 | 32GB | 2.28 | 125 TFLOPS | v100 | 旧架构 |
| L20 | 48GB | 3.68 | 119.5 TFLOPS | l20 | 中等显存 |
| A800-80GB | 80GB | 4.98 | 312 TFLOPS | a800-80g | 大模型 |
| H20 | 96GB | 6.98 | — | h20 | 企业级 |
| H800 | 80GB | 8.88 | 756 TFLOPS | h800 | ⭐ 旗舰 |

> spec_uuid 字段用于创建实例 API 的 `gpu_spec_uuid` 参数

### 最近实例费用 (PRO 6000 — 恒源云)
```
实例: PRO 6000 (connect.westd.seetacloud.com:19380)
运行时长: ~6h (80M→200M→Qwen2训练)
消耗: ~¥35
费用总结: 全部烧光, 余额不足, 已释放
```

---

## 11. 已知 Bug 与陷阱

### Bug 1: deploy 脚本 sftp 重连问题
```
原因: sftp.close() 后再次 with sftp.open() 报错
修复: close 后重新 sftp = ssh.open_sftp() 再写
状态: 已修复 ✅
```

### Bug 2-5: (历史 Bug, 见 v4 文档)

### Bug 6: s_tau_fused backward 梯度爆炸 🔴 已修复
```
发现: 2026-04-30 | 平台: RTX 5090 (sm_120, PyTorch 2.8.0+cu128)
严重度: CRITICAL — 导致 τ 完全不学

根因 1 — AMP 精度毒药:
  autocast → scores fp16 → 融合算子跟随 fp16
  旧路径 pow() 混合精度 → 自动 upcast fp32
  修复: 核心计算强制 fp32, 仅输出 cast 回原 dtype

根因 2 — clamp 梯度断点:
  scores ≤ ε 被 clamp → 公式 a_k/s_k 在 s_k≈ε 处产生大值
  修复: 保存 clamp_mask, backward 置零被 clamp 位置

修复后: |∇logτ| Fused:0.003~0.009 vs Old:0.003~0.012 ✅
速度: Fused 0.54ms vs Old 0.89ms → 1.64× 加速
状态: ✅ 已修复 (s_tau_fused.py v2/v3/v4)
```

### Bug 7: Triton 3.4 在 Blackwell sm_120 上 backward crash ⛔
```
发现: 2026-04-30 | 平台: RTX 5090 (sm_120)
现象: fwd kernel 正常, bwd kernel 调用 tl.sum → CUDA illegal memory access
根因: Triton 3.4 对 Blackwell 的归约指令实验性支持有 bug
影响: Triton 版算子无法在 5090 上使用
降级: 使用 v4 autograd.Function 版 (1.19× vs softmax, 已足够)
状态: ⛔ 等待 Triton 更新, 不阻塞项目
```

### Bug 9: Tempered softmax τ 梯度永久 NaN ⛔ (已定论)
```
发现: 2026-04-30 | 严重度: FATAL — 公式层面不可修复
现象: 4 seeds × 200 ep, τ 全程冻结在 ln(2)=0.693
根因: IEEE 754: 因果 mask 的 -inf 产生 softmax 梯度中 a_i·s_i = 0·(-∞) = NaN
      这不是 FP 精度问题, FP64 也无法避免 — 是逻辑正确性级别的问题
详见: TEMPERED_NAN_POSTMORTEM.md (完整事后分析, 含梯度推导 + s^τ 对比)
结论: tempered softmax 被否决, s^τ 的 clamp 机制是唯一优雅解
状态: ⛔ 定论, 不再修复
```

### Bug 8: GPT-2 char-level 数据训练无效 🟡
```
原因: 用 ord(c)%50257 作为 token ID, 50257 词表映射到 ~100 个实际字符
      10MB 数据 vs 124M 模型 → 严重过拟合, PPL≈1.0, τ 无梯度信号
修复: 使用真实 GPT-2 tokenizer + Wikitext-103 数据集
经验: 验证 s^τ 学习能力需要足够难的任务 (PPL>20)
状态: ✅ 已修复 (train_gpt2_real.py)
```

### 开发经验总结

```
1. AMP 精度坑:
   - autocast 会自动降级 dtype, 自定义 backward 必须手动 upcast
   - 经验法则: 自定义 autograd.Function 内部永远用 fp32

2. IEEE 754 与注意力 mask 的冲突:
   - 因果 mask 的 -inf 与 softmax 梯度中 a_i·s_i 产生 0·(-∞)=NaN
   - 这不是精度问题 — FP64 也 NaN — 这是逻辑正确性级别
   - 详析: TEMPERED_NAN_POSTMORTEM.md
   - 教训: 可学习参数通过 log 驱动比线性尺度数值上更安全 (log-sensitivity)

3. Blackwell (sm_120) 兼容性:
   - PyTorch 2.8+cu128: 基础功能正常
   - Triton 3.4: forward 可用, backward 不稳定
   - CUDA 12.8 原生: 正常 (v5 CUDA C++ 预期兼容)

4. 数据集重要性:
   - char-level 数据信息量太低 → τ 学不动
   - 验证 s^τ 需要真实 tokenizer + 大语料 (PPL>20 才有 τ 梯度)

5. 算子优化天花板:
   - pow() 本身 = exp + mul + log, 三条指令少不了
   - autograd.Function 极限 ~1.19× vs softmax
   - 再往下压需要 CUDA C++ 手写 (预期 ~1.10-1.12×)
   - Triton 不是银弹 — 架构兼容性问题

6. s^τ ↔ softmax 等价性:
   - 双向解析映射存在 (见 §6)
   - 但权重不能直接转换 (非线性变换)
   - 推理时可互换 forward 公式
   - 重要应用: 加载 pretrained GPT-2 → 换成 s^τ → 多一个 τ 维度可控

7. 监控设计模式:
   - watch_tiny.py: 双行读取 (status.txt dashboard + tiny_train.log 实时日志)
   - status.txt 写入用 write_status(**kw) 键值对格式, 对齐字段名
   - watch 脚本只读不写, 零干扰
   - 每 5 秒刷新, 内容变化才刷新屏幕 (避免闪烁)

8. 数据集下载经验 (魔搭/hf-mirror 踩坑):
   - MsDataset.load('name', namespace='ns', split='train') — namespace 参数必须传
   - modelscope[framework] 依赖 addict, 需提前 pip install addict
   - streaming=True 可边下边读, 但依网速可能仍慢
   - 远端优先使用本地 shakespeare.txt (零延迟), 替代方案: 合成数据 (零依赖)
   - hf-mirror.com 国内可用, 但一次性大文件下载可能因无卡实例 OOM kill
   - 最佳实践: 先 ssh 执行 pip install + 数据下载测试, 确认无误再 nohup 启动训练

9. 远程调试陷阱:
   - SSH exec_command 用单引号包裹命令行可避免 PowerSell 引号逃逸
   - 实际调试路径: 写本地 .py 脚本 → sftp.put 上传 → exec_command 执行 → 看日志
   - 不要在 SSH 命令里嵌套 Python -c 带复杂字符串, 用文件上传代替
   - nohup 启动后等至少 8 秒再查进程, 此时若进程不存在说明启动时崩溃

10. 训练脚本健壮性:
     - 所有 pip install 加 -q 和 2>/dev/null, 不影响 stdout 日志
     - write_status() 用 **kw 参数, 自动对齐 python→watch 字段
     - 模型权重间隔保存, 用 best_ppl 筛选
     - 训练完显式 torch.cuda.empty_cache() + gc.collect(), 避免显存泄漏

11. AutoDL 实例清除 (防止烧钱):
      - 列出实例: D:\python\python.exe scripts\autodl_api.py list
      - 获取SSH/密码: D:\python\python.exe scripts\autodl_api.py snapshot
      - 查询状态: D:\python\python.exe scripts\autodl_api.py status
      - 关机实例: D:\python\python.exe scripts\autodl_api.py stop
      - 一键释放: D:\python\python.exe scripts\autodl_api.py release
      - 强制清除: D:\python\python.exe scripts\nuke.py
      - 所有实例停止后余额不再扣费, 但镜像存储仍占少量费用
      - 实例释放前确认已下载结果 (harvest.py), 释放后数据不可恢复
      - 实例 UUID 在 list 或 snapshot 输出中可见

12. FP8 `torch._scaled_mm` 自实现六连坑 (2026-05-01):
    - 起因: torchao 与 torch 2.8.0 不兼容, 决定手写 FP8 Linear
    - 目标: `torch._scaled_mm(x_fp8, w_fp8, scale_x, scale_w)` 实现 FP8 matmul
    - 坑位 1: 矩阵乘法后 device 不对
      - 错误: `mat2 is on cpu`
      - 根因: `replace_linear` 创建新 `FP8Linear` 后没 `.to(device)`
      - 修复: `new = FP8Linear(...)` → `new = new.to(device)`
    - 坑位 2: `_scaled_mm` 只接受 2D 矩阵
      - 错误: `mat1 must be a matrix`
      - 根因: transformer 的 Linear 层输入可能是 3D `[B, T, D]` 或 4D `[B, H, T, D]`
      - 修复: 记录 shape → `x.reshape(-1, dim)` 展平 → `out.reshape(original_shape)`
    - 坑位 3: cuBLASLt 矩阵布局要求
      - 错误: `Only multiplication of row-major and column-major matrices is supported`
      - 根因: `weight.t().contiguous()` 生成了 row-major, 但 cuBLASLt 要求 mat2 是 column-major
      - 修复: `torch.empty_strided((k, n), (1, k), dtype=FP8_DTYPE)` → `copy_(weight.t())`
    - 坑位 4: 维度必须 16 对齐 (tensor core)
      - 错误: `mat2 shape (896x2389) must be divisible by 16`
      - 根因: `2389 % 16 = 5`, Blackwell tensor core 要求对齐
      - 修复: `_pad16(n) = (16 - n%16) % 16`, weight 补零 + 输入 `F.pad` + 输出 slice
    - 坑位 5: view 上的 in-place 操作
      - 错误: `a view of a leaf Variable that requires grad is being used in an in-place operation`
      - 根因: `self.weight[:out, :in].zero_()` 在 autograd 的 view 上 in-place
      - 修复: `self.weight.data[:out, :in].zero_()` 绕过 autograd
    - 坑位 6: backward dtype 不匹配
      - 错误: `expected mat1 and mat2 to have the same dtype, but got: c10::BFloat16 != float`
      - 根因: `_scaled_mm` 输出 BF16, 但 `weight` 是 FP32, `grad_output(BF16) @ weight(FP32)` 炸
      - 修复: `grad_x = grad_output @ weight.to(dtype)` 统一转 BF16
    - 最终代码结构:
      ```python
      class _FP8MatMul(torch.autograd.Function):
          forward:  x → FP8 → _scaled_mm → BF16 out  (column-major weight)
          backward: grad_output @ weight.to(dtype) BF16  (STE)
      class FP8Linear(nn.Module):
          weight: (out+pad) × (in+pad), 自动 16 对齐
          forward: reshape → pad → _FP8MatMul → slice → reshape back
      ```
    - 教训: torch._scaled_mm 是底层 cuBLASLt 封装, 跟普通 matmul 行为不同
       - 必须 column-major layout, 维度 16 对齐, 只支持 2D
       - backward 要自己管 dtype 一致性
     - 最终结论: ❌ 放弃了, 纯 BF16 训练
       - FP8 比 BF16 多 9GB 显存 (38G vs 29G), 速度更慢 (2.9 vs 4.7 st/s)
       - 我们实现的是"计算时转 FP8, weight 存 FP32"→ 没省显存
       - weight 本身存 FP8 才省显存, 但需要改 training loop 结构
       - 纯 BF16 完全够用: 200M 模型只占 29G / 96G, 速度 4.7st/s 稳
     - 状态: ✅ 已从 deploy_pkg 移除

13. AutoDL API 创建实例 (实测结论):
    - API 实测: list/status/snapshot/stop/release 全部 ✅ 通过
    - 创建实例需要: gpu_spec_uuid + image_uuid (私有镜像)
    - 5090 正确 spec UUID: 5090-p (格式: pro6000-p, 4090D 同理)
    - 基础镜像 UUID (base-image-xxx) 属于弹性部署 API, Pro API 不可用
    - image_uuid 需先通过 Web 控制台创建实例 → 保存为私有镜像 → 获取 UUID
    - 关键端点: /api/v1/dev/instance/pro/create (POST)
    - 创建成功返回 pro-xxxxxxxxxxxx 格式的实例 ID
    - 创建后立即计费, 测试时务必及时 stop + release
```

### AutoDL 开发者 API 容器部署（弹性部署）

> 适合场景: 训练脚本稳定后, 通过 API 自动创建/启动/管理 GPU 容器, 无需手动 SSH
> 当前 Token (已配置在 hy_config.py): 见下方, 控制台也可重新生成

#### API 基础信息 (已验证)
```
Host:   https://api.autodl.com
Token  已配置在 hy_config.py 中
鉴权:   headers = {"Authorization": "your_token"}
可用端点 (容器实例Pro, ✅ 测试通过):
  POST /api/v1/dev/instance/pro/list       → 实例列表
  GET  /api/v1/dev/instance/pro/status     → 实例运行状态 (需要 instance_uuid)
  GET  /api/v1/dev/instance/pro/snapshot   → 实例详情 (SSH/密码/价格/使用率)
  POST /api/v1/dev/instance/pro/stop       → 关机实例
  POST /api/v1/dev/instance/pro/start      → 开机实例
  POST /api/v1/dev/instance/pro/release    → 释放实例
  POST /api/v1/dev/instance/pro/create     → 创建实例 (未测试, 见下方规格)
库存查询: Web 控制台直接查看, API 侧可尝试 /api/v1/dev/instance/stock (需企业认证)
本地脚本: D:\python\python.exe scripts\autodl_api.py [list|snapshot|status|specs|stop|release]
```

#### GPU 算力规格 ID (创建实例用，详见 §10 完整表格)

创建实例时 `gpu_spec_uuid` 可用的常见值:

| GPU | spec_uuid | 价格 |
|:---|:---:|:---:|
| PRO 6000 | pro6000-p | ¥5.98/h |
| RTX 5090 | 5090-p | ¥2.88/h |
| RTX 4090D | 4090D | ¥1.88/h |
| RTX 4090 | 4090 | ¥1.98/h |
| RTX 3090 | 3090 | ¥1.32/h |
| H800 | h800 | ¥8.88/h |

#### 保存私有镜像 (获取 image_uuid 的唯一方式)

> AutoDL **不** 支持外部导入镜像。必须通过 Web 控制台操作。

**步骤:**

1. **创建一个实例**（在算力市场租用一台机器）
2. **配置环境**：装好所有依赖 (PyTorch + 数据集 + 代码等)
   - 代码建议放 `/root/epx/`（系统盘，会随镜像保存）
   - 大文件放 `/root/autodl-tmp/`（数据盘，**不会**随镜像保存）
3. **关机实例** — 必须关机才能保存镜像
4. **保存镜像**: 控制台 → 实例列表 → 对应实例 → 「更多操作」→「保存镜像」
   ![](https://aka.doubaocdn.com/s/s5ca1wKzdK)
5. **命名镜像** — 给个容易识别的名字，如 `s-tau-env-v1`
6. **获取 UUID**: 控制台 →「镜像」菜单 → 找到刚才保存的镜像 → 记录其 UUID
   - UUID 格式: `image-xxxxxxxxxxxx`
7. **API 调用** — 用此 UUID 作为 `image_uuid` 参数创建新实例

> ⚠️ 保存的只有系统盘数据（/root 下的文件）。`/root/autodl-tmp` 不会保存。
> ⚠️ 私有镜像迁移到新地区首次创建较慢（需公网传输），但创建过程不计费。

**当前已保存的私有镜像:**
| 镜像名 | UUID | 说明 |
|:---|:---|:---|
| τ1111 | `image-401b2d24be` | 基础 PyTorch 环境, 已保存 |

**验证私有镜像列表 (API):**
```python
import requests
headers = {"Authorization": "your_token"}
r = requests.post('https://api.autodl.com/api/v1/dev/image/private/list',
                  json={"page_index": 1, "page_size": 20}, headers=headers)
print(r.json())
# 返回的 data.list[].image_uuid 就是你需要的值
```

#### Python SDK (社区封装)
```bash
pip install autodl-api
```
```python
from autodl import AutoDLElasticDeployment
client = AutoDLElasticDeployment("your_token")
```

#### 可用接口一览

**容器实例 Pro API (✅ 开发者 Token 可用)**
| 接口 | 方法 | 用途 |
|:----|:----|:-----|
| `/api/v1/dev/instance/pro/list` | POST | 实例列表 |
| `/api/v1/dev/instance/pro/status` | GET | 实例状态 |
| `/api/v1/dev/instance/pro/snapshot` | GET | 实例详情 (SSH/密码) |
| `/api/v1/dev/instance/pro/stop` | POST | 关机 |
| `/api/v1/dev/instance/pro/start` | POST | 开机 |
| `/api/v1/dev/instance/pro/release` | POST | 释放 |
| `/api/v1/dev/instance/pro/create` | POST | 创建 (需私有镜像 UUID) |
| `/api/v1/dev/image/private/list` | POST | 私有镜像列表 |

**弹性部署 API (❌ 需企业认证)**
| 接口 | 方法 | 用途 |
|:----|:----|:-----|
| `/api/v1/dev/deployment` | POST | 创建部署 |
| `/api/v1/dev/instance/stock` | POST | 查询 GPU 库存 |
| `/api/v1/dev/instance/blacklist` | POST | 调度黑名单 |

#### 创建部署示例 (直接 POST)

```python
import requests
headers = {"Authorization": "your_token", "Content-Type": "application/json"}
url = "https://api.autodl.com/api/v1/dev/deployment"
body = {
    "name": "s-tau-training",
    "deployment_type": "Container",          # 一次性任务用 Container, 服务用 ReplicaSet
    "reuse_container": True,                 # 复用已停止容器, 提升启动速度
    "container_template": {
        "dc_list": ["westDC2", "westDC3"],   # 地区: 西北企业区等
        "gpu_name_set": ["RTX 4090D"],
        "gpu_num": 1,
        "cuda_v_from": 118,                  # CUDA 11.8 整数编码
        "cuda_v_to": 128,
        "memory_size_from": 10,
        "memory_size_to": 96,
        "cpu_num_from": 1,
        "cpu_num_to": 16,
        "price_from": 1,                     # 单位: 元*1000, 0.1元=100
        "price_to": 9000,
        "image_uuid": "image-xxxxxxx",       # 私有镜像或公共镜像UUID
        "cmd": "cd /root/epx && python -u train_quick.py",
    }
}
resp = requests.post(url, json=body, headers=headers)
print(resp.json())
```

#### Container vs ReplicaSet vs Job

| 类型 | 生命周期 | 用途 |
|:----|:--------|:----|
| **Container** | cmd 结束容器即终止 | ✅ 一次性训练任务 |
| **ReplicaSet** | 容器异常自动拉起, 维持副本数 | 服务部署 (API推理) |
| **Job** | 并行多个容器执行相同任务 | 批量实验/超参搜索 |

#### 关键实践要点

1. **镜像管理**: 私有镜像需在 AutoDL 网页创建保存, 不支持外部导入. 公共基础镜像对应 UUID 见附录
2. **文件存储**: 跨实例共享存储挂载在同地区容器中, 适合存放代码/模型; 小文件读写性能差 (~100MB/s 大文件带宽)
3. **启动命令**: cmd 结束容器即停止释放, **不要后台执行** (`python app.py &` 会立刻结束)
4. **复用容器**: `reuse_container=True` 大幅提升启动速度, 但注意旧容器文件残留
5. **算力规格**: 创建实例时 `gpu_spec_uuid` 字段用已知 ID (pro6000-p/4090D/3090 等). 库存量建议 Web 控制台查看 (API 弹性部署 stock 端点需企业认证)
6. **成本控制**: 设置 `price_from/price_to` 过滤预算范围; 训练完及时 stop/delete 部署
7. **实例清除**: API 释放比网页更快; 本地调 `autodl_api.py release` 一键清空. 强制清除用 `nuke.py`
8. **注意事项**: release 后数据不可恢复, 务必先 harvest 下载结果. Token 泄露可能产生费用, 勿提交到公开仓库

#### 与当前项目的结合设想
```
Python 脚本自动流程:
  1. token = os.getenv("AUTODL_TOKEN")
  2. client = AutoDLElasticDeployment(token)
  3. stock = client.get_gpu_stock("westDC2", 128)  # 查 5090/4090D 库存
  4. deployment_uuid = client.create_container_deployment(...)  # 创建训练
  5. containers = client.query_containers(deployment_uuid)     # 等 running
  6. ssh 进去 tail -f 日志 / 或 watch_tiny 监控
  7. 训练完 client.get_containers 拿结果 / stop_deployment
  8. 下载结果 → 释放部署
```

---

## 12. 后续方向

### 近期 (恢复实验后)

| 任务 | 优先级 | 说明 |
|:-----|:------:|:-----|
| 完成 200M s^τ 训练 | 🔴 | 恒源云 PRO 6000, softmax 已跑完 (PPL=68.27), 直接 resume s^tau |
| 200M τ 移动验证 | 🔴 | Qwen2 151k vocab 预期 PPL >> 100, τ 应显著移动 |
| s^τ ↔ softmax 等价性验证 + 注入实验 | 🟡 | 已验证 ✅ GPT-2 / Qwen3 / Qwen3.5 三架构通过, 见 S_TAU_INJECTION_REPORT.md |
| PPL 基准测量 (Phase 1) | 🟡 | 已完成 ✅ softmax=51.48 vs s^τ τ=1=61.29(+19%), τ>1: ~200-215(+300%) |
| Qwen2 vs GPT-2 BPE 词表 τ 对比 | 🟡 | 同模型同数据, 仅换 tokenizer, 看 τ 分布差异 |
| τ 相图大模型验证 | 🟡 | 200M 896d×14H 的 τ(d_head, PE, L) 验证小模型相图预测 |

### 中期

| 方向 | 说明 |
|:-----|:-----|
| GPT-2 长序列外推 (8K/16K) | s^τ 在 OOD 长度上可能优于 softmax |
| 3B+ 模型上的 s^τ | 推理一致性 → 预训练 GPT-2 权重直接换 s^τ |
| τ 的信息论解释 | τ 与注意力熵的关系 |

### 算子工程

```
当前状态:
  autograd.Function v4:  1.19× vs softmax ✅ 稳定
  CUDA C++ v5:          ~1.12× (预期) ⏳ 待编译
  Triton:               ❌ Blackwell 不兼容

推荐:
  短期: 用 v4 跑完所有实验
  中期: 验证 CUDA C++ v5, 不上 Triton
  长期: 如果 Triton 修了 Blackwell bug, 再试
```

## 12-b. 可复用组件清单 (2026-05-01 从 gsa-epxa11111 审查 + 本地整理)

### 核心积木 (f:\τ\experiments\s_tau_lab.py)

| 组件 | 类型 | 说明 |
|---|---|---|
| `stau_norm(scores, tau)` | 函数 | s^τ 归一化 (clamp+pow+normalize)，纯 numpy |
| `softmax_norm(scores, T)` | 函数 | 带温度 softmax，纯 numpy |
| `entropy(probs)` | 函数 | 注意力熵 (bits) |
| `effective_n(probs)` | 函数 | 有效数量 1/Σp² |
| `BanditEnv` | 类 | N 臂老虎机 (Bernoulli 奖励) |
| `run_bandit()` | 函数 | 通用 bandit 运行器 (可插拔 score_fn) |
| `bandit_sweep()` | 函数 | 参数扫描 |
| `plot_regret_curves()` | 函数 | 累积 regret 曲线 |
| `plot_prob_snapshot()` | 函数 | s^τ vs softmax 概率快照 |

### 外地项目可复用组件 (F:\gsa-epxa11111\epx-b112)

以下是从外地项目审查中识别的可复用组件，**仅参考，不动原文件**：

| 组件 | 来源 | 状态 |
|---|---|---|
| `VariableLengthAPIterator` | `data.py` | 🔵 数据管道基石，待提取 |
| `apply_attn_norm()` | `run_tau_context.py` | 🟢 已在 f:\τ 中实现 |
| `get_tau(L)` 公式 | `run_tau_context.py` | 🟡 理论公式，待验证 |
| `ModelConfig` dataclass | `config.py` | 🔵 配置系统参考模式 |
| `optimal_tau_from_variance()` | `_gpt2_empirical_tau.py` | 🟡 理论推导，可纳入 THEORY.md |
| `extract_all_taus()` | `_gpt2_tau_analysis.py` | 🟡 训练后分析工具 |

### 外地项目关键实验结果 (已验证，来自 51 模型)

| 发现 | 证据 |
|---|---|
| RoPE 推高 τ (dh16: +140%) | dh×PE 扫描 8 配置 |
| τ 是配置决定的吸引子 (std=1.97%) | 8-seed 收敛性 |
| PPL < 20 时 τ 无法学习 | char-level vs BPE vs 全尺寸 |
| τ(L) 非单调: L256 峰值 | L 扫描 (128~2048) |
| 每层 τ U 型分布 (浅/深层高, 中层低) | per-layer τ 提取 |

---

## 13. 附录: 关键数学公式

### s^τ 归一化
```
a_i = clamp(s_i, ε)^τ / Σ_j clamp(s_j, ε)^τ
τ = softplus(log_tau) + 1.0  (τ ∈ (1, ∞))
```

### 梯度公式 (解析, 手写 backward)
```
∂/∂s_k = τ · a_k / s_k · (g_k − Σ_j a_j·g_j)    [s_k > ε]
∂/∂τ   = ⟨g⊙A, log(s)⟩ − ⟨A, log(s)⟩·⟨A, g⟩
```

### s^τ → softmax 等价映射
```
σ_i = τ · log(clamp(s_i, ε)) + C
softmax(σ_i) = s^τ(s_i)  对任意 i 成立
```

### Tempered softmax (已否决) — 详见 [TEMPERED_NAN_POSTMORTEM.md](TEMPERED_NAN_POSTMORTEM.md)
```
a_i = softmax(τ · s_i)    其中 τ = softplus(log_tau)
→ IEEE 754: 因果 mask -inf 产生 a_i·s_i = 0·(-∞) = NaN
→ τ 永久冻结在 ln(2), 公式层面不可修复
→ 不可用于带因果 mask 的注意力
```
