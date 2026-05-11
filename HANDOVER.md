# τ 项目交接文档 (v6 — τ-injection + s^τ 双线并行)

> **最后更新: 2026-05-11**
> **仓库**: [rwkv--](https://github.com/inkamrais-hub/rwkv--) (本仓库) · [tau-atth111](https://github.com/inkamrais-hub/tau-atth111) (s^τ 独立仓库)
> **状态**: 🟢 工程完成 · 🟡 论文待发布

---

## 目录
1. [项目定位](#1)
2. [两条主线](#2)
3. [文件结构](#3)
4. [AutoDL 远端操作](#4)
5. [代码库清理记录](#5)
6. [CUDA Kernel 版本链](#6)
7. [GitHub 仓库清单](#7)
8. [论文与文档索引](#8)
9. [待办事项](#9)
10. [命令速查](#10)

---

## 1. 项目定位

**τ 项目** 围绕一个核心概念展开：**可学习的注意力缩放参数 τ**。

两个方向：
- **s^τ**：用幂律归一化替代 softmax，τ 是逐头可学习的"焦距旋钮"
- **τ-injection**：在预训练模型的注意力节点注入 τ，通过 PPL 响应诊断架构信息瓶颈

两者使用同一套 CUDA 算子基础设施，但面向不同的学术问题。

---

## 2. 两条主线

### 2.1 τ-injection — RWKV-7 架构诊断

**论文**: [PAPER.md](PAPER.md)（英文）· [PAPER_CN.md](PAPER_CN.md)（中文）· [PAPER.html](PAPER.html)（HTML 带图）

**核心发现**:
| 发现 | 数据 |
|---|---|
| v-injection 最优 | PPL −6.81% (0.4B), −5.59% (1.5B), −6.58% (2.9B) |
| WKV 有效秩 | 1–3 / 64，深层退化至 rank-1 |
| 6 层阻尼链 | 信号衰减 ~15× vs softmax |
| g 门控陷阱 | 注入后 PPL +1.6% — 已训练优化 |
| 跨域鲁棒性 | 0.4B 跨 EN→ZH 保留 79% |
| 生成质量 | 重复 n-gram 从 4.2% → 0.0% |

**实验覆盖**: 0.4B / 1.5B / 2.9B 三个规模，wiki-text / CMRC / HumanEval 三个域

**代码**: `deploy_pkg/tau_injection/` + `rwkv/experiments/`

### 2.2 s^τ — 可学习幂律注意力归一化

**理论**: [THEORY.md](THEORY.md)（完整数学推导）

**核心验证**:
| 验证 | 结果 |
|---|---|
| CUDA Kernel (v12) | RTX 5090 上 1.2× vs softmax |
| GPT-2 124M 训练 | τ 正常收敛（8 seeds, std<0.1） |
| GPT-2 零训练替换 | 直接换 s^τ 不重训练也能跑 |
| Qwen3-1.7B 全微调 | s^τ 替换 softmax 通过 |
| SDXL 图像生成 | 零训练替换，KL=1.57 |

**独立仓库**: [tau-atth111](https://github.com/inkamrais-hub/tau-atth111)

---

## 3. 文件结构

```
f:\τ\                              ← 主仓库 (rwkv--)
├── PAPER.md / PAPER_CN.md / PAPER.html   ← τ-injection 论文
├── README.md                              ← 论文主页（摘要 + 6 张图）
├── REPORT.md                              ← 技术报告 (~13K words)
├── THEORY.md                              ← s^τ 理论推导
├── HANDOVER.md                            ← 本文档
│
├── deploy_pkg/
│   ├── tau_injection/                     ← τ-injection 核心包
│   │   ├── model.py                       ←   模型加载 + 前向
│   │   ├── optimize.py                    ←   梯度优化器
│   │   ├── eval.py                        ←   PPL 评估
│   │   ├── generation.py                  ←   生成测试
│   │   └── visualize.py                   ←   可视化
│   └── attention_mechanisms/              ← s^τ 算子库
│       ├── s_tau_fused.py                 ←   v4 autograd 融合算子
│       ├── s_tau_cuda_kernel.py           ←   v5 CUDA C++ 行内编译
│       ├── s_tau_cuda_kernel_v12.py       ←   v12 最终 CUDA kernel 迭代
│       ├── s_tau_cuda_kernel_v12_1~3.py   ←   v12 子版本
│       ├── s_tau_fused_attention_v14~v16  ←   fused attention 版本链
│       └── s_tau_fused_bwd.py             ←   backward 算子
│
├── rwkv/
│   ├── experiments/                       ← τ-injection 实验脚本
│   │   ├── run_all.py                     ←   一键复现
│   │   ├── experiment_01~05               ←   分步实验
│   │   └── ε_supplement.py                ←   补充实验
│   └── 可视化/                            ← 6 张中文字幕 PNG
│       ├── 图一_有效秩随层深变化.png
│       ├── 图二_断崖曲线.png
│       ├── 图三_注入点对比.png
│       ├── 图四_奇异值谱.png
│       ├── 图五_输出范数变化.png
│       └── 图六_预测熵变化.png
│
├── scripts/                               ← AutoDL 运维脚本
│   ├── check.py                           ←   查看远端实例状态
│   ├── nuke.py                            ←   释放所有实例
│   ├── watch.py                           ←   持续监控训练
│   ├── status.py                          ←   实例快照
│   ├── harvest.py                         ←   拉取远端数据
│   ├── report_scan.py                     ←   分析相图
│   ├── report_long.py                     ←   分析长实验
│   ├── autodl_api.py                      ←   API 封装 (secrets)
│   ├── hy_config.py                       ←   恒源云配置 (secrets)
│   └── archive/                           ←   历史运维脚本
│
├── stau_release/                          ← s^τ 独立仓库本地镜像
│   └── → 已推送到 tau-atth111
│
├── experiments/                           ← s^τ 早期实验
│   ├── s_tau_lab.py                       ←   积木化实验模块
│   ├── tau_bandit.py                      ←   多臂赌博机 demo
│   ├── tau_attention_gif.py               ←   注意力热力图
│   └── contrastive/                       ←   对比学习实验
│
├── data/                  (gitignored)     ← HuggingFace 数据集
├── ms_weights/            (gitignored)     ← RWKV-7 模型权重 (17 GB)
├── modelscope_cache/      (gitignored)     ← ModelScope 缓存
└── project_assets/        (gitignored)     ← 大资源文件
```

---

## 4. AutoDL 远端操作

### API 端点
```
base: https://api.autodl.com
认证: scripts/autodl_api.py (HEADER)
实例: pro-77757062a9f9 (L20), pro-77757457d678 (L20)
```

### 常用脚本

| 脚本 | 用途 |
|------|------|
| `D:\python\python.exe scripts/check.py` | 查看远端实例状态 + 训练日志 |
| `D:\python\python.exe scripts/nuke.py` | 强制释放所有实例 |
| `D:\python\python.exe scripts/report_scan.py` | 分析相图实验 |
| `D:\python\python.exe scripts/report_long.py` | 分析长文本实验 |
| `D:\python\python.exe scripts/watch.py` | 轮询监控训练进度 |
| `D:\python\python.exe scripts/harvest.py` | 拉取远端实验结果 |
| `D:\python\python.exe scripts/status.py` | 实例快照 |

### 环境配置

```python
# scripts/autodl_api.py
ENDPOINT: https://api.autodl.com
AUTH: 通过 HEADER 认证
MODEL_PATH: /root/autodl-tmp/
DATA_PATH: /root/autodl-fs/
```

### CUDA JIT 编译环境

```powershell
# 启动 Python 前执行
D:\c++\VC\Auxiliary\Build\vcvars64.bat
$env:PATH = "D:\c++\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"
```

---

## 5. 代码库清理记录 (2026-05-11)

| 操作 | 数量 |
|------|------|
| 删除根目录垃圾脚本 (.py/.txt/.json/.log) | 111 文件 |
| 删除 deploy_pkg 旧版 CUDA kernel (<v12) | 7 文件 |
| 删除 experiments 归档实验 | 25 文件 |
| 删除 scripts 弃用脚本 | 68 文件 |
| 删除无用子目录 | 7 个 |
| 更新 .gitignore | 68 行全覆盖 |
| 恢复 kernel v12 + fused attention v14~v16 | 10 文件 (从回收站) |

**可恢复**: 全部已删除文件在 Windows 回收站中，未清空。

---

## 6. CUDA Kernel 版本链

### s^τ 归一化算子

| 版本 | 文件 | 状态 |
|------|------|------|
| v1 | 初始 autograd.Function | 🔴 已废弃 (fp16 梯度爆炸) |
| v2 | fp32 强制修复 | 🔴 已废弃 |
| v3 | 优化 autograd | 🔴 已废弃 |
| v4 | torch.compile 加速 | 🟢 保留 (s_tau_fused.py) |
| v5 | CUDA C++ 行内编译 | 🟢 保留 (s_tau_cuda_kernel.py) |
| v6~v10 | 中间迭代 | 🔴 回收站 |
| v12 | 最终 CUDA kernel | 🟢 保留 (v12 ~ v12_3) |

### Fused Attention 实现

| 版本 | 状态 |
|------|------|
| v13 | 🔴 回收站 |
| v14~v16 | 🟢 保留 |
| v15_tk64 / v15_train / v16_train | 🟢 保留 |

---

## 7. GitHub 仓库清单

### rwkv-- (主仓库)
- URL: `https://github.com/inkamrais-hub/rwkv--`
- Remote: `rwkv` (SSH: git@ssh.github.com:inkamrais-hub/rwkv--.git)
- 内容: τ-injection 论文 + 实验 + AutoDL 脚本 + s^τ 算子

### tau-atth111 (s^τ 独立仓库)
- URL: `https://github.com/inkamrais-hub/tau-atth111`
- Remote: 在 `stau_release/` 子目录中
- 内容: s^τ 核心包 + CUDA kernel + 训练验证 + 理论文档

---

## 8. 论文与文档索引

| 文档 | 内容 | 状态 |
|------|------|------|
| [PAPER.md](PAPER.md) | τ-injection 英文论文 | ✅ 完成 |
| [PAPER_CN.md](PAPER_CN.md) | τ-injection 中文论文 | ✅ 完成 |
| [PAPER.html](PAPER.html) | 学术 HTML 版 (含 6 张图) | ✅ 完成 |
| [README.md](README.md) | 论文主页 (GitHub 首页) | ✅ 完成 |
| [REPORT.md](REPORT.md) | 技术报告 (~13K words) | ✅ 完成 |
| [THEORY.md](THEORY.md) | s^τ 数学推导 | ✅ 完成 |
| [SOLO_CHALLENGE_DRAFT.md](SOLO_CHALLENGE_DRAFT.md) | SOLO 挑战投稿草稿 | 🟡 草稿 |
| [FORUM_POST.md](FORUM_POST.md) | RWKV 论坛帖子草稿 | 🟡 草稿 |
| [STAU_RESEARCH.md](STAU_RESEARCH.md) | s^τ 研究方法论 | 🟡 笔记 |
| [EXPERIMENT_REPORT.md](EXPERIMENT_REPORT.md) | 51 模型实验报告 | ✅ 参考 |
| [TEMPERED_NAN_POSTMORTEM.md](TEMPERED_NAN_POSTMORTEM.md) | Tempered 死锁分析 | ✅ 参考 |

---

## 9. 待办事项

### 高优先级
- [ ] arXiv 预印本提交（注册 + endorsement）
- [ ] RWKV 社区曝光（QQ 群 332381861 或论坛发帖）
- [ ] ChinaXiv 备选路径（已拒：需要机构邮箱）

### 中优先级
- [ ] 7B 模型 τ-injection 实验（需云 GPU ~5 元/小时）
- [ ] s^τ 独立论文撰写
- [ ] HANDOVER.md 中的 AutoDL 实例重新激活（如需远端实验）

### 低优先级
- [ ] s^τ CUDA kernel 合并 v12_1~v12_3 为单一最终版
- [ ] SDXL 训练验证（s^τ 替换 + 训练）
- [ ] PHI-3 / LLaMA 跨架构 τ-injection

---

## 10. 命令速查

```powershell
# AutoDL 运维
D:\python\python.exe scripts/check.py         # 查看远端
D:\python\python.exe scripts/nuke.py          # 释放实例
D:\python\python.exe scripts/report_scan.py   # 相图分析
D:\python\python.exe scripts/report_long.py   # Long 实验

# Git (主仓库)
git push rwkv main                             # 推送到 GitHub
git pull rwkv main --rebase                    # 拉取远端

# CUDA JIT (启动 Python 前执行)
D:\c++\VC\Auxiliary\Build\vcvars64.bat
$env:PATH = "D:\c++\VC\Tools\MSVC\14.50.35717\bin\Hostx64\x64;$env:PATH"

# 复现 τ-injection
python rwkv/experiments/run_all.py --model 0.4B --steps 10

# s^τ 独立仓库
cd stau_release/
git push -u origin main
```