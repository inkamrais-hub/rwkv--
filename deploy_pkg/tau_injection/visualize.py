"""Visualization utilities for τ-injection analysis.
Generates publication-quality figures in Chinese."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams["font.family"] = "Microsoft YaHei"
plt.rcParams["axes.unicode_minus"] = False
plt.style.use("seaborn-v0_8-darkgrid")

OUT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "rwkv", "可视化")
os.makedirs(OUT_DIR, exist_ok=True)

def plot_effective_rank(layer_ranks_90, layer_ranks_95, layer_ranks_99, model_name, out_dir=None):
    """Plot effective rank decay across layers."""
    out_dir = out_dir or OUT_DIR
    layers = range(len(layer_ranks_90))
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(layers, layer_ranks_90, "o-", label="90% 能量", linewidth=2, markersize=6)
    ax.plot(layers, layer_ranks_95, "s-", label="95% 能量", linewidth=2, markersize=6)
    ax.plot(layers, layer_ranks_99, "D-", label="99% 能量", linewidth=2, markersize=6)
    ax.axhline(y=1.0, color="red", linestyle="--", alpha=0.5, label="Rank-1")
    ax.set_xlabel("层深度")
    ax.set_ylabel("有效秩 (N=64)")
    ax.set_title(f"RWKV-7-{model_name} WKV 状态有效秩随层深变化")
    ax.legend()
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    path = os.path.join(out_dir, f"图一_有效秩随层深变化_{model_name}.png" if model_name else "图一_有效秩随层深变化.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_cliff_curve(history, out_dir=None):
    """Plot train vs val PPL over gradient descent steps."""
    out_dir = out_dir or OUT_DIR
    steps = [h["step"] for h in history]
    train_ppl = [h["train_ppl"] for h in history]
    val_ppl = [h["val_ppl"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(steps, train_ppl, label="训练 PPL", linewidth=2, color="tab:blue")
    ax.plot(steps, val_ppl, label="验证 PPL", linewidth=2, color="tab:orange")
    ax.set_xlabel("梯度下降步数")
    ax.set_ylabel("PPL")
    ax.set_title("τ 优化断崖曲线: 训练 vs 验证")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "图二_断崖曲线.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_injection_comparison(results, out_dir=None):
    """Bar chart: PPL delta for each injection configuration."""
    out_dir = out_dir or OUT_DIR
    tags = list(results.keys())
    deltas = [results[t] for t in tags]
    colors = ["#2ecc71" if d < 0 else "#e74c3c" for d in deltas]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(tags, deltas, color=colors, edgecolor="white", linewidth=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xlabel("注入配置")
    ax.set_ylabel("PPL 变化 (%)")
    ax.set_title("τ 注入点对比: PPL 改善率")
    for bar, d in zip(bars, deltas):
        ax.text(bar.get_x() + bar.get_width() / 2, d + (0.3 if d >= 0 else -0.8),
                f"{d:+.2f}%", ha="center", fontsize=10, fontweight="bold")
    fig.tight_layout()
    path = os.path.join(out_dir, "图三_注入点对比.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_singular_values(layer_svals, highlight_layers=None, out_dir=None):
    """Singular value spectrum for selected layers."""
    out_dir = out_dir or OUT_DIR
    highlight_layers = highlight_layers or [0, len(layer_svals) // 2, len(layer_svals) - 1]
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(highlight_layers)))
    for li, color in zip(highlight_layers, colors):
        if li < len(layer_svals) and layer_svals[li] is not None:
            sv = np.sort(layer_svals[li])[::-1]
            sv_norm = sv / sv[0] if sv[0] > 0 else sv
            ax.semilogy(range(1, len(sv_norm) + 1), sv_norm, "o-", color=color,
                        label=f"L{li} (σ₁={sv[0]:.1f})", linewidth=2, markersize=4)
    ax.set_xlabel("奇异值序号")
    ax.set_ylabel("归一化奇异值 (σ/σ₁)")
    ax.set_title("WKV 状态矩阵奇异值谱")
    ax.legend()
    ax.set_ylim(bottom=1e-3)
    fig.tight_layout()
    path = os.path.join(out_dir, "图四_奇异值谱.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_output_norms(base_norms, tau_norms, out_dir=None):
    """Output norm delta per layer: base vs τ."""
    out_dir = out_dir or OUT_DIR
    layers = range(len(base_norms))
    delta = [100 * (t - b) / max(b, 1e-12) for b, t in zip(base_norms, tau_norms)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(layers, base_norms, "o-", label="基线 (τ=1)", linewidth=2, markersize=5)
    ax1.plot(layers, tau_norms, "s-", label="τ 优化后", linewidth=2, markersize=5)
    ax1.set_xlabel("层深度")
    ax1.set_ylabel("输出范数均值")
    ax1.set_title("输出范数: 基线 vs τ")
    ax1.legend()

    colors = ["#e74c3c" if d < 0 else "#2ecc71" for d in delta]
    ax2.bar(layers, delta, color=colors, edgecolor="white", linewidth=0.5)
    ax2.axhline(y=0, color="black", linewidth=0.8)
    ax2.set_xlabel("层深度")
    ax2.set_ylabel("范数变化 (%)")
    ax2.set_title("τ 注入后输出范数变化")
    fig.tight_layout()
    path = os.path.join(out_dir, "图五_输出范数变化.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path

def plot_entropy_change(base_stats, tau_stats, out_dir=None):
    """Prediction entropy + top-5 mass: base vs τ."""
    out_dir = out_dir or OUT_DIR
    metrics = ["entropy_mean", "top5_mass_mean"]
    labels = ["预测熵", "Top-5 质量"]
    base_vals = [base_stats.get(m, 0) for m in metrics]
    tau_vals = [tau_stats.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, base_vals, width, label="基线 (τ=1)", color="tab:blue", edgecolor="white")
    ax.bar(x + width / 2, tau_vals, width, label="τ 优化后", color="tab:orange", edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_title("预测分布变化: 基线 vs τ")
    ax.legend()
    fig.tight_layout()
    path = os.path.join(out_dir, "图六_预测熵变化.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path