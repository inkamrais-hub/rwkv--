"""
s^τ Attention Heatmap Animation

Sweeps τ from 0.5 to 8.0 on synthetic attention scores,
captures heatmaps at each τ, and generates a GIF showing
how attention patterns sharpen as τ increases.

Also generates a companion comparison: s^τ vs softmax at fixed τ.
"""
import numpy as np
import os, sys
from pathlib import Path

EPS = 1e-8

def stau_norm(scores, tau):
    clamped = np.maximum(scores, EPS)
    powered = np.power(clamped, tau)
    s = powered.sum(axis=-1, keepdims=True)
    return np.divide(powered, s, where=s > 0, out=np.ones_like(powered) / powered.shape[-1])


def softmax(scores, temp=1.0):
    s = scores - scores.max(axis=-1, keepdims=True)
    exps = np.exp(s / temp)
    return exps / exps.sum(axis=-1, keepdims=True)


def make_synthetic_scores(n_queries=12, n_keys=12, seed=42):
    """
    Generate synthetic attention scores with structure:
    - Some queries attend to nearby keys (local pattern)
    - One query attends to a specific far-away key (long-range)
    - Background noise
    """
    rng = np.random.RandomState(seed)
    scores = np.zeros((n_queries, n_keys))

    for q in range(n_queries):
        # Local attention: nearby tokens get higher scores
        for k in range(n_keys):
            dist = abs(q - k)
            local_score = 2.0 * np.exp(-dist / 2.0)
            noise = rng.randn() * 0.3
            scores[q, k] = local_score + noise

        # Long-range pattern: query 5 attends strongly to key 11
        if q == 5:
            scores[q, 11] += 4.0
            scores[q, 10] += 2.0
            scores[q, 0] += 1.5

    return scores


def create_stau_gif(scores, tau_range, save_path, fps=3):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.animation import FuncAnimation, PillowWriter
    except ImportError:
        print("[error] matplotlib not available; install with: pip install matplotlib pillow")
        return

    n_q, n_k = scores.shape

    fig, axes = plt.subplots(2, len(tau_range), figsize=(len(tau_range) * 2.5, 5))
    fig.suptitle("s^tau: How Attention Sharpens with τ", fontsize=13, fontweight='bold', y=0.98)

    softmax_attn = softmax(scores)

    for i, tau in enumerate(tau_range):
        stau_attn = stau_norm(scores, tau)

        # Row 0: s^tau
        ax0 = axes[0, i]
        im0 = ax0.imshow(stau_attn, cmap='YlOrRd', vmin=0, vmax=1.0, aspect='auto')
        ax0.set_title(f'τ = {tau:.1f}', fontsize=10)
        if i == 0:
            ax0.set_ylabel('s^tau Norm\nQueries', fontsize=9)
        ax0.set_xticks([]); ax0.set_yticks([])
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        # Row 1: softmax (same for all columns — reference)
        ax1 = axes[1, i]
        im1 = ax1.imshow(softmax_attn, cmap='YlOrRd', vmin=0, vmax=1.0, aspect='auto')
        if i == 0:
            ax1.set_ylabel('Softmax\nQueries', fontsize=9)
        ax1.set_xlabel('Keys', fontsize=8)
        ax1.set_xticks([]); ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])

    png_path = str(save_path).replace('.gif', '_grid.png')
    plt.savefig(png_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Grid saved to: {png_path}")

    # --- Create animated GIF ---
    fig_anim, ax_anim = plt.subplots(figsize=(6, 5))

    def animate(frame_idx):
        ax_anim.clear()
        tau = tau_range[frame_idx]
        stau_attn = stau_norm(scores, tau)
        im = ax_anim.imshow(stau_attn, cmap='YlOrRd', vmin=0, vmax=0.5,
                            aspect='auto', interpolation='nearest')
        ax_anim.set_title(f's^τ Attention Heatmap: τ = {tau:.1f}',
                          fontsize=13, fontweight='bold')
        ax_anim.set_xlabel('Keys'); ax_anim.set_ylabel('Queries')
        cbar = fig_anim.colorbar(im, ax=ax_anim, shrink=0.85)
        cbar.set_label('Attention Weight')
        return [im]

    anim = FuncAnimation(fig_anim, animate, frames=len(tau_range),
                         interval=1000//fps, blit=False)

    gif_path = str(save_path)
    anim.save(gif_path, writer=PillowWriter(fps=fps), dpi=100)
    plt.close()
    print(f"  GIF saved to: {gif_path}")


def create_stau_vs_softmax_comparison(scores, taus_compare, save_path):
    """Side-by-side: s^τ vs softmax at selected τ values."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    n = len(taus_compare)
    fig, axes = plt.subplots(3, n, figsize=(n * 2.8, 7.5))
    fig.suptitle("s^tau vs Softmax: Direct Comparison", fontsize=13, fontweight='bold')

    softmax_attn = softmax(scores)

    for i, tau in enumerate(taus_compare):
        stau_attn = stau_norm(scores, tau)
        diff = stau_attn - softmax_attn

        vmax = max(stau_attn.max(), softmax_attn.max())
        diff_abs_max = max(abs(diff.min()), abs(diff.max()))

        ax0 = axes[0, i]
        im0 = ax0.imshow(stau_attn, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
        ax0.set_title(f's^tau  tau={tau:.1f}', fontsize=9)
        if i == 0: ax0.set_ylabel('Queries', fontsize=9)
        ax0.set_xticks([]); ax0.set_yticks([])
        plt.colorbar(im0, ax=ax0, fraction=0.046, pad=0.04)

        ax1 = axes[1, i]
        im1 = ax1.imshow(softmax_attn, cmap='YlOrRd', vmin=0, vmax=vmax, aspect='auto')
        if i == 0: ax1.set_ylabel('Softmax\nQueries', fontsize=9)
        ax1.set_xticks([]); ax1.set_yticks([])
        plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)

        ax2 = axes[2, i]
        im2 = ax2.imshow(diff, cmap='RdBu_r', vmin=-diff_abs_max, vmax=diff_abs_max,
                         aspect='auto')
        if i == 0: ax2.set_ylabel('Delta (s^tau - SM)\nQueries', fontsize=9)
        ax2.set_xlabel('Keys', fontsize=8)
        ax2.set_xticks([]); ax2.set_yticks([])
        plt.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    plt.tight_layout(rect=[0, 0, 1, 0.93])
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Comparison saved to: {save_path}")


def create_entropy_vs_tau_plot(scores, tau_fine, save_path):
    """Plot entropy and max weight as functions of tau."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    def _entropy(probs):
        return -np.sum(probs * np.log2(np.maximum(probs, 1e-12)), axis=-1)

    entropies = np.zeros((len(tau_fine), scores.shape[0]))
    max_weights = np.zeros((len(tau_fine), scores.shape[0]))

    for i, tau in enumerate(tau_fine):
        attn = stau_norm(scores, tau)
        entropies[i] = _entropy(attn)
        max_weights[i] = attn.max(axis=-1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle("s^tau Attention: Entropy and Max Weight vs τ", fontsize=13, fontweight='bold')

    for q in range(scores.shape[0]):
        ax1.plot(tau_fine, entropies[:, q], alpha=0.6, linewidth=0.8,
                 label=f'q={q}' if q < 5 else None)
    ax1.set_title('Entropy H(attn) vs τ')
    ax1.set_xlabel('τ'); ax1.set_ylabel('Entropy (bits)')
    ax1.grid(True, alpha=0.3)

    for q in range(scores.shape[0]):
        ax2.plot(tau_fine, max_weights[:, q], alpha=0.6, linewidth=0.8)
    ax2.set_title('Max Attention Weight vs τ')
    ax2.set_xlabel('τ'); ax2.set_ylabel('max(a)')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Entropy plot saved to: {save_path}")


if __name__ == '__main__':
    OUT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("  s^τ Attention Heatmap Animation")
    print("=" * 60)

    # Generate synthetic structured attention scores
    scores = make_synthetic_scores(n_queries=12, n_keys=12, seed=42)
    print(f"  Score shape: {scores.shape}")
    print(f"  Score range: [{scores.min():.2f}, {scores.max():.2f}]")
    print()

    # 1. Static grid: s^τ (varying τ) vs softmax
    tau_coarse = [0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0]
    gif_path = OUT_DIR / 'tau_attention.gif'
    create_stau_gif(scores, tau_coarse, save_path=gif_path)

    # 2. Side-by-side comparison at key τ values
    taus_compare = [1.0, 2.0, 5.0]
    cmp_path = OUT_DIR / 'tau_vs_softmax.png'
    create_stau_vs_softmax_comparison(scores, taus_compare, save_path=cmp_path)

    # 3. Entropy and max weight vs τ (fine sweep)
    tau_fine = np.linspace(0.3, 10.0, 60)
    ent_path = OUT_DIR / 'tau_entropy.png'
    create_entropy_vs_tau_plot(scores, tau_fine, save_path=ent_path)

    print()
    print("  All outputs in:", OUT_DIR)