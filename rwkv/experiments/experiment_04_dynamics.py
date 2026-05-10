"""Experiment 4: Cliff experiment + effective rank + sparsity analysis."""
import json, math, sys, os
import numpy as np

TAU_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TAU_DIR, "deploy_pkg"))
from tau_injection import (
    load_rwkv7, optimize_tau_track, rwkv7_fwd_detailed,
    compute_effective_rank, compute_gini, analyze_token_dynamics,
    init_tokenizer, get_device, load_eval_texts,
)
from tau_injection.visualize import (
    plot_effective_rank, plot_cliff_curve, plot_singular_values,
    plot_output_norms, plot_entropy_change,
)

DEV = get_device()
tok = init_tokenizer()

def run(name, path):
    w, nL, C, H, N = load_rwkv7(path, name)
    texts = load_eval_texts()
    all_ids = [tok.encode(t)[:8] for t in texts]
    n_train = max(len(all_ids) * 3 // 4, 8)
    train_flat = [x for ids in all_ids[:n_train] for x in ids]
    val_list = all_ids[n_train:]

    print(f"  Train: {n_train} texts ({len(train_flat)} tok), Val: {len(val_list)} texts")

    tau_list, best_tau, history = optimize_tau_track(
        train_flat, val_list, w, nL, C, H, N, steps=80, lr=0.05)
    print(f"  Best val PPL at step {min(range(len(history)), key=lambda i: history[i]['val_ppl'])}: "
          f"{min(h['val_ppl'] for h in history):.2f}")

    plot_cliff_curve(history)

    logits_base, snap_base = rwkv7_fwd_detailed(all_ids[0], w, nL, C, H, N)
    logits_tau, snap_tau = rwkv7_fwd_detailed(all_ids[0], w, nL, C, H, N, tau_v=best_tau)

    ranks_90, ranks_95, ranks_99 = [], [], []
    for sv in snap_base["state_svals"]:
        ranks_90.append(compute_effective_rank(sv, 0.90) if sv is not None else 0)
        ranks_95.append(compute_effective_rank(sv, 0.95) if sv is not None else 0)
        ranks_99.append(compute_effective_rank(sv, 0.99) if sv is not None else 0)

    print(f"  Effective rank (90%): L0={ranks_90[0]:.1f} → L{nL-1}={ranks_90[-1]:.1f}")
    print(f"  Effective rank (99%): L0={ranks_99[0]:.1f} → L{nL-1}={ranks_99[-1]:.1f}")

    plot_effective_rank(ranks_90, ranks_95, ranks_99, name)

    highlight = [0, nL // 3, 2 * nL // 3, nL - 1]
    plot_singular_values(snap_base["state_svals"], highlight_layers=highlight)

    plot_output_norms(snap_base["out_norms"], snap_tau["out_norms"])

    base_dyn = analyze_token_dynamics(logits_base)
    tau_dyn = analyze_token_dynamics(logits_tau)
    print(f"  Entropy: base={base_dyn['entropy_mean']:.3f} → tau={tau_dyn['entropy_mean']:.3f}")
    print(f"  Top-5 mass: base={base_dyn['top5_mass_mean']:.3f} → tau={tau_dyn['top5_mass_mean']:.3f}")
    plot_entropy_change(base_dyn, tau_dyn)

    with open(os.path.join(os.path.dirname(__file__), "..", f"tau_dynamics_{name}.json"), "w") as f:
        json.dump(history, f, indent=2)

    return history, ranks_90, base_dyn, tau_dyn

if __name__ == "__main__":
    run("test", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")