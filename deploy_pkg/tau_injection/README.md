# τ-Injection: Diagnosing RWKV-7's Information Bottleneck

A lightweight toolkit for probing the information bottleneck of RWKV-7 (and
similar recurrent architectures) via per-channel multiplicative scaling (τ).

## Quick Start

```python
from tau_injection import (
    load_rwkv7, optimize_tau, eval_ppl,
    compute_effective_rank, generate
)

# 1. Load a pretrained RWKV-7 model
w, nL, C, H, N = load_rwkv7("RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")

# 2. Optimize τ on a few sentences (10 steps, ~5 seconds on RTX 3060)
train_ids = tok.encode("The future of AI will depend on how we design")
best_tau = optimize_tau(train_ids, w, nL, C, H, N, steps=10, lr=0.05)

# 3. Evaluate improvement
ppl_base = eval_ppl([val_ids], w, nL, C, H, N)
ppl_tau  = eval_ppl([val_ids], w, nL, C, H, N, tau_v=best_tau)
print(f"PPL improvement: {(ppl_tau - ppl_base) / ppl_base * 100:.2f}%")

# 4. Generate with optimized τ
text = generate("The future of AI is", w, nL, C, H, N, inject={"v": best_tau})
```

## API Overview

| Module | Key Functions |
|:-------|:--------------|
| `model.py` | `load_rwkv7`, `rwkv7_fwd`, `rwkv7_fwd_inject`, `rwkv7_fwd_detailed` |
| `optimize.py` | `optimize_tau` (v-only), `optimize_inject` (multi-point), `optimize_tau_track` (tracking) |
| `eval.py` | `eval_ppl`, `compute_effective_rank`, `compute_gini`, `analyze_token_dynamics` |
| `generation.py` | `generate`, `generate_compare` |
| `visualize.py` | `plot_effective_rank`, `plot_cliff_curve`, `plot_injection_comparison`, `plot_singular_values`, `plot_output_norms`, `plot_entropy_change` |

## Injection Points

| Key | Tensor Shape | Description |
|:----|:------------:|:------------|
| `v` | [H, N] | Value vector before WKV recurrence — **canonical point** |
| `g` | [H, N] | Output gate after GroupNorm — **harmful** (already optimized) |
| `output` | [1, C] | Final output projection — bypasses all damping |
| `rk` | [H, N] | r_k shortcut — negligible effect |

## Reproducing Paper Results

Run from the project root:

```
D:\python\python.exe rwkv/experiments/run_all.py
```

This executes all five experiments from the paper:
1. k-injection grid search (baseline)
2. v-injection gradient descent (main result)
3. Multi-point injection sweep (8 configurations)
4. Cliff experiment + effective rank + sparsity
5. Generation quality comparison

## Requirements

- torch >= 2.0.0
- numpy >= 1.24.0
- matplotlib >= 3.7.0
- rwkv >= 0.8.0 (for tokenizer)
- GPU with >= 6 GB VRAM (tested on RTX 3060 Laptop)

## Citation

If you use τ-injection in your research, please cite:

```
@techreport{tau-project-2026,
  title  = {Probing the Information Bottleneck of RWKV-7 via Per-Channel τ-Injection},
  author = {τ Project},
  year   = {2026},
  note   = {Original research project. Code: deploy_pkg/tau_injection/}
}
```