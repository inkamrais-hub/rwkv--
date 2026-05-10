import torch
import math
import numpy as np
from .model import rwkv7_fwd_detailed, _dev


def eval_ppl(tok_ids_list, w, nL, C, H, N, tau_v=None, inject=None):
    """Evaluate perplexity on a list of token sequences."""
    dev = _dev()
    tl, tt = 0.0, 0
    with torch.no_grad():
        for ids in tok_ids_list:
            if inject is not None:
                from .model import rwkv7_fwd_inject
                logits = rwkv7_fwd_inject(ids, w, nL, C, H, N, inject=inject)
            elif tau_v is not None:
                logits, _ = rwkv7_fwd_detailed(ids, w, nL, C, H, N, tau_v=tau_v)
            else:
                logits, _ = rwkv7_fwd_detailed(ids, w, nL, C, H, N)
            ids_t = torch.tensor(ids, device=dev)
            L = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:]).item()
            tl += L * (len(ids) - 1)
            tt += len(ids) - 1
    return math.exp(tl / max(tt, 1))


def compute_effective_rank(svals, threshold=0.9):
    """Effective rank: number of singular values capturing `threshold` of energy.
    svals: [H, N] or [N] numpy array of singular values."""
    svals = np.atleast_2d(svals)
    total = np.sum(svals**2, axis=-1)
    cumsum = np.cumsum(np.sort(svals**2, axis=-1)[:, ::-1], axis=-1)
    ranks = []
    for h in range(svals.shape[0]):
        ratio = cumsum[h] / max(total[h], 1e-12)
        r = np.searchsorted(ratio, threshold) + 1
        ranks.append(min(r, svals.shape[1]))
    return np.mean(ranks)


def compute_gini(x):
    """Gini coefficient: 0 = uniform, 1 = all mass in one element."""
    x = np.asarray(x).flatten()
    x_sorted = np.sort(x)
    n = len(x_sorted)
    if n == 0 or x_sorted.sum() == 0:
        return 0.0
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * x_sorted)) / (n * np.sum(x_sorted)) - (n + 1) / n


def analyze_token_dynamics(logits):
    """Per-position prediction statistics.
    Returns dict with entropy, top-5 mass, sparsity metrics."""
    probs = torch.softmax(logits.float(), dim=-1)
    entropies = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    top5, _ = torch.topk(probs, 5, dim=-1)
    top5_mass = top5.sum(dim=-1)
    sparsity = (probs > 0.01).float().sum(dim=-1) / probs.shape[-1]
    return {
        "entropy_mean": entropies.mean().item(),
        "entropy_std": entropies.std().item(),
        "top5_mass_mean": top5_mass.mean().item(),
        "sparsity_mean": sparsity.mean().item(),
    }