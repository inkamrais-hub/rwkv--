"""Experiment 3: Multi-point injection sweep.
Tests all 8 injection combinations to map the bottleneck."""
import time, sys, os

TAU_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TAU_DIR, "deploy_pkg"))
from tau_injection import load_rwkv7, optimize_inject, eval_ppl, init_tokenizer, get_device, load_eval_texts

DEV = get_device()
tok = init_tokenizer()

CONFIGS = [
    ("v",          ["v"],                "value (WKV input)"),
    ("g",          ["g"],                "g gate (after GN)"),
    ("output",     ["output"],           "output proj (final)"),
    ("rk",         ["rk"],               "r_k (direct shortcut)"),
    ("v+g",        ["v", "g"],           "v + g combined"),
    ("v+output",   ["v", "output"],      "v + output combined"),
    ("g+output",   ["g", "output"],      "g + output combined"),
    ("v+g+output", ["v", "g", "output"], "v + g + output (triple)"),
]

def run(name, path):
    w, nL, C, H, N = load_rwkv7(path, name)
    texts = load_eval_texts()
    ids_list = [tok.encode(t)[:8] for t in texts]
    n_train = 8
    train_flat = [x for ids in ids_list[:n_train] for x in ids]
    val_list = ids_list[n_train:]

    base_ppl = eval_ppl(val_list, w, nL, C, H, N)
    print(f"  Baseline val PPL: {base_ppl:.2f}")

    results = {}
    for tag, keys, desc in CONFIGS:
        print(f"  [{tag}] {desc} ...")
        t0 = time.time()
        best = optimize_inject(train_flat, w, nL, C, H, N, keys, steps=20, lr=0.05)
        ppl = eval_ppl(val_list, w, nL, C, H, N, inject=best)
        delta = (ppl - base_ppl) / base_ppl * 100
        elapsed = time.time() - t0
        results[tag] = delta
        bar = "█" * max(1, int(-delta * 3)) if delta < 0 else "▁" * 5
        print(f"    PPL={ppl:.2f} Δ={delta:+.2f}% {bar} ({elapsed:.1f}s)")

    best_tag = min(results, key=results.get)
    print(f"\n  Best: {best_tag} ({results[best_tag]:+.2f}%)")
    return results

if __name__ == "__main__":
    run("test", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")