#!/usr/bin/env python3
"""Generation quality: base vs v-τ vs v+output-τ vs v+g+output-τ"""
import torch, sys, os, math, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tau_injection_sweep import load_rwkv7, rwkv7_fwd_inject, optimize_inject, DEV
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tok = TRIE_TOKENIZER(r"D:\python\lib\site-packages\rwkv\rwkv_vocab_v20230424.txt")

PROMPTS = [
    "The future of artificial intelligence",
    "The secret to building great software is",
    "Mathematics is beautiful because",
    "The relationship between consciousness and matter",
    "A wise person once said:",
]

OPT_TEXTS = [
    "Deep learning has revolutionized the field of computer vision",
    "Quantum computing will transform cryptography and drug discovery",
    "Climate change requires global cooperation across all nations",
    "The history of science shows that paradigm shifts are inevitable",
]

@torch.no_grad()
def generate(prefix, w, nL, C, H, N, inject=None, max_new=60, temperature=0.7, top_k=40):
    ids = tok.encode(prefix)
    for _ in range(max_new):
        logits = rwkv7_fwd_inject(ids, w, nL, C, H, N, inject=inject)
        logits_last = logits[-1].float()
        if temperature > 0:
            logits_last = logits_last / temperature
            v, _ = torch.topk(logits_last, top_k)
            logits_last[logits_last < v[-1]] = -float("inf")
            probs = torch.softmax(logits_last, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
        else:
            ids.append(torch.argmax(logits_last).item())
    return tok.decode(ids[len(tok.encode(prefix)):])

def run_model(name, path):
    w, nL, C, H, N = load_rwkv7(path, name)
    train_flat = [x for t in OPT_TEXTS for x in tok.encode(t)[:8]]
    print(f"  Optimizing on {len(train_flat)} tokens...")

    print("  [base]  baseline (τ=1)")
    best_v = optimize_inject(train_flat, w, nL, C, H, N, ["v"], steps=15, lr=0.05)
    print("  [v]     value injection only")
    best_vo = optimize_inject(train_flat, w, nL, C, H, N, ["v", "output"], steps=15, lr=0.05)
    print("  [v+o]   v + output dual injection")
    best_all = optimize_inject(train_flat, w, nL, C, H, N, ["v", "g", "output"], steps=15, lr=0.05)
    print("  [all]   v + g + output triple injection")

    configs = [
        ("base", None),
        ("v",    best_v),
        ("v+o",  best_vo),
        ("all",  best_all),
    ]

    print(f"\n{'='*70}")
    for prompt in PROMPTS:
        print(f"\n  Prompt: \"{prompt}\"")
        for tag, inj in configs:
            g = generate(prompt, w, nL, C, H, N, inject=inj, max_new=50, temperature=0.7)
            print(f"  [{tag:>4}] {prompt}{g}")

if __name__ == "__main__":
    for name, path in [
        ("0.4B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth"),
        ("1.5B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"),
    ]:
        print(f"\n{'#'*70}\n# RWKV-7-{name}\n{'#'*70}")
        run_model(name, path)