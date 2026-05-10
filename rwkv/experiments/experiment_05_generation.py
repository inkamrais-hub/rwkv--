"""Experiment 5: Generation quality comparison."""
import sys, os

TAU_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TAU_DIR, "deploy_pkg"))
from tau_injection import load_rwkv7, optimize_inject, generate, init_tokenizer, get_device

DEV = get_device()
tok = init_tokenizer()

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

def run(name, path):
    w, nL, C, H, N = load_rwkv7(path, name)
    train_flat = [x for t in OPT_TEXTS for x in tok.encode(t)[:8]]

    best_v = optimize_inject(train_flat, w, nL, C, H, N, ["v"], steps=15, lr=0.05)
    best_vo = optimize_inject(train_flat, w, nL, C, H, N, ["v", "output"], steps=15, lr=0.05)
    best_all = optimize_inject(train_flat, w, nL, C, H, N, ["v", "g", "output"], steps=15, lr=0.05)

    configs = [
        ("base", None),
        ("v",    best_v),
        ("v+o",  best_vo),
        ("all",  best_all),
    ]

    for prompt in PROMPTS:
        print(f"\n  Prompt: \"{prompt}\"")
        for tag, inj in configs:
            g = generate(prompt, w, nL, C, H, N, inject=inj, max_new=50, temperature=0.7)
            print(f"  [{tag:>4}] {prompt}{g}")

if __name__ == "__main__":
    run("test", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")