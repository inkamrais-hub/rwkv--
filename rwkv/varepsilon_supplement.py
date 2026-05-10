"""Supplementary experiments: cross-domain robustness, repeat n-gram, error bars.
Reuses tau_injection package. Quick run on 0.4B (~2 min total)."""
import sys, os, time, json, math
from collections import Counter

TAU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(TAU_DIR, "deploy_pkg"))

import torch
import numpy as np
from tau_injection import (
    load_rwkv7, optimize_tau, eval_ppl, generate,
    init_tokenizer, get_device, load_eval_texts,
)
from tau_injection.visualize import OUT_DIR

DEV = get_device()
tok = init_tokenizer()
MODEL_PATH = r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth"
MODEL_PATH_15B = r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"

# ── Cross-domain texts ──
ZH_TEXTS = [
    "人工智能的未来将取决于我们如何设计",
    "量子计算有望彻底改变密码学和优化领域",
    "深度学习在医疗保健中的应用包括医学影像分析",
    "气候变化需要各国政府和企业立即采取行动",
    "区块链技术实现了去中心化的信任和透明度",
    "可再生能源如太阳能和风能正变得具有成本竞争力",
    "意识的理解仍然是神经科学中的一个挑战",
    "现代经济学在很大程度上依赖统计模型和数据",
]

CODE_TEXTS = [
    "def quicksort(arr): if len(arr) <= 1: return arr",
    "class Transformer(nn.Module): def __init__(self, d_model):",
    "async function fetchData(url) { const response = await",
    "#include <stdio.h>\nint main() { printf(\"Hello World\");",
    "SELECT user_id, COUNT(*) FROM orders GROUP BY user_id",
    "import numpy as np\nX = np.random.randn(100, 10)",
    "docker run -d -p 8080:80 --name nginx nginx:latest",
    "def loss_fn(pred, target): return F.cross_entropy(pred, target)",
]

# ── Generation prompts for repeat n-gram ──
GEN_PROMPTS = [
    "The future of artificial intelligence",
    "The secret to building great software is",
    "Mathematics is beautiful because",
]


def repeat_ngram_ratio(text, n=2):
    """Ratio of repeated n-grams to total n-grams (lower = less repetitive)."""
    words = text.split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    counts = Counter(ngrams)
    repeats = sum(c - 1 for c in counts.values() if c > 1)
    return repeats / max(len(ngrams), 1)


def run(name, path, n_repeats=3):
    w, nL, C, H, N = load_rwkv7(path, name)

    # ── 1. Cross-domain optimization ──
    en_texts = load_eval_texts()[:8]
    en_ids = [tok.encode(t)[:8] for t in en_texts]
    zh_ids = [tok.encode(t)[:8] for t in ZH_TEXTS]
    code_ids = [tok.encode(t)[:12] for t in CODE_TEXTS]

    train_flat = [x for ids in en_ids[:6] for x in ids]
    val_en = en_ids[6:]
    val_zh = zh_ids
    val_code = code_ids

    t0 = time.time()
    best_tau = optimize_tau(train_flat, w, nL, C, H, N, steps=15, lr=0.05)
    opt_time = time.time() - t0

    ppl_en_base = eval_ppl(val_en, w, nL, C, H, N)
    ppl_en_tau  = eval_ppl(val_en, w, nL, C, H, N, tau_v=best_tau)
    ppl_zh_base = eval_ppl(val_zh, w, nL, C, H, N)
    ppl_zh_tau  = eval_ppl(val_zh, w, nL, C, H, N, tau_v=best_tau)
    ppl_code_base = eval_ppl(val_code, w, nL, C, H, N)
    ppl_code_tau  = eval_ppl(val_code, w, nL, C, H, N, tau_v=best_tau)

    cross_domain = {
        "en_delta%": round((ppl_en_tau - ppl_en_base) / ppl_en_base * 100, 2),
        "zh_delta%": round((ppl_zh_tau - ppl_zh_base) / ppl_zh_base * 100, 2),
        "code_delta%": round((ppl_code_tau - ppl_code_base) / ppl_code_base * 100, 2),
        "opt_time_s": round(opt_time, 1),
    }

    # ── 2. Error bars (n_repeats) ──
    ppl_samples_base = []
    ppl_samples_tau = []
    for _ in range(n_repeats):
        ppl_samples_base.append(eval_ppl(val_en, w, nL, C, H, N))
        ppl_samples_tau.append(eval_ppl(val_en, w, nL, C, H, N, tau_v=best_tau))

    error_bars = {
        "base_mean": round(np.mean(ppl_samples_base), 2),
        "base_std": round(np.std(ppl_samples_base), 2),
        "tau_mean": round(np.mean(ppl_samples_tau), 2),
        "tau_std": round(np.std(ppl_samples_tau), 2),
        "delta%": round((np.mean(ppl_samples_tau) - np.mean(ppl_samples_base))
                        / np.mean(ppl_samples_base) * 100, 2),
    }

    # ── 3. Repeat n-gram ──
    rep_results = {}
    for prompt in GEN_PROMPTS[:2]:
        base_text = generate(prompt, w, nL, C, H, N, inject=None,
                            max_new=50, temperature=0.7)
        tau_text = generate(prompt, w, nL, C, H, N,
                           inject={"v": best_tau},
                           max_new=50, temperature=0.7)
        for n in [2, 3]:
            key = f"{prompt[:20]}_repeat-{n}gram"
            rep_results.setdefault(key, {})
            rep_results[key]["base"] = round(repeat_ngram_ratio(base_text, n), 4)
            rep_results[key]["tau"] = round(repeat_ngram_ratio(tau_text, n), 4)

    results = {
        "model": name,
        "cross_domain": cross_domain,
        "error_bars": error_bars,
        "repeat_ngram": rep_results,
    }
    json_path = os.path.join(os.path.dirname(TAU_DIR), f"tau_supplement_{name}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)

    # ── Print summary ──
    print(f"\n{'='*60}")
    print(f"  {name} Supplement Results")
    print(f"{'='*60}")
    print(f"  Cross-domain (τ opt on EN → eval on):")
    print(f"    EN:   base={ppl_en_base:.2f} → tau={ppl_en_tau:.2f}  Δ={cross_domain['en_delta%']:+.2f}%")
    print(f"    ZH:   base={ppl_zh_base:.2f} → tau={ppl_zh_tau:.2f}  Δ={cross_domain['zh_delta%']:+.2f}%")
    print(f"    Code: base={ppl_code_base:.2f} → tau={ppl_code_tau:.2f}  Δ={cross_domain['code_delta%']:+.2f}%")
    print(f"\n  Error bars ({n_repeats} repeats on EN val):")
    print(f"    Base: {error_bars['base_mean']:.2f} ± {error_bars['base_std']:.2f}")
    print(f"    Tau:  {error_bars['tau_mean']:.2f} ± {error_bars['tau_std']:.2f}")
    print(f"    Δ:    {error_bars['delta%']:+.2f}%")
    print(f"\n  Repeat n-gram ratio (lower = less repetitive):")
    for k, v in rep_results.items():
        print(f"    {k}: base={v['base']:.4f} → tau={v['tau']:.4f}")

    return results


if __name__ == "__main__":
    r1 = run("0.4B", MODEL_PATH)
    print("\n" + "=" * 60)
    r2 = run("1.5B", MODEL_PATH_15B)
    print(f"\n  Results saved: tau_supplement_0.4B.json, tau_supplement_1.5B.json")