"""Experiment 1: k-injection grid search (baseline).
Replicates the standard s^τ approach on RWKV-7 key weights."""
import torch, math, time, sys, os

TAU_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TAU_DIR, "deploy_pkg"))
from tau_injection import load_rwkv7, rwkv7_fwd, eval_ppl, init_tokenizer, get_device, load_eval_texts

DEV = get_device()
tok = init_tokenizer()

def run(name, path):
    w, nL, C, H, N = load_rwkv7(path, name)
    texts = load_eval_texts()
    ids_list = [tok.encode(t)[:8] for t in texts]
    train_list, val_list = ids_list[:8], ids_list[8:]

    base_ppl = eval_ppl(val_list, w, nL, C, H, N)
    print(f"  Baseline PPL: {base_ppl:.2f}")

    best_overall = (float("inf"), None, None, None)

    for kw in ["key", "value", "output"]:
        for tau_val in [0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
            w_mod = dict(w)
            for k in w_mod:
                if isinstance(w_mod[k], torch.Tensor) and f".{kw}.weight" in k:
                    w_mod[k] = w_mod[k] * tau_val
            ppl = eval_ppl(val_list, w_mod, nL, C, H, N, tau_v=None)
            delta = (ppl - base_ppl) / base_ppl * 100
            if ppl < best_overall[0]:
                best_overall = (ppl, tau_val, kw, delta)
            if abs(tau_val - 1.0) < 0.01:
                continue

    print(f"  Best: τ={best_overall[1]:.1f} at {best_overall[2]} Δ={best_overall[3]:+.2f}%")

if __name__ == "__main__":
    run("test", list(MODELS.values())[0] if 'MODELS' in dir() else r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")