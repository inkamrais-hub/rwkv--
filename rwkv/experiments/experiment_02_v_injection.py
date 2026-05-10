"""Experiment 2: v-injection gradient descent (main result).
Exact gradient τ optimization on the value channel."""
import math, time, sys, os

TAU_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(TAU_DIR, "deploy_pkg"))
from tau_injection import load_rwkv7, optimize_tau, eval_ppl, init_tokenizer, get_device, load_eval_texts

DEV = get_device()
tok = init_tokenizer()

def run(name, path):
    w, nL, C, H, N = load_rwkv7(path, name)
    texts = load_eval_texts()
    ids_list = [tok.encode(t)[:8] for t in texts]

    n_train = max(len(ids_list) * 3 // 4, 8)
    train_flat = [x for ids in ids_list[:n_train] for x in ids]
    val_list = ids_list[n_train:]

    base_ppl = eval_ppl(val_list, w, nL, C, H, N)
    print(f"  Baseline PPL: {base_ppl:.2f}")

    t0 = time.time()
    best_tau = optimize_tau(train_flat, w, nL, C, H, N, steps=10, lr=0.05)
    elapsed = time.time() - t0

    opt_ppl = eval_ppl(val_list, w, nL, C, H, N, tau_v=best_tau)
    delta = (opt_ppl - base_ppl) / base_ppl * 100

    tau_vals = torch.stack([t.flatten() for t in best_tau])
    print(f"  τ stats: mean={tau_vals.mean().item():.4f} std={tau_vals.std().item():.4f} "
          f"range=[{tau_vals.min().item():.3f}, {tau_vals.max().item():.3f}]")
    print(f"  Optimized PPL: {opt_ppl:.2f}  Δ={delta:+.2f}%  time={elapsed:.1f}s")
    return best_tau, base_ppl, opt_ppl, delta

if __name__ == "__main__":
    run("test", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")