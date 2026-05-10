#!/usr/bin/env python3
"""RWKV-7 τ dynamics deep analysis: cliff effect, internal distribution, effective rank, attention, sparsity"""
import torch, sys, os, math, time, json
import numpy as np

TAU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TAU_DIR)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU: {torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'}")

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tok = TRIE_TOKENIZER(r"D:\python\lib\site-packages\rwkv\rwkv_vocab_v20230424.txt")

TEXTS = [
    "The future of artificial intelligence will depend on how we design systems that can learn",
    "Deep learning applications in healthcare include medical image analysis and diagnosis",
    "Natural language processing has evolved significantly with transformer models and attention",
    "Quantum computing promises to revolutionize cryptography and optimization problems",
    "Climate change requires immediate action from governments and corporations worldwide",
    "The history of mathematics stretches back thousands of years to ancient civilizations",
    "Modern economics relies heavily on statistical models and data-driven decision making",
    "Space exploration has led to numerous technological innovations benefiting life on Earth",
    "The human brain contains approximately eighty six billion neurons forming complex networks",
    "Renewable energy sources like solar and wind are becoming increasingly cost competitive",
    "Machine learning algorithms can now generate realistic images and coherent text",
    "The theory of evolution by natural selection explains the diversity of life on Earth",
    "Blockchain technology enables decentralized trust and transparent transaction records",
    "Understanding consciousness remains one of the greatest challenges in neuroscience",
    "Global supply chains have been disrupted by recent geopolitical events and pandemics",
    "The development of agriculture marked a fundamental shift in human civilization",
    "Particle physics explores the fundamental constituents of matter and their interactions",
    "Cybersecurity threats continue to evolve as technology becomes more deeply integrated",
]

def load_rwkv7(path, name):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    nL = max(int(k.split(".")[1]) for k in raw if k.startswith("blocks.")) + 1
    H, N = int(raw["blocks.0.att.r_k"].shape[0]), int(raw["blocks.0.att.r_k"].shape[1])
    C = H * N
    w = {}
    for k, v in raw.items():
        v = v.squeeze()
        if k.endswith(".att.r_k"):
            v = v.flatten()
        w[k] = v.float()
    w["emb.weight"] = torch.nn.functional.layer_norm(
        w["emb.weight"], (C,), weight=w["blocks.0.ln0.weight"], bias=w["blocks.0.ln0.bias"])
    if "blocks.0.att.v0" not in w:
        w["blocks.0.att.v0"] = torch.empty(0)
        w["blocks.0.att.v1"] = torch.empty(0)
        w["blocks.0.att.v2"] = torch.empty(0)
    print(f"  {name}: {nL}L C={C} H={H} N={N}")
    return w, nL, C, H, N

def rwkv7_fwd_detailed(tok_ids, w, nL, C, H, N, tau_list=None):
    """Forward with full state capture. Returns (logits, snapshots_dict)."""
    T = len(tok_ids)
    tok_ids_cpu = tok_ids.cpu() if isinstance(tok_ids, torch.Tensor) else tok_ids
    x = w["emb.weight"][tok_ids_cpu].float().to(DEV)
    if tau_list is not None:
        x = x.requires_grad_(True)

    snap = {"x_per_layer": [], "state_svals": [], "state_norms": [], "out_norms": []}

    for bi in range(nL):
        bp = f"blocks.{bi}"
        att = f"{bp}.att"
        ffn = f"{bp}.ffn"

        d = {}
        for k in w:
            if k.startswith(bp) and isinstance(w[k], torch.Tensor):
                d[k] = w[k].to(DEV)

        xx = torch.nn.functional.layer_norm(x, (C,),
            weight=d[f"{bp}.ln1.weight"], bias=d[f"{bp}.ln1.bias"])
        sx = torch.cat((torch.zeros(1, C, device=DEV), xx[:-1])) - xx

        xr = xx + sx * d[f"{att}.x_r"]
        xw = xx + sx * d[f"{att}.x_w"]
        xk = xx + sx * d[f"{att}.x_k"]
        xv = xx + sx * d[f"{att}.x_v"]
        xa = xx + sx * d[f"{att}.x_a"]
        xg = xx + sx * d[f"{att}.x_g"]

        r = xr @ d[f"{att}.receptance.weight"].T
        k = xk @ d[f"{att}.key.weight"].T
        v = xv @ d[f"{att}.value.weight"].T

        w_param = torch.tanh(xw @ d[f"{att}.w1"]) @ d[f"{att}.w2"]
        decay = torch.exp(-0.606531 * torch.sigmoid(
            (d[f"{att}.w0"] + w_param).float())).to(DEV)

        a = torch.sigmoid(d[f"{att}.a0"] + (xa @ d[f"{att}.a1"]) @ d[f"{att}.a2"])
        g = torch.sigmoid(xg @ d[f"{att}.g1"]) @ d[f"{att}.g2"]

        kk = torch.nn.functional.normalize(
            (k * d[f"{att}.k_k"]).view(T, H, N), dim=-1, p=2.0).view(T, C)
        k_mod = k * (1 + (a - 1) * d[f"{att}.k_a"])

        if bi > 0:
            v0_sq = d[f"{att}.v0"]
            if v0_sq.numel() > 0:
                v = v + (torch.zeros_like(v) - v) * torch.sigmoid(
                    v0_sq + (xv @ d[f"{att}.v1"]) @ d[f"{att}.v2"])

        out_a_list = []
        state_att = torch.zeros(H, N, N, device=DEV, dtype=torch.float32)
        tauv = tau_list[bi] if tau_list is not None else None

        v_h = v.view(T, H, N)
        if tauv is not None:
            v_h = v_h * tauv.view(1, H, N)

        state_snaps = []
        for t_idx in range(T):
            rt = r[t_idx].view(H, N)
            kt = k_mod[t_idx].view(H, N)
            vt = v_h[t_idx]
            kkt = kk[t_idx].view(H, N)
            at_ = a[t_idx].view(H, N)
            dt = decay[t_idx].view(H, N)

            vk = vt.unsqueeze(2) * kt.unsqueeze(1)
            ab = (-kkt).unsqueeze(2) * (kkt * at_).unsqueeze(1)
            state_att = state_att * dt.unsqueeze(1) + state_att @ ab + vk
            o = (state_att @ rt.unsqueeze(2)).view(C)
            out_a_list.append(o)

            if T <= 32 or t_idx % max(1, T // 8) == 0:
                state_snaps.append(state_att.detach().clone().cpu())

        out_a = torch.stack(out_a_list, dim=0)

        r_k_d = d[f"{att}.r_k"].flatten()
        xx_gn = torch.nn.functional.group_norm(
            out_a, num_groups=H, weight=d[f"{att}.ln_x.weight"],
            bias=d[f"{att}.ln_x.bias"], eps=64e-5)
        rk_res = ((r * k_mod * r_k_d).view(T, H, N).sum(dim=-1, keepdim=True)
                  * v_h).view(T, C)
        xx_gn = (xx_gn + rk_res) * g
        x = x + xx_gn @ d[f"{att}.output.weight"].T

        xx2 = torch.nn.functional.layer_norm(x, (C,),
            weight=d[f"{bp}.ln2.weight"], bias=d[f"{bp}.ln2.bias"])
        sx_ffn = torch.cat((torch.zeros(1, C, device=DEV), xx2[:-1])) - xx2
        xk_ffn = xx2 + sx_ffn * d[f"{ffn}.x_k"]
        k_ffn = torch.relu(xk_ffn @ d[f"{ffn}.key.weight"].T) ** 2
        x = x + k_ffn @ d[f"{ffn}.value.weight"].T

        snap["x_per_layer"].append(x.detach().cpu().norm(dim=-1).mean().item())
        snap["out_norms"].append(out_a.detach().cpu().norm(dim=-1).mean().item())
        if state_snaps:
            sv_norms = [s.norm().item() for s in state_snaps]
            snap["state_norms"].append(np.mean(sv_norms))
            fst = state_snaps[-1].float()
            svals = torch.linalg.svdvals(fst).cpu().numpy()
            snap["state_svals"].append(svals)

    x_norm = torch.nn.functional.layer_norm(
        x, (C,), weight=w["ln_out.weight"].to(DEV),
        bias=w.get("ln_out.bias", torch.zeros(C)).to(DEV))
    logits = x_norm @ w["head.weight"].T.float().to(DEV)
    return logits, snap

def eval_ppl(tok_ids_list, w, nL, C, H, N, tau_list=None):
    tl, tt = 0.0, 0
    with torch.no_grad():
        for ids in tok_ids_list:
            logits, _ = rwkv7_fwd_detailed(ids, w, nL, C, H, N, tau_list=tau_list)
            ids_t = torch.tensor(ids, device=DEV)
            L = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:]).item()
            tl += L * (len(ids) - 1)
            tt += len(ids) - 1
    return math.exp(tl / max(tt, 1))

def optimize_tau_track(train_ids, val_ids_list, w, nL, C, H, N, steps=80, lr=0.05, reg=0.001):
    """Run GD, track train loss + val PPL at every step."""
    tau_list = [torch.ones(H, N, device=DEV, requires_grad=True) for _ in range(nL)]
    ids_t = torch.tensor(train_ids, device=DEV)

    history = []
    best_val_ppl = float("inf")
    best_tau_snapshot = None
    for step_i in range(steps):
        logits, _ = rwkv7_fwd_detailed(train_ids, w, nL, C, H, N, tau_list=tau_list)
        loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:])
        grads = torch.autograd.grad(loss, tau_list, create_graph=False)
        del logits

        with torch.no_grad():
            gn = sum(g.norm().item() for g in grads if g is not None)
            tch = sum((t.data - 1.0).norm().item() for t in tau_list)
            for bi, g in enumerate(grads):
                if g is not None:
                    tau_list[bi].data -= lr * (g + reg * (tau_list[bi].data - 1.0))
                    tau_list[bi].data.clamp_(0.5, 2.0)
                tau_list[bi].grad = None

        taus_d = [t.detach().clone() for t in tau_list]
        val_ppl = eval_ppl(val_ids_list, w, nL, C, H, N, tau_list=taus_d)
        train_ppl = math.exp(loss.item())

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_tau_snapshot = [t.detach().clone() for t in tau_list]

        history.append({
            "step": step_i, "loss": loss.item(), "train_ppl": train_ppl,
            "val_ppl": val_ppl, "grad_norm": gn, "tau_change": tch,
        })
        if step_i % 10 == 0 or step_i < 5:
            print(f"  step {step_i:3d}: train={train_ppl:.2f} val={val_ppl:.2f} τΔ={tch:.4f}")

        del loss, grads
        torch.cuda.empty_cache()

    return tau_list, best_tau_snapshot, history

def compute_effective_rank(svals, threshold=0.9):
    """Effective rank: r where cumulative energy > threshold."""
    total = np.sum(svals**2, axis=-1)
    cumsum = np.cumsum(np.sort(svals**2, axis=-1)[:, ::-1], axis=-1)
    ranks = []
    for h in range(svals.shape[0]):
        ratio = cumsum[h] / max(total[h], 1e-12)
        r = np.searchsorted(ratio, threshold) + 1
        ranks.append(min(r, svals.shape[1]))
    return np.mean(ranks)

def analyze_token_dynamics(logits):
    """Per-position: entropy, top-5 prob mass, repetition tendency."""
    probs = torch.softmax(logits.float(), dim=-1)
    entropies = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
    top5, _ = torch.topk(probs, 5, dim=-1)
    top5_mass = top5.sum(dim=-1)
    sparsity = ((probs > 0.01).float().sum(dim=-1) / probs.shape[-1])
    return {
        "entropy_mean": entropies.mean().item(),
        "entropy_std": entropies.std().item(),
        "top5_mass_mean": top5_mass.mean().item(),
        "sparsity_mean": sparsity.mean().item(),
    }

def run_analysis(model_name, path):
    w, nL, C, H, N = load_rwkv7(path, model_name)

    all_ids_list = [tok.encode(t)[:8] for t in TEXTS]
    n_train = max(len(all_ids_list) * 3 // 4, 8)
    train_list = all_ids_list[:n_train]
    val_list = all_ids_list[n_train:]
    train_flat = [x for ids in train_list for x in ids]
    val_flat = [x for ids in val_list for x in ids]
    print(f"  Train: {n_train} texts ({len(train_flat)} tok), Val: {len(val_list)} texts ({len(val_flat)} tok)")

    print(f"\n{'='*60}")
    print(f"1. CLIFF EXPERIMENT: 80-step GD with train/val tracking")
    print(f"{'='*60}")
    t_final, t_best, history = optimize_tau_track(train_flat, val_list, w, nL, C, H, N, steps=80, lr=0.05)

    vals = [h["val_ppl"] for h in history]
    best_step = int(np.argmin(vals))
    best_val = vals[best_step]
    start_val = vals[0]
    print(f"\n  Baseline val PPL: {start_val:.2f}")
    print(f"  Best val PPL:    {best_val:.2f} at step {best_step} ({ (best_val-start_val)/start_val*100:+.2f}%)")
    print(f"  Final val PPL:   {vals[-1]:.2f} ({ (vals[-1]-start_val)/start_val*100:+.2f}%)")

    cliff_found = False
    for i in range(best_step + 1, len(vals) - 1):
        if vals[i] > best_val * 1.02 and vals[i+1] > vals[i]:
            print(f"  CLIFF DETECTED: val PPL rises at step {i} (best was {best_step})")
            cliff_found = True
            break
    if not cliff_found:
        print(f"  No sharp cliff — val PPL stable or gradual decline")

    best_taus = t_best if t_best is not None else [t.detach().clone() for t in t_final]

    print(f"\n{'='*60}")
    print(f"2. INTERNAL DISTRIBUTION: baseline vs optimized τ")
    print(f"{'='*60}")
    sample_ids = all_ids_list[0]
    with torch.no_grad():
        _, snap_base = rwkv7_fwd_detailed(sample_ids, w, nL, C, H, N)
        _, snap_opt = rwkv7_fwd_detailed(sample_ids, w, nL, C, H, N, tau_list=best_taus)

    print(f"\n  Layer output L2 norm (mean across positions):")
    print(f"  {'Layer':>6} {'Base':>10} {'Opt':>10} {'Δ%':>8}")
    for bi in range(nL):
        b, o = snap_base["x_per_layer"][bi], snap_opt["x_per_layer"][bi]
        print(f"  {bi:>6} {b:>10.4f} {o:>10.4f} {(o-b)/max(b,1e-8)*100:>+7.1f}%")

    print(f"\n  WKV state norm (final timestep, mean across heads):")
    print(f"  {'Layer':>6} {'Base':>10} {'Opt':>10} {'Δ%':>8}")
    for bi in range(nL):
        b, o = snap_base["state_norms"][bi], snap_opt["state_norms"][bi]
        print(f"  {bi:>6} {b:>10.4f} {o:>10.4f} {(o-b)/max(b,1e-8)*100:>+7.1f}%")

    print(f"\n{'='*60}")
    print(f"3. EFFECTIVE RANK & ATTENTION SPECTRUM")
    print(f"{'='*60}")

    def eff_rank_at(svals, thresh):
        return compute_effective_rank(svals, thresh)

    key_layers = [0, nL//3, 2*nL//3, nL-1]
    for bi in key_layers:
        b_sv = snap_base["state_svals"][bi]
        o_sv = snap_opt["state_svals"][bi]
        for tname, th in [("90%", 0.90), ("95%", 0.95), ("99%", 0.99)]:
            br = eff_rank_at(b_sv, th)
            or_ = eff_rank_at(o_sv, th)
            print(f"  L{bi} {tname}: base={br:.1f} opt={or_:.1f}  Δ={or_-br:+.1f}")

    top5_sv_b = snap_base["state_svals"][-1][0][:5]
    top5_sv_o = snap_opt["state_svals"][-1][0][:5]
    print(f"\n  Last layer, head 0 top-5 singvals: base={top5_sv_b} opt={top5_sv_o}")

    print(f"\n{'='*60}")
    print(f"4. TOKEN PREDICTION DYNAMICS")
    print(f"{'='*60}")
    all_ids_t = torch.tensor(all_ids_list[0], device=DEV)
    with torch.no_grad():
        lb, _ = rwkv7_fwd_detailed(all_ids_list[0], w, nL, C, H, N)
        lo, _ = rwkv7_fwd_detailed(all_ids_list[0], w, nL, C, H, N, tau_list=best_taus)
    tb = analyze_token_dynamics(lb)
    to = analyze_token_dynamics(lo)
    for k in tb:
        print(f"  {k}: base={tb[k]:.4f} opt={to[k]:.4f}")

    print(f"\n{'='*60}")
    print(f"5. SPARSITY: Gini coefficient of state_att")
    print(f"{'='*60}")
    def gini(x):
        s = torch.sort(x.flatten())[0]
        n = s.shape[0]
        if torch.sum(s) == 0:
            return 0.0
        return (2 * torch.sum((torch.arange(1, n+1, device=s.device).float() * s)) / (n * torch.sum(s)) - (n + 1) / n).item()

    key_layers = [0, nL//3, 2*nL//3, nL-1]
    for name, snap in [("base", snap_base), ("opt", snap_opt)]:
        print(f"  {name}:")
        for bi in key_layers:
            sv = snap["state_svals"][bi]
            g_h0 = gini(torch.tensor(sv[0]))
            g_all = gini(torch.tensor(sv.ravel()))
            print(f"    L{bi}: Gini(h0)={g_h0:.3f} Gini(all)={g_all:.3f}")

    full_val_ppl = eval_ppl(val_list, w, nL, C, H, N)
    full_val_opt = eval_ppl(val_list, w, nL, C, H, N, tau_list=best_taus)
    print(f"\n  Full validation PPL: base={full_val_ppl:.2f} → opt={full_val_opt:.2f} ({(full_val_opt-full_val_ppl)/full_val_ppl*100:+.2f}%)")

    return {
        "model": model_name, "baseline_val": full_val_ppl, "opt_val": full_val_opt,
        "best_step": best_step, "history": history,
        "layer_out_delta": [(snap_opt["x_per_layer"][i] - snap_base["x_per_layer"][i]) / max(snap_base["x_per_layer"][i], 1e-8) * 100 for i in range(nL)],
        "state_norm_delta": [(snap_opt["state_norms"][i] - snap_base["state_norms"][i]) / max(snap_base["state_norms"][i], 1e-8) * 100 for i in range(nL)],
        "token_dynamics": {"base": tb, "opt": to},
    }

if __name__ == "__main__":
    paths = [
        ("RWKV-7-0.4B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth"),
        ("RWKV-7-1.5B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"),
    ]
    all_results = {}
    for name, path in paths:
        if not os.path.exists(path):
            print(f"\nSKIP {name}: not found")
            continue
        print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
        all_results[name] = run_analysis(name, path)

    with open("rwkv/tau_dynamics_analysis.json", "w") as f:
        json.dump(all_results, f, indent=2, default=lambda x: x.tolist() if isinstance(x, (np.ndarray, torch.Tensor)) else str(x))
    print(f"\nSaved: rwkv/tau_dynamics_analysis.json")