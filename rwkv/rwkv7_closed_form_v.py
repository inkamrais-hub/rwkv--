#!/usr/bin/env python3
"""RWKV-7 closed-form τ on VALUE (not key).
v is NOT in ab term → out = s@r is strictly linear in τ⊙v → one gradient step gives exact ∂L/∂τ"""
import torch, sys, os, math, time

TAU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TAU_DIR)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU: {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'CPU'}")

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tok = TRIE_TOKENIZER(r"D:\python\lib\site-packages\rwkv\rwkv_vocab_v20230424.txt")

EVAL_TEXTS = [
    "The future of artificial intelligence will depend on how we design",
    "Deep learning applications in healthcare include medical image analysis",
    "Natural language processing has evolved significantly with transformer models",
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

def rwkv7_fwd(tok_ids, w, nL, C, H, N, tau_list=None):
    """Forward pass. If tau_list provided, scale v at each layer by tau (in-graph)."""
    T = len(tok_ids)
    tok_ids_cpu = tok_ids.cpu() if isinstance(tok_ids, torch.Tensor) else tok_ids
    x = w["emb.weight"][tok_ids_cpu].float().to(DEV)
    if tau_list is not None:
        x = x.requires_grad_(True)

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

        if bi == 0:
            pass
        else:
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

    x_norm = torch.nn.functional.layer_norm(
        x, (C,), weight=w["ln_out.weight"].to(DEV),
        bias=w.get("ln_out.bias", torch.zeros(C)).to(DEV))
    logits = x_norm @ w["head.weight"].T.float().to(DEV)
    return logits

def eval_ppl(tok_ids_list, w, nL, C, H, N, tau_list=None):
    tl, tt = 0.0, 0
    with torch.no_grad():
        for ids in tok_ids_list:
            logits = rwkv7_fwd(ids, w, nL, C, H, N, tau_list=tau_list)
            ids_t = torch.tensor(ids, device=DEV)
            L = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:]).item()
            tl += L * (len(ids) - 1)
            tt += len(ids) - 1
    return math.exp(tl / max(tt, 1))

def optimize_tau(train_ids, w, nL, C, H, N, steps=10, lr=0.1, reg=0.001):
    """Gradient descent on τ (v-injection). Returns optimized τ list (detached)."""
    tau_list = [torch.ones(H, N, device=DEV, requires_grad=True) for _ in range(nL)]
    ids_t = torch.tensor(train_ids, device=DEV)

    for step_i in range(steps):
        logits = rwkv7_fwd(train_ids, w, nL, C, H, N, tau_list=tau_list)
        loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:])

        grads = torch.autograd.grad(loss, tau_list, create_graph=False)

        if step_i == 0 or step_i == steps - 1 or step_i % (max(1, steps // 5)) == 0:
            with torch.no_grad():
                taus = [t.detach() for t in tau_list]
                ppl = eval_ppl([train_ids], w, nL, C, H, N, tau_list=taus)
            print(f"    step {step_i:2d}: loss={loss.item():.4f} PPL={ppl:.2f}")

        del logits, loss
        torch.cuda.empty_cache()

        with torch.no_grad():
            for bi, g in enumerate(grads):
                if g is not None:
                    tau_list[bi].data -= lr * (g + reg * (tau_list[bi].data - 1.0))
                    tau_list[bi].data.clamp_(0.5, 2.0)
                tau_list[bi].grad = None

    return [t.detach().clone() for t in tau_list]

def main(model_name, path):
    w, nL, C, H, N = load_rwkv7(path, model_name)
    eval_ids_list = [tok.encode(t)[:32] for t in EVAL_TEXTS]

    baseline = eval_ppl(eval_ids_list, w, nL, C, H, N)
    print(f"  Baseline PPL: {baseline:.2f}")

    all_ids = [x for ids in eval_ids_list for x in ids]
    if H * N > 2048:
        opt_ids = eval_ids_list[0]
        print(f"  Large model (C={C}), using {len(opt_ids)} tokens for optimization")
    else:
        opt_ids = all_ids
    print(f"\nPhase 1: optimizing τ on {len(opt_ids)} tokens ({H} heads × {N} dims per layer)...")
    t0 = time.time()
    taus_opt = optimize_tau(opt_ids, w, nL, C, H, N, steps=10, lr=0.1)
    print(f"  Done in {time.time()-t0:.1f}s")

    print(f"\nPhase 2: re-evaluating with optimized τ...")
    ppl_opt = eval_ppl(eval_ids_list, w, nL, C, H, N, tau_list=taus_opt)
    d = (ppl_opt - baseline) / baseline * 100

    print(f"\n=== Results ===")
    print(f"  Baseline:  {baseline:.2f}")
    print(f"  v-τ opt:   {ppl_opt:.2f}  ({d:+.2f}%)")
    for bi in range(nL):
        t = taus_opt[bi]
        print(f"  L{bi}: τ mean={t.mean():.3f} std={t.std():.3f} range=[{t.min():.3f},{t.max():.3f}]")

    return ppl_opt, d, taus_opt

if __name__ == "__main__":
    paths = [
        ("RWKV-7-0.4B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth"),
        ("RWKV-7-1.5B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"),
        ("RWKV-7-2.9B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth"),
    ]
    results = {}
    for name, path in paths:
        if not os.path.exists(path):
            print(f"\nSKIP {name}: not found")
            continue
        print(f"\n{'='*60}\n{name}\n{'='*60}")
        ppl, delta, taus = main(name, path)
        results[name] = {"ppl": ppl, "delta": delta}

    print(f"\n{'='*60}\nSummary\n{'='*60}")
    for k, v in results.items():
        print(f"  {k}: PPL={v['ppl']:.2f} ({v['delta']:+.2f}%)")