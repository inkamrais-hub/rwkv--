#!/usr/bin/env python3
"""RWKV-7 injection point sweep: find the least-damped path for s^τ"""
import torch, sys, os, math, time

TAU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TAU_DIR)
DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"GPU: {torch.cuda.get_device_name(0) if DEV=='cuda' else 'CPU'}")

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
tok = TRIE_TOKENIZER(r"D:\python\lib\site-packages\rwkv\rwkv_vocab_v20230424.txt")

EVAL_TEXTS = [
    "The future of artificial intelligence will depend on how we design", "Deep learning applications in healthcare include medical image analysis",
    "Natural language processing has evolved significantly with transformer models", "Quantum computing promises to revolutionize cryptography and optimization",
    "Climate change requires immediate action from governments and corporations", "The history of mathematics stretches back thousands of years",
    "Modern economics relies heavily on statistical models and data", "Space exploration has led to numerous technological innovations",
    "The human brain contains approximately eighty six billion neurons", "Renewable energy sources like solar and wind are becoming cost competitive",
    "Machine learning algorithms can now generate realistic images and text", "The theory of evolution explains the diversity of life",
    "Blockchain technology enables decentralized trust and transparency", "Understanding consciousness remains a challenge in neuroscience",
    "Global supply chains have been disrupted by geopolitical events",
]

def load_rwkv7(path, name):
    raw = torch.load(path, map_location="cpu", weights_only=True)
    nL = max(int(k.split(".")[1]) for k in raw if k.startswith("blocks.")) + 1
    H = int(raw["blocks.0.att.r_k"].shape[0])
    N = int(raw["blocks.0.att.r_k"].shape[1])
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

def rwkv7_fwd_inject(tok_ids, w, nL, C, H, N, inject=None):
    """
    inject: dict with keys 'v', 'g', 'output', 'rk' -> list of per-layer tensors
    """
    T = len(tok_ids)
    tok_ids_cpu = tok_ids.cpu() if isinstance(tok_ids, torch.Tensor) else tok_ids
    x = w["emb.weight"][tok_ids_cpu].float().to(DEV)
    has_inject = inject is not None and any(inject.get(k) is not None for k in inject)
    if has_inject:
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
        xv_ = xx + sx * d[f"{att}.x_v"]
        xa = xx + sx * d[f"{att}.x_a"]
        xg_ = xx + sx * d[f"{att}.x_g"]

        r = xr @ d[f"{att}.receptance.weight"].T
        k = xk @ d[f"{att}.key.weight"].T
        v = xv_ @ d[f"{att}.value.weight"].T
        w_param = torch.tanh(xw @ d[f"{att}.w1"]) @ d[f"{att}.w2"]
        decay = torch.exp(-0.606531 * torch.sigmoid(
            (d[f"{att}.w0"] + w_param).float())).to(DEV)
        a = torch.sigmoid(d[f"{att}.a0"] + (xa @ d[f"{att}.a1"]) @ d[f"{att}.a2"])
        g = torch.sigmoid(xg_ @ d[f"{att}.g1"]) @ d[f"{att}.g2"]
        kk = torch.nn.functional.normalize(
            (k * d[f"{att}.k_k"]).view(T, H, N), dim=-1, p=2.0).view(T, C)
        k_mod = k * (1 + (a - 1) * d[f"{att}.k_a"])
        if bi > 0:
            v0_sq = d[f"{att}.v0"]
            if v0_sq.numel() > 0:
                v = v + (torch.zeros_like(v) - v) * torch.sigmoid(
                    v0_sq + (xv_ @ d[f"{att}.v1"]) @ d[f"{att}.v2"])

        out_a_list = []
        state_att = torch.zeros(H, N, N, device=DEV, dtype=torch.float32)
        tau_v = inject.get("v", [None]*nL)[bi] if inject else None
        v_h = v.view(T, H, N)
        if tau_v is not None:
            v_h = v_h * tau_v.view(1, H, N)

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
            out_a_list.append((state_att @ rt.unsqueeze(2)).view(C))
        out_a = torch.stack(out_a_list, dim=0)

        xx_gn = torch.nn.functional.group_norm(
            out_a, num_groups=H, weight=d[f"{att}.ln_x.weight"],
            bias=d[f"{att}.ln_x.bias"], eps=64e-5)

        r_k_d = d[f"{att}.r_k"].reshape(H, N)
        tau_rk = inject.get("rk", [None]*nL)[bi] if inject else None
        if tau_rk is not None:
            r_k_d = (r_k_d * tau_rk).flatten()
        else:
            r_k_d = r_k_d.flatten()
        rk_res = ((r * k_mod * r_k_d).view(T, H, N).sum(dim=-1, keepdim=True) * v_h).view(T, C)
        xx_gn = xx_gn + rk_res

        tau_g = inject.get("g", [None]*nL)[bi] if inject else None
        if tau_g is not None:
            g = g * tau_g.view(1, C)
        xx_gn = xx_gn * g

        att_out = xx_gn @ d[f"{att}.output.weight"].T
        tau_out = inject.get("output", [None]*nL)[bi] if inject else None
        if tau_out is not None:
            att_out = att_out * tau_out.view(1, C)
        x = x + att_out

        xx2 = torch.nn.functional.layer_norm(x, (C,),
            weight=d[f"{bp}.ln2.weight"], bias=d[f"{bp}.ln2.bias"])
        sx_ffn = torch.cat((torch.zeros(1, C, device=DEV), xx2[:-1])) - xx2
        xk_ffn = xx2 + sx_ffn * d[f"{ffn}.x_k"]
        x = x + (torch.relu(xk_ffn @ d[f"{ffn}.key.weight"].T) ** 2) @ d[f"{ffn}.value.weight"].T

    x_norm = torch.nn.functional.layer_norm(x, (C,),
        weight=w["ln_out.weight"].to(DEV),
        bias=w.get("ln_out.bias", torch.zeros(C)).to(DEV))
    return x_norm @ w["head.weight"].T.float().to(DEV)

def eval_ppl(tok_ids_list, w, nL, C, H, N, inject=None):
    tl, tt = 0.0, 0
    with torch.no_grad():
        for ids in tok_ids_list:
            logits = rwkv7_fwd_inject(ids, w, nL, C, H, N, inject=inject)
            ids_t = torch.tensor(ids, device=DEV)
            L = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:]).item()
            tl += L * (len(ids) - 1)
            tt += len(ids) - 1
    return math.exp(tl / max(tt, 1))

def optimize_inject(train_ids, w, nL, C, H, N, inject_keys, steps=20, lr=0.05):
    """Optimize tau for specified injection points. Returns best inject dict."""
    inject_params = {}
    for key in inject_keys:
        if key == "v":
            inject_params[key] = [torch.ones(H, N, device=DEV, requires_grad=True) for _ in range(nL)]
        elif key == "g":
            inject_params[key] = [torch.ones(H, N, device=DEV, requires_grad=True) for _ in range(nL)]
        elif key == "output":
            inject_params[key] = [torch.ones(1, C, device=DEV, requires_grad=True) for _ in range(nL)]
        elif key == "rk":
            inject_params[key] = [torch.ones(H, N, device=DEV, requires_grad=True) for _ in range(nL)]

    all_params = [p for k in inject_params for p in inject_params[k]]
    ids_t = torch.tensor(train_ids, device=DEV)
    best_loss, best_inject = float("inf"), None

    for step_i in range(steps):
        inject_cur = {k: inject_params[k] for k in inject_params}
        logits = rwkv7_fwd_inject(train_ids, w, nL, C, H, N, inject=inject_cur)
        loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:])
        grads = torch.autograd.grad(loss, all_params, create_graph=False)

        with torch.no_grad():
            for i, p in enumerate(all_params):
                if grads[i] is not None:
                    p.data -= lr * (grads[i] + 0.001 * (p.data - 1.0))
                    p.data.clamp_(0.2, 5.0)
                p.grad = None

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inject = {k: [t.detach().clone() for t in inject_params[k]] for k in inject_params}

        del logits, loss, grads
        torch.cuda.empty_cache()

    return best_inject

def test_injection(model_name, path):
    w, nL, C, H, N = load_rwkv7(path, model_name)
    ids_list = [tok.encode(t)[:8] for t in EVAL_TEXTS]
    n_train = 8
    train_flat = [x for ids in ids_list[:n_train] for x in ids]
    val_list = ids_list[n_train:]

    print(f"  Train: {n_train} texts ({len(train_flat)} tok), Val: {len(val_list)} texts")

    base_ppl = eval_ppl(val_list, w, nL, C, H, N)
    print(f"  Baseline val PPL: {base_ppl:.2f}")

    configs = [
        ("v",          ["v"],      "value (WKV input)"),
        ("g",          ["g"],      "g gate (after GN)"),
        ("output",     ["output"], "output proj (final)"),
        ("rk",         ["rk"],     "r_k (direct shortcut)"),
        ("v+g",        ["v", "g"], "v + g combined"),
        ("v+output",   ["v", "output"], "v + output combined"),
        ("g+output",   ["g", "output"], "g + output combined"),
        ("v+g+output", ["v", "g", "output"], "v + g + output (triple)"),
    ]

    results = {}
    for tag, keys, desc in configs:
        print(f"\n  [{tag}] {desc} ...")
        t0 = time.time()
        best = optimize_inject(train_flat, w, nL, C, H, N, keys, steps=20, lr=0.05)
        ppl = eval_ppl(val_list, w, nL, C, H, N, inject=best)
        delta = (ppl - base_ppl) / base_ppl * 100
        bar = "█" * max(1, int(-delta * 3)) if delta < 0 else "▁" * 5
        print(f"    PPL: {ppl:.2f} ({(ppl - base_ppl)/base_ppl*100:+.2f}%)  {bar}  ({time.time()-t0:.1f}s)")
        results[tag] = {"ppl": ppl, "delta": delta, "desc": desc}

    return results

if __name__ == "__main__":
    paths = [
        ("RWKV-7-0.4B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth"),
        ("RWKV-7-1.5B", r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"),
    ]
    all_results = {}
    for name, path in paths:
        if not os.path.exists(path):
            continue
        print(f"\n{'#'*60}\n# {name}\n{'#'*60}")
        all_results[name] = test_injection(name, path)

    print(f"\n{'#'*60}\n# SUMMARY\n{'#'*60}")
    for model, res in all_results.items():
        print(f"\n{model}:")
        best_tag = min(res, key=lambda k: res[k]["ppl"])
        for tag, r in sorted(res.items(), key=lambda x: x[1]["ppl"]):
            star = " ★" if tag == best_tag else ""
            print(f"  {tag:>12}: {r['ppl']:.2f} ({r['delta']:+.2f}%){star} — {r['desc']}")