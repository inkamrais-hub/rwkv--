#!/usr/bin/env python3
"""RWKV-7 s^tau unified injection test — supports 0.1B through 2.9B.
Uses the proven batch-forward approach (same as rwkv7_0.4b_test.py)."""
import torch, sys, os, math, json, time
torch.set_grad_enabled(False)

TAU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TAU_DIR)

DEV = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {torch.cuda.get_device_name(0) if DEV == 'cuda' else 'CPU'}")

EVAL_TEXTS = [
    "The future of artificial intelligence will depend on how we design",
    "Deep learning applications in healthcare include medical image analysis",
    "Natural language processing has evolved significantly with transformer models",
]

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
_vocab = r"D:\python\lib\site-packages\rwkv\rwkv_vocab_v20230424.txt"
tok = TRIE_TOKENIZER(_vocab)
print(f"World tokenizer OK (vocab={_vocab})")

def load_rwkv7(path, name):
    print(f"\n{'='*60}\nLoading {name}\n{'='*60}")
    t0 = time.time()
    raw = torch.load(path, map_location="cpu", weights_only=True)
    nL = max(int(k.split(".")[1]) for k in raw if k.startswith("blocks.")) + 1
    r_k_raw = raw["blocks.0.att.r_k"]
    H, N = int(r_k_raw.shape[0]), int(r_k_raw.shape[1])
    C = H * N
    nv = raw["emb.weight"].shape[0]

    w = {}
    for k, v in raw.items():
        v = v.squeeze()
        if k.endswith(".att.r_k"):
            v = v.flatten()
        w[k] = v.float()

    ln0_w = w["blocks.0.ln0.weight"]
    ln0_b = w["blocks.0.ln0.bias"]
    w["emb.weight"] = torch.nn.functional.layer_norm(
        w["emb.weight"], (C,), weight=ln0_w, bias=ln0_b)

    if "blocks.0.att.v0" not in w:
        w["blocks.0.att.v0"] = torch.empty(0)
        w["blocks.0.att.v1"] = torch.empty(0)
        w["blocks.0.att.v2"] = torch.empty(0)

    print(f"  {nL} layers  C={C}  H={H}  N={N}  V={nv}")
    print(f"  Load time: {time.time()-t0:.1f}s")
    return w, nL, C, H, N, nv

def rwkv7_fwd(tok_ids, w, nL, C, H, N):
    T = len(tok_ids)
    x = w["emb.weight"][tok_ids].float()
    v_first = torch.zeros(C)

    for bi in range(nL):
        bp = f"blocks.{bi}"
        att = f"{bp}.att"
        ffn = f"{bp}.ffn"

        xx = torch.nn.functional.layer_norm(x, (C,),
            weight=w[f"{bp}.ln1.weight"], bias=w[f"{bp}.ln1.bias"])
        sx = torch.cat((torch.zeros(1, C).to(x.device), xx[:-1, :])) - xx

        xr = xx + sx * w[f"{att}.x_r"].squeeze()
        xw = xx + sx * w[f"{att}.x_w"].squeeze()
        xk = xx + sx * w[f"{att}.x_k"].squeeze()
        xv = xx + sx * w[f"{att}.x_v"].squeeze()
        xa = xx + sx * w[f"{att}.x_a"].squeeze()
        xg = xx + sx * w[f"{att}.x_g"].squeeze()

        r  = xr @ w[f"{att}.receptance.weight"].T
        k  = xk @ w[f"{att}.key.weight"].T
        v  = xv @ w[f"{att}.value.weight"].T

        w_param = torch.tanh(xw @ w[f"{att}.w1"]) @ w[f"{att}.w2"]
        w0 = w[f"{att}.w0"].squeeze()
        decay = torch.exp(-0.606531 * torch.sigmoid((w0 + w_param).float()))
        decay = decay.to(x.device)

        a = torch.sigmoid(w[f"{att}.a0"].squeeze() + (xa @ w[f"{att}.a1"]) @ w[f"{att}.a2"])
        g = torch.sigmoid(xg @ w[f"{att}.g1"]) @ w[f"{att}.g2"]

        kk = torch.nn.functional.normalize(
            (k * w[f"{att}.k_k"].squeeze()).view(T, H, N), dim=-1, p=2.0).view(T, C)
        k_mod = k * (1 + (a - 1) * w[f"{att}.k_a"].squeeze())

        if bi == 0:
            v_first = v.clone()
        else:
            v = v + (v_first - v) * torch.sigmoid(
                w[f"{att}.v0"].squeeze() + (xv @ w[f"{att}.v1"]) @ w[f"{att}.v2"])

        out_a = torch.zeros(T, C, device=x.device)
        state_att = torch.zeros(H, N, N, device=x.device, dtype=torch.float32)
        for t in range(T):
            rt = r[t].view(H, N)
            kt = k_mod[t].view(H, N)
            vt = v[t].view(H, N)
            kkt = kk[t].view(H, N)
            at_ = a[t].view(H, N)
            dt = decay[t].view(H, N)

            vk = vt.unsqueeze(2) * kt.unsqueeze(1)
            ab = (-kkt).unsqueeze(2) * (kkt * at_).unsqueeze(1)
            state_att = state_att * dt.unsqueeze(1) + state_att @ ab + vk
            out_a[t] = (state_att @ rt.unsqueeze(2)).view(C)

        xx_gn = torch.nn.functional.group_norm(
            out_a, num_groups=H,
            weight=w[f"{att}.ln_x.weight"],
            bias=w[f"{att}.ln_x.bias"], eps=64e-5)
        r_k = w[f"{att}.r_k"].flatten().to(x.device)
        rk_res = ((r * k_mod * r_k).view(T, H, N).sum(dim=-1, keepdim=True)
                  * v.view(T, H, N)).view(T, C)
        xx_gn = (xx_gn + rk_res) * g
        x = x + xx_gn @ w[f"{att}.output.weight"].T

        xx = torch.nn.functional.layer_norm(x, (C,),
            weight=w[f"{bp}.ln2.weight"], bias=w[f"{bp}.ln2.bias"])
        sx_ffn = torch.cat((torch.zeros(1, C).to(x.device), xx[:-1, :])) - xx
        xk_ffn = xx + sx_ffn * w[f"{ffn}.x_k"].squeeze()
        k_ffn = torch.relu(xk_ffn @ w[f"{ffn}.key.weight"].T) ** 2
        x = x + k_ffn @ w[f"{ffn}.value.weight"].T

    x = torch.nn.functional.layer_norm(
        x, (C,), weight=w["ln_out.weight"],
        bias=w.get("ln_out.bias"))
    return x @ w["head.weight"].T.float()

def eval_ppl(texts, w, nL, C, H, N, max_len=32):
    tl, tt = 0.0, 0
    for txt in texts:
        ids = tok.encode(txt)[:max_len]
        if len(ids) < 3:
            continue
        logits = rwkv7_fwd(ids, w, nL, C, H, N)
        ids_t = torch.tensor(ids)
        loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:])
        tl += loss.item() * (len(ids) - 1)
        tt += len(ids) - 1
    return math.exp(tl / max(tt, 1))

def swprint(point, tau, ppl, base, dt=0):
    d = (ppl - base) / base * 100
    print(f"    {point:20s} tau={str(tau):>6s}  PPL={ppl:.2f}  {d:+.2f}%  ({dt:.1f}s)")

def run_model(path, name):
    w, nL, C, H, N, nv = load_rwkv7(path, name)

    test_ids = tok.encode("Hello world")[:4]
    t0 = time.time()
    lo = rwkv7_fwd(test_ids, w, nL, C, H, N)
    nan = torch.isnan(lo).any().item()
    print(f"  Quick fwd: shape={list(lo.shape)} nan={nan} ({time.time()-t0:.1f}s)")
    if nan:
        print("  FATAL: NaN detected!")
        return

    baseline = eval_ppl(EVAL_TEXTS, w, nL, C, H, N)
    print(f"  Baseline PPL: {baseline:.2f}")
    if baseline > 500:
        print("  WARNING: PPL suspiciously high, test results may be unreliable")

    R = {"model": name, "baseline_ppl": baseline, "C": C, "H": H, "N": N, "V": nv,
         "layers": nL, "sweep": [], "gen": {}}

    sweeps = [
        ("att.w0 (decay)", "blocks.{bi}.att.w0",
         [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5]),
        ("att.key.weight", "blocks.{bi}.att.key.weight",
         [0.5, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5, 2.0, 3.0, 5.0]),
        ("att.value.weight", "blocks.{bi}.att.value.weight",
         [0.7, 0.9, 0.95, 1.0, 1.05, 1.1]),
        ("att.output.weight", "blocks.{bi}.att.output.weight",
         [0.9, 0.95, 1.0, 1.05, 1.1]),
        ("att.receptance.weight", "blocks.{bi}.att.receptance.weight",
         [0.7, 0.9, 0.95, 1.0, 1.05, 1.1]),
        ("att.g1 (gate)", "blocks.{bi}.att.g1",
         [0.7, 0.9, 1.0, 1.1]),
    ]

    print(f"\n  Injection sweep ({len(sweeps)} points):")
    t0 = time.time()
    for pname, tmpl, taus in sweeps:
        for tau in taus:
            sv = []
            for bi in range(nL):
                k = tmpl.format(bi=bi)
                if k in w:
                    sv.append((k, w[k].clone()))
                    w[k] = w[k] * tau
            t1 = time.time()
            p = eval_ppl(EVAL_TEXTS, w, nL, C, H, N)
            R["sweep"].append({"point": pname, "tau": tau, "ppl": p,
                               "delta": (p - baseline) / baseline * 100})
            swprint(pname, tau, p, baseline, time.time() - t1)
            for k, v in sv:
                w[k] = v

    print(f"\n  Combined injections:")
    combos = [
        ("decay=0.3", [("blocks.{bi}.att.w0", 0.3)]),
        ("decay=0.5", [("blocks.{bi}.att.w0", 0.5)]),
        ("key=2.0", [("blocks.{bi}.att.key.weight", 2.0)]),
        ("key=3.0", [("blocks.{bi}.att.key.weight", 3.0)]),
        ("key=5.0", [("blocks.{bi}.att.key.weight", 5.0)]),
        ("decay=0.3+key=3.0", [("blocks.{bi}.att.w0", 0.3),
                               ("blocks.{bi}.att.key.weight", 3.0)]),
        ("decay=0.3+key=5.0", [("blocks.{bi}.att.w0", 0.3),
                               ("blocks.{bi}.att.key.weight", 5.0)]),
        ("decay=0.5+key=2.0", [("blocks.{bi}.att.w0", 0.5),
                               ("blocks.{bi}.att.key.weight", 2.0)]),
        ("decay=0.5+val=0.95", [("blocks.{bi}.att.w0", 0.5),
                                ("blocks.{bi}.att.value.weight", 0.95)]),
    ]
    for cname, params in combos:
        sv = []
        for tmpl, tau in params:
            for bi in range(nL):
                k = tmpl.format(bi=bi)
                if k in w:
                    sv.append((k, w[k].clone()))
                    w[k] = w[k] * tau
        p = eval_ppl(EVAL_TEXTS, w, nL, C, H, N)
        d = (p - baseline) / baseline * 100
        R["sweep"].append({"point": cname, "tau": "combo", "ppl": p, "delta": d})
        print(f"    {cname:25s}  PPL={p:.2f}  {d:+.2f}%")
        for k, v in sv:
            w[k] = v

    sweep_time = time.time() - t0
    print(f"  Sweep total: {sweep_time:.0f}s")

    print(f"\n  Gen test:")
    def gen7(prompt, max_new=40, temp=0.8, top_k=40):
        ids = tok.encode(prompt)
        torch.manual_seed(42)
        for _ in range(max_new):
            logits = rwkv7_fwd(ids, w, nL, C, H, N)
            logits = logits[-1] / temp
            if top_k > 0:
                vals, _ = logits.topk(top_k)
                logits[logits < vals[-1]] = float("-inf")
            probs = torch.softmax(logits.float(), dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
        return tok.decode(ids)

    gprompt = "The future of artificial intelligence"
    gconfigs = [
        ("Baseline", []),
        ("decay=0.3", [("blocks.{bi}.att.w0", 0.3)]),
        ("key=3.0", [("blocks.{bi}.att.key.weight", 3.0)]),
        ("decay=0.3+key=3.0", [("blocks.{bi}.att.w0", 0.3),
                               ("blocks.{bi}.att.key.weight", 3.0)]),
    ]
    for gname, params in gconfigs:
        sv = []
        for tmpl, tau in params:
            for bi in range(nL):
                k = tmpl.format(bi=bi)
                if k in w:
                    sv.append((k, w[k].clone()))
                    w[k] = w[k] * tau
        text = gen7(gprompt)
        for k, v in sv:
            w[k] = v
        R["gen"][gname] = text[:200]
        print(f"    [{gname}] {text[:120]}")

    negs = [r for r in R["sweep"] if r["delta"] < 0]
    if negs:
        best = min(negs, key=lambda r: r["delta"])
        print(f"\n  BEST: {best['point']} tau={best['tau']}  PPL={best['ppl']:.2f}  {best['delta']:+.2f}%")

    out_path = os.path.join(TAU_DIR, "rwkv", name.replace("-", "_").lower() + "_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(R, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {out_path}")

MODELS = [
    (r"ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth",
     "RWKV-7-0.4B"),
    (r"ms_weights\rwkv-7-world\RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth",
     "RWKV-7-1.5B"),
    (r"ms_weights\rwkv-7-world\RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth",
     "RWKV-7-2.9B"),
]

for rel_path, name in MODELS:
    full_path = os.path.join(TAU_DIR, rel_path)
    if not os.path.exists(full_path):
        print(f"\nSKIP {name}: not found at {full_path}")
        continue
    try:
        run_model(full_path, name)
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\nERROR {name}: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "=" * 60)
print("All done!")