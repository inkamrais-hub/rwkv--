"""
RWKV-4-Pile-430M s^τ injection + generation quality test.
Also tests RWKV-7 0.4B.
"""
import torch, sys, os, math, json, time
sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, r"F:\τ")
torch.manual_seed(42)

# ===== RWKV-4-430M =====
P = os.path.join(r"F:\τ", "ms_weights", "_ms_cache", "RWKV", "rwkv-4-pile-430m", "RWKV-4-Pile-430M-20220808-8066.pth")
w = torch.load(P, map_location="cpu", weights_only=True)
NB = max(int(k.split(".")[1]) for k in w if k.startswith("blocks."))
C = w["emb.weight"].shape[1]
V = w["emb.weight"].shape[0]
print(f"RWKV-4-430M: {NB+1}L, {C}d, {V}V")

def rwkv_fwd(w, tok_ids):
    """RWKV v4 forward — log-space WKV."""
    T = len(tok_ids)
    x = w["emb.weight"][tok_ids].float()
    C = x.shape[1]
    if "blocks.0.ln0.weight" in w:
        x = torch.layer_norm(x, [C], w["blocks.0.ln0.weight"].float(), w["blocks.0.ln0.bias"].float())
    for bi in range(NB + 1):
        bp = f"blocks.{bi}"
        xx = torch.layer_norm(x, [C], w[f"{bp}.ln1.weight"].float(), w[f"{bp}.ln1.bias"].float())
        mk = w[f"{bp}.att.time_mix_k"].squeeze().float()
        mv = w[f"{bp}.att.time_mix_v"].squeeze().float()
        mr = w[f"{bp}.att.time_mix_r"].squeeze().float()
        kw = w[f"{bp}.att.key.weight"].float()
        vw = w[f"{bp}.att.value.weight"].float()
        rw = w[f"{bp}.att.receptance.weight"].float()
        ow = w[f"{bp}.att.output.weight"].float()
        td = w[f"{bp}.att.time_decay"].float()
        tf = w[f"{bp}.att.time_first"].float()
        state = torch.zeros(C)
        log_aa = tf.clone()
        bb = torch.zeros(C)
        out_a = torch.zeros(T, C)
        for t in range(T):
            xt = xx[t]
            xk = xt * mk + state * (1 - mk)
            xv = xt * mv + state * (1 - mv)
            xr = xt * mr + state * (1 - mr)
            kt = xk @ kw.T
            vt = xv @ vw.T
            rt = xr @ rw.T
            log_old = log_aa + td
            log_new = kt
            mx = torch.max(log_old, log_new)
            log_aa = mx + torch.log1p(torch.exp(torch.min(log_old, log_new) - mx))
            scale_old = torch.exp(log_old - log_aa)
            scale_new = torch.exp(log_new - log_aa)
            bb = bb * scale_old + scale_new * vt
            out_a[t] = torch.sigmoid(rt) * bb
            state = xt
        x = x + out_a @ ow.T
        xx = torch.layer_norm(x, [C], w[f"{bp}.ln2.weight"].float(), w[f"{bp}.ln2.bias"].float())
        mk = w[f"{bp}.ffn.time_mix_k"].squeeze().float()
        mr = w[f"{bp}.ffn.time_mix_r"].squeeze().float()
        kw = w[f"{bp}.ffn.key.weight"].float()
        rw = w[f"{bp}.ffn.receptance.weight"].float()
        vw = w[f"{bp}.ffn.value.weight"].float()
        state = torch.zeros(C)
        out_f = torch.zeros(T, C)
        for t in range(T):
            xt = xx[t]
            xk = xt * mk + state * (1 - mk)
            xr = xt * mr + state * (1 - mr)
            kt = torch.relu(xk @ kw.T) ** 2
            rt = torch.sigmoid(xr @ rw.T)
            out_f[t] = rt * (kt @ vw.T)
            state = xt
        x = x + out_f
    x = torch.layer_norm(x, [C], w["ln_out.weight"].float(), w["ln_out.bias"].float())
    return x @ w["head.weight"].T.float()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(r"F:\τ\ms_weights\rwkv-4-169m", trust_remote_code=True)

def eval_ppl(texts, max_len=64):
    tl, tt = 0.0, 0
    with torch.no_grad():
        for txt in texts:
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_len)
            ids = enc["input_ids"][0]
            if len(ids) < 3: continue
            logits = rwkv_fwd(w, ids)
            loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids[1:])
            tl += loss.item() * (len(ids) - 1)
            tt += len(ids) - 1
    return math.exp(tl / max(tt, 1))

# ===== Verification =====
print("\n" + "=" * 70)
print("CODE VERIFICATION")
print("=" * 70)
test_ids = tok("Hello world this is a test", return_tensors="pt")["input_ids"][0]
with torch.no_grad():
    lo = rwkv_fwd(w, test_ids)
print(f"Forward: shape={lo.shape}, NaN={torch.isnan(lo).any()}")

sv = w["blocks.0.att.time_decay"].clone()
w["blocks.0.att.time_decay"] = w["blocks.0.att.time_decay"] * 1.0
with torch.no_grad():
    lo2 = rwkv_fwd(w, test_ids)
w["blocks.0.att.time_decay"] = sv
diff = (lo - lo2).abs().max().item()
print(f"tau=1.0 identity: max_diff={diff:.2e} {'PASS' if diff < 1e-5 else 'FAIL'}")

texts = [
    "The future of artificial intelligence will depend on how we design",
    "Deep learning applications in healthcare include medical image analysis",
    "Natural language processing has evolved significantly with transformer models",
]

base = eval_ppl(texts, max_len=32)
print(f"Baseline PPL: {base:.2f}")

# ===== Injection Sweep =====
print("\n" + "=" * 70)
print("INJECTION SWEEP")
print("=" * 70)
R = []

def sweep(name, tmpl, taus):
    print(f"\n  {name}:")
    for t in taus:
        sv = []
        for bi in range(NB + 1):
            k = tmpl.format(bi=bi)
            sv.append(w[k].clone()); w[k] = w[k] * t
        p = eval_ppl(texts, max_len=32); dp = (p - base) / base * 100
        R.append({"point": name, "tau": t, "ppl": p, "delta_pct": dp})
        print(f"    tau={t:5.2f}  PPL={p:.2f}  {dp:+.2f}%")
        for i, bi in enumerate(range(NB + 1)):
            w[tmpl.format(bi=bi)] = sv[i]

t0 = time.time()
sweep("att.time_decay", "blocks.{bi}.att.time_decay", [0.3,0.5,0.7,0.9,1.0,1.1,1.3])
sweep("att.key.weight", "blocks.{bi}.att.key.weight", [0.9,0.95,1.0,1.05,1.1,1.5,2.0])
sweep("att.value.weight", "blocks.{bi}.att.value.weight", [0.9,1.0,1.1])
sweep("att.output.weight", "blocks.{bi}.att.output.weight", [0.9,1.0,1.1])
sweep("att.time_first", "blocks.{bi}.att.time_first", [0.5,1.0,1.5])
print(f"\nSweep: {time.time()-t0:.0f}s")

# Combined
print("\n  Combined:")
combos = [
    ("decay=0.5", [("blocks.{bi}.att.time_decay", 0.5)]),
    ("decay=0.3", [("blocks.{bi}.att.time_decay", 0.3)]),
    ("key=2.0", [("blocks.{bi}.att.key.weight", 2.0)]),
    ("key=3.0", [("blocks.{bi}.att.key.weight", 3.0)]),
    ("decay=0.5+key=2.0", [("blocks.{bi}.att.time_decay", 0.5), ("blocks.{bi}.att.key.weight", 2.0)]),
    ("decay=0.3+key=2.0", [("blocks.{bi}.att.time_decay", 0.3), ("blocks.{bi}.att.key.weight", 2.0)]),
]
for cname, params in combos:
    sv = []
    for tmpl, tau in params:
        for bi in range(NB + 1):
            k = tmpl.format(bi=bi)
            sv.append((k, w[k].clone())); w[k] = w[k] * tau
    p = eval_ppl(texts, max_len=32); dp = (p - base) / base * 100
    R.append({"point": cname, "tau": "combo", "ppl": p, "delta_pct": dp})
    print(f"    {cname:25s}  PPL={p:.2f}  {dp:+.2f}%")
    for k, v in sv: w[k] = v

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Baseline PPL={base:.2f}")
for p in sorted(set(r["point"] for r in R)):
    bs = [r for r in R if r["point"] == p]
    b = min(bs, key=lambda x: x["ppl"])
    ts = b["tau"] if isinstance(b["tau"], str) else f'{b["tau"]:5.2f}'
    print(f"  {p:25s}: tau={ts:>8s} PPL={b['ppl']:.2f} delta={b['delta_pct']:+.2f}%")

# ===== Generation =====
print("\n" + "=" * 70)
print("GENERATION QUALITY")
print("=" * 70)

def gen(prompt, max_new=40, temp=0.8, top_k=40):
    ids = tok(prompt, return_tensors="pt")["input_ids"][0].tolist()
    torch.manual_seed(42)
    for _ in range(max_new):
        with torch.no_grad():
            logits = rwkv_fwd(w, torch.tensor(ids))
        logits = logits[-1] / temp
        if top_k > 0:
            vals, _ = logits.topk(top_k)
            logits[logits < vals[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        ids.append(torch.multinomial(probs, 1).item())
    return tok.decode(ids)

gprompts = ["The future of artificial intelligence", "In a distant galaxy,", "Scientists have discovered that"]
gconfigs = [
    ("Baseline", []),
    ("decay=0.5", [("blocks.{bi}.att.time_decay", 0.5)]),
    ("decay=0.3", [("blocks.{bi}.att.time_decay", 0.3)]),
    ("key=2.0", [("blocks.{bi}.att.key.weight", 2.0)]),
    ("decay=0.5+key=2.0", [("blocks.{bi}.att.time_decay", 0.5), ("blocks.{bi}.att.key.weight", 2.0)]),
]

all_gen = {}
for prompt in gprompts:
    print(f"\n{'='*60}\nPrompt: {prompt}\n{'='*60}")
    all_gen[prompt] = {}
    for gname, params in gconfigs:
        sv = []
        for tmpl, tau in params:
            for bi in range(NB + 1):
                k = tmpl.format(bi=bi)
                sv.append((k, w[k].clone())); w[k] = w[k] * tau
        text = gen(prompt)
        for k, v in sv: w[k] = v
        all_gen[prompt][gname] = text
        print(f"\n[{gname}]\n{text}")

out = {"model": "RWKV-4-Pile-430M", "layers": NB+1, "dim": C, "vocab": V,
       "base_ppl": base, "sweep": R, "generation": all_gen}
op = os.path.join(r"F:\τ", "rwkv", "rwkv4_430m_results.json")
with open(op, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {op}")

# ===== RWKV-7 0.4B =====
print("\n\n" + "#" * 70)
print("RWKV-7 0.4B")
print("#" * 70)

P7 = r"F:\τ\ms_weights\rwkv-7-world\RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth"
w7 = torch.load(P7, map_location="cpu", weights_only=True)
NB7 = max(int(k.split(".")[1]) for k in w7 if k.startswith("blocks."))
C7 = w7["emb.weight"].shape[1]
V7 = w7["emb.weight"].shape[0]
H7 = 16  # heads
N7 = C7 // H7  # 64
print(f"RWKV-7-0.4B: {NB7+1}L, {C7}d, {V7}V, {H7}H")

def rwkv7_fwd(w, tok_ids):
    """RWKV v7 forward — based on v6 adapted for v7 params."""
    T = len(tok_ids)
    x = w["emb.weight"][tok_ids].float()
    C = x.shape[1]
    H = H7
    N = C // H
    for bi in range(NB7 + 1):
        bp = f"blocks.{bi}"
        # Layer norm
        xx = torch.layer_norm(x, [C], w[f"{bp}.ln1.weight"].float(), w[f"{bp}.ln1.bias"].float())
        # State mixing (v7 uses direct scalar mixing, not tanh)
        sx = rwkv7_fwd._sx if rwkv7_fwd._sx is not None else torch.zeros(C)
        sx_diff = sx - xx
        # v7 mixing: x_param is [1,1,D], broadcasts over T
        xr = xx + sx_diff * w[f"{bp}.att.x_r"].squeeze()
        xw = xx + sx_diff * w[f"{bp}.att.x_w"].squeeze()
        xk = xx + sx_diff * w[f"{bp}.att.x_k"].squeeze()
        xv = xx + sx_diff * w[f"{bp}.att.x_v"].squeeze()
        xa = xx + sx_diff * w[f"{bp}.att.x_a"].squeeze()
        xg = xx + sx_diff * w[f"{bp}.att.x_g"].squeeze()
        # Projections
        r = xr @ w[f"{bp}.att.receptance.weight"].float().T  # [T, D]
        k = xk @ w[f"{bp}.att.key.weight"].float().T         # [T, D]
        v = xv @ w[f"{bp}.att.value.weight"].float().T       # [T, D]
        # Gate (v7: SiLU of low-rank projection)
        g_in = xg @ w[f"{bp}.att.g1"].float()
        g = torch.sigmoid(g_in @ w[f"{bp}.att.g2"].float())
        # Value modulation (optional, layer 0 doesn't have it)
        if f"{bp}.att.v0" in w:
            v0 = w[f"{bp}.att.v0"].squeeze().float()
            v1 = w[f"{bp}.att.v1"].float()
            v2 = w[f"{bp}.att.v2"].float()
            v = v * (v0 + torch.tanh(v @ v1) @ v2)
        # Decay: w0 + low-rank(wx)
        w0 = w[f"{bp}.att.w0"].squeeze().float()  # [D]
        w1 = w[f"{bp}.att.w1"].float()  # [D, 64]
        w2 = w[f"{bp}.att.w2"].float()  # [64, D]
        # Bonus
        a0 = w[f"{bp}.att.a0"].squeeze().float()  # [D]
        a1 = w[f"{bp}.att.a1"].float()  # [D, 64]
        a2 = w[f"{bp}.att.a2"].float()  # [64, D]
        # Key modulation
        kk = w[f"{bp}.att.k_k"].squeeze().float()  # [D]
        ka = w[f"{bp}.att.k_a"].squeeze().float()  # [D]
        # Receptance-key interaction
        rk = w[f"{bp}.att.r_k"].float()  # [H, N]
        # Layer norm in attention
        lx_w = w[f"{bp}.att.ln_x.weight"].float()
        lx_b = w[f"{bp}.att.ln_x.bias"].float()
        ow = w[f"{bp}.att.output.weight"].float()
        # State: [H, N, N]
        state = torch.zeros(H, N, N)
        out_a = torch.zeros(T, C)
        for t in range(T):
            rt = r[t].view(H, N)  # [H, N]
            kt = k[t].view(H, N)  # [H, N]
            vt = v[t].view(H, N)  # [H, N]
            # Key modulation
            kk_val = kk.view(H, N)
            ka_val = ka.view(H, N)
            kt_boost = kt * (1 + kk_val * (ka_val + 0.5))
            # Decay
            wx_t = xw[t]
            decay_mod = torch.tanh(wx_t @ w1) @ w2  # [D]
            decay = w0.view(H, N) + decay_mod.view(H, N)
            decay = torch.exp(-torch.exp(decay))
            # Bonus
            ax_t = xa[t]
            bonus_mod = torch.tanh(ax_t @ a1) @ a2  # [D]
            bonus = a0.view(H, N) * (bonus_mod + 1).view(H, N)
            # WKV state update
            kv = vt.unsqueeze(2) * kt_boost.unsqueeze(1)  # [H, N, N]
            bonus_kv = bonus.unsqueeze(2) * kt_boost.unsqueeze(1)  # [H, N, N]
            # r modulation
            rk_val = rk  # [H, N]
            r_mod = rt * (1 + rk_val)
            # Output
            out_t = r_mod.unsqueeze(1) @ (bonus_kv + state)  # [H, 1, N]
            out_t = out_t.squeeze(1)  # [H, N]
            # State update
            state = decay.unsqueeze(2) * state + kv
            # Group norm + gate
            out_flat = out_t.flatten()
            out_flat = torch.layer_norm(out_flat, [C], lx_w, lx_b)
            out_flat = out_flat * g[t]
            out_a[t] = out_flat @ ow.T
            state_last = xx[t]
        x = x + out_a
        # FFN (v7 style: mixing with previous state)
        xx = torch.layer_norm(x, [C], w[f"{bp}.ln2.weight"].float(), w[f"{bp}.ln2.bias"].float())
        xk_ffn = xx + (state_last - xx[-1]) * w[f"{bp}.ffn.x_k"].squeeze().float()
        k_ffn = torch.relu(xk_ffn @ w[f"{bp}.ffn.key.weight"].float().T) ** 2
        out_f = k_ffn @ w[f"{bp}.ffn.value.weight"].float().T
        x = x + out_f
        rwkv7_fwd._sx = xx[-1]
    rwkv7_fwd._sx = None
    x = torch.layer_norm(x, [C], w["ln_out.weight"].float(), w["ln_out.bias"].float())
    return x @ w["head.weight"].T.float()

# Test RWKV-7 forward
print("\nTesting RWKV-7 forward...")
tok7 = AutoTokenizer.from_pretrained(r"F:\τ\ms_weights\rwkv-4-169m", trust_remote_code=True)
test_ids7 = tok7("Hello world", return_tensors="pt")["input_ids"][0]
rwkv7_fwd._sx = None
with torch.no_grad():
    lo7 = rwkv7_fwd(w7, test_ids7)
print(f"  Forward: shape={lo7.shape}, NaN={torch.isnan(lo7).any()}")
if not torch.isnan(lo7).any():
    print(f"  Top token: '{tok7.decode([lo7[-1].argmax().item()])}'")

    # Quick PPL
    def eval_ppl7(texts, max_len=32):
        tl, tt = 0.0, 0
        with torch.no_grad():
            for txt in texts:
                enc = tok7(txt, return_tensors="pt", truncation=True, max_length=max_len)
                ids = enc["input_ids"][0]
                if len(ids) < 3: continue
                rwkv7_fwd._sx = None
                logits = rwkv7_fwd(w7, ids)
                loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids[1:])
                tl += loss.item() * (len(ids) - 1)
                tt += len(ids) - 1
        return math.exp(tl / max(tt, 1))

    texts7 = [
        "The future of artificial intelligence will depend on how we design",
        "Deep learning applications in healthcare include medical image analysis",
        "Natural language processing has evolved significantly with transformer models",
    ]
    base7 = eval_ppl7(texts7, max_len=32)
    print(f"  Baseline PPL: {base7:.2f}")

    # Injection sweep on key points
    print("\n  RWKV-7 injection sweep:")
    R7 = []
    for name, tmpl, taus in [
        ("w0 (decay)", "blocks.{bi}.att.w0", [0.3, 0.5, 0.7, 1.0, 1.5]),
        ("key.weight", "blocks.{bi}.att.key.weight", [0.9, 1.0, 1.1, 1.5, 2.0]),
        ("output.weight", "blocks.{bi}.att.output.weight", [0.9, 1.0, 1.1]),
    ]:
        print(f"\n    {name}:")
        for t in taus:
            sv = []
            for bi in range(NB7 + 1):
                k = tmpl.format(bi=bi)
                sv.append(w7[k].clone())
                w7[k] = w7[k] * t
            p = eval_ppl7(texts7, max_len=32)
            dp = (p - base7) / base7 * 100
            R7.append({"point": name, "tau": t, "ppl": p, "delta_pct": dp})
            print(f"      tau={t:5.2f}  PPL={p:.2f}  {dp:+.2f}%")
            for i, bi in enumerate(range(NB7 + 1)):
                w7[tmpl.format(bi=bi)] = sv[i]

    # Generation
    print("\n  RWKV-7 Generation:")
    def gen7(prompt, max_new=40, temp=0.8, top_k=40):
        ids = tok7(prompt, return_tensors="pt")["input_ids"][0].tolist()
        torch.manual_seed(42)
        for _ in range(max_new):
            rwkv7_fwd._sx = None
            with torch.no_grad():
                logits = rwkv7_fwd(w7, torch.tensor(ids))
            logits = logits[-1] / temp
            if top_k > 0:
                vals, _ = logits.topk(top_k)
                logits[logits < vals[-1]] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            ids.append(torch.multinomial(probs, 1).item())
        return tok7.decode(ids)

    for prompt in ["The future of artificial intelligence", "In a distant galaxy,"]:
        print(f"\n  Prompt: {prompt}")
        # Baseline
        print(f"  [Baseline] {gen7(prompt)}")
        # w0=0.5
        sv = []
        for bi in range(NB7 + 1):
            k = f"blocks.{bi}.att.w0"
            sv.append(w7[k].clone()); w7[k] = w7[k] * 0.5
        print(f"  [w0=0.5]  {gen7(prompt)}")
        for i, bi in enumerate(range(NB7 + 1)):
            w7[f"blocks.{bi}.att.w0"] = sv[i]
        # key=1.5
        sv = []
        for bi in range(NB7 + 1):
            k = f"blocks.{bi}.att.key.weight"
            sv.append(w7[k].clone()); w7[k] = w7[k] * 1.5
        print(f"  [key=1.5] {gen7(prompt)}")
        for i, bi in enumerate(range(NB7 + 1)):
            w7[f"blocks.{bi}.att.key.weight"] = sv[i]

    # Save RWKV-7 results
    out7 = {"model": "RWKV-7-World-0.4B", "layers": NB7+1, "dim": C7, "vocab": V7,
            "base_ppl": base7, "sweep": R7}
    op7 = os.path.join(r"F:\τ", "rwkv", "rwkv7_0.4b_results.json")
    with open(op7, "w", encoding="utf-8") as f:
        json.dump(out7, f, indent=2, ensure_ascii=False)
    print(f"\n  Saved to {op7}")
else:
    print("  NaN in forward — implementation needs fixing")
