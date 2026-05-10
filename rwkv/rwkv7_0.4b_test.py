import torch, sys, os, math, json, time
sys.stdout.reconfigure(encoding="utf-8")

# Find tau dir dynamically
TAU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TAU_DIR)
torch.manual_seed(42)

# ===== Load RWKV-7 0.4B =====
P7 = os.path.join(TAU_DIR, "ms_weights", "rwkv-7-world", "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")
print(f"Loading RWKV-7-0.4B from {P7}...")
w7_raw = torch.load(P7, map_location="cpu", weights_only=True)

# Detect architecture
NB7 = max(int(k.split(".")[1]) for k in w7_raw if k.startswith("blocks."))
H7, N7 = w7_raw["blocks.0.att.r_k"].shape
C7 = H7 * N7
V7 = w7_raw["emb.weight"].shape[0]
print(f"RWKV-7-0.4B: {NB7+1}L, {C7}d, {V7}V, {H7}H, head_size={N7}")

# ===== Weight preprocessing =====
w7 = {}
for k, v in w7_raw.items():
    v = v.squeeze()
    if k.endswith(".att.r_k"):
        v = v.flatten()
    w7[k] = v.float()

# Pre-normalize embedding
ln0_w = w7["blocks.0.ln0.weight"]
ln0_b = w7["blocks.0.ln0.bias"]
w7["emb.weight"] = torch.nn.functional.layer_norm(
    w7["emb.weight"].float(), (C7,), weight=ln0_w, bias=ln0_b
)
w7["blocks.0.att.v0"] = torch.empty(0)
w7["blocks.0.att.v1"] = torch.empty(0)
w7["blocks.0.att.v2"] = torch.empty(0)
print("Weights preprocessed.")

# ===== Debug helper =====
DEBUG = False
def dbg(name, t):
    if DEBUG:
        print(f"  {name:30s}: shape={list(t.shape)}, mean={t.float().mean():.6f}, std={t.float().std():.6f}, "
              f"min={t.float().min():.4f}, max={t.float().max():.4f}, nan={torch.isnan(t).any()}")

# ===== RWKV-7 Forward (pure Python, matching rwkv package non-CUDA path) =====
def rwkv7_fwd(tok_ids, w=w7, nb=NB7, c=C7, h=H7, n=N7):
    T = len(tok_ids)
    x = w["emb.weight"][tok_ids].float()
    v_first = torch.zeros(c)

    for bi in range(nb + 1):
        bp = f"blocks.{bi}"
        att = f"{bp}.att"
        ffn = f"{bp}.ffn"
        state_att = torch.zeros(h, n, n)

        # Attention
        xx = torch.nn.functional.layer_norm(
            x, (c,), weight=w[f"{bp}.ln1.weight"], bias=w[f"{bp}.ln1.bias"]
        )
        sx = torch.cat((torch.zeros(c).unsqueeze(0), xx[:-1, :])) - xx

        xr = xx + sx * w[f"{att}.x_r"].squeeze()
        xw = xx + sx * w[f"{att}.x_w"].squeeze()
        xk = xx + sx * w[f"{att}.x_k"].squeeze()
        xv = xx + sx * w[f"{att}.x_v"].squeeze()
        xa = xx + sx * w[f"{att}.x_a"].squeeze()
        xg = xx + sx * w[f"{att}.x_g"].squeeze()

        r = xr @ w[f"{att}.receptance.weight"].T
        k = xk @ w[f"{att}.key.weight"].T
        v = xv @ w[f"{att}.value.weight"].T

        w_param = torch.tanh(xw @ w[f"{att}.w1"]) @ w[f"{att}.w2"]
        w0 = w[f"{att}.w0"].squeeze()
        decay = torch.exp(-0.606531 * torch.sigmoid((w0 + w_param).float()))

        a = torch.sigmoid(w[f"{att}.a0"].squeeze() + (xa @ w[f"{att}.a1"]) @ w[f"{att}.a2"])
        g = torch.sigmoid(xg @ w[f"{att}.g1"]) @ w[f"{att}.g2"]

        kk = torch.nn.functional.normalize(
            (k * w[f"{att}.k_k"].squeeze()).view(T, h, n), dim=-1, p=2.0
        ).view(T, c)
        k_mod = k * (1 + (a - 1) * w[f"{att}.k_a"].squeeze())

        if bi == 0:
            v_first = v.clone()
        else:
            v = v + (v_first - v) * torch.sigmoid(
                w[f"{att}.v0"].squeeze() + (xv @ w[f"{att}.v1"]) @ w[f"{att}.v2"]
            )

        out_a = torch.zeros(T, c)
        for t in range(T):
            rt = r[t].view(h, n)
            kt = k_mod[t].view(h, n)
            vt = v[t].view(h, n)
            kkt = kk[t].view(h, n)
            at = a[t].view(h, n)
            dt = decay[t].view(h, n)
            vk = vt.unsqueeze(2) * kt.unsqueeze(1)
            ab = (-kkt).unsqueeze(2) * (kkt * at).unsqueeze(1)
            state_att = state_att * dt.unsqueeze(1) + state_att @ ab.float() + vk.float()
            out_a[t] = (state_att @ rt.unsqueeze(2)).view(c)

        xx_gn = torch.nn.functional.group_norm(
            out_a.view(T, c), num_groups=h,
            weight=w[f"{att}.ln_x.weight"], bias=w[f"{att}.ln_x.bias"], eps=64e-5
        )
        r_k = w[f"{att}.r_k"].flatten()
        rk_res = ((r * k_mod * r_k).view(T, h, n).sum(dim=-1, keepdim=True) * v.view(T, h, n)).view(T, c)
        xx_gn = (xx_gn + rk_res) * g
        x = x + xx_gn @ w[f"{att}.output.weight"].T

        # FFN
        xx = torch.nn.functional.layer_norm(
            x, (c,), weight=w[f"{bp}.ln2.weight"], bias=w[f"{bp}.ln2.bias"]
        )
        sx_ffn = torch.cat((torch.zeros(c).unsqueeze(0), xx[:-1, :])) - xx
        xk_ffn = xx + sx_ffn * w[f"{ffn}.x_k"].squeeze()
        k_ffn = torch.relu(xk_ffn @ w[f"{ffn}.key.weight"].T) ** 2
        x = x + k_ffn @ w[f"{ffn}.value.weight"].T

    x = torch.nn.functional.layer_norm(x, (c,), weight=w["ln_out.weight"], bias=w["ln_out.bias"])
    return x @ w["head.weight"].T.float()

# ===== Tokenizer (RWKV World tokenizer for v7) =====
from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
_world_vocab = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "lib", "site-packages", "rwkv", "rwkv_vocab_v20230424.txt")
if not os.path.exists(_world_vocab):
    # fallback
    for p in [r"D:\python\lib\site-packages\rwkv\rwkv_vocab_v20230424.txt",
              r"C:\Python310\Lib\site-packages\rwkv\rwkv_vocab_v20230424.txt"]:
        if os.path.exists(p):
            _world_vocab = p
            break
tok = TRIE_TOKENIZER(_world_vocab)
print(f"World tokenizer loaded from: {_world_vocab}")
print(f"Vocab test: 'Hello world' -> {tok.encode('Hello world')} -> {[tok.decode([i]) for i in tok.encode('Hello world')]}")

# ===== Verification =====
print("\n" + "=" * 70)
print("RWKV-7 FORWARD VERIFICATION")
print("=" * 70)
test_ids = tok.encode("Hello world")
print(f"Token IDs: {test_ids} -> {[tok.decode([i]) for i in test_ids]}")
t0 = time.time()
with torch.no_grad():
    lo = rwkv7_fwd(test_ids)
dt = time.time() - t0
print(f"Forward: shape={lo.shape}, NaN={torch.isnan(lo).any()}, {dt:.1f}s")
if torch.isnan(lo).any():
    print("NaN detected! Exiting.")
    sys.exit(1)
print(f"Top-5 last pos: {[(tok.decode([i.item()]), f'{lo[-1][i].item():.2f}') for i in lo[-1].topk(5).indices]}")

# Quick PPL
def eval_ppl7(texts, max_len=32):
    tl, tt = 0.0, 0
    with torch.no_grad():
        for txt in texts:
            ids = tok.encode(txt)[:max_len]
            if len(ids) < 3: continue
            logits = rwkv7_fwd(ids)
            ids_t = torch.tensor(ids)
            loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:])
            tl += loss.item() * (len(ids) - 1)
            tt += len(ids) - 1
    return math.exp(tl / max(tt, 1))

texts7 = [
    "The future of artificial intelligence will depend on how we design",
    "Deep learning applications in healthcare include medical image analysis",
    "Natural language processing has evolved significantly with transformer models",
]

print(f"\nEvaluating baseline PPL (3 texts, max_len=32)...")
t0 = time.time()
base7 = eval_ppl7(texts7, max_len=32)
print(f"Baseline PPL: {base7:.2f} ({time.time()-t0:.0f}s)")

# ===== Injection Sweep =====
print("\n" + "=" * 70)
print("INJECTION SWEEP")
print("=" * 70)
R7 = []

def sweep7(name, tmpl, taus):
    print(f"\n  {name}:")
    for t in taus:
        sv = []
        for bi in range(NB7 + 1):
            k = tmpl.format(bi=bi)
            if k in w7:
                sv.append((k, w7[k].clone()))
                w7[k] = w7[k] * t
        p = eval_ppl7(texts7, max_len=32)
        dp = (p - base7) / base7 * 100
        R7.append({"point": name, "tau": t, "ppl": p, "delta_pct": dp})
        print(f"    tau={t:5.2f}  PPL={p:.2f}  {dp:+.2f}%")
        for k, v in sv:
            w7[k] = v

t0 = time.time()
# w0 controls decay base rate (scalar per-dim)
sweep7("att.w0 (decay base)", "blocks.{bi}.att.w0", [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.5])
# key.weight controls write strength
sweep7("att.key.weight", "blocks.{bi}.att.key.weight", [0.5, 0.9, 0.95, 1.0, 1.05, 1.1, 1.5, 2.0])
# value.weight
sweep7("att.value.weight", "blocks.{bi}.att.value.weight", [0.9, 0.95, 1.0, 1.05, 1.1])
# output.weight
sweep7("att.output.weight", "blocks.{bi}.att.output.weight", [0.9, 0.95, 1.0, 1.05, 1.1])
# g1 (gate low-rank projection)
sweep7("att.g1", "blocks.{bi}.att.g1", [0.9, 1.0, 1.1])
# receptance
sweep7("att.receptance.weight", "blocks.{bi}.att.receptance.weight", [0.9, 1.0, 1.1])
print(f"\nSweep: {time.time()-t0:.0f}s")

# Combined
print("\n  Combined:")
combos7 = [
    ("w0=0.5", [("blocks.{bi}.att.w0", 0.5)]),
    ("key=1.5", [("blocks.{bi}.att.key.weight", 1.5)]),
    ("key=2.0", [("blocks.{bi}.att.key.weight", 2.0)]),
    ("w0=0.5+key=1.5", [("blocks.{bi}.att.w0", 0.5), ("blocks.{bi}.att.key.weight", 1.5)]),
    ("w0=0.5+key=2.0", [("blocks.{bi}.att.w0", 0.5), ("blocks.{bi}.att.key.weight", 2.0)]),
    ("w0=0.7+key=1.5", [("blocks.{bi}.att.w0", 0.7), ("blocks.{bi}.att.key.weight", 1.5)]),
    ("w0=0.7+val=0.95", [("blocks.{bi}.att.w0", 0.7), ("blocks.{bi}.att.value.weight", 0.95)]),
]
for cname, params in combos7:
    sv = []
    for tmpl, tau in params:
        for bi in range(NB7 + 1):
            k = tmpl.format(bi=bi)
            if k in w7:
                sv.append((k, w7[k].clone()))
                w7[k] = w7[k] * tau
    p = eval_ppl7(texts7, max_len=32)
    dp = (p - base7) / base7 * 100
    R7.append({"point": cname, "tau": "combo", "ppl": p, "delta_pct": dp})
    print(f"    {cname:25s}  PPL={p:.2f}  {dp:+.2f}%")
    for k, v in sv:
        w7[k] = v

# Summary
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Baseline PPL={base7:.2f}")
for p in sorted(set(r["point"] for r in R7)):
    bs = [r for r in R7 if r["point"] == p]
    b = min(bs, key=lambda x: x["ppl"])
    ts = b["tau"] if isinstance(b["tau"], str) else f'{b["tau"]:5.2f}'
    print(f"  {p:25s}: tau={ts:>8s} PPL={b['ppl']:.2f} delta={b['delta_pct']:+.2f}%")

# ===== Generation Quality =====
print("\n" + "=" * 70)
print("GENERATION QUALITY")
print("=" * 70)

def gen7(prompt, max_new=40, temp=0.8, top_k=40):
    ids = tok.encode(prompt)
    torch.manual_seed(42)
    for _ in range(max_new):
        with torch.no_grad():
            logits = rwkv7_fwd(ids)
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
    ("w0=0.5", [("blocks.{bi}.att.w0", 0.5)]),
    ("w0=0.7", [("blocks.{bi}.att.w0", 0.7)]),
    ("key=1.5", [("blocks.{bi}.att.key.weight", 1.5)]),
    ("w0=0.5+key=1.5", [("blocks.{bi}.att.w0", 0.5), ("blocks.{bi}.att.key.weight", 1.5)]),
]

all_gen = {}
for prompt in gprompts:
    print(f"\n{'='*60}\nPrompt: {prompt}\n{'='*60}")
    all_gen[prompt] = {}
    for gname, params in gconfigs:
        sv = []
        for tmpl, tau in params:
            for bi in range(NB7 + 1):
                k = tmpl.format(bi=bi)
                if k in w7:
                    sv.append((k, w7[k].clone()))
                    w7[k] = w7[k] * tau
        text = gen7(prompt)
        for k, v in sv:
            w7[k] = v
        all_gen[prompt][gname] = text
        print(f"\n[{gname}]\n{text}")

# Save results
out7 = {
    "model": "RWKV-7-World-0.4B",
    "layers": NB7+1, "dim": C7, "vocab": V7, "heads": H7, "head_size": N7,
    "base_ppl": base7, "sweep": R7, "generation": all_gen
}
op7 = os.path.join(TAU_DIR, "rwkv", "rwkv7_0.4b_results.json")
with open(op7, "w", encoding="utf-8") as f:
    json.dump(out7, f, indent=2, ensure_ascii=False)
print(f"\nSaved to {op7}")
print("Done.")
