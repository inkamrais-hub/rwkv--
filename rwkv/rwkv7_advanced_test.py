import torch, sys, os, math, json, time
sys.stdout.reconfigure(encoding='utf-8')

TAU_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TAU_DIR)
torch.manual_seed(42)

P7 = os.path.join(TAU_DIR, "ms_weights", "rwkv-7-world", "RWKV-x070-World-0.4B-v2.9-20250107-ctx4096.pth")
print("Loading RWKV-7-0.4B...")
w7_raw = torch.load(P7, map_location='cpu', weights_only=True)
NB7 = max(int(k.split(".")[1]) for k in w7_raw if k.startswith("blocks."))
H7, N7 = w7_raw["blocks.0.att.r_k"].shape
C7 = H7 * N7
V7 = w7_raw["emb.weight"].shape[0]
print(f"RWKV-7-0.4B: {NB7+1}L, {C7}d, {V7}V, {H7}H")

w7 = {}
for k, v in w7_raw.items():
    v = v.squeeze()
    if k.endswith(".att.r_k"): v = v.flatten()
    w7[k] = v.float()
ln0_w = w7["blocks.0.ln0.weight"]
ln0_b = w7["blocks.0.ln0.bias"]
w7["emb.weight"] = torch.nn.functional.layer_norm(w7["emb.weight"].float(), (C7,), weight=ln0_w, bias=ln0_b)
w7["blocks.0.att.v0"] = torch.empty(0)
w7["blocks.0.att.v1"] = torch.empty(0)
w7["blocks.0.att.v2"] = torch.empty(0)
print("Preprocessed.")

from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
_world_vocab = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "python", "lib", "site-packages", "rwkv", "rwkv_vocab_v20230424.txt")
if not os.path.exists(_world_vocab):
    _alt = os.path.join("D:" + chr(92), "python", "lib", "site-packages", "rwkv", "rwkv_vocab_v20230424.txt")
    if os.path.exists(_alt): _world_vocab = _alt
tok = TRIE_TOKENIZER(_world_vocab)

def eval_ppl7(texts, max_len=32, fwd_fn=None):
    tl, tt = 0.0, 0
    with torch.no_grad():
        for txt in texts:
            ids = tok.encode(txt)[:max_len]
            if len(ids) < 3: continue
            logits = fwd_fn(ids)
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

def rwkv7_fwd(tok_ids, w=w7, nb=NB7, c=C7, h=H7, n=N7,
              layer_taus=None, recur_tau=None):
    T = len(tok_ids)
    x = w["emb.weight"][tok_ids].float()
    v_first = torch.zeros(c)
    if layer_taus is None: layer_taus = {}

    for bi in range(nb + 1):
        bp = f"blocks.{bi}"; att = f"{bp}.att"; ffn = f"{bp}.ffn"
        state_att = torch.zeros(h, n, n)
        lt = layer_taus.get(bi, {})

        xx = torch.nn.functional.layer_norm(x, (c,), weight=w[f"{bp}.ln1.weight"], bias=w[f"{bp}.ln1.bias"])
        sx = torch.cat((torch.zeros(c).unsqueeze(0), xx[:-1, :])) - xx

        xr = xx + sx * w[f"{att}.x_r"].squeeze()
        xw = xx + sx * w[f"{att}.x_w"].squeeze()
        xk = xx + sx * w[f"{att}.x_k"].squeeze()
        xv = xx + sx * w[f"{att}.x_v"].squeeze()
        xa = xx + sx * w[f"{att}.x_a"].squeeze()
        xg = xx + sx * w[f"{att}.x_g"].squeeze()

        kw = w[f"{att}.key.weight"]
        vw = w[f"{att}.value.weight"]
        rw_mat = w[f"{att}.receptance.weight"]
        ow = w[f"{att}.output.weight"]
        if "key.weight" in lt: kw = kw * lt["key.weight"]
        if "value.weight" in lt: vw = vw * lt["value.weight"]
        if "receptance.weight" in lt: rw_mat = rw_mat * lt["receptance.weight"]
        if "output.weight" in lt: ow = ow * lt["output.weight"]

        r = xr @ rw_mat.T; k = xk @ kw.T; v = xv @ vw.T

        w_param = torch.tanh(xw @ w[f"{att}.w1"]) @ w[f"{att}.w2"]
        w0 = w[f"{att}.w0"].squeeze()
        if "w0" in lt: w0 = w0 * lt["w0"]
        decay = torch.exp(-0.606531 * torch.sigmoid((w0 + w_param).float()))

        a = torch.sigmoid(w[f"{att}.a0"].squeeze() + (xa @ w[f"{att}.a1"]) @ w[f"{att}.a2"])
        g = torch.sigmoid(xg @ w[f"{att}.g1"]) @ w[f"{att}.g2"]

        kk = torch.nn.functional.normalize((k * w[f"{att}.k_k"].squeeze()).view(T, h, n), dim=-1, p=2.0).view(T, c)
        k_mod = k * (1 + (a - 1) * w[f"{att}.k_a"].squeeze())

        if bi == 0: v_first = v.clone()
        else:
            v = v + (v_first - v) * torch.sigmoid(w[f"{att}.v0"].squeeze() + (xv @ w[f"{att}.v1"]) @ w[f"{att}.v2"])

        out_a = torch.zeros(T, c)
        for t in range(T):
            rt = r[t].view(h, n); kt = k_mod[t].view(h, n); vt = v[t].view(h, n)
            kkt = kk[t].view(h, n); at = a[t].view(h, n); dt = decay[t].view(h, n)
            vk = vt.unsqueeze(2) * kt.unsqueeze(1)
            ab = (-kkt).unsqueeze(2) * (kkt * at).unsqueeze(1)
            if recur_tau is not None:
                state_att = (state_att * (dt.unsqueeze(1) ** recur_tau)) + state_att @ ab.float() + vk.float()
            else:
                state_att = state_att * dt.unsqueeze(1) + state_att @ ab.float() + vk.float()
            out_a[t] = (state_att @ rt.unsqueeze(2)).view(c)

        xx_gn = torch.nn.functional.group_norm(out_a.view(T, c), num_groups=h, weight=w[f"{att}.ln_x.weight"], bias=w[f"{att}.ln_x.bias"], eps=64e-5)
        r_k = w[f"{att}.r_k"].flatten()
        rk_res = ((r * k_mod * r_k).view(T, h, n).sum(dim=-1, keepdim=True) * v.view(T, h, n)).view(T, c)
        xx_gn = (xx_gn + rk_res) * g
        x = x + xx_gn @ ow.T

        xx = torch.nn.functional.layer_norm(x, (c,), weight=w[f"{bp}.ln2.weight"], bias=w[f"{bp}.ln2.bias"])
        sx_ffn = torch.cat((torch.zeros(c).unsqueeze(0), xx[:-1, :])) - xx
        xk_ffn = xx + sx_ffn * w[f"{ffn}.x_k"].squeeze()
        k_ffn = torch.relu(xk_ffn @ w[f"{ffn}.key.weight"].T) ** 2
        x = x + k_ffn @ w[f'{ffn}.value.weight'].T

    x = torch.nn.functional.layer_norm(x, (c,), weight=w["ln_out.weight"], bias=w["ln_out.bias"])
    return x @ w["head.weight"].T.float()
# ===== BASELINE =====
print("" + "=" * 70)
print("RWKV-7 0.4B ADVANCED INJECTION")
print("=" * 70)
t0 = time.time()
base7 = eval_ppl7(texts7, max_len=32, fwd_fn=rwkv7_fwd)
print(f"Baseline PPL: {base7:.4f} ({time.time()-t0:.0f}s)")
R = []

# ===== METHOD 1: PER-LAYER SCAN =====
print("" + "=" * 70)
print("METHOD 1: PER-LAYER key.weight SCAN (tau=0.9)")
print("=" * 70)
layer_key_sens = []
for li in range(NB7 + 1):
    t = time.time()
    p = eval_ppl7(texts7, max_len=32, fwd_fn=lambda ids, _li=li: rwkv7_fwd(ids, layer_taus={_li: {"key.weight": 0.9}}))
    dp = (p - base7) / base7 * 100
    layer_key_sens.append((li, p, dp))
    mark = " <--" if dp < -0.3 else ""
    print(f"  Layer {li:2d}: PPL={p:.4f}  {dp:+.3f}%{mark}  ({time.time()-t:.1f}s)")
layer_key_sens.sort(key=lambda x: x[2])
print("Top 5:")
for li, p, dp in layer_key_sens[:5]:
    print(f"  Layer {li:2d}: {dp:+.3f}%")

print("" + "-" * 50)
print("PER-LAYER w0 SCAN (tau=0.5)")
print("-" * 50)
layer_w0_sens = []
for li in range(NB7 + 1):
    t = time.time()
    p = eval_ppl7(texts7, max_len=32, fwd_fn=lambda ids, _li=li: rwkv7_fwd(ids, layer_taus={_li: {"w0": 0.5}}))
    dp = (p - base7) / base7 * 100
    layer_w0_sens.append((li, p, dp))
    mark = " <--" if dp < -0.3 else ""
    print(f"  Layer {li:2d}: PPL={p:.4f}  {dp:+.3f}%{mark}  ({time.time()-t:.1f}s)")
layer_w0_sens.sort(key=lambda x: x[2])
print("Top 5:")
for li, p, dp in layer_w0_sens[:5]:
    print(f"  Layer {li:2d}: {dp:+.3f}%")

top_key = [li for li, _, _ in layer_key_sens[:3]]
top_w0 = [li for li, _, _ in layer_w0_sens[:3]]
print(f"Top key layers: {top_key}")
print(f"Top w0 layers: {top_w0}")

print("Multi-layer key injection:")
for tv in [0.7, 0.8, 0.9, 0.95]:
    lt = {li: {"key.weight": tv} for li in top_key}
    p = eval_ppl7(texts7, max_len=32, fwd_fn=lambda ids, _lt=lt: rwkv7_fwd(ids, layer_taus=_lt))
    dp = (p - base7) / base7 * 100
    R.append({"method": "top3_key", "tau": tv, "ppl": p, "delta_pct": dp})
    print(f"  tau={tv}: PPL={p:.4f}  {dp:+.3f}%")

print("Multi-layer w0 injection:")
for tv in [0.3, 0.5, 0.7]:
    lt = {li: {"w0": tv} for li in top_w0}
    p = eval_ppl7(texts7, max_len=32, fwd_fn=lambda ids, _lt=lt: rwkv7_fwd(ids, layer_taus=_lt))
    dp = (p - base7) / base7 * 100
    R.append({"method": "top3_w0", "tau": tv, "ppl": p, "delta_pct": dp})
    print(f"  tau={tv}: PPL={p:.4f}  {dp:+.3f}%")

print("Combined top layers:")
for kt, wt in [(0.9, 0.5), (0.8, 0.5), (0.9, 0.7), (0.95, 0.3)]:
    lt = {}
    for li in top_key: lt[li] = {"key.weight": kt}
    for li in top_w0:
        if li in lt: lt[li]["w0"] = wt
        else: lt[li] = {"w0": wt}
    p = eval_ppl7(texts7, max_len=32, fwd_fn=lambda ids, _lt=lt: rwkv7_fwd(ids, layer_taus=_lt))
    dp = (p - base7) / base7 * 100
    R.append({"method": "top3_combined", "tau": f"key={kt}+w0={wt}", "ppl": p, "delta_pct": dp})
    print(f"  key={kt}+w0={wt}: PPL={p:.4f}  {dp:+.3f}%")

# ===== METHOD 2: RECURRENCE INJECTION =====
print("" + "=" * 70)
print("METHOD 2: RECURRENCE INJECTION (state * decay^tau)")
print("=" * 70)
for tv in [0.3, 0.5, 0.7, 0.8, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
    t = time.time()
    p = eval_ppl7(texts7, max_len=32, fwd_fn=lambda ids, _t=tv: rwkv7_fwd(ids, recur_tau=_t))
    dp = (p - base7) / base7 * 100
    R.append({"method": "recur_tau", "tau": tv, "ppl": p, "delta_pct": dp})
    mark = " <--" if dp < -0.5 else ""
    print(f"  tau={tv:4.1f}: PPL={p:.4f}  {dp:+.3f}%  ({time.time()-t:.1f}s){mark}")

# ===== METHOD 3: COMBINED =====
print("" + "=" * 70)
print("METHOD 3: COMBINED PER-LAYER + RECURRENCE")
print("=" * 70)
recur_results = [r for r in R if r["method"] == "recur_tau"]
best_recur = min(recur_results, key=lambda x: x["ppl"])
best_rtau = best_recur['tau']
print(f"Best recur: tau={best_rtau} PPL={best_recur['ppl']:.4f}")
for kt in [0.8, 0.9, 0.95]:
    for rt in [0.5, 0.7, 0.8, 0.9]:
        lt = {li: {"key.weight": kt} for li in top_key}
        p = eval_ppl7(texts7, max_len=32, fwd_fn=lambda ids, _lt=lt, _rt=rt: rwkv7_fwd(ids, layer_taus=_lt, recur_tau=_rt))
        dp = (p - base7) / base7 * 100
        R.append({"method": "key+recur", "tau": f"key={kt}+recur={rt}", "ppl": p, "delta_pct": dp})
        print(f"  key={kt}+recur={rt}: PPL={p:.4f}  {dp:+.3f}%")

# ===== SUMMARY =====
print("" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"Baseline PPL = {base7:.4f}")
R_sorted = sorted(R, key=lambda x: x["ppl"])
for r in R_sorted[:10]:
    ts = str(r["tau"]) if isinstance(r["tau"], str) else f'{r["tau"]:.2f}'
    print(f'  {r["method"]:25s} tau={ts:>20s}  PPL={r["ppl"]:.4f}  {r["delta_pct"]:+.3f}%')


# ===== GENERATION =====
print("" + "=" * 70)
print("GENERATION QUALITY")
print("=" * 70)
def gen7(prompt, max_new=40, temp=0.8, top_k=40, fwd_fn=None):
    ids = tok.encode(prompt)
    torch.manual_seed(42)
    for _ in range(max_new):
        with torch.no_grad(): logits = fwd_fn(ids)
        logits = logits[-1] / temp
        if top_k > 0:
            vals, _ = logits.topk(top_k)
            logits[logits < vals[-1]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        ids.append(torch.multinomial(probs, 1).item())
    return tok.decode(ids)

prompt = "The future of artificial intelligence"
print(f"Prompt: {prompt}")
print(f"[Baseline] {gen7(prompt, fwd_fn=rwkv7_fwd)}")
lt_best = {li: {"key.weight": 0.9} for li in top_key}
print(f"[top3 key=0.9] {gen7(prompt, fwd_fn=lambda ids: rwkv7_fwd(ids, layer_taus=lt_best))}")
print(f"[recur tau={best_rtau}] {gen7(prompt, fwd_fn=lambda ids: rwkv7_fwd(ids, recur_tau=best_rtau))}")
print(f"[combined] {gen7(prompt, fwd_fn=lambda ids: rwkv7_fwd(ids, layer_taus=lt_best, recur_tau=best_rtau))}")

out = {"model": "RWKV-7-World-0.4B", "baseline_ppl": base7,
       "top_key_layers": top_key, "top_w0_layers": top_w0, "results": R,
       "layer_key_sensitivity": layer_key_sens, "layer_w0_sensitivity": layer_w0_sens}
op = os.path.join(TAU_DIR, "rwkv", "rwkv7_0.4b_advanced_results.json")
with open(op, "w", encoding="utf-8") as f:
    json.dump(out, f, indent=2, ensure_ascii=False)
print(f"Saved to {op}")
print("Done.")
