#!/usr/bin/env python3
"""RWKV-7 s^tau test v5 - corrected state management matching official RWKV-7."""
import os, sys, json, time
import numpy as np
import torch
import torch.nn.functional as F
torch.set_grad_enabled(False)

MODEL_DIR = "/root/models"
RESULTS_DIR = "/root/results"
os.makedirs(RESULTS_DIR, exist_ok=True)

EVAL_TEXTS = [
    "The quick brown fox jumps over the lazy dog. This sentence contains every letter of the alphabet.",
    "Scientists have discovered a new species of deep-sea fish that produces its own light through bioluminescence.",
    "In a distant galaxy, a dying star created an enormous explosion that illuminated the entire nebula.",
]
TOKENIZER = None
try:
    from rwkv.rwkv_tokenizer import TRIE_TOKENIZER
    TOKENIZER = TRIE_TOKENIZER("/usr/local/lib/python3.11/dist-packages/rwkv/rwkv_vocab_v20230424.txt")
    print("Tokenizer OK")
except Exception as e:
    print("Tok err:", e)

def tok(text):
    if TOKENIZER: return TOKENIZER.encode(text)
    return [ord(c) % 65536 for c in text][:128]

DEV = "cuda"

def load_model(path):
    print("Loading %s ..." % os.path.basename(path))
    t0 = time.time()
    w = torch.load(path, map_location="cpu", weights_only=True)
    nL = max(int(k.split(".")[1]) for k in w if k.startswith("blocks.")) + 1
    # Extract H, N BEFORE any processing (r_k is [H, N] in raw .pth)
    r_k_raw = w["blocks.0.att.r_k"]
    if r_k_raw.dim() >= 2:
        H, N = r_k_raw.shape[0], r_k_raw.shape[1]
    else:
        H, N = r_k_raw.shape[0], 1
    # Pre-process embedding with ln0 (matching official)
    C = w["blocks.0.ln1.weight"].shape[0]
    emb = F.layer_norm(w["emb.weight"].float(), (C,),
                        weight=w["blocks.0.ln0.weight"].float(),
                        bias=w["blocks.0.ln0.bias"].float())
    w["emb.weight"] = emb
    # Transpose weights matching official rwkv package (line 259-260)
    for k in list(w.keys()):
        if any(x in k for x in ["key.weight", "value.weight", "receptance.weight", "output.weight", "head.weight"]):
            w[k] = w[k].t().contiguous()
        if k.endswith("att.r_k"):
            w[k] = w[k].flatten()
        if isinstance(w[k], torch.Tensor):
            w[k] = w[k].squeeze().to(DEV)
    torch.cuda.synchronize()
    print("  %dL %dH %dN %.1fs GPU:%.1fGB" % (nL, H, N, time.time()-t0, torch.cuda.memory_allocated()/1e9))
    return w, nL, H, N

def get_params(w, H, N):
    C = w["blocks.0.ln1.weight"].shape[0]
    nv = w["emb.weight"].shape[0]
    return C, H, N, nv

def cppl(lo, ids):
    sl = lo[:-1].float()
    lp = torch.log_softmax(sl, -1)
    tgt = torch.tensor(ids[1:], device=sl.device)
    return float(torch.exp(-lp[range(len(ids)-1), tgt].mean()))

def rwkv_fwd(ids, w, nL, C, H, N, lt=None, rt=None, debug=False):
    """RWKV-7 forward matching official implementation.
    State: [x_prev_tmix[C], kv[H,N,N], x_prev_ffn[C]] per layer.
    Weights already transposed during load (matching official)."""
    x = w["emb.weight"][ids]
    state = []
    for _ in range(nL):
        state.append(torch.zeros(C, device=DEV))
        state.append(torch.zeros(H, N, N, device=DEV))
        state.append(torch.zeros(C, device=DEV))
    v_first = None
    outs = []
    for t in range(len(ids)):
        xt = x[t]
        for li in range(nL):
            B = "blocks." + str(li)
            att = B + ".att."
            ffn = B + ".ffn."
            # TMix LayerNorm
            xm = F.layer_norm(xt.float(), (C,),
                              weight=w[B+".ln1.weight"].float(),
                              bias=w[B+".ln1.bias"].float())
            if debug and t == 0 and li < 3:
                print("  L%d ln1: [%.3f, %.3f]" % (li, xm.min().item(), xm.max().item()))
            # Delta from per-layer x_prev
            x_prev = state[li*3+0].float()
            xx = x_prev - xm
            # 6-way mixing with residual (weights pre-transposed)
            xr  = xm + xx * w[att+"x_r"].squeeze().float()
            xw_ = xm + xx * w[att+"x_w"].squeeze().float()
            xk_ = xm + xx * w[att+"x_k"].squeeze().float()
            xv_ = xm + xx * w[att+"x_v"].squeeze().float()
            xa_ = xm + xx * w[att+"x_a"].squeeze().float()
            xg_ = xm + xx * w[att+"x_g"].squeeze().float()
            # Projections (weights transposed: [out, in])
            r = xr @ w[att+"receptance.weight"].float()
            k = xk_ @ w[att+"key.weight"].float()
            v = xv_ @ w[att+"value.weight"].float()
            if debug and t == 0 and li < 3:
                print("  L%d r: [%.3f, %.3f] k: [%.3f, %.3f] v: [%.3f, %.3f]" % (
                    li, r.min().item(), r.max().item(), k.min().item(), k.max().item(), v.min().item(), v.max().item()))
                sys.stdout.flush()
            # Data-dependent decay
            w_tilde = torch.tanh(xw_ @ w[att+"w1"].float()) @ w[att+"w2"].float()
            decay = torch.exp(-0.606531 * torch.sigmoid((w[att+"w0"].squeeze().float() + w_tilde).float()))
            # Key modulation
            a = torch.sigmoid(w[att+"a0"].squeeze().float() + (xa_ @ w[att+"a1"].float()) @ w[att+"a2"].float())
            g = torch.sigmoid(xg_ @ w[att+"g1"].float()) @ w[att+"g2"].float()
            if debug and t == 0 and li < 3:
                print("  L%d g: [%.3f, %.3f] decay: [%.4f, %.4f]" % (
                    li, g.min().item(), g.max().item(), decay.min().item(), decay.max().item()))
                sys.stdout.flush()
            kk = F.normalize((k * w[att+"k_k"].squeeze().float()).view(H, N), dim=-1, p=2.0).view(C)
            k = k * (1 + (a - 1) * w[att+"k_a"].squeeze().float())
            # Value residual
            if li == 0:
                v_first = v.clone()
            else:
                v = v + (v_first - v) * torch.sigmoid(w[att+"v0"].squeeze().float() + (xv_ @ w[att+"v1"].float()) @ w[att+"v2"].float())
            # s^tau injection
            l = lt.get(li) if lt else None
            if l and "key.weight" in l: k = k * l["key.weight"]
            if l and "w0" in l: decay = decay ** l["w0"]
            if rt is not None: decay = decay ** rt
            # WKV recurrence (matching official exactly)
            s = state[li*3+1]
            vk = v.view(H, N, 1) @ k.view(H, 1, N)
            ab = (-kk).view(H, N, 1) @ (kk * a).view(H, 1, N)
            s = s * decay.view(H, 1, N) + s @ ab.float() + vk.float()
            yy = (s.float() @ r.view(H, N, 1)).view(C)
            if debug and t == 0 and li < 3:
                print("  L%d state_out: [%.3f, %.3f] nan=%s" % (li, yy.min().item(), yy.max().item(), torch.isnan(yy).any().item()))
                sys.stdout.flush()
            # GroupNorm
            yy = F.group_norm(yy.view(1, C), num_groups=H, weight=w[att+"ln_x.weight"].float(), bias=w[att+"ln_x.bias"].float(), eps=64e-5).view(C)
            # r_k residual
            yy = yy + ((r * k * w[att+"r_k"].float()).view(H, N).sum(dim=-1, keepdim=True) * v.view(H, N)).view(C)
            # Gate and output (weight pre-transposed)
            ao = (yy * g) @ w[att+"output.weight"].float()
            if debug and t == 0 and li < 3:
                print("  L%d tmix_out: [%.3f, %.3f] nan=%s" % (li, ao.min().item(), ao.max().item(), torch.isnan(ao).any().item()))
                sys.stdout.flush()
            # Update state
            state[li*3+0] = xm.detach()
            state[li*3+1] = s.detach()
            # TMix residual
            xt = xt + ao
            if debug and t == 0 and li < 3:
                print("  L%d after_tmix: [%.3f, %.3f] nan=%s" % (li, xt.min().item(), xt.max().item(), torch.isnan(xt).any().item()))
                sys.stdout.flush()
            # CMix LayerNorm
            xn = F.layer_norm(xt.float(), (C,), weight=w[B+".ln2.weight"].float(), bias=w[B+".ln2.bias"].float())
            # FFN with per-layer x_prev (after TMix+residual of prev token)
            x_ffn_prev = state[li*3+2].float()
            xx_ffn = x_ffn_prev - xn
            k_ffn = xn + xx_ffn * w[ffn+"x_k"].squeeze().float()
            k_ffn = torch.relu(k_ffn @ w[ffn+"key.weight"].float()) ** 2
            xt = xt + k_ffn @ w[ffn+"value.weight"].float()
            if debug and t == 0 and li < 3:
                print("  L%d after_ffn: [%.3f, %.3f] nan=%s" % (li, xt.min().item(), xt.max().item(), torch.isnan(xt).any().item()))
                sys.stdout.flush()
            state[li*3+2] = xt.detach()
        # Final LayerNorm + LM head (head pre-transposed)
        if "ln_out.weight" in w:
            xt = F.layer_norm(xt.float(), (C,), weight=w["ln_out.weight"].float(), bias=w["ln_out.bias"].float())
        hf = xt.float() @ w["head.weight"].float()
        outs.append(hf)
    return torch.stack(outs), state

def run_tests(mp, mn):
    print(); print("=" * 60); print("Test: %s" % mn); print("=" * 60)
    t0 = time.time()
    w, nL, H, N = load_model(mp)
    C, H, N, nv = get_params(w, H, N)
    print("  L=%d C=%d H=%d N=%d V=%d" % (nL, C, H, N, nv))
    eids = [tok(t) for t in EVAL_TEXTS]
    print("  tokens: %s" % [len(x) for x in eids])
    sys.stdout.flush()
    # Warmup
    t1 = time.time()
    _ = rwkv_fwd(eids[0][:4], w, nL, C, H, N)
    torch.cuda.synchronize()
    print("  warmup: %.2fs" % (time.time()-t1))
    sys.stdout.flush()
    # Baseline
    print("\n[Baseline]")
    bp = []
    for i, ids in enumerate(eids):
        t1 = time.time()
        p = cppl(rwkv_fwd(ids, w, nL, C, H, N)[0], ids)
        bp.append(p)
        print("  t%d PPL=%.4f (%.1fs)" % (i, p, time.time()-t1))
        sys.stdout.flush()
    b = float(np.mean(bp))
    print("  MEAN=%.4f" % b)
    sys.stdout.flush()
    R = {"model": mn, "bppl": b, "lk": [], "lw": [], "tk": [], "rc": [], "gen": {}}
    # Key sensitivity
    print("\n[Key tau=0.9]")
    lk = []
    for li in range(nL):
        t1 = time.time()
        m = float(np.mean([cppl(rwkv_fwd(ids, w, nL, C, H, N, {li: {"key.weight": 0.9}})[0], ids) for ids in eids]))
        lk.append([li, m, (m-b)/b*100])
        print("  L%2d %.4f %+.2f%% (%.1fs)" % (li, m, lk[-1][2], time.time()-t1))
        sys.stdout.flush()
    lk.sort(key=lambda x: x[2])
    R["lk"] = lk
    t3 = [x[0] for x in lk[:3]]
    print("  Top3: %s" % t3)
    # w0 sensitivity
    print("\n[w0 tau=0.7]")
    lw = []
    for li in range(nL):
        t1 = time.time()
        m = float(np.mean([cppl(rwkv_fwd(ids, w, nL, C, H, N, {li: {"w0": 0.7}})[0], ids) for ids in eids]))
        lw.append([li, m, (m-b)/b*100])
        print("  L%2d %.4f %+.2f%% (%.1fs)" % (li, m, lw[-1][2], time.time()-t1))
        sys.stdout.flush()
    lw.sort(key=lambda x: x[2])
    R["lw"] = lw
    t3w = [x[0] for x in lw[:3]]
    print("  Top3w: %s" % t3w)
    # Top3 key injection
    print("\n[Top3 key]")
    for tau in [0.7, 0.8, 0.9, 0.95]:
        lt = {li: {"key.weight": tau} for li in t3}
        m = float(np.mean([cppl(rwkv_fwd(ids, w, nL, C, H, N, lt)[0], ids) for ids in eids]))
        d = (m-b)/b*100
        R["tk"].append({"t": tau, "p": m, "d": d})
        print("  %.2f %.4f %+.2f%%" % (tau, m, d))
        sys.stdout.flush()
    # Combined
    print("\n[Combined]")
    for kt, wt in [(0.8, 0.5), (0.9, 0.5), (0.8, 0.3)]:
        lt = {li: {"key.weight": kt, "w0": wt} for li in t3}
        m = float(np.mean([cppl(rwkv_fwd(ids, w, nL, C, H, N, lt)[0], ids) for ids in eids]))
        d = (m-b)/b*100
        R["tk"].append({"t": "k%.1f+w%.1f" % (kt, wt), "p": m, "d": d})
        print("  k%.1f+w%.1f %.4f %+.2f%%" % (kt, wt, m, d))
        sys.stdout.flush()
    # Recurrence sweep
    print("\n[Recurrence]")
    for tau in [0.3, 0.5, 0.7, 0.9, 1.0, 1.1, 1.3, 1.5, 2.0]:
        t1 = time.time()
        m = float(np.mean([cppl(rwkv_fwd(ids, w, nL, C, H, N, rt=tau)[0], ids) for ids in eids]))
        d = (m-b)/b*100
        R["rc"].append({"t": tau, "p": m, "d": d})
        print("  %.1f %.4f %+.2f%% (%.1fs)" % (tau, m, d, time.time()-t1))
        sys.stdout.flush()
    # Generation
    print("\n[Gen]")
    gids = tok("The future of artificial intelligence")
    for nm, cfg in [("base", {}), ("k09", {"lt": {l: {"key.weight": 0.9} for l in t3}}), ("r2", {"rt": 2.0})]:
        g = list(gids)
        for _ in range(80):
            lo, _ = rwkv_fwd(g, w, nL, C, H, N, **cfg)
            tv, ti = torch.topk(lo[-1].float(), 40)
            g.append(ti[torch.multinomial(torch.softmax(tv / 0.8, -1), 1)].item())
        try:
            tx = TOKENIZER.decode(g[len(gids):]) if TOKENIZER else ""
        except: tx = ""
        if not tx: tx = "".join(chr(min(max(c, 32), 126)) for c in g[len(gids):] if 32 <= c < 127)
        R["gen"][nm] = tx[:300]
        print("  [%s] %s" % (nm, tx[:100]))
        sys.stdout.flush()
    op = os.path.join(RESULTS_DIR, mn.replace("-", "_") + ".json")
    with open(op, "w") as f:
        json.dump(R, f, indent=2)
    print("\nSaved: %s (%.0fs = %.1fmin)" % (op, time.time()-t0, (time.time()-t0)/60))
    sys.stdout.flush()

def quicktest():
    """Quick sanity check - compares v5 against official rwkv package."""
    print("RWKV-7 s^tau v5 - quicktest")
    p = os.path.join(MODEL_DIR, "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth")
    w, nL, H, N = load_model(p)
    C, H, N, nv = get_params(w, H, N)
    print("  L=%d C=%d H=%d N=%d V=%d" % (nL, C, H, N, nv))
    # Print key weight shapes for debugging
    print("\n  [Weight shapes L0]")
    for k in sorted(w.keys()):
        if 'blocks.0.att.' in k:
            print("  %s: %s" % (k, list(w[k].shape)))
    ids = tok("The quick brown fox jumps over the lazy dog")[:8]
    print("  ids:", ids)
    # Test 1: Official package
    print("\n  [Test 1: Official]")
    try:
        import rwkv.model as rm
        official = rm.RWKV_x070(p, "cuda fp32")
        state = official.generate_zero_state()
        oo = []
        for idx in ids:
            out, state = official.forward_one(idx, state)
            oo.append(out)
        oo = torch.stack(oo)
        off_ppl = cppl(oo, ids)
        print("  PPL: %.4f (nan=%s)" % (off_ppl, torch.isnan(oo).any().item()))
    except Exception as e:
        print("  FAILED: %s" % e)
        off_ppl = float('inf')
    # Test 2: V5
    print("\n  [Test 2: v5]")
    lo, st = rwkv_fwd(ids, w, nL, C, H, N, debug=True)
    print("  shape:", list(lo.shape))
    print("  nan:", torch.isnan(lo).any().item())
    if not torch.isnan(lo).any():
        print("  range: [%.3f, %.3f]" % (lo.min().item(), lo.max().item()))
        p_val = cppl(lo, ids)
        print("  PPL: %.4f" % p_val)
        # Cosine similarity per token
        print("\n  [Per-token cos]")
        for t in range(min(len(oo), len(lo))):
            cos = torch.nn.functional.cosine_similarity(
                oo[t].float().unsqueeze(0), lo[t].float().unsqueeze(0)).item()
            l2 = (oo[t].float() - lo[t].float()).norm().item()
            print("  t%d: cos=%.6f L2=%.4f" % (t, cos, l2))
        print("  quicktest OK!" if p_val < 100 else "  WARNING: PPL too high!")
    else:
        for t in range(len(ids)):
            if torch.isnan(lo[t]).any():
                print("  First NaN at token %d" % t)
                break

def validate_against_official(path, name):
    """Compare v5 forward against official rwkv package on short sequence."""
    print("\n" + "=" * 60)
    print("VALIDATE: %s" % name)
    print("=" * 60)
    # Load official
    import rwkv.model as rm
    official = rm.RWKV_x070(path, "cuda fp32")
    # Load v5
    w, nL, H, N = load_model(path)
    C, _, _, _ = get_params(w, H, N)
    ids = tok("The quick brown fox")[:6]
    print("  ids:", ids)
    # Official forward
    print("\n  [Official]")
    state = official.generate_zero_state()
    oo = []
    for idx in ids:
        out, state = official.forward_one(idx, state)
        oo.append(out)
    oo = torch.stack(oo)
    print("  nan=%s range=[%.4f,%.4f] PPL=%.4f" % (
        torch.isnan(oo).any().item(), oo.min().item(), oo.max().item(), cppl(oo, ids)))
    # V5 forward
    print("\n  [v5]")
    vo, _ = rwkv_fwd(ids, w, nL, C, H, N, debug=True)
    print("  nan=%s" % torch.isnan(vo).any().item())
    if not torch.isnan(vo).any():
        print("  range=[%.4f,%.4f] PPL=%.4f" % (vo.min().item(), vo.max().item(), cppl(vo, ids)))
        # Per-token comparison
        print("\n  [Per-token]")
        for t in range(min(len(oo), len(vo))):
            o, v = oo[t].float(), vo[t].float()
            cos = torch.nn.functional.cosine_similarity(o.unsqueeze(0), v.unsqueeze(0)).item()
            l2 = (o - v).norm().item()
            md = (o - v).abs().max().item()
            print("  t%d: cos=%.6f L2=%.4f max=%.4f" % (t, cos, l2, md))
        return cppl(vo, ids)
    else:
        print("  NaN detected - need to debug!")
        return float('inf')

if __name__ == "__main__":
    if "--quick" in sys.argv:
        quicktest()
        sys.exit(0)
    if "--validate" in sys.argv:
        print("RWKV-7 v5 Validation Mode")
        print("Device: %s" % torch.cuda.get_device_name(0))
        sys.stdout.flush()
        for p, n in [
            (os.path.join(MODEL_DIR, "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"), "RWKV-7-1.5B"),
            (os.path.join(MODEL_DIR, "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth"), "RWKV-7-2.9B"),
        ]:
            if os.path.exists(p):
                validate_against_official(p, n)
                torch.cuda.empty_cache()
            else:
                print("SKIP: %s" % p)
        print("\nValidation done!")
        sys.exit(0)
    print("RWKV-7 s^tau v5 (corrected state)")
    print("Device: %s" % torch.cuda.get_device_name(0))
    sys.stdout.flush()
    for p, n in [
        (os.path.join(MODEL_DIR, "RWKV-x070-World-1.5B-v3-20250127-ctx4096.pth"), "RWKV-7-1.5B"),
        (os.path.join(MODEL_DIR, "RWKV-x070-World-2.9B-v3-20250211-ctx4096.pth"), "RWKV-7-2.9B"),
    ]:
        if os.path.exists(p):
            run_tests(p, n)
            torch.cuda.empty_cache()
        else:
            print("SKIP: %s" % p)
    print("\nAll done!")
