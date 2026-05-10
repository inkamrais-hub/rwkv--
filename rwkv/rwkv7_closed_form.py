#!/usr/bin/env python3
"""RWKV-7 WKV closed-form τ solver.
1 forward → retain out_a → backward → capture grad → solve per-head τ* → inject → verify PPL"""
import torch, sys, os, math, json, time

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

def rwkv7_fwd_full(tok_ids, w, nL, C, H, N):
    """Full forward returning (logits, out_as, Ss, Rs). out_as kept alive for gradient."""
    T = len(tok_ids)
    tok_ids_cpu = tok_ids.cpu() if isinstance(tok_ids, torch.Tensor) else tok_ids
    x = w["emb.weight"][tok_ids_cpu].float().to(DEV).requires_grad_(True)
    out_as = []   # each is [T, C] kept alive
    S_per_layer = []
    R_per_layer = []

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
        Ss, Rs = [], []
        state_att = torch.zeros(H, N, N, device=DEV, dtype=torch.float32)

        for t_idx in range(T):
            rt = r[t_idx].view(H, N)
            kt = k_mod[t_idx].view(H, N)
            vt = v[t_idx].view(H, N)
            kkt = kk[t_idx].view(H, N)
            at_ = a[t_idx].view(H, N)
            dt = decay[t_idx].view(H, N)

            vk = vt.unsqueeze(2) * kt.unsqueeze(1)
            ab = (-kkt).unsqueeze(2) * (kkt * at_).unsqueeze(1)
            state_att = state_att * dt.unsqueeze(1) + state_att @ ab + vk
            o = (state_att @ rt.unsqueeze(2)).view(C)
            out_a_list.append(o)

            Ss.append(state_att.detach().clone().cpu())
            Rs.append(rt.detach().clone().cpu())

        out_a = torch.stack(out_a_list, dim=0)
        if torch.is_grad_enabled():
            out_a.retain_grad()
        out_as.append(out_a)
        S_per_layer.append(Ss)
        R_per_layer.append(Rs)

        r_k_d = d[f"{att}.r_k"].flatten()
        xx_gn = torch.nn.functional.group_norm(
            out_a, num_groups=H, weight=d[f"{att}.ln_x.weight"],
            bias=d[f"{att}.ln_x.bias"], eps=64e-5)
        rk_res = ((r * k_mod * r_k_d).view(T, H, N).sum(dim=-1, keepdim=True)
                  * v.view(T, H, N)).view(T, C)
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

    return logits, out_as, S_per_layer, R_per_layer

def solve_tau_layer(Ss, Rs, grad_tensor, H, N, C, step=0.05, reg=0.01):
    """Ss/Rs: list of [H,N,N]/[H,N] tensors (CPU). grad: [T,C] on device. Returns τ ∈ R^C."""
    tau = torch.ones(C, device=DEV)
    T = len(Ss)
    if T == 0 or grad_tensor is None:
        return tau

    for hi in range(H):
        offset = hi * N
        G = grad_tensor[:, offset:offset+N].to(DEV)
        A = torch.zeros(T * N, N, device=DEV)
        b = torch.zeros(T * N, device=DEV)

        for t in range(T):
            si = t * N
            S = Ss[t][hi].to(DEV)
            R = Rs[t][hi].to(DEV)
            for i in range(N):
                A[si+i] = S[i] * R
            b[si:si+N] = G[t]

        AtA = A.T @ A
        Atb = A.T @ b
        lam = max(AtA.trace().item(), 0.01) * reg
        Rg = torch.eye(N, device=DEV) * lam

        try:
            dp = torch.linalg.solve(AtA + Rg, Atb)
        except Exception:
            dp = torch.zeros(N, device=DEV)

        tau_h = 1.0 - dp * step
        tau_h = torch.clamp(tau_h, 0.5, 2.0)
        tau[offset:offset+N] = tau_h

    return tau

def eval_ppl(tok_ids_list, w, nL, C, H, N, tau_list=None):
    tl, tt = 0.0, 0
    with torch.no_grad():
        for ids in tok_ids_list:
            if tau_list is not None:
                for bi in range(nL):
                    k = f"blocks.{bi}.att.key.weight"
                    w[k] = w[k] * tau_list[bi].cpu()
            logits, out_as, _, _ = rwkv7_fwd_full(ids, w, nL, C, H, N)
            if tau_list is not None:
                for bi in range(nL):
                    k = f"blocks.{bi}.att.key.weight"
                    w[k] = w[k] / tau_list[bi].cpu()
            ids_t = torch.tensor(ids, device=DEV)
            L = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:]).item()
            tl += L * (len(ids) - 1)
            tt += len(ids) - 1
    return math.exp(tl / max(tt, 1))

def main(model_name, path):
    torch.set_grad_enabled(True)
    w, nL, C, H, N = load_rwkv7(path, model_name)
    eval_ids_list = [tok.encode(t)[:32] for t in EVAL_TEXTS]

    baseline = eval_ppl(eval_ids_list, w, nL, C, H, N)
    print(f"  Baseline PPL: {baseline:.2f}")

    all_ids = torch.tensor([x for ids in eval_ids_list for x in ids], device=DEV)
    print(f"\nPhase 1: fwd+bwd ({len(all_ids)} tokens)...")
    t0 = time.time()
    logits, out_as, Ss_all, Rs_all = rwkv7_fwd_full(all_ids, w, nL, C, H, N)
    ids_t = all_ids
    loss = torch.nn.functional.cross_entropy(logits[:-1], ids_t[1:])
    loss.backward()
    print(f"  Done in {time.time()-t0:.1f}s, loss={loss.item():.4f}")

    print(f"\nPhase 2: solving per-layer τ*...")
    taus = []
    for bi in range(nL):
        grad = out_as[bi].grad
        if grad is None:
            print(f"  L{bi}: grad=None → τ=1")
            taus.append(torch.ones(C, device=DEV))
            continue
        tau = solve_tau_layer(Ss_all[bi], Rs_all[bi], grad, H, N, C)
        taus.append(tau)
        print(f"  L{bi}: τ mean={tau.mean():.3f} std={tau.std():.3f} "
              f"range=[{tau.min():.3f},{tau.max():.3f}]")

    torch.cuda.empty_cache()

    print(f"\nPhase 3: injecting τ and re-evaluating...")
    t0 = time.time()
    ppl_tau = eval_ppl(eval_ids_list, w, nL, C, H, N, tau_list=taus)
    d = (ppl_tau - baseline) / baseline * 100
    print(f"  Closed-form PPL: {ppl_tau:.2f} ({d:+.2f}%)  [{time.time()-t0:.1f}s]")

    grid_best = {
        "0.4B": "PPL 21.88 (-3.21% key τ=0.9)",
        "1.5B": "PPL 19.11 (-1.66% key τ=0.9)",
        "2.9B": "PPL 17.17 (-3.16% output τ=1.1)",
    }
    gb = grid_best.get(model_name.split("-")[-1], "N/A")
    print(f"\n=== Results ===")
    print(f"  Baseline:  {baseline:.2f}")
    print(f"  Closed τ:  {ppl_tau:.2f}  ({d:+.2f}%)")
    print(f"  Best grid: {gb}")

    return ppl_tau, d, taus

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