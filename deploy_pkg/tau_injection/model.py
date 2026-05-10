import torch
import torch.nn.functional as F
import numpy as np
from .utils import get_device

DEV = None

def _dev():
    global DEV
    if DEV is None:
        DEV = get_device()
    return DEV

def load_rwkv7(path, name=""):
    """Load RWKV-7 weights from .pth file.
    Returns (weight_dict, n_layers, C_dim, H_heads, N_head_dim)."""
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
    w["emb.weight"] = F.layer_norm(
        w["emb.weight"], (C,), weight=w["blocks.0.ln0.weight"], bias=w["blocks.0.ln0.bias"])
    if "blocks.0.att.v0" not in w:
        for suf in ["v0", "v1", "v2"]:
            w[f"blocks.0.att.{suf}"] = torch.empty(0)
    if name:
        print(f"  {name}: {nL}L C={C} H={H} N={N}")
    return w, nL, C, H, N

def rwkv7_fwd(tok_ids, w, nL, C, H, N, tau_v=None):
    """Minimal forward pass. If tau_v provided: list of [H,N] tensors scaling v per layer.
    Returns logits [T, vocab]. Includes gradient tracking when tau_v with requires_grad."""
    dev = _dev()
    T = len(tok_ids)
    tok_ids_cpu = tok_ids.cpu() if isinstance(tok_ids, torch.Tensor) else tok_ids
    x = w["emb.weight"][tok_ids_cpu].float().to(dev)
    if tau_v is not None:
        x = x.requires_grad_(True)

    for bi in range(nL):
        bp = f"blocks.{bi}"
        att = f"{bp}.att"
        ffn = f"{bp}.ffn"

        d = {k: v.to(dev) for k, v in w.items() if k.startswith(bp) and isinstance(v, torch.Tensor)}

        xx = F.layer_norm(x, (C,), weight=d[f"{bp}.ln1.weight"], bias=d[f"{bp}.ln1.bias"])
        sx = torch.cat((torch.zeros(1, C, device=dev), xx[:-1])) - xx

        xr, xw, xk, xv, xa, xg = [xx + sx * d[f"{att}.x_{c}"] for c in ["r", "w", "k", "v", "a", "g"]]

        r = xr @ d[f"{att}.receptance.weight"].T
        k = xk @ d[f"{att}.key.weight"].T
        v = xv @ d[f"{att}.value.weight"].T

        w_param = torch.tanh(xw @ d[f"{att}.w1"]) @ d[f"{att}.w2"]
        decay = torch.exp(-0.606531 * torch.sigmoid((d[f"{att}.w0"] + w_param).float())).to(dev)

        a = torch.sigmoid(d[f"{att}.a0"] + (xa @ d[f"{att}.a1"]) @ d[f"{att}.a2"])
        g = torch.sigmoid(xg @ d[f"{att}.g1"]) @ d[f"{att}.g2"]

        kk = F.normalize((k * d[f"{att}.k_k"]).view(T, H, N), dim=-1, p=2.0).view(T, C)
        k_mod = k * (1 + (a - 1) * d[f"{att}.k_a"])

        if bi > 0 and d[f"{att}.v0"].numel() > 0:
            v = v + (torch.zeros_like(v) - v) * torch.sigmoid(
                d[f"{att}.v0"] + (xv @ d[f"{att}.v1"]) @ d[f"{att}.v2"])

        out_a = []
        state_att = torch.zeros(H, N, N, device=dev, dtype=torch.float32)

        v_h = v.view(T, H, N)
        tauv = tau_v[bi] if tau_v is not None else None
        if tauv is not None:
            v_h = v_h * tauv.view(1, H, N)

        for t_idx in range(T):
            rt = r[t_idx].view(H, N)
            kt = k_mod[t_idx].view(H, N)
            vt = v_h[t_idx]
            kkt_ = kk[t_idx].view(H, N)
            at_ = a[t_idx].view(H, N)
            dt = decay[t_idx].view(H, N)

            vk = vt.unsqueeze(2) * kt.unsqueeze(1)
            ab = (-kkt_).unsqueeze(2) * (kkt_ * at_).unsqueeze(1)
            state_att = state_att * dt.unsqueeze(1) + state_att @ ab + vk
            out_a.append((state_att @ rt.unsqueeze(2)).view(C))

        out_a = torch.stack(out_a, dim=0)

        r_k_d = d[f"{att}.r_k"].flatten()
        xx_gn = F.group_norm(out_a, num_groups=H, weight=d[f"{att}.ln_x.weight"],
                             bias=d[f"{att}.ln_x.bias"], eps=64e-5)
        rk_res = ((r * k_mod * r_k_d).view(T, H, N).sum(dim=-1, keepdim=True) * v_h).view(T, C)
        xx_gn = (xx_gn + rk_res) * g
        x = x + xx_gn @ d[f"{att}.output.weight"].T

        xx2 = F.layer_norm(x, (C,), weight=d[f"{bp}.ln2.weight"], bias=d[f"{bp}.ln2.bias"])
        sx_ffn = torch.cat((torch.zeros(1, C, device=dev), xx2[:-1])) - xx2
        xk_ffn = xx2 + sx_ffn * d[f"{ffn}.x_k"]
        x = x + (torch.relu(xk_ffn @ d[f"{ffn}.key.weight"].T) ** 2) @ d[f"{ffn}.value.weight"].T

    x_norm = F.layer_norm(x, (C,), weight=w["ln_out.weight"].to(dev),
                          bias=w.get("ln_out.bias", torch.zeros(C)).to(dev))
    return x_norm @ w["head.weight"].T.float().to(dev)

def rwkv7_fwd_inject(tok_ids, w, nL, C, H, N, inject=None):
    """Forward with multi-point injection.
    inject: dict with keys 'v','g','output','rk' -> list of per-layer tensors.
    Returns logits [T, vocab]."""
    dev = _dev()
    T = len(tok_ids)
    tok_ids_cpu = tok_ids.cpu() if isinstance(tok_ids, torch.Tensor) else tok_ids
    x = w["emb.weight"][tok_ids_cpu].float().to(dev)
    has_inject = inject is not None and any(inject.get(k) is not None for k in inject)
    if has_inject:
        x = x.requires_grad_(True)

    for bi in range(nL):
        bp = f"blocks.{bi}"
        att = f"{bp}.att"
        ffn = f"{bp}.ffn"

        d = {k: v.to(dev) for k, v in w.items() if k.startswith(bp) and isinstance(v, torch.Tensor)}

        xx = F.layer_norm(x, (C,), weight=d[f"{bp}.ln1.weight"], bias=d[f"{bp}.ln1.bias"])
        sx = torch.cat((torch.zeros(1, C, device=dev), xx[:-1])) - xx

        xr, xw, xk, xv, xa, xg = [xx + sx * d[f"{att}.x_{c}"] for c in ["r", "w", "k", "v", "a", "g"]]

        r = xr @ d[f"{att}.receptance.weight"].T
        k = xk @ d[f"{att}.key.weight"].T
        v = xv @ d[f"{att}.value.weight"].T

        w_param = torch.tanh(xw @ d[f"{att}.w1"]) @ d[f"{att}.w2"]
        decay = torch.exp(-0.606531 * torch.sigmoid((d[f"{att}.w0"] + w_param).float())).to(dev)

        a = torch.sigmoid(d[f"{att}.a0"] + (xa @ d[f"{att}.a1"]) @ d[f"{att}.a2"])
        g_raw = torch.sigmoid(xg @ d[f"{att}.g1"]) @ d[f"{att}.g2"]

        kk = F.normalize((k * d[f"{att}.k_k"]).view(T, H, N), dim=-1, p=2.0).view(T, C)
        k_mod = k * (1 + (a - 1) * d[f"{att}.k_a"])

        if bi > 0 and d[f"{att}.v0"].numel() > 0:
            v = v + (torch.zeros_like(v) - v) * torch.sigmoid(
                d[f"{att}.v0"] + (xv @ d[f"{att}.v1"]) @ d[f"{att}.v2"])

        tau_v = inject["v"][bi] if inject and inject.get("v") else None
        tau_g = inject["g"][bi] if inject and inject.get("g") else None
        tau_rk = inject["rk"][bi] if inject and inject.get("rk") else None
        tau_out = inject["output"][bi] if inject and inject.get("output") else None

        out_a = []
        state_att = torch.zeros(H, N, N, device=dev, dtype=torch.float32)

        v_h = v.view(T, H, N)
        if tau_v is not None:
            v_h = v_h * tau_v.view(1, H, N)

        for t_idx in range(T):
            rt = r[t_idx].view(H, N)
            kt = k_mod[t_idx].view(H, N)
            vt = v_h[t_idx]
            kkt_ = kk[t_idx].view(H, N)
            at_ = a[t_idx].view(H, N)
            dt = decay[t_idx].view(H, N)

            vk = vt.unsqueeze(2) * kt.unsqueeze(1)
            ab = (-kkt_).unsqueeze(2) * (kkt_ * at_).unsqueeze(1)
            state_att = state_att * dt.unsqueeze(1) + state_att @ ab + vk
            out_a.append((state_att @ rt.unsqueeze(2)).view(C))

        out_a = torch.stack(out_a, dim=0)

        r_k_d = d[f"{att}.r_k"].flatten()
        if tau_rk is not None:
            r_k_d = r_k_d * tau_rk.reshape(H, N).flatten()

        xx_gn = F.group_norm(out_a, num_groups=H, weight=d[f"{att}.ln_x.weight"],
                             bias=d[f"{att}.ln_x.bias"], eps=64e-5)
        rk_res = ((r * k_mod * r_k_d).view(T, H, N).sum(dim=-1, keepdim=True) * v_h).view(T, C)
        xx_gn = xx_gn + rk_res

        g = g_raw
        if tau_g is not None:
            g = g * tau_g.view(1, H, N).view(T, C)
        xx_gn = xx_gn * g

        output_weight = d[f"{att}.output.weight"]
        if tau_out is not None:
            output_weight = output_weight * tau_out.unsqueeze(-1)
        x = x + xx_gn @ output_weight.T

        xx2 = F.layer_norm(x, (C,), weight=d[f"{bp}.ln2.weight"], bias=d[f"{bp}.ln2.bias"])
        sx_ffn = torch.cat((torch.zeros(1, C, device=dev), xx2[:-1])) - xx2
        xk_ffn = xx2 + sx_ffn * d[f"{ffn}.x_k"]
        x = x + (torch.relu(xk_ffn @ d[f"{ffn}.key.weight"].T) ** 2) @ d[f"{ffn}.value.weight"].T

    x_norm = F.layer_norm(x, (C,), weight=w["ln_out.weight"].to(dev),
                          bias=w.get("ln_out.bias", torch.zeros(C)).to(dev))
    return x_norm @ w["head.weight"].T.float().to(dev)

def rwkv7_fwd_detailed(tok_ids, w, nL, C, H, N, tau_v=None):
    """Forward with full state capture for analysis.
    Returns (logits [T,vocab], snapshots dict with per-layer metrics)."""
    dev = _dev()
    T = len(tok_ids)
    tok_ids_cpu = tok_ids.cpu() if isinstance(tok_ids, torch.Tensor) else tok_ids
    x = w["emb.weight"][tok_ids_cpu].float().to(dev)
    if tau_v is not None:
        x = x.requires_grad_(True)

    snap = {"x_per_layer": [], "state_svals": [], "state_norms": [], "out_norms": []}

    for bi in range(nL):
        bp = f"blocks.{bi}"
        att = f"{bp}.att"
        ffn = f"{bp}.ffn"

        d = {k: v.to(dev) for k, v in w.items() if k.startswith(bp) and isinstance(v, torch.Tensor)}

        xx = F.layer_norm(x, (C,), weight=d[f"{bp}.ln1.weight"], bias=d[f"{bp}.ln1.bias"])
        sx = torch.cat((torch.zeros(1, C, device=dev), xx[:-1])) - xx

        xr, xw, xk, xv, xa, xg = [xx + sx * d[f"{att}.x_{c}"] for c in ["r", "w", "k", "v", "a", "g"]]

        r = xr @ d[f"{att}.receptance.weight"].T
        k = xk @ d[f"{att}.key.weight"].T
        v = xv @ d[f"{att}.value.weight"].T

        w_param = torch.tanh(xw @ d[f"{att}.w1"]) @ d[f"{att}.w2"]
        decay = torch.exp(-0.606531 * torch.sigmoid((d[f"{att}.w0"] + w_param).float())).to(dev)

        a = torch.sigmoid(d[f"{att}.a0"] + (xa @ d[f"{att}.a1"]) @ d[f"{att}.a2"])
        g = torch.sigmoid(xg @ d[f"{att}.g1"]) @ d[f"{att}.g2"]

        kk = F.normalize((k * d[f"{att}.k_k"]).view(T, H, N), dim=-1, p=2.0).view(T, C)
        k_mod = k * (1 + (a - 1) * d[f"{att}.k_a"])

        if bi > 0 and d[f"{att}.v0"].numel() > 0:
            v = v + (torch.zeros_like(v) - v) * torch.sigmoid(
                d[f"{att}.v0"] + (xv @ d[f"{att}.v1"]) @ d[f"{att}.v2"])

        out_a = []
        state_att = torch.zeros(H, N, N, device=dev, dtype=torch.float32)

        v_h = v.view(T, H, N)
        tauv = tau_v[bi] if tau_v is not None else None
        if tauv is not None:
            v_h = v_h * tauv.view(1, H, N)

        state_snaps = []
        for t_idx in range(T):
            rt = r[t_idx].view(H, N)
            kt = k_mod[t_idx].view(H, N)
            vt = v_h[t_idx]
            kkt_ = kk[t_idx].view(H, N)
            at_ = a[t_idx].view(H, N)
            dt = decay[t_idx].view(H, N)

            vk = vt.unsqueeze(2) * kt.unsqueeze(1)
            ab = (-kkt_).unsqueeze(2) * (kkt_ * at_).unsqueeze(1)
            state_att = state_att * dt.unsqueeze(1) + state_att @ ab + vk
            out_a.append((state_att @ rt.unsqueeze(2)).view(C))

            if T <= 32 or t_idx % max(1, T // 8) == 0:
                state_snaps.append(state_att.detach().clone().cpu())

        out_a = torch.stack(out_a, dim=0)

        r_k_d = d[f"{att}.r_k"].flatten()
        xx_gn = F.group_norm(out_a, num_groups=H, weight=d[f"{att}.ln_x.weight"],
                             bias=d[f"{att}.ln_x.bias"], eps=64e-5)
        rk_res = ((r * k_mod * r_k_d).view(T, H, N).sum(dim=-1, keepdim=True) * v_h).view(T, C)
        xx_gn = (xx_gn + rk_res) * g
        x = x + xx_gn @ d[f"{att}.output.weight"].T

        xx2 = F.layer_norm(x, (C,), weight=d[f"{bp}.ln2.weight"], bias=d[f"{bp}.ln2.bias"])
        sx_ffn = torch.cat((torch.zeros(1, C, device=dev), xx2[:-1])) - xx2
        xk_ffn = xx2 + sx_ffn * d[f"{ffn}.x_k"]
        x = x + (torch.relu(xk_ffn @ d[f"{ffn}.key.weight"].T) ** 2) @ d[f"{ffn}.value.weight"].T

        snap["x_per_layer"].append(x.detach().cpu().norm(dim=-1).mean().item())
        snap["out_norms"].append(out_a.detach().cpu().norm(dim=-1).mean().item())
        if state_snaps:
            sv_norms = [s.norm().item() for s in state_snaps]
            snap["state_norms"].append(np.mean(sv_norms))
            fst = state_snaps[-1].float()
            snap["state_svals"].append(torch.linalg.svdvals(fst).cpu().numpy())

    x_norm = F.layer_norm(x, (C,), weight=w["ln_out.weight"].to(dev),
                          bias=w.get("ln_out.bias", torch.zeros(C)).to(dev))
    logits = x_norm @ w["head.weight"].T.float().to(dev)
    return logits, snap