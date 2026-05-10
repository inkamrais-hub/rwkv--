import torch
import math
from .model import rwkv7_fwd, rwkv7_fwd_inject, rwkv7_fwd_detailed, _dev
from .eval import eval_ppl


def optimize_tau(train_ids, w, nL, C, H, N, steps=10, lr=0.05, reg=0.001, clamp_range=(0.2, 5.0)):
    """Optimize per-layer τ for v-injection. Returns best tau_list."""
    dev = _dev()
    tau_list = [torch.ones(H, N, device=dev, requires_grad=True) for _ in range(nL)]
    ids_t = torch.tensor(train_ids, device=dev)
    best_loss = float("inf")
    best_tau = None

    for step_i in range(steps):
        logits = rwkv7_fwd(train_ids, w, nL, C, H, N, tau_v=tau_list)
        loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:])
        grads = torch.autograd.grad(loss, tau_list, create_graph=False)
        del logits

        with torch.no_grad():
            for bi, g in enumerate(grads):
                if g is not None:
                    tau_list[bi].data -= lr * (g + reg * (tau_list[bi].data - 1.0))
                    tau_list[bi].data.clamp_(*clamp_range)
                tau_list[bi].grad = None

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_tau = [t.detach().clone() for t in tau_list]

        del loss, grads
        torch.cuda.empty_cache()

    return best_tau


def optimize_inject(train_ids, w, nL, C, H, N, inject_keys, steps=20, lr=0.05, reg=0.001):
    """Optimize τ for multi-point injection. inject_keys: list of 'v','g','output','rk'.
    Returns best inject dict {key: [per-layer tensors]}."""
    dev = _dev()
    inject_params = {}
    for key in inject_keys:
        if key in ("v", "g", "rk"):
            inject_params[key] = [torch.ones(H, N, device=dev, requires_grad=True) for _ in range(nL)]
        elif key == "output":
            inject_params[key] = [torch.ones(1, C, device=dev, requires_grad=True) for _ in range(nL)]

    all_params = [p for k in inject_params for p in inject_params[k]]
    ids_t = torch.tensor(train_ids, device=dev)
    best_loss = float("inf")
    best_inject = None

    for step_i in range(steps):
        inject_cur = {k: inject_params[k] for k in inject_params}
        logits = rwkv7_fwd_inject(train_ids, w, nL, C, H, N, inject=inject_cur)
        loss = torch.nn.functional.cross_entropy(logits[:-1].float(), ids_t[1:])
        grads = torch.autograd.grad(loss, all_params, create_graph=False)

        with torch.no_grad():
            for i, p in enumerate(all_params):
                if grads[i] is not None:
                    p.data -= lr * (grads[i] + reg * (p.data - 1.0))
                    p.data.clamp_(0.2, 5.0)
                p.grad = None

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_inject = {k: [t.detach().clone() for t in inject_params[k]] for k in inject_params}

        del logits, loss, grads
        torch.cuda.empty_cache()

    return best_inject


def optimize_tau_track(train_ids, val_ids_list, w, nL, C, H, N, steps=80, lr=0.05, reg=0.001):
    """Run GD tracking train loss + val PPL at every step. Returns (tau_list, best_tau, history)."""
    dev = _dev()
    tau_list = [torch.ones(H, N, device=dev, requires_grad=True) for _ in range(nL)]
    ids_t = torch.tensor(train_ids, device=dev)

    history = []
    best_val_ppl = float("inf")
    best_tau_snapshot = None

    for step_i in range(steps):
        logits, _ = rwkv7_fwd_detailed(train_ids, w, nL, C, H, N, tau_v=tau_list)
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
        val_ppl = eval_ppl(val_ids_list, w, nL, C, H, N, tau_v=taus_d)
        train_ppl = math.exp(loss.item())

        if val_ppl < best_val_ppl:
            best_val_ppl = val_ppl
            best_tau_snapshot = [t.detach().clone() for t in tau_list]

        history.append({
            "step": step_i, "loss": loss.item(), "train_ppl": train_ppl,
            "val_ppl": val_ppl, "grad_norm": gn, "tau_change": tch,
        })

        del loss, grads
        torch.cuda.empty_cache()

    return tau_list, best_tau_snapshot, history