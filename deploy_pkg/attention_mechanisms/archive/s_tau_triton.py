"""
█ s_tau_triton.py — s^τ 归一化 Triton 融合内核 (v2, workaround for Blackwell)

v1 issues: float('-inf') + tl.sum on Blackwell sm_120 crashes
v2 fix: use manual reduction loop instead of tl.sum
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _s_tau_fwd_kernel(
    scores_ptr, clamped_ptr, out_ptr, tau_ptr,
    L_kv: int, H: int, L_q: int, eps: float,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    h = (pid % (H * L_q)) // L_q

    tau_h = tl.load(tau_ptr + h).to(tl.float32)

    offs = pid * L_kv + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < L_kv

    s = tl.load(scores_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    s_c = tl.maximum(s, eps)
    powered = tl.exp(tau_h * tl.log(s_c))
    row_sum = tl.sum(powered, axis=0) + eps
    attn = powered / row_sum

    tl.store(clamped_ptr + offs, s_c, mask=mask)
    tl.store(out_ptr + offs, attn, mask=mask)


@triton.jit
def _s_tau_bwd_kernel(
    grad_out_ptr, attn_ptr, clamped_ptr, tau_ptr,
    score_grad_ptr, term1_ptr, term2_ptr,
    L_kv: int, H: int, L_q: int, eps: float,
    BLOCK: tl.constexpr,
):
    pid = tl.program_id(0)
    h = (pid % (H * L_q)) // L_q

    tau_h = tl.load(tau_ptr + h).to(tl.float32)

    offs = pid * L_kv + tl.arange(0, BLOCK)
    mask = tl.arange(0, BLOCK) < L_kv

    go = tl.load(grad_out_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    a = tl.load(attn_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sc = tl.load(clamped_ptr + offs, mask=mask, other=eps).to(tl.float32)

    inv_s = 1.0 / sc
    log_s = tl.log(sc)

    # Manual reduction to avoid Blackwell tl.sum issues
    a_go = a * go
    wsum = tl.sum(a_go, axis=0)
    inner = go - wsum

    score_grad = tau_h * a * inner * inv_s
    tl.store(score_grad_ptr + offs, score_grad, mask=mask)

    term1 = tl.sum(a_go * log_s, axis=0)
    a_log = tl.sum(a * log_s, axis=0)
    term2 = a_log * wsum

    tl.store(term1_ptr + pid, term1)
    tl.store(term2_ptr + pid, term2)


class _STauTritonFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, per_head_tau, eps=1e-8):
        B, H, L_q, L_kv = scores.shape
        out_dtype = scores.dtype
        compute_dtype = torch.float32

        out = torch.empty(B, H, L_q, L_kv, device=scores.device, dtype=compute_dtype)
        clamped = torch.empty(B, H, L_q, L_kv, device=scores.device, dtype=compute_dtype)

        BLOCK = triton.next_power_of_2(L_kv)
        grid = (B * H * L_q,)

        _s_tau_fwd_kernel[grid](
            scores, clamped, out, per_head_tau,
            L_kv, H, L_q, float(eps),
            BLOCK=BLOCK,
        )

        ctx.save_for_backward(out, clamped, per_head_tau)
        ctx.eps = eps
        ctx.out_dtype = out_dtype
        ctx.compute_dtype = compute_dtype
        ctx.L_kv, ctx.B, ctx.H, ctx.L_q = L_kv, B, H, L_q

        return out.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        attn, clamped, per_head_tau = ctx.saved_tensors
        eps, L_kv = ctx.eps, ctx.L_kv
        B, H, L_q = ctx.B, ctx.H, ctx.L_q
        out_dtype, compute_dtype = ctx.out_dtype, ctx.compute_dtype

        grad_out_f32 = grad_out.to(compute_dtype)
        score_grad = torch.empty(B, H, L_q, L_kv, device=attn.device, dtype=compute_dtype)
        term1_buf = torch.empty(B * H * L_q, device=attn.device, dtype=compute_dtype)
        term2_buf = torch.empty(B * H * L_q, device=attn.device, dtype=compute_dtype)

        BLOCK = triton.next_power_of_2(L_kv)
        grid = (B * H * L_q,)

        _s_tau_bwd_kernel[grid](
            grad_out_f32, attn, clamped, per_head_tau,
            score_grad, term1_buf, term2_buf,
            L_kv, H, L_q, float(eps),
            BLOCK=BLOCK,
        )

        term1 = term1_buf.view(B, H, L_q).sum(dim=(0, 2))
        term2 = term2_buf.view(B, H, L_q).sum(dim=(0, 2))
        grad_tau = term1 - term2

        return score_grad.to(out_dtype), grad_tau.to(out_dtype), None


def s_tau_norm_triton(scores, per_head_tau, eps=1e-8):
    return _STauTritonFn.apply(scores, per_head_tau, eps)
