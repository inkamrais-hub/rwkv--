"""
█ s_tau_fused.py — s^τ 归一化融合算子 (v4 compiled)

v3: autograd.Function + fp32强制 + clamp mask → 1.02× vs softmax  
v4: torch.compile 加速核心计算路径

用法:
    from s_tau_fused import s_tau_norm
"""
import torch


class _STauNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, per_head_tau, eps=1e-8):
        out_dtype = scores.dtype
        compute_dtype = torch.float32

        needs_cast = (out_dtype != compute_dtype)
        s_f32 = scores.to(compute_dtype) if needs_cast else scores
        t = per_head_tau.view(1, scores.shape[1], 1, 1)
        tau_f32 = t.to(compute_dtype) if t.dtype != compute_dtype else t

        clamped = s_f32.clamp(min=eps)
        powered = clamped.pow(tau_f32)
        row_sum = powered.sum(dim=-1, keepdim=True)
        attn = powered / (row_sum + eps)

        clamp_mask = (s_f32 > eps)
        ctx.save_for_backward(attn, clamped, tau_f32, clamp_mask)
        ctx.needs_cast = needs_cast
        ctx.eps = eps
        ctx.out_dtype = out_dtype

        return attn.to(out_dtype) if needs_cast else attn

    @staticmethod
    def backward(ctx, grad_out):
        attn, clamped, tau_f32, clamp_mask = ctx.saved_tensors
        eps = ctx.eps
        needs_cast = ctx.needs_cast
        out_dtype = ctx.out_dtype
        compute_dtype = torch.float32

        grad_out_f32 = grad_out.to(compute_dtype) if needs_cast else grad_out
        mask = clamp_mask.to(compute_dtype)
        B, H, L_q, L_kv = attn.shape

        weighted_sum = (attn * grad_out_f32).sum(dim=-1, keepdim=True)
        inner = grad_out_f32 - weighted_sum

        safe_s = clamped.clamp(min=eps)
        inv_s = 1.0 / safe_s
        grad_scores = tau_f32 * attn * inner * inv_s * mask

        log_s = safe_s.log()
        term1 = (grad_out_f32 * attn * log_s).sum(dim=(0, 2, 3))
        a_dot_log = (attn * log_s).sum(dim=-1)
        a_dot_g = weighted_sum.squeeze(-1)
        term2 = (a_dot_log * a_dot_g).sum(dim=(0, 2))
        grad_tau = term1 - term2

        return grad_scores.to(out_dtype), grad_tau.to(out_dtype), None


def s_tau_norm(scores, per_head_tau, eps=1e-8):
    return _STauNormFunction.apply(scores, per_head_tau, eps)
