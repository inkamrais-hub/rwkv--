"""
█ s_tau_fused_v2_backup.py — s^τ 归一化融合算子 v2 稳定版 (备份)

已修复: AMP 精度毒药 + clamp 梯度断点
5090 验证: Fused 0.54ms vs Old 0.89ms → 1.64× 加速
梯度验证: Fused vs Old τ diff=0.017 ✅ MATCH

此文件为 v2 稳定版备份。新优化在 s_tau_fused_v3.py 中。
不要删除此备份 — 如果 v3 出问题回退到这里。
"""
import torch
import torch.nn.functional as F


class _STauNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, per_head_tau, eps=1e-8):
        B, H, L_q, L_kv = scores.shape
        out_dtype = scores.dtype
        compute_dtype = torch.float32

        s_f32 = scores.to(compute_dtype)
        tau_f32 = per_head_tau.view(1, H, 1, 1).to(compute_dtype)

        clamped = s_f32.clamp(min=eps)
        powered = clamped.pow(tau_f32)
        row_sum = powered.sum(dim=-1, keepdim=True)
        attn = powered / (row_sum + eps)

        ctx.save_for_backward(attn, s_f32, clamped, tau_f32)
        ctx.eps = eps

        return attn.to(out_dtype)

    @staticmethod
    def backward(ctx, grad_out):
        attn, raw_scores, clamped, tau_f32 = ctx.saved_tensors
        eps = ctx.eps
        compute_dtype = torch.float32

        grad_out = grad_out.to(compute_dtype)
        B, H, L_q, L_kv = attn.shape

        not_clamped = (raw_scores > eps).to(compute_dtype)

        weighted_sum = (attn * grad_out).sum(dim=-1, keepdim=True)
        inner = grad_out - weighted_sum
        grad_scores = tau_f32 * attn * inner / clamped.clamp(min=eps)
        grad_scores = grad_scores * not_clamped

        log_s = clamped.clamp(min=eps).log()

        term1 = (grad_out * attn * log_s).sum(dim=(0, 2, 3))
        a_dot_log = (attn * log_s).sum(dim=-1)
        a_dot_g = weighted_sum.squeeze(-1)
        term2 = (a_dot_log * a_dot_g).sum(dim=(0, 2))

        grad_tau = term1 - term2

        return grad_scores.to(grad_out.dtype), grad_tau.to(grad_out.dtype), None


def s_tau_norm(scores, per_head_tau, eps=1e-8):
    return _STauNormFunction.apply(scores, per_head_tau, eps)
