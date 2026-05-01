"""
三种注意力机制实现（实数版本）

1. StandardAttention - 标准点积注意力（基线）
2. VMFAttention - von Mises-Fisher 核注意力
3. GrassmannAttention - 实数 Grassmann 投影注意力（Cayley参数化）

所有 Attention 支持 RoPE（旋转位置编码）用于长度外推。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rotary_emb(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return torch.cat([
        x1 * cos[..., :half] - x2 * sin[..., :half],
        x1 * sin[..., :half] + x2 * cos[..., :half],
    ], dim=-1)


def apply_attn_norm(scores, dim=-1, norm_type='softmax', eps=1e-8, per_head_tau=None,
                   use_fused=True):
    if norm_type == 'softmax':
        return F.softmax(scores, dim=dim)
    elif norm_type == 'l1':
        scores_finite = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        scores_abs = scores_finite.abs()
        return scores_abs / (scores_abs.sum(dim=dim, keepdim=True) + eps)
    elif norm_type == 'l2':
        scores_finite = torch.where(torch.isfinite(scores), scores, torch.zeros_like(scores))
        return F.normalize(scores_finite, p=2, dim=dim, eps=eps)
    elif norm_type == 'learned':
        if per_head_tau is None:
            raise ValueError(
                "norm_type='learned' requires per_head_tau to be set. "
                "Ensure the attention module has per_head_log_tau nn.Parameter."
            )
        if use_fused:
            from .s_tau_fused import s_tau_norm
            return s_tau_norm(scores, per_head_tau, eps=eps)

        clamped = scores.clamp(min=eps)
        tau = per_head_tau.view(1, -1)
        while tau.dim() < clamped.dim():
            tau = tau.unsqueeze(-1)
        powered = clamped.pow(tau)
        return powered / (powered.sum(dim=dim, keepdim=True) + eps)
    elif norm_type == 'tempered':
        tau = per_head_tau.view(1, -1)
        while tau.dim() < scores.dim():
            tau = tau.unsqueeze(-1)
        return F.softmax(scores * tau, dim=dim)
    else:
        raise ValueError(f"Unknown norm_type: {norm_type}")


class StandardAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0,
                 norm_type: str = 'softmax'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.dropout_p = dropout
        self.norm_type = norm_type

        if norm_type in ('learned', 'tempered'):
            self.per_head_log_tau = nn.Parameter(torch.zeros(num_heads))
        else:
            self.per_head_log_tau = None

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False, rotary=None):
        B, L, D = x.shape
        H = self.num_heads
        dh = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, H, dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, dh).permute(0, 2, 1, 3)

        if rotary is not None:
            cos, sin = rotary
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            scores = scores + attn_mask
        else:
            L_k = k.size(2)
            mask = torch.triu(
                torch.full((L, L_k), float('-inf'), device=x.device),
                diagonal=L_k - L + 1
            )
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        log_tau = getattr(self, 'per_head_log_tau', None)
        if log_tau is not None:
            offset = 0.0 if self.norm_type == 'tempered' else 1.0
            tau = F.softplus(log_tau) + offset
        else:
            tau = None
        attn_weights = apply_attn_norm(scores, dim=-1, norm_type=self.norm_type, per_head_tau=tau)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if use_cache:
            return out, present_kv
        return out


class VMFAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.0,
                 kappa_init: float = 1.0, kappa_learnable: bool = True,
                 norm_type: str = 'softmax'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.dropout_p = dropout
        self.norm_type = norm_type

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        log_kappa_init = math.log(max(kappa_init, 1e-6))
        if kappa_learnable:
            self.log_kappa = nn.Parameter(torch.full((1,), log_kappa_init))
        else:
            self.register_buffer('log_kappa', torch.full((1,), log_kappa_init))
        self.kappa_min = 0.01
        self.kappa_max = 100.0

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.0)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)

    @property
    def kappa(self):
        return torch.clamp(torch.exp(self.log_kappa), self.kappa_min, self.kappa_max)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False, rotary=None):
        B, L, D = x.shape
        H = self.num_heads
        dh = self.head_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, H, dh).permute(0, 2, 1, 3)
        k = k.view(B, L, H, dh).permute(0, 2, 1, 3)
        v = v.view(B, L, H, dh).permute(0, 2, 1, 3)

        if rotary is not None:
            cos, sin = rotary
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=2)
            v = torch.cat([past_v, v], dim=2)

        present_kv = (k, v) if use_cache else None

        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(k, dim=-1)

        cos_sim = torch.matmul(q_norm, k_norm.transpose(-2, -1))
        scores = self.kappa * cos_sim

        if attn_mask is not None:
            scores = scores + attn_mask
        else:
            L_k = k.size(2)
            mask = torch.triu(
                torch.full((L, L_k), float('-inf'), device=x.device),
                diagonal=L_k - L + 1
            )
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = apply_attn_norm(scores, dim=-1, norm_type=self.norm_type)
        attn_weights = self.attn_dropout(attn_weights)

        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if use_cache:
            return out, present_kv
        return out


class GrassmannAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 8, rank: int = 2,
                 dropout: float = 0.0, eps: float = 1e-6,
                 metric_mode: str = 'diagonal', norm_type: str = 'softmax'):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.rank = rank
        self.head_dim = dim // num_heads
        self.sub_dim = self.head_dim // rank
        self.eps = eps
        self.dropout_p = dropout
        self.metric_mode = metric_mode
        self.norm_type = norm_type

        if self.head_dim % rank != 0:
            raise ValueError(f"head_dim {self.head_dim} must be divisible by rank {rank}")
        if metric_mode not in ('full', 'diagonal'):
            raise ValueError(f"metric_mode must be 'full' or 'diagonal'")

        self.qkv_proj = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        d_eff = self.sub_dim
        R = self.rank
        if metric_mode == 'full':
            self.scale = math.sqrt(d_eff) / (R ** 2)
        else:
            self.scale = math.sqrt(d_eff) / R

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.qkv_proj.weight, gain=1.5)
        nn.init.xavier_uniform_(self.out_proj.weight, gain=1.0)

    def _cayley_orthogonalize(self, X: torch.Tensor) -> torch.Tensor:
        M = torch.matmul(X.transpose(-2, -1), X)
        M_f32 = M.float()
        eigenvalues, eigenvectors = torch.linalg.eigh(M_f32)
        eigenvalues = eigenvalues.clamp(min=self.eps)
        D_inv_sqrt = 1.0 / torch.sqrt(eigenvalues)
        M_inv_sqrt = torch.einsum('nij,nj,nkj->nik', eigenvectors, D_inv_sqrt, eigenvectors)
        U = torch.matmul(X, M_inv_sqrt)
        return U

    def _orthogonalize(self, X: torch.Tensor) -> torch.Tensor:
        return self._cayley_orthogonalize(X)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False, rotary=None):
        B, L, D = x.shape
        H, R, d_c = self.num_heads, self.rank, self.sub_dim

        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        q = q.view(B, L, H, self.head_dim)
        k = k.view(B, L, H, self.head_dim)
        v = v.view(B, L, H, self.head_dim)

        if rotary is not None:
            cos, sin = rotary
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            q = apply_rotary_emb(q, cos, sin)
            k = apply_rotary_emb(k, cos, sin)
            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)

        q = q.view(B, L, H, R, d_c)
        k = k.view(B, L, H, R, d_c)
        v = v.view(B, L, H, self.head_dim)

        B_L_H = B * L * H
        q_2d = q.reshape(B_L_H, d_c, R)
        k_2d = k.reshape(B_L_H, d_c, R)
        q_2d = self._orthogonalize(q_2d)
        k_2d = self._orthogonalize(k_2d)
        q = q_2d.reshape(B, L, H, R, d_c)
        k = k_2d.reshape(B, L, H, R, d_c)

        if past_kv is not None:
            past_k, past_v = past_kv
            k = torch.cat([past_k, k], dim=1)
            v = torch.cat([past_v, v], dim=1)

        present_kv = (k, v) if use_cache else None
        L_k = k.size(1)

        q = q.permute(0, 2, 3, 1, 4)
        k = k.permute(0, 2, 3, 1, 4)

        if self.metric_mode == 'full':
            scores = self._compute_full_metric(q, k)
        else:
            scores = self._compute_diagonal_metric(q, k)

        if attn_mask is not None:
            scores = scores + attn_mask
        else:
            mask = self._get_causal_mask(L, L_k, x.device)
            scores = scores + mask.unsqueeze(0).unsqueeze(0)

        attn_weights = apply_attn_norm(scores, dim=-1, norm_type=self.norm_type)
        if self.attn_dropout is not None and self.training:
            attn_weights = self.attn_dropout(attn_weights)

        v = v.permute(0, 2, 1, 3)
        out = torch.matmul(attn_weights, v)
        out = out.permute(0, 2, 1, 3).contiguous().view(B, L, D)
        out = self.out_proj(out)

        if use_cache:
            return out, present_kv
        return out

    def _compute_full_metric(self, q, k):
        B, H, R, L, d_c = q.shape
        L_k = k.size(3)
        q = q.permute(0, 1, 3, 2, 4)
        k = k.permute(0, 1, 3, 2, 4)
        q_flat = q.reshape(B * H, L, R, d_c)
        k_flat = k.reshape(B * H, L_k, R, d_c)
        inner = torch.einsum('birm,bjsm->bijrs', q_flat, k_flat)
        scores = (inner ** 2).sum(dim=(-2, -1))
        scores = scores.reshape(B, H, L, L_k) * self.scale
        return scores

    def _compute_diagonal_metric(self, q, k):
        inner = torch.matmul(q, k.transpose(-2, -1))
        scores = (inner ** 2).sum(dim=2) * self.scale
        return scores

    def _get_causal_mask(self, L, L_k, device):
        mask = torch.triu(
            torch.full((L, L_k), float('-inf'), device=device),
            diagonal=L_k - L + 1
        )
        return mask
