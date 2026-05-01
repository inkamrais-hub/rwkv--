"""
ATTH 模型（参考 ATTH-v2 架构）
支持三种注意力机制 + RoPE（旋转位置编码）
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .attention import StandardAttention, VMFAttention, GrassmannAttention
from .attention_complex import ComplexStandardAttention, ComplexVMFAttention, ComplexGrassmannCayleyAttention
from .attention_equivariant import EquivariantVMFAttention, EquivariantGrassmannAttention
from .attention_complex_equiv import (
    ComplexEquivariantGrassmannAttention,
    ComplexFullEquivariantGrassmannAttention,
    ComplexMeanEquivariantGrassmannAttention,
)


class AngleEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.amplitude = nn.Parameter(torch.ones(dim))

    def forward(self, x, mod_vals):
        B, L = x.shape
        half_d = self.dim // 2
        m = mod_vals.float().reshape(B, 1, 1)
        v = x.float().unsqueeze(-1)
        idx = torch.arange(1, half_d + 1, device=x.device).float()
        freqs = idx / m.clamp(min=1)
        angles = 2 * math.pi * v * freqs
        emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if self.dim % 2 == 1:
            emb = torch.cat([emb, torch.zeros(B, L, 1, device=x.device)], dim=-1)
        return emb * self.amplitude.unsqueeze(0).unsqueeze(0)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.pow(2).mean(-1, keepdim=True).clamp(min=self.eps).sqrt()
        return x / rms * self.scale


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_mult: float = 4.0):
        super().__init__()
        hidden_dim = int(dim * ffn_mult * 2 / 3)
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 4096, base: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len

    def forward(self, seq_len: int, device: torch.device):
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()


ATTN_REGISTRY = {
    'standard': StandardAttention,
    'vmf': VMFAttention,
    'grassmann': GrassmannAttention,
    'complex_standard': ComplexStandardAttention,
    'complex_vmf': ComplexVMFAttention,
    'complex_grassmann': ComplexGrassmannCayleyAttention,
    'equiv_vmf': EquivariantVMFAttention,
    'equiv_grassmann': EquivariantGrassmannAttention,
    'complex_equiv_grassmann': ComplexEquivariantGrassmannAttention,
    'complex_full_equiv_grassmann': ComplexFullEquivariantGrassmannAttention,
    'complex_mean_equiv_grassmann': ComplexMeanEquivariantGrassmannAttention,
}


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, attn_type: str = 'standard',
                 dropout: float = 0.0, ffn_mult: float = 4.0, **attn_kwargs):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        attn_cls = ATTN_REGISTRY[attn_type]
        self.attn = attn_cls(dim=dim, num_heads=num_heads, dropout=dropout, **attn_kwargs)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_mult)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, past_kv=None, use_cache=False, rotary=None):
        residual = x
        x = self.norm1(x)
        attn_out = self.attn(x, attn_mask=attn_mask, past_kv=past_kv,
                             use_cache=use_cache, rotary=rotary)
        if use_cache:
            attn_out, present_kv = attn_out
        x = residual + self.dropout(attn_out)
        residual = x
        x = self.norm2(x)
        x = residual + self.dropout(self.ffn(x))
        if use_cache:
            return x, present_kv
        return x


class ATTHModel(nn.Module):
    def __init__(self, vocab_size: int, dim: int = 256, num_layers: int = 6,
                 num_heads: int = 8, max_seq_len: int = 512, dropout: float = 0.1,
                 ffn_mult: float = 4.0, tie_weights: bool = False,
                 attn_type: str = 'standard', use_rope: bool = False,
                 use_pos_emb: bool = True, angle_emb: bool = False,
                 **attn_kwargs):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.use_pos_emb = use_pos_emb
        self.angle_emb = angle_emb

        if angle_emb:
            self.token_emb = AngleEmbedding(dim)
        else:
            self.token_emb = nn.Embedding(vocab_size, dim)
        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim, max_seq_len)
        elif use_pos_emb:
            self.pos_emb = nn.Embedding(max_seq_len, dim)

        self.layers = nn.ModuleList([
            TransformerBlock(dim, num_heads, attn_type, dropout, ffn_mult, **attn_kwargs)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(dim)
        self.lm_head = nn.Linear(dim, vocab_size, bias=False)

        if tie_weights and not angle_emb:
            self.lm_head.weight = self.token_emb.weight

        self._init_weights()

    def _init_weights(self):
        if hasattr(self.token_emb, 'weight'):
            nn.init.normal_(self.token_emb.weight, mean=0.0, std=0.02)
        if hasattr(self, 'pos_emb'):
            nn.init.normal_(self.pos_emb.weight, mean=0.0, std=0.02)

    def forward(self, x, attn_mask=None, return_logits=True):
        B, L = x.shape
        device = x.device

        if self.angle_emb:
            mod_vals = x[:, 0:1]
            h = self.token_emb(x, mod_vals)
        else:
            h = self.token_emb(x)

        if self.use_rope:
            cos, sin = self.rotary(L, device)
            rotary = (cos, sin)
            for layer in self.layers:
                h = layer(h, attn_mask=attn_mask, rotary=rotary)
        elif self.use_pos_emb:
            pos = torch.arange(L, device=device).unsqueeze(0)
            h = h + self.pos_emb(pos)
            for layer in self.layers:
                h = layer(h, attn_mask=attn_mask)
        else:
            for layer in self.layers:
                h = layer(h, attn_mask=attn_mask)

        h = self.norm(h)

        if return_logits:
            logits = self.lm_head(h)
            return logits
        return h
