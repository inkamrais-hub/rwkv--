"""
75M ATTH Model + s^τ fused 集成

用法:
    from model_tiny import build_model
    model = build_model(vocab_size=256, norm='learned', max_seq=512)
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn as nn
from attention_mechanisms.model import ATTHModel


def build_model(vocab_size, norm='learned', max_seq=2048, dim=768,
                n_layers=11, n_heads=12, dropout=0.0, use_rope=False,
                tie_weights=False):
    kwargs = dict(
        use_pos_emb=not use_rope,
        use_rope=use_rope,
        attn_type='standard',
        norm_type=norm,
    )
    return ATTHModel(vocab_size, dim=dim, num_layers=n_layers,
                     num_heads=n_heads, max_seq_len=max_seq,
                     dropout=dropout, tie_weights=tie_weights, **kwargs)


def get_tau_params(model):
    return [p for n, p in model.named_parameters() if 'log_tau' in n]

def get_other_params(model):
    return [p for n, p in model.named_parameters() if 'log_tau' not in n]


def make_optimizer(model, norm='learned', lr=6e-4, wd=0.01):
    other = get_other_params(model)
    groups = [{'params': other, 'lr': lr, 'weight_decay': wd}]
    if norm == 'learned':
        tau = get_tau_params(model)
        if tau:
            groups.append({'params': tau, 'lr': 1e-2, 'weight_decay': 0.0})
    return torch.optim.AdamW(groups, betas=(0.9, 0.95), fused=True)


def count_params(model):
    return sum(p.numel() for p in model.parameters())


def get_tau_values(model):
    result = {}
    for i, layer in enumerate(model.layers):
        attn = layer.attn
        if hasattr(attn, 'per_head_log_tau') and attn.per_head_log_tau is not None:
            t = torch.nn.functional.softplus(attn.per_head_log_tau.detach()) + 1.0
            result[i] = {'mean': round(t.mean().item(), 4),
                          'vals': [round(v, 4) for v in t.tolist()]}
    return result


def get_avg_tau(model):
    taus = get_tau_values(model)
    if not taus:
        return 0.0
    return sum(v['mean'] for v in taus.values()) / len(taus)
