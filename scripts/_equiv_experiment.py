"""
s^τ ↔ softmax 等价性验证 + 解剖实验

三部分:
  Part 1: 随机 score 张量数值验证等价性定理
  Part 2: 加载真实 200M checkpoint, 提取注意力对比
  Part 3: τ 解剖 — 变化 τ 对注意力模式的影响

用法:
    python scripts/_equiv_experiment.py          # 完整实验
    python scripts/_equiv_experiment.py --quick  # 只跑 Part 1+3 (不加载大模型)
"""
import sys, os, math, json
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deploy_pkg'))
import torch
import torch.nn.functional as F
import numpy as np

# ========== 配置 ==========
CKPT = 'f:/τ/project_assets/tiny_results/model_softmax_best.pt'
EPS = 1e-8
N_HEADS = 14
DIM = 896
N_LAYERS = 12
HEAD_DIM = DIM // N_HEADS  # 64
VOCAB = 50257
MAX_SEQ = 2048
DEVICE = 'cpu'

# ====================================================================
# Part 1: 数值验证 s^τ ↔ softmax 等价性
# ====================================================================
def part1_numerical_verification():
    print('='*60)
    print('PART 1: 数值验证等价性定理')
    print('='*60)

    rng = torch.Generator().manual_seed(42)
    tau = 3.5  # 典型的 τ 值

    for B, T in [(1, 8), (4, 32), (2, 128)]:
        scores = torch.randn(B, N_HEADS, T, T, generator=rng) * 2.0

        # 应用因果 mask
        mask = torch.triu(torch.full((T, T), float('-inf'), dtype=scores.dtype), diagonal=1)
        masked = scores + mask

        # s^τ 注意力
        clamped = masked.clamp(min=EPS)
        s_tau = clamped.pow(tau)
        attn_stau = s_tau / (s_tau.sum(dim=-1, keepdim=True) + EPS)

        # 等价 softmax: σ = τ · log(clamp(s, ε))
        transformed = tau * torch.log(masked.clamp(min=EPS))
        attn_softmax_via_stau = F.softmax(transformed, dim=-1)

        diff = (attn_stau - attn_softmax_via_stau).abs().max().item()
        ok = '✅' if diff < 1e-5 else '❌'
        print(f'  B={B:>2d} T={T:>4d}  max|diff|={diff:.2e}  {ok}  (τ={tau})')

    # 验证 s^τ → softmax 映射的唯一性
    print()
    print('  === 反向验证: softmax → s^τ ===')
    for B, T in [(2, 16), (1, 64)]:
        scores = torch.randn(B, N_HEADS, T, T, generator=rng)
        mask = torch.triu(torch.full((T, T), float('-inf'), dtype=scores.dtype), diagonal=1)
        masked = scores + mask
        attn_softmax = F.softmax(masked, dim=-1)

        # 对任意 τ, 构造 s_i = exp(σ_i / τ)
        for tau_val in [1.0, 2.0, 5.0]:
            s_constructed = torch.exp(masked / tau_val)
            clamped = s_constructed.clamp(min=EPS)
            s_tau = clamped.pow(tau_val)
            attn_stau_from_softmax = s_tau / (s_tau.sum(dim=-1, keepdim=True) + EPS)

            diff = (attn_softmax - attn_stau_from_softmax).abs().max().item()
            ok = '✅' if diff < 1e-5 else '❌'
            print(f'  B={B} T={T} τ={tau_val:.1f}  max|diff|={diff:.2e}  {ok}')

    # 验证 ε 的敏感度
    print()
    print('  === ε 敏感度分析 ===')
    T = 64
    for eps in [1e-2, 1e-4, 1e-6, 1e-8, 1e-12]:
        scores = torch.randn(2, N_HEADS, T, T, generator=rng) * 3.0
        mask = torch.triu(torch.full((T, T), float('-inf'), dtype=scores.dtype), diagonal=1)
        masked = scores + mask
        clamped = masked.clamp(min=eps)
        s_tau = clamped.pow(tau)
        attn_stau = s_tau / (s_tau.sum(dim=-1, keepdim=True) + eps)
        transformed = tau * torch.log(masked.clamp(min=eps))
        attn_via_softmax = F.softmax(transformed, dim=-1)
        diff = (attn_stau - attn_via_softmax).abs().max().item()
        safe_count = (masked < eps).sum().item()  # 被 clamp 的位置数
        print(f'  ε={eps:.0e}  clamped={safe_count:>5d}  max|diff|={diff:.2e}')


# ====================================================================
# Part 2: 真实 checkpoint 验证
# ====================================================================
def part2_real_checkpoint():
    print()
    print('='*60)
    print('PART 2: 真实 200M checkpoint 验证')
    print('='*60)

    if not os.path.exists(CKPT):
        print(f'  ⚠️  未找到 checkpoint: {CKPT}')
        print(f'  File size: {os.path.getsize(CKPT)/1e9:.2f}GB')
        print('  跳过 Part 2')
        return

    print(f'  加载 checkpoint ({os.path.getsize(CKPT)/1e9:.2f}GB)...')
    sd = torch.load(CKPT, map_location='cpu', weights_only=True)
    print(f'  状态字典: {len(sd)} 个键')

    # 构建 softmax 模型 + 加载权重
    from model_tiny import build_model
    model = build_model(vocab_size=VOCAB, norm='softmax', dim=DIM,
                        n_layers=N_LAYERS, n_heads=N_HEADS, max_seq=MAX_SEQ,
                        use_rope=False, tie_weights=True)
    model.load_state_dict(sd, strict=True)
    model.eval()

    # 构建 s^τ 模型 + 加载兼容权重
    model_stau = build_model(vocab_size=VOCAB, norm='learned', dim=DIM,
                             n_layers=N_LAYERS, n_heads=N_HEADS, max_seq=MAX_SEQ,
                             use_rope=False, tie_weights=True)
    missing, _ = model_stau.load_state_dict(sd, strict=False)
    tau_keys = [k for k in missing if 'log_tau' in k]
    print(f'  s^τ 模型初始化: 加载 {len(sd)-len(missing)}/{len(sd)} 权重层, tau 新参 {len(tau_keys)} 个')
    model_stau.eval()

    # 模型前向对比
    T = 64
    x = torch.randint(100, 200, (2, T))
    print(f'  输入: {x.shape}')

    with torch.no_grad():
        out_sm = model(x, return_logits=True)
        logits_sm = out_sm[:, -1, :]
        ppl_sm = float(F.cross_entropy(logits_sm, x[:, -1]).exp())

        out_stau = model_stau(x, return_logits=True)
        logits_stau = out_stau[:, -1, :]
        ppl_stau = float(F.cross_entropy(logits_stau, x[:, -1]).exp())

        logit_diff = (out_sm - out_stau).abs().max().item()
        print(f'  softmax PPL = {ppl_sm:.2f}')
        print(f'  s^τ (τ≈1.0) PPL = {ppl_stau:.2f}')
        print(f'  最大 logit 差异 = {logit_diff:.4f}')
        print(f'  → {"两阶段训练初始 PPL 会上升 (预期行为)" if ppl_stau > ppl_sm else "意外一致"}')
        print(f'  → s^1 ≠ softmax, 这是正常的, τ 会通过学习补偿差异')

    return model, model_stau


# ====================================================================
# Part 3: τ 解剖 — 注意力熵 vs τ
# ====================================================================
def part3_tau_dissection():
    print()
    print('='*60)
    print('PART 3: τ 解剖 — 注意力熵 vs τ')
    print('='*60)

    rng = torch.Generator().manual_seed(42)
    T = 128
    scores = torch.randn(1, N_HEADS, T, T, generator=rng) * 3.0
    mask = torch.triu(torch.full((T, T), float('-inf'), dtype=scores.dtype), diagonal=1)
    masked = scores + mask

    taus = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0, 10.0]

    print(f'  {"τ":>6s}  {"entropy_mean":>12s}  {"entropy_min":>11s}  {"entropy_max":>11s}  {"max_attn":>9s}')
    print(f'  {"-"*55}')
    for tau in taus:
        clamped = masked.clamp(min=EPS)
        s_tau = clamped.pow(tau)
        attn = s_tau / (s_tau.sum(dim=-1, keepdim=True) + EPS)
        H = -(attn * torch.log(attn.clamp(min=1e-12))).sum(dim=-1)
        H_max = math.log(T)
        H_norm = H / H_max
        max_weight = attn.max(dim=-1).values
        print(f'  {tau:>6.1f}  {H_norm.mean().item():>12.4f}  {H_norm.min().item():>11.4f}  {H_norm.max().item():>11.4f}  {max_weight.mean().item():>9.4f}')

    # 注意力可视化 (一个 head, 一个 query position)
    print()
    print('  === Attention maps (head=0, query=T//2) ===')
    head_idx = 0
    qpos = T // 2
    for tau in [1.0, 2.0, 4.0, 10.0]:
        clamped = masked.clamp(min=EPS)
        s_tau = clamped.pow(tau)
        attn = s_tau / (s_tau.sum(dim=-1, keepdim=True) + EPS)
        attn_row = attn[0, head_idx, qpos, :qpos+1]
        entropy = -(attn_row * torch.log(attn_row.clamp(min=1e-12))).sum().item()
        print(f'  τ={tau:.1f}  entropy={entropy:.3f}  top3={attn_row.topk(3).values.tolist()}')


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    quick = '--quick' in sys.argv

    part1_numerical_verification()

    if not quick:
        part2_real_checkpoint()
    else:
        print()
        print('  (--quick 模式, 跳过 Part 2)')

    part3_tau_dissection()
    print()
    print('='*60)
    print('ALL DONE')
