"""
加载 model_softmax_best.pt → 换 s^τ → 调 τ 旋钮看生成效果

用法:
    python scripts/_tau_generate.py                     # 完整实验
    python scripts/_tau_generate.py --prompt "hello"    # 自定义 prompt
"""
import sys, os, math
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deploy_pkg'))
import torch
import torch.nn.functional as F

CKPT = 'f:/τ/project_assets/tiny_results/model_softmax_best.pt'
VOCAB = 50257
DIM = 896
N_LAYERS = 12
N_HEADS = 14
MAX_SEQ = 2048
DEVICE = 'cpu'

# ===== 1. 加载 tokenizer =====
print('[1] Loading GPT-2 tokenizer...')
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from transformers import GPT2TokenizerFast
tok = GPT2TokenizerFast.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
print(f'    vocab_size={tok.vocab_size}, pad={tok.pad_token_id}, eos={tok.eos_token_id}')

# ===== 2. 构建 s^τ 模型 + 加载 softmax 权重 =====
print('[2] Building s^tau model from softmax checkpoint...')
from model_tiny import build_model

model = build_model(vocab_size=VOCAB, norm='learned', dim=DIM,
                    n_layers=N_LAYERS, n_heads=N_HEADS, max_seq=MAX_SEQ,
                    use_rope=False, tie_weights=True)

sd = torch.load(CKPT, map_location=DEVICE, weights_only=True)
missing, _ = model.load_state_dict(sd, strict=False)
print(f'    Loaded {len(sd)-len(missing)}/{len(sd)} weights, tau heads: {len([k for k in missing if "log_tau" in k])} new')
model.eval()

# ===== 3. τ 控制函数 =====
def set_tau(model, tau_value):
    """Override all tau heads to a fixed value."""
    with torch.no_grad():
        for name, p in model.named_parameters():
            if 'log_tau' in name:
                # τ = softplus(log_tau) + 1, 反解 log_tau
                # softplus(x) + 1 = τ → softplus(x) = τ - 1
                # x = ln(exp(τ-1) - 1)
                t = tau_value - 1.0
                if t > 0:
                    p.fill_(math.log(math.expm1(t)))
                else:
                    p.fill_(-10.0)  # softplus(-10) ≈ 0, τ ≈ 1
    taus = get_tau_values(model)
    avg = sum(v['mean'] for v in taus.values()) / len(taus) if taus else 0
    print(f'    τ set to {tau_value:.1f} (actual mean={avg:.3f})')

def get_tau_values(model):
    result = {}
    for i, layer in enumerate(model.layers):
        attn = layer.attn
        if hasattr(attn, 'per_head_log_tau') and attn.per_head_log_tau is not None:
            t = F.softplus(attn.per_head_log_tau.detach()) + 1.0
            result[i] = {'mean': round(t.mean().item(), 4),
                          'vals': [round(v, 4) for v in t.tolist()]}
    return result

# ===== 4. 生成函数 =====
@torch.no_grad()
def generate(model, prompt, max_new=50, temperature=0.9, top_k=40):
    model.eval()
    input_ids = tok.encode(prompt, return_tensors='pt')
    if input_ids.size(1) == 0:
        input_ids = torch.tensor([[tok.eos_token_id]])

    generated = input_ids.clone()
    for _ in range(max_new):
        # 截断到 max_seq
        x = generated[:, -MAX_SEQ:]
        logits = model(x, return_logits=True)[:, -1, :]

        # temperature + top-k sampling
        logits = logits / temperature
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = float('-inf')

        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)

        if next_id.item() == tok.eos_token_id:
            break

    return tok.decode(generated[0].tolist(), skip_special_tokens=True)

# ===== 5. 主实验 =====
prompt = sys.argv[2] if len(sys.argv) > 2 else "The future of AI is"
print(f'\n[3] Generating with prompt: "{prompt}"')
print(f'    max_new=100, temperature=0.8\n')

tau_values = [1.0, 2.0, 3.5, 5.0]
results = {}

for tau in tau_values:
    set_tau(model, tau)
    text = generate(model, prompt, max_new=30, temperature=0.8)
    results[tau] = text
    print(f'\n{"="*60}')
    print(f'τ={tau:>5.1f}  |  {text}')
    print(f'{"="*60}')

# ===== 6. 注意力熵分析 =====
print(f'\n\n[4] Attention entropy analysis across τ values:')
print(f'{"τ":>6s}  {"entropy_mean":>12s}  {"max_attn":>9s}  {"困惑度":>8s}')
print(f'{"-"*40}')

@torch.no_grad()
def analyze_attention(model, prompt):
    input_ids = tok.encode(prompt, return_tensors='pt')[:, :32]
    x = input_ids
    B, T = x.shape
    h = model.token_emb(x)
    if hasattr(model, 'pos_emb'):
        pos = torch.arange(T).unsqueeze(0)
        h = h + model.pos_emb(pos)
    for layer in model.layers:
        h = layer(h)
    attn = model.layers[-1].attn
    W = attn.qkv_proj.weight  # (2688, 896)
    head_dim = DIM // N_HEADS
    # h: (B, T, DIM) → scores: (B, N_HEADS, T, T)
    q = (h @ W[:DIM].T).view(B, T, N_HEADS, head_dim).transpose(1, 2)
    k = (h @ W[DIM:2*DIM].T).view(B, T, N_HEADS, head_dim).transpose(1, 2)
    scores = (q @ k.transpose(-2, -1)) / (head_dim ** 0.5)
    mask = torch.triu(torch.full((T, T), float('-inf'), dtype=scores.dtype), diagonal=1)
    masked = scores + mask

    for tau_test in [1.0, 2.0, 3.5, 5.0, 10.0]:
        s_tau = masked.clamp(min=1e-8).pow(tau_test)
        attn_w = s_tau / (s_tau.sum(dim=-1, keepdim=True) + 1e-8)
        H = -(attn_w * torch.log(attn_w.clamp(min=1e-12))).sum(dim=-1)
        H_max = math.log(T)
        H_norm = H / H_max
        max_w = attn_w.max(dim=-1).values.mean().item()
        print(f'  {tau_test:>6.1f}  {H_norm.mean().item():>12.4f}  {max_w:>9.4f}')

analyze_attention(model, prompt)
print('\nDone!')
