"""
加载 GPT-2 small 124M pretrained → 替换 softmax → s^τ → 拧 τ 旋钮看效果

用法:
    python scripts/_gpt2_stau.py
"""
import sys, os, math, threading
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deploy_pkg'))
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

EPS = 1e-8

# 全局状态: 当前 τ 值 (per-attention-module, 但全设一样简化)
_current_tau = 1.0

def s_tau_softmax(x, dim=-1, dtype=None):
    """替代 torch.softmax: 4D 张量用 s^τ, 其余走原版."""
    if x.dim() == 4:
        clamped = x.clamp(min=EPS)
        powered = clamped.pow(_current_tau)
        out = powered / (powered.sum(dim=dim, keepdim=True) + EPS)
        if dtype is not None:
            out = out.to(dtype)
        return out
    return torch._orig_softmax(x, dim=dim, dtype=dtype)


# ===== 1. 加载 GPT-2 =====
print('[1] Loading GPT-2 small (124M)...')
tok = AutoTokenizer.from_pretrained('gpt2')
tok.pad_token = tok.eos_token
model = AutoModelForCausalLM.from_pretrained('gpt2', torch_dtype=torch.float32, attn_implementation='eager')
model.eval()
n = sum(p.numel() for p in model.parameters())
print(f'    vocab={tok.vocab_size}, params={n/1e6:.1f}M')

# ===== 2. Monkey-patch =====
print('[2] Patching torch.softmax → s^tau...')

# 保存原始 softmax
torch._orig_softmax = torch.softmax

import transformers.models.gpt2.modeling_gpt2 as gpt2_module
orig_attn_forward = gpt2_module.GPT2Attention.forward

def patched_attn_forward(self, *args, **kwargs):
    global _current_tau
    # 设置全局 τ (每个 head 独立, 但这里统一)
    torch.softmax = s_tau_softmax
    try:
        return orig_attn_forward(self, *args, **kwargs)
    finally:
        torch.softmax = torch._orig_softmax

gpt2_module.GPT2Attention.forward = patched_attn_forward
print('    Replaced GPT2Attention.forward with s^tau wrapper')

# ===== 3. τ 控制 =====
def set_tau(val):
    global _current_tau
    _current_tau = val
    print(f'    τ={val:.1f}')

# ===== 4. 生成 =====
@torch.no_grad()
def generate(prompt, max_new=50, temperature=0.9, top_k=40):
    input_ids = tok.encode(prompt, return_tensors='pt')
    generated = input_ids.clone()
    for _ in range(max_new):
        x = generated[:, -1024:]
        logits = model(x).logits[:, -1, :]
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

# ===== 5. s^τ τ=1.0 vs softmax (预期不同, s^1 ≠ softmax) =====
print('\n[3] s^τ τ=1.0 vs softmax (s^1 = clamp/sum ≠ exp/sum)\n')

for p in ["The future of AI is", "I believe the meaning of life is"]:
    set_tau(1.0)
    t = generate(p, max_new=30, temperature=0.8)
    print(f'  "{p}"')
    print(f'  τ=1.0 → {t}\n')

# ===== 6. 扫 τ =====
print('[4] τ sweep\n')

for tau in [1.0, 2.0, 3.5, 5.0, 10.0]:
    set_tau(tau)
    t = generate("The future of AI is", max_new=40, temperature=0.8)
    print(f'  τ={tau:>5.1f}  |  {t}\n')

# ===== 7. 第二个 prompt =====
print('[5] Second prompt\n')
for tau in [1.0, 2.0, 3.5, 5.0, 10.0]:
    set_tau(tau)
    t = generate("I believe the meaning of life is", max_new=40, temperature=0.8)
    print(f'  τ={tau:>5.1f}  |  {t}\n')

# ===== 8. 注意力分析 =====
print('[6] Attention entropy (force eager mode)\n')

# 需要 eager 模式才能 output_attentions
model.config._attn_implementation = 'eager'

@torch.no_grad()
def analyze(prompt, tau_val):
    set_tau(tau_val)
    input_ids = tok.encode(prompt, return_tensors='pt')
    model.config.output_attentions = True
    outputs = model(input_ids, output_attentions=True)
    model.config.output_attentions = False
    attentions = outputs.attentions
    print(f'  τ={tau_val:>5.1f}:')
    for i, attn in enumerate(attentions[:3]):
        T = attn.size(-1)
        H = -(attn * torch.log(attn.clamp(min=1e-12))).sum(dim=-1)
        H_max = math.log(T)
        H_norm = H / H_max
        max_w = attn.max(dim=-1).values
        print(f'    layer {i}: entropy={H_norm.mean().item():.4f}  max_attn={max_w.mean().item():.4f}')

analyze("The future of AI is", 1.0)
analyze("The future of AI is", 3.5)
analyze("The future of AI is", 10.0)

print('\nDone!')
