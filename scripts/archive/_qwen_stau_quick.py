"""
Qwen3 s^tau 中文快速实验 — 直接出结果
"""
import sys, os, math
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F

EPS = 1e-8
_current_tau = 1.0

def s_tau_softmax(x, dim=-1, dtype=None):
    if x.dim() == 4:
        clamped = x.clamp(min=EPS)
        powered = clamped.pow(_current_tau)
        out = powered / (powered.sum(dim=dim, keepdim=True) + EPS)
        if dtype is not None:
            out = out.to(dtype)
        return out
    return torch._orig_softmax(x, dim=dim, dtype=dtype)

# patch
torch._orig_softmax = torch.softmax

def set_tau(val):
    global _current_tau
    _current_tau = val

# 加载
print('加载 Qwen3-0.6B...', flush=True)
from modelscope import AutoModelForCausalLM, AutoTokenizer

# 两个模型对比: 原始的 vs patched的
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)

# 1. 原始模型 (不patch)
model_orig = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True)
model_orig.eval()

# 2. s^tau 模型 (patch)
torch.softmax = s_tau_softmax  # 全局替换
model_stau = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True)
model_stau.eval()

print(f'词表={len(tok)}', flush=True)

@torch.no_grad()
def generate(model, prompt, max_new=20, temp=0.7, top_k=40):
    input_ids = tok.encode(prompt, return_tensors='pt')
    generated = input_ids.clone()
    for i in range(max_new):
        x = generated[:, -1024:]
        logits = model(x).logits[:, -1, :]
        logits = logits / temp
        if top_k > 0:
            vals, _ = torch.topk(logits, top_k)
            logits[logits < vals[:, -1:]] = float('-inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)
        generated = torch.cat([generated, next_id], dim=1)
        if next_id.item() == tok.eos_token_id:
            break
    return tok.decode(generated[0].tolist(), skip_special_tokens=True)

# === 验证: τ=1.0 和原始模型是否一致 ===
print('\n[验证] τ=1.0 ≈ 原始 softmax ?', flush=True)
prompts = ['人生的意义是什么？', '人工智能的未来']
for p in prompts:
    set_tau(1.0)
    torch.softmax = s_tau_softmax
    t1 = generate(model_stau, p, max_new=15, temp=0.7)
    torch.softmax = torch._orig_softmax
    t2 = generate(model_orig, p, max_new=15, temp=0.7)
    match = '✅' if t1 == t2 else '⚠️ diff'
    print(f'  "{p[:20]}"', flush=True)
    print(f'  s^τ τ=1.0: {t1}', flush=True)
    print(f'  original:  {t2}', flush=True)
    print(f'  {match}', flush=True)
    print()

# === τ 扫描 ===
print('[实验] τ 扫描', flush=True)
torch.softmax = s_tau_softmax  # 切回 s^tau

for p in prompts[:2]:
    print(f'--- "{p}" ---', flush=True)
    for tau in [1.0, 2.0, 3.5, 5.0, 10.0]:
        set_tau(tau)
        t = generate(model_stau, p, max_new=25, temp=0.7)
        print(f'  τ={tau:>5.1f} | {t}', flush=True)
    print()

print('Done!', flush=True)
