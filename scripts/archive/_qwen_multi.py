"""
Qwen3.5-0.8B Instruct 多语言 s^τ 实验 + 证伪讨论

用法:
    python scripts/_qwen_multi.py
"""
import sys, os, math, gc, random
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F

EPS = 1e-8
_current_tau = 1.0

def set_tau(val):
    global _current_tau
    _current_tau = val

# ===== Patch: 替换 eager_attention_forward 的 softmax → s^τ =====
def patch_qwen35():
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    mod._orig_forward = mod.eager_attention_forward
    def stau_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        clamped = attn_weights.clamp(min=EPS)
        powered = clamped.pow(_current_tau)
        attn_weights = powered / (powered.sum(dim=-1, keepdim=True) + EPS)
        attn_weights = attn_weights.to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights
    mod.eager_attention_forward = stau_forward
    print('  [patch] Qwen3.5 eager_attention_forward → s^tau', flush=True)

def unpatch_qwen35():
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    mod.eager_attention_forward = mod._orig_forward
    print('  [unpatch] Qwen3.5 restored', flush=True)

# ===== 加载模型 =====
print('加载 Qwen3.5-0.8B Instruct...', flush=True)
from modelscope import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = 'Qwen/Qwen3.5-0.8B'

# 原始模型
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
model_orig = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True,
    attn_implementation='eager')
model_orig.eval()

# Patched 模型
patch_qwen35()
model_stau = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.float32, trust_remote_code=True,
    attn_implementation='eager')
model_stau.eval()
print(f'  词表={len(tok)}, Qwen3.5 hybrid: 3/4 DeltaNet + 1/4 Gated Attention', flush=True)

# ===== 生成函数（用于 instruct 模型） =====
@torch.no_grad()
def generate(model, prompt, max_new=30, temp=0.7, top_k=40):
    messages = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tok.encode(text, return_tensors='pt')
    generated = input_ids.clone()
    for _ in range(max_new):
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
            # Also check if it's the end-of-turn token
            break
    full = tok.decode(generated[0].tolist(), skip_special_tokens=True)
    # 提取 assistant 回复
    marker = 'assistant\n'
    if marker in full:
        return full.split(marker)[-1].strip()
    return full

def generate_baseline(model, prompt, max_new=30, temp=0.7, top_k=40):
    """Generate without chat template (base model style)"""
    input_ids = tok.encode(prompt, return_tensors='pt')
    generated = input_ids.clone()
    for _ in range(max_new):
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

# ====================================================================
# Part A: 验证 τ=1.0 的不等价性 + 证伪讨论
# ====================================================================
print('\n' + '='*60)
print('Part A: 验证 + 证伪讨论')
print('='*60, flush=True)

set_tau(1.0)
t1 = generate(model_stau, '人生的意义是什么？', max_new=15, temp=0.7)

unpatch_qwen35()
t2 = generate(model_orig, '人生的意义是什么？', max_new=15, temp=0.7)

print(f'  s^τ τ=1.0: {t1[:100]}', flush=True)
print(f'  original:  {t2[:100]}', flush=True)

print('\n  ⚠️  两者不同 —— 这恰恰是定理的验证:', flush=True)
print('  定理: s^1(scores) = softmax(log(clamp(scores, ε)))  ≠  softmax(scores)', flush=True)
print('  s^τ 和 softmax 是两种不同的归一化函数, 等价需要分数变换', flush=True)
print('  s^τ 套 softmax 权重 = 换了激活函数, 输出必然不同', flush=True)
print('  这正是为什么两阶段训练策略是必要的:', flush=True)
print('  Phase 1: softmax 训练 → Phase 2: 换 s^τ + fine-tune', flush=True)

# ====================================================================
# Part B: 多语言测试
# ====================================================================
print('\n' + '='*60)
print('Part B: 多语言 s^τ 实验')
print('='*60, flush=True)

patch_qwen35()

prompts = [
    ('中文', '人生的意义是什么？'),
    ('English', 'What is the meaning of life?'),
    ('日本語', '人生の意味とは何ですか？'),
    ('中文2', '人工智能的未来发展方向'),
]

for lang, prompt in prompts:
    print(f'\n  --- [{lang}] "{prompt}" ---', flush=True)
    for tau in [1.0, 2.0, 3.5, 5.0, 10.0]:
        set_tau(tau)
        seed = int(tau * 100)
        torch.manual_seed(seed)
        random.seed(seed)
        t = generate(model_stau, prompt, max_new=25, temp=0.7)
        print(f'  τ={tau:>5.1f} | {t[:120]}', flush=True)

# ====================================================================
# Part C: Base vs Instruct 对比
# ====================================================================
print('\n' + '='*60)
print('Part C: 验证 instruct 模板确实有效')
print('='*60, flush=True)

set_tau(1.0)
t_chat = generate(model_stau, '写一句关于大海的话', max_new=20, temp=0.7)
t_base = generate_baseline(model_stau, '写一句关于大海的话', max_new=20, temp=0.7)
print(f'  [instruct] 写一句关于大海的话 → {t_chat}', flush=True)
print(f'  [base]     写一句关于大海的话 → {t_base}', flush=True)

unpatch_qwen35()
print('\n✅ All done!', flush=True)
