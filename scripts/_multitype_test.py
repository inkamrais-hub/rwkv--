"""
Qwen3.5-0.8B 多类型文本 s^τ 对比实验

测试 5 种文本类型: 诗歌/技术/故事/代码/哲学
对比: 原始 softmax vs s^τ (τ=1.0, 3.5, 10.0)
"""
import sys, os, gc
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM, AutoTokenizer

EPS = 1e-8
_current_tau = 1.0
DEVICE = 'cuda'

out = []
def log(s):
    out.append(str(s))
    print(s, flush=True)

def set_tau(v):
    global _current_tau
    _current_tau = v

orig_fwd = None

def patch():
    global orig_fwd
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    orig_fwd = mod.eager_attention_forward
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    def fwd(module, query, key, value, attn_mask, scaling, dropout=0.0, **kwargs):
        ks = repeat_kv(key, module.num_key_value_groups)
        vs = repeat_kv(value, module.num_key_value_groups)
        aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attn_mask is not None: aw = aw + attn_mask
        c = aw.clamp(min=EPS)
        p = c.pow(_current_tau)
        aw = p / (p.sum(dim=-1, keepdim=True) + EPS)
        aw = aw.to(query.dtype)
        aw = F.dropout(aw, p=dropout, training=module.training)
        ao = torch.matmul(aw, vs).transpose(1, 2).contiguous()
        return ao, aw
    mod.eager_attention_forward = fwd

def unpatch():
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    mod.eager_attention_forward = orig_fwd

log('Loading Qwen3.5-0.8B...')
tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)
patch()
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3.5-0.8B', torch_dtype=torch.float16, trust_remote_code=True,
    attn_implementation='eager', device_map='auto')
model.eval()
log(f'Vocab={len(tok)}')

@torch.no_grad()
def gen(prompt, max_new=15, temp=0.7, top_k=40):
    msg = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, return_tensors='pt').to(DEVICE)
    g = ids.clone()
    for _ in range(max_new):
        x = g[:, -1024:]
        lo = model(x).logits[:, -1, :]
        lo = lo / temp
        vals, _ = torch.topk(lo, top_k)
        lo[lo < vals[:, -1:]] = float('-inf')
        pr = F.softmax(lo, dim=-1)
        ni = torch.multinomial(pr, 1)
        g = torch.cat([g, ni], dim=1)
        if ni.item() == tok.eos_token_id: break
    full = tok.decode(g[0].tolist(), skip_special_tokens=True)
    marker = 'assistant\n'
    return full.split(marker)[-1].strip() if marker in full else full[-200:]

# ===== 5 种文本类型测试 =====
tests = [
    ('诗歌',   '以「月」为题写一首七言绝句'),
    ('技术',   '用通俗的语言解释什么是注意力机制'),
    ('故事',   '写一个关于人工智能觉醒的微小说开头'),
    ('代码',   '用Python写一个快速排序函数'),
    ('哲学',   '自由意志是否存在？'),
]

log('\n' + '='*70)
log('s^τ 跨文本类型对比: softmax(原始) vs τ=1.0 vs τ=3.5 vs τ=10.0')
log('='*70)

for dtype, prompt in tests:
    log(f'\n─── [{dtype}] ───')
    log(f'  Prompt: {prompt}')

    # 原始 softmax
    unpatch()
    torch.manual_seed(42)
    t_orig = gen(prompt)
    log(f'  [softmax]  {t_orig[:100]}')

    # s^τ 不同 τ
    patch()
    for tau in [1.0, 3.5, 10.0]:
        set_tau(tau)
        torch.manual_seed(42)
        t = gen(prompt)
        diff = '← 同softmax?' if t == t_orig and tau == 1.0 else ''
        log(f'  [s^τ τ={tau:>4.1f}]  {t[:100]} {diff}')

unpatch()
log('\nDone! 结果已保存')

with open('f:/τ/multitype_stau.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
