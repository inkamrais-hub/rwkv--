import sys, os
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch, gc
import torch.nn.functional as F
from modelscope import AutoModelForCausalLM, AutoTokenizer

EPS = 1e-8
_current_tau = 1.0
DEVICE = 'cuda'

def set_tau(v):
    global _current_tau
    _current_tau = v

def patch():
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    mod._orig_forward = mod.eager_attention_forward
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    def fwd(module, query, key, value, attn_mask, scaling, dropout=0.0, **kwargs):
        ks = repeat_kv(key, module.num_key_value_groups)
        vs = repeat_kv(value, module.num_key_value_groups)
        aw = torch.matmul(query, ks.transpose(2, 3)) * scaling
        if attn_mask is not None:
            aw = aw + attn_mask
        c = aw.clamp(min=EPS)
        p = c.pow(_current_tau)
        aw = p / (p.sum(dim=-1, keepdim=True) + EPS)
        aw = aw.to(query.dtype)
        aw = F.dropout(aw, p=dropout, training=module.training)
        ao = torch.matmul(aw, vs).transpose(1, 2).contiguous()
        return ao, aw
    mod.eager_attention_forward = fwd
    print('[patch]', flush=True)

def unpatch():
    import transformers.models.qwen3_5.modeling_qwen3_5 as mod
    mod.eager_attention_forward = mod._orig_forward

print('Loading...', flush=True)
MODEL = 'Qwen/Qwen3.5-0.8B'
tok = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
patch()
model = AutoModelForCausalLM.from_pretrained(
    MODEL, torch_dtype=torch.float16, trust_remote_code=True,
    attn_implementation='eager', device_map='auto')
model.eval()
print(f'Vocab={len(tok)}', flush=True)

@torch.no_grad()
def gen(prompt, max_new=12, top_k=40):
    msg = [{'role': 'user', 'content': prompt}]
    text = tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
    ids = tok.encode(text, return_tensors='pt').to(DEVICE)
    g = ids.clone()
    for _ in range(max_new):
        x = g[:, -1024:]
        lo = model(x).logits[:, -1, :]
        lo = lo / 0.7
        vals, _ = torch.topk(lo, top_k)
        lo[lo < vals[:, -1:]] = float('-inf')
        pr = F.softmax(lo, dim=-1)
        ni = torch.multinomial(pr, 1)
        g = torch.cat([g, ni], dim=1)
        if ni.item() == tok.eos_token_id:
            break
    full = tok.decode(g[0].tolist(), skip_special_tokens=True)
    m = 'assistant\n'
    return full.split(m)[-1].strip() if m in full else full

print('\n=== Test 1: τ=1.0 → should differ from softmax ===', flush=True)
set_tau(1.0)
try:
    t1 = gen('你好', max_new=10)
    print(f's^τ τ=1.0: {t1[:80]}', flush=True)
except Exception as e:
    print(f'ERROR: {e}', flush=True)
    import traceback; traceback.print_exc()

unpatch()
torch.cuda.empty_cache()
gc.collect()

print('\n=== Test 2: original softmax ===', flush=True)
try:
    t2 = gen('你好', max_new=10)
    print(f'softmax:   {t2[:80]}', flush=True)
except Exception as e:
    print(f'ERROR: {e}', flush=True)

print('\nDone!', flush=True)
