"""
Qwen3 + Qwen3.5 s^τ 实验 — 正确的 patch 方式

替换 eager_attention_forward 的 nn.functional.softmax → s^τ
"""
import sys, os, math, gc
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F

EPS = 1e-8
_current_tau = 1.0

def set_tau(val):
    global _current_tau
    _current_tau = val

# ===== s^τ attention forward =====
def make_stau_forward(orig_forward, repeat_kv):
    """包装 eager_attention_forward → s^τ 版本"""
    def stau_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
        key_states = repeat_kv(key, module.num_key_value_groups)
        value_states = repeat_kv(value, module.num_key_value_groups)
        attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask
        # s^τ 替换 softmax
        clamped = attn_weights.clamp(min=EPS)
        powered = clamped.pow(_current_tau)
        attn_weights = powered / (powered.sum(dim=-1, keepdim=True) + EPS)
        attn_weights = attn_weights.to(query.dtype)
        attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
        attn_output = torch.matmul(attn_weights, value_states)
        attn_output = attn_output.transpose(1, 2).contiguous()
        return attn_output, attn_weights
    return stau_forward

# ===== 打 patch =====
def patch_qwen3():
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    from transformers.models.qwen3.modeling_qwen3 import repeat_kv
    qwen3_mod.eager_attention_forward = make_stau_forward(qwen3_mod.eager_attention_forward, repeat_kv)
    print('  [patch] Qwen3 eager_attention_forward → s^tau', flush=True)

def patch_qwen35():
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_mod
    from transformers.models.qwen3_5.modeling_qwen3_5 import repeat_kv
    qwen35_mod.eager_attention_forward = make_stau_forward(qwen35_mod.eager_attention_forward, repeat_kv)
    print('  [patch] Qwen3.5 eager_attention_forward → s^tau', flush=True)

def unpatch_qwen3():
    import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
    qwen3_mod.eager_attention_forward = qwen3_mod._orig_eager_attention_forward
    print('  [unpatch] Qwen3 restored', flush=True)

def unpatch_qwen35():
    import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_mod
    qwen35_mod.eager_attention_forward = qwen35_mod._orig_eager_attention_forward
    print('  [unpatch] Qwen3.5 restored', flush=True)

# 保存原始函数
import transformers.models.qwen3.modeling_qwen3 as qwen3_mod
qwen3_mod._orig_eager_attention_forward = qwen3_mod.eager_attention_forward
import transformers.models.qwen3_5.modeling_qwen3_5 as qwen35_mod
qwen35_mod._orig_eager_attention_forward = qwen35_mod.eager_attention_forward

# ====================================================================
# Qwen3 实验
# ====================================================================
def run_qwen3():
    print('\n' + '='*60)
    print('Qwen3-0.6B 中文 s^τ (正确 patch)')
    print('='*60, flush=True)

    from modelscope import AutoModelForCausalLM, AutoTokenizer

    # 原始模型
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B', trust_remote_code=True)
    model_orig = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation='eager')
    model_orig.eval()

    # Patched 模型 (重新加载确保干净)
    patch_qwen3()
    model_stau = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation='eager')
    model_stau.eval()

    @torch.no_grad()
    def generate(model, prompt, max_new=20, temp=0.7, top_k=40):
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

    # 验证 τ=1.0 ≡ softmax
    print('\n[验证] τ=1.0 ≈ 原始 softmax?', flush=True)
    set_tau(1.0)
    t1 = generate(model_stau, '人生的意义是什么？', max_new=15, temp=0.7)
    # unpatch 后用原始模型
    unpatch_qwen3()
    t2 = generate(model_orig, '人生的意义是什么？', max_new=15, temp=0.7)
    match = '✅' if t1 == t2 else '⚠️ diff (seed?)'
    print(f'  s^τ τ=1.0: {t1}', flush=True)
    print(f'  original:  {t2}', flush=True)
    print(f'  {match}', flush=True)

    # τ 扫描
    print('\n[实验] τ 扫描', flush=True)
    patch_qwen3()
    prompts = ['人生的意义是什么？', '人工智能的未来发展方向']
    for p in prompts:
        print(f'  --- "{p}" ---', flush=True)
        for tau in [1.0, 2.0, 3.5, 5.0, 10.0]:
            set_tau(tau)
            t = generate(model_stau, p, max_new=20, temp=0.7)
            print(f'  τ={tau:>5.1f} | {t}', flush=True)
        print()

    unpatch_qwen3()
    del model_orig, model_stau, tok
    gc.collect()


# ====================================================================
# Qwen3.5 多模态实验
# ====================================================================
def run_qwen35():
    print('\n' + '='*60)
    print('Qwen3.5-4B 多模态中文 s^τ')
    print('='*60, flush=True)
    print('  (Qwen3.5 混合架构: 3/4 Gated DeltaNet + 1/4 Gated Attention)', flush=True)
    print('  patch 影响 Gated Attention 层 ≈ 8/32 层)', flush=True)

    from modelscope import AutoModelForCausalLM, AutoTokenizer

    patch_qwen35()
    tok = AutoTokenizer.from_pretrained('Qwen/Qwen3.5-4B', trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3.5-4B', torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation='eager')
    model.eval()

    with torch.no_grad():
        def generate(prompt, max_new=20, temp=0.7, top_k=40):
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

        print('\n[实验] Qwen3.5 τ 扫描', flush=True)
        prompts = ['人生的意义是什么？', '人工智能的未来']
        for p in prompts:
            print(f'  --- "{p}" ---', flush=True)
            for tau in [1.0, 2.0, 3.5, 5.0, 10.0]:
                set_tau(tau)
                t = generate(p, max_new=20, temp=0.7)
                print(f'  τ={tau:>5.1f} | {t}', flush=True)
            print()

    unpatch_qwen35()
    del model, tok
    gc.collect()


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    quick = '--quick' in sys.argv

    run_qwen3()
    print('\n✅ Qwen3 done, now Qwen3.5...', flush=True)
    if not quick:
        run_qwen35()
    else:
        print('  (--quick 模式, 跳过 Qwen3.5)')

    print('\nAll done!', flush=True)
