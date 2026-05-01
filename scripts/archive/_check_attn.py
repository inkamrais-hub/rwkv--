import os, json, sys
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from modelscope import AutoModelForCausalLM
import torch, inspect

out = []
def log(s):
    out.append(s)
    print(s, flush=True)

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True)

cfg = model.config.to_dict()
log('=== CONFIG ===')
for k in ['architectures','model_type','hidden_size','num_attention_heads','num_key_value_heads','num_hidden_layers','attn_implementation','use_flash_attn']:
    if k in cfg:
        log(f'  {k}: {cfg[k]}')

log('\n=== ATTN MODULES ===')
for name, mod in model.named_modules():
    n = type(mod).__name__.lower()
    if 'attn' in n and 'qk' not in n and 'rope' not in n:
        log(f'  {name}: {type(mod).__name__}')

log('\n=== softmax in forward ===')
for name, mod in model.named_modules():
    if type(mod).__name__ == 'Qwen3Attention':
        src = inspect.getsource(mod.forward)
        for i, line in enumerate(src.split('\n')):
            if 'softmax' in line.lower():
                log(f'  {name} L{i}: {line.strip()[:200]}')
        # also check for _scaled_dot_product
        for i, line in enumerate(src.split('\n')):
            if 'scaled_dot' in line.lower() or 'sdpa' in line.lower():
                log(f'  {name} L{i}: {line.strip()[:200]}')
                break
        break

log('\n=== EAGER ===')
try:
    model2 = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True, attn_implementation='eager')
    for name, mod in model2.named_modules():
        if type(mod).__name__ == 'Qwen3Attention':
            src = inspect.getsource(mod.forward)
            for i, line in enumerate(src.split('\n')):
                if 'softmax' in line.lower():
                    log(f'  EAGER L{i}: {line.strip()[:200]}')
            break
except Exception as e:
    log(f'  ERROR: {e}')

with open('f:/τ/attn_check_output.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))
log('\nWRITTEN')
