import os
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from modelscope import AutoModelForCausalLM
import torch

model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True)

lines = []
def add(s):
    lines.append(str(s))

add(f'Config _attn_implementation: {model.config._attn_implementation}')
use_flash = getattr(model.config, 'use_flash_attn', 'N/A')
add(f'Config use_flash_attn: {use_flash}')
add('')

# 列出所有子模块类型
types_seen = set()
for name, mod in model.named_modules():
    t = type(mod).__name__
    if t not in types_seen:
        types_seen.add(t)
        add(f'  {t}')

add(f'\nTotal unique types: {len(types_seen)}')

# 找注意力层
add('\n--- Looking for attention ---')
for name, mod in model.named_modules():
    nl = name.lower()
    if 'layers' in nl and any(x in nl for x in ['attn', 'self_attn', 'attention', 'mha']):
        add(f'{name}: {type(mod).__name__}')
        parent_name = name.rpartition('.')[0]
        if parent_name:
            parent = model
            for part in parent_name.split('.'):
                parent = getattr(parent, part)
            for n, m in parent.named_children():
                add(f'  child: {n}: {type(m).__name__}')
        break

# 直接看第一个 decoder layer
add('\n--- First decoder layer children ---')
first_layer = model.model.layers[0]
for n, m in first_layer.named_children():
    add(f'  {n}: {type(m).__name__}')

with open('f:/τ/attn_check2.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('DONE', flush=True)
