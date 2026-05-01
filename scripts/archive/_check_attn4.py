import os
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from modelscope import AutoModelForCausalLM
import torch, inspect

model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True,
    attn_implementation='eager')

attn = model.model.layers[0].self_attn
src = inspect.getsource(type(attn))

with open('f:/τ/attn_full.txt', 'w', encoding='utf-8') as f:
    f.write(f'=== {type(attn).__name__} full source ===\n')
    f.write(src)
    
    # 在 model 模块 level 搜 softmax
    f.write('\n\n=== softmax in whole model module ===\n')
    model_src = inspect.getsource(type(model))
    for i, line in enumerate(model_src.split('\n')):
        if 'softmax' in line.lower():
            f.write(f'  L{i}: {line.strip()[:200]}\n')

print('DONE', flush=True)
