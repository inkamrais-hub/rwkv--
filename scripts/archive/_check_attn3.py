import os
os.environ['MODELSCOPE_CACHE'] = 'f:/τ/modelscope_cache'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
from modelscope import AutoModelForCausalLM
import torch, inspect

# 强制 eager 模式
model = AutoModelForCausalLM.from_pretrained(
    'Qwen/Qwen3-0.6B', torch_dtype=torch.float32, trust_remote_code=True,
    attn_implementation='eager')

lines = []
def add(s):
    lines.append(str(s))

add(f'eager mode _attn_implementation: {model.config._attn_implementation}')

# 找 self_attn 的 forward 中 softmax 调用
attn = model.model.layers[0].self_attn
src = inspect.getsource(attn.forward)
add(f'\nQwen3Attention.forward lines: {len(src.split(chr(10)))}')
for i, line in enumerate(src.split('\n')):
    if any(x in line.lower() for x in ['softmax', 'scaled_dot', 'sdpa']):
        add(f'  L{i}: {line.strip()[:200]}')

# 也检查 F.softmax 和 torch.softmax 引用
add('\n--- Softmax references in module ---')
mod_src = inspect.getsource(type(attn))
for i, line in enumerate(mod_src.split('\n')):
    if 'softmax' in line.lower():
        add(f'  {line.strip()[:200]}')

with open('f:/τ/attn_check3.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(lines))
print('DONE', flush=True)
