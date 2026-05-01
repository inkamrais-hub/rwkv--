"""Inspect checkpoint architecture"""
import torch, os

ckpt = 'f:/τ/project_assets/tiny_results/model_softmax_best.pt'
sd = torch.load(ckpt, map_location='cpu', weights_only=True)
print(f'Keys ({len(sd)}):')
for k, v in list(sd.items())[:20]:
    print(f'  {k}: {tuple(v.shape)}')
print('  ...')
print(f'\nTotal params: {sum(v.numel() for v in sd.values())/1e6:.1f}M')
print(f'File size: {os.path.getsize(ckpt)/1e9:.2f}GB')

dim = sd.get('tok_embed.weight', sd.get('wte.weight', None))
if dim is not None:
    print(f'\nEmbedding: {dim.shape} → vocab={dim.shape[0]}, dim={dim.shape[1]}')

# Check for tie_weights
has_lm_head = any('lm_head' in k for k in sd)
print(f'Has lm_head: {has_lm_head}')

# Check attention structure
attn_keys = [k for k in sd if 'attn' in k.lower() and 'weight' in k]
print(f'\nAttention weights:')
for k in attn_keys[:10]:
    print(f'  {k}: {tuple(sd[k].shape)}')
