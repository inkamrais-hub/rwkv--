"""
Qwen3 + Qwen3.5 (多模态) s^τ 实验 — 中文生成 + 图文注意力分析

用法:
    python scripts/_qwen_stau.py                    # Qwen3-0.6B 全量中文实验
    python scripts/_qwen_stau.py --qwen35           # Qwen3.5-4B 多模态实验
    python scripts/_qwen_stau.py --quick            # 快速模式 (少生成)
"""
import sys, os, math, gc
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import torch
import torch.nn.functional as F

EPS = 1e-8
_current_tau = 1.0

# ===== s^τ softmax 替换函数 =====
def s_tau_softmax(x, dim=-1, dtype=None):
    if x.dim() == 4:
        clamped = x.clamp(min=EPS)
        powered = clamped.pow(_current_tau)
        out = powered / (powered.sum(dim=dim, keepdim=True) + EPS)
        if dtype is not None:
            out = out.to(dtype)
        return out
    return torch._orig_softmax(x, dim=dim, dtype=dtype)

def set_tau(val):
    global _current_tau
    _current_tau = val

def patch_model():
    """替换 torch.softmax → s^tau"""
    torch._orig_softmax = torch.softmax
    torch.softmax = s_tau_softmax

def unpatch_model():
    """恢复原始 softmax"""
    if hasattr(torch, '_orig_softmax'):
        torch.softmax = torch._orig_softmax


# ====================================================================
# Part A: Qwen3 中文实验
# ====================================================================
def run_qwen3_experiment():
    print('='*60)
    print('Qwen3-0.6B 中文 s^τ 实验 (魔搭)')
    print('='*60)

    from modelscope import AutoModelForCausalLM, AutoTokenizer

    print('[1] 加载 Qwen3-0.6B (魔搭)...')
    model_name = 'Qwen/Qwen3-0.6B'
    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float32, trust_remote_code=True,
        attn_implementation='eager'
    )
    model.eval()
    n = sum(p.numel() for p in model.parameters())
    print(f'    词表={len(tok)}, 参数量={n/1e6:.1f}M')

    print('[2] 打 patch: softmax → s^tau...')
    patch_model()

    print('[3] 验证等价性: τ=1.0 应与原始 softmax 一致\n')
    @torch.no_grad()
    def generate(prompt, max_new=30, temp=0.7, top_k=40):
        # base 模型: 直接 encode prompt
        input_ids = tok.encode(prompt, return_tensors='pt')
        generated = input_ids.clone()
        for _ in range(max_new):
            x = generated[:, -2048:]
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

    # 中文 prompts
    prompts = [
        "人生的意义是什么？",
        "人工智能的未来发展方向",
        "用一句话描述量子计算",
    ]

    for p in prompts:
        set_tau(1.0)
        t = generate(p, max_new=30, temp=0.7)
        print(f'  问题: "{p}"')
        print(f'  τ=1.0 → {t}\n')

    print('[4] τ 扫描 — 看中文生成变化\n')
    tau_values = [1.0, 2.0, 3.5, 5.0, 10.0]

    for p in prompts[:2]:
        print(f'  ─── "{p}" ───')
        for tau in tau_values:
            set_tau(tau)
            t = generate(p, max_new=40, temp=0.7)
            print(f'  τ={tau:>5.1f}  |  {t}')
        print()

    # 注意力熵分析
    print('[5] 注意力熵分析 (中文输入)\n')
    @torch.no_grad()
    def analyze_entropy(prompt, tau_val):
        set_tau(tau_val)
        input_ids = tok.encode(prompt, return_tensors='pt')[:, :64]
        model.config.output_attentions = True
        outputs = model(input_ids, output_attentions=True)
        model.config.output_attentions = False
        if outputs.attentions is None or len(outputs.attentions) == 0:
            return
        attentions = outputs.attentions
        print(f'  τ={tau_val:>5.1f}:')
        for i, attn in enumerate(attentions[:3]):
            T = attn.size(-1)
            H = -(attn * torch.log(attn.clamp(min=1e-12))).sum(dim=-1)
            H_max = math.log(T)
            H_norm = H / H_max
            max_w = attn.max(dim=-1).values
            print(f'    layer {i}: entropy={H_norm.mean().item():.4f}  max_attn={max_w.mean().item():.4f}')

    analyze_entropy("人生的意义是什么？", 1.0)
    analyze_entropy("人生的意义是什么？", 3.5)
    analyze_entropy("人生的意义是什么？", 10.0)

    unpatch_model()
    del model, tok
    gc.collect()
    return True


# ====================================================================
# Part B: Qwen3.5 多模态实验
# ====================================================================
def run_qwen35_multimodal():
    print('='*60)
    print('Qwen3.5-4B 多模态 s^τ 实验')
    print('='*60)
    print()
    print('  Qwen3.5 使用混合注意力架构:')
    print('  - Gated DeltaNet (3/4 层): 线性注意力, 无 softmax')
    print('  - Gated Attention (1/4 层): 标准 softmax → 可 patch')
    print()
    print('  多模态: 图像 → 视觉 token → 与文本 token 一起进入 transformer')
    print('  → s^τ 会同时影响文本和图像的注意力!\n')

    from transformers import AutoModelForCausalLM, AutoProcessor

    print('[1] 加载 Qwen3.5-4B...')
    model_name = 'Qwen/Qwen3.5-4B-Instruct'
    try:
        processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        model.eval()
    except Exception as e:
        print(f'  ⚠️  加载失败: {e}')
        print('  可能原因: transformers 版本与 Qwen3.5 不兼容')
        print('  需要 transformers>=5.6.0 (或对应 Qwen3.5 版本)')
        return False

    n = sum(p.numel() for p in model.parameters())
    print(f'  参数量={n/1e6:.1f}M')

    print('[2] 打 patch: softmax → s^tau (仅影响 Gated Attention 层)...')
    patch_model()

    print('[3] 中文文本生成 (纯文本, 无需图像)\n')
    @torch.no_grad()
    def generate_text(prompt, max_new=30, temp=0.7):
        messages = [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], return_tensors='pt', padding=True, truncation=True)
        input_ids = inputs['input_ids']
        generated = input_ids.clone()
        for _ in range(max_new):
            x = generated[:, -2048:]
            logits = model(x).logits[:, -1, :]
            logits = logits / temp
            vals, _ = torch.topk(logits, 40)
            logits[logits < vals[:, -1:]] = float('-inf')
            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            generated = torch.cat([generated, next_id], dim=1)
            if next_id.item() in [processor.tokenizer.eos_token_id, 151645]:
                break
        return processor.tokenizer.decode(generated[0].tolist(), skip_special_tokens=True)

    prompts_cn = ["人生的意义是什么？", "人工智能的未来"]
    tau_values = [1.0, 2.0, 3.5, 5.0]

    for p in prompts_cn:
        print(f'  ─── "{p}" ───')
        for tau in tau_values:
            set_tau(tau)
            t = generate_text(p, max_new=30, temp=0.7)
            print(f'  τ={tau:>5.1f}  |  {t}')
        print()

    print('[4] 多模态: 图像+文本推理 (需要提供图像 URL)\n')
    print('  由于本地无 GPU, 图像推理会很慢...')
    print('  但概念上 s^τ 会同时影响:')
    print('    - 文本 token 之间的注意力')
    print('    - 图像 token 之间的注意力')
    print('    - 文本↔图像 跨模态注意力')
    print()
    print('  → τ 越高, 模型越聚焦于图像中最显著的区域')
    print('  → τ 越低, 模型越均匀地分布注意力到整个图像')
    print()
    print('  等有 GPU 资源时可以实测多模态效果!')

    unpatch_model()
    del model, processor
    gc.collect()
    return True


# ====================================================================
# Main
# ====================================================================
if __name__ == '__main__':
    quick = '--quick' in sys.argv
    do_qwen35 = '--qwen35' in sys.argv

    if do_qwen35:
        run_qwen35_multimodal()
    else:
        ok = run_qwen3_experiment()
        if ok:
            print('\n  ✅ Qwen3 实验完成!')
            print('  💡 加上 --qwen35 参数即可跑 Qwen3.5 多模态:')
            print('     python scripts/_qwen_stau.py --qwen35')
        else:
            print('\n  ❌ Qwen3 实验失败')

    print('\nDone!')
