"""
█ train_gpt2_real.py — GPT-2 124M 训练 (真实 tokenizer + 数据)

数据:  HuggingFace datasets → wikitext-103
分词:  GPT-2 tokenizer (50257 词表)
架构:  GPT-2 124M + RoPE + s^τ / softmax
训练:  BF16 Amp, 512→1024 curriculum
Eval:  加载 HF GPT-2 pretrained 权重对比

用法:
    /root/miniconda3/bin/python3 -u train_gpt2_real.py --norm learned
"""
import os, sys, math, time, json, argparse, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

os.environ['TOKENIZERS_PARALLELISM'] = 'false'
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

RESULTS_DIR = '/root/epx/gpt2_results'
os.makedirs(RESULTS_DIR, exist_ok=True)

# ─── 参数 ───
VOCAB_SIZE = 50257
N_LAYER = 12
N_EMBD = 768
N_HEAD = 12
MAX_CTX = 1024
EVAL_CTXS = [128, 512, 1024, 2048]

# ─── RoPE ───
def precompute_freqs(dim, max_pos, theta=10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_pos).float()
    return torch.outer(t, freqs)

def apply_rotary(x, cos, sin):
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    cos = cos[:x.shape[-2]].reshape(1, 1, -1, half)
    sin = sin[:x.shape[-2]].reshape(1, 1, -1, half)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)

# ─── Attention ───
class STauAttention(nn.Module):
    def __init__(self, norm_type='learned'):
        super().__init__()
        self.norm_type = norm_type
        if norm_type in ('learned', 'tempered'):
            self.per_head_log_tau = nn.Parameter(torch.zeros(N_HEAD))
        self.c_attn = nn.Linear(N_EMBD, 3 * N_EMBD, bias=False)
        self.c_proj = nn.Linear(N_EMBD, N_EMBD, bias=False)

    def forward(self, x, freqs_cos, freqs_sin, attn_mask=None):
        B, T, C = x.shape
        qkv = self.c_attn(x).reshape(B, T, 3, N_HEAD, C // N_HEAD)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]

        q = apply_rotary(q.transpose(1, 2), freqs_cos, freqs_sin)
        k = apply_rotary(k.transpose(1, 2), freqs_cos, freqs_sin)
        v = v.transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * (C // N_HEAD) ** -0.5
        if attn_mask is not None:
            scores = scores + attn_mask[:, :, :T, :T]

        if self.norm_type == 'softmax':
            attn_weights = F.softmax(scores, dim=-1)
        elif self.norm_type == 'learned':
            tau = F.softplus(self.per_head_log_tau) + 1.0
            try:
                from attention_mechanisms.s_tau_fused import s_tau_norm
                attn_weights = s_tau_norm(scores, tau)
            except:
                clamped = scores.clamp(min=1e-8)
                tau_b = tau.view(1, -1, 1, 1)
                powered = clamped.pow(tau_b)
                attn_weights = powered / (powered.sum(dim=-1, keepdim=True) + 1e-8)

        return self.c_proj(attn_weights.to(v.dtype) @ v).transpose(1, 2).reshape(B, T, C)

class Block(nn.Module):
    def __init__(self, norm_type):
        super().__init__()
        self.ln1 = nn.LayerNorm(N_EMBD)
        self.attn = STauAttention(norm_type)
        self.ln2 = nn.LayerNorm(N_EMBD)
        self.mlp = nn.Sequential(
            nn.Linear(N_EMBD, 4 * N_EMBD, bias=False),
            nn.GELU(),
            nn.Linear(4 * N_EMBD, N_EMBD, bias=False),
        )

    def forward(self, x, fc, fs, mask=None):
        x = x + self.attn(self.ln1(x), fc, fs, mask)
        x = x + self.mlp(self.ln2(x))
        return x

class GPT2(nn.Module):
    def __init__(self, norm_type='learned'):
        super().__init__()
        self.wte = nn.Embedding(VOCAB_SIZE, N_EMBD)
        self.freqs_cos = None
        self.freqs_sin = None
        self.layers = nn.ModuleList([Block(norm_type) for _ in range(N_LAYER)])
        self.ln_f = nn.LayerNorm(N_EMBD)
        self.lm_head = nn.Linear(N_EMBD, VOCAB_SIZE, bias=False)
        self.wte.weight = self.lm_head.weight
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.zeros_(m.bias)
            torch.nn.init.ones_(m.weight)

    def forward(self, idx, targets=None, ctx_len=None):
        B, T = idx.shape
        if ctx_len is None: ctx_len = MAX_CTX
        if self.freqs_cos is None or self.freqs_cos.shape[0] < ctx_len:
            freqs = precompute_freqs(N_EMBD // N_HEAD, max(ctx_len, 8192))
            self.freqs_cos = freqs.cos().to(idx.device)
            self.freqs_sin = freqs.sin().to(idx.device)

        cos = self.freqs_cos[:ctx_len].to(idx.dtype)
        sin = self.freqs_sin[:ctx_len].to(idx.dtype)
        mask = torch.triu(torch.full((ctx_len, ctx_len), float('-inf'),
                           device=idx.device), diagonal=1).unsqueeze(0).unsqueeze(0)

        x = self.wte(idx)
        for layer in self.layers:
            x = layer(x, cos, sin, mask)
        logits = self.lm_head(self.ln_f(x))
        if targets is not None:
            loss = F.cross_entropy(logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1))
            return logits, loss
        return logits

# ─── 数据 ───
def load_tokenizer():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained('gpt2')
    tok.pad_token = tok.eos_token
    return tok

def load_dataset(tokenizer):
    from datasets import load_dataset
    print('Loading wikitext-103...')
    ds = load_dataset('wikitext', 'wikitext-103-raw-v1', split='train')
    val = load_dataset('wikitext', 'wikitext-103-raw-v1', split='validation')
    print(f'Train: {len(ds)} lines, Val: {len(val)} lines')
    return ds, val

def tokenize_fn(batch, tokenizer):
    texts = [t for t in batch['text'] if t.strip()]
    if not texts:
        return {'input_ids': [], 'attention_mask': []}
    enc = tokenizer(texts, truncation=True, max_length=MAX_CTX + 1, padding=False)
    return {'input_ids': enc['input_ids']}

class TokenizedDataset(torch.utils.data.Dataset):
    def __init__(self, hf_ds, tokenizer, max_samples=200000):
        print('Tokenizing...')
        self.tokens = []
        for i, example in enumerate(hf_ds):
            if max_samples and i >= max_samples:
                break
            text = example['text']
            if len(text.strip()) < 10:
                continue
            ids = tokenizer.encode(text, truncation=True, max_length=MAX_CTX + 1)
            self.tokens.append(torch.tensor(ids, dtype=torch.long))
            if (i + 1) % 10000 == 0:
                print(f'  tokenized {i+1}...')
        print(f'Tokenized {len(self.tokens)} sequences')

    def __len__(self):
        return len(self.tokens)

    def __getitem__(self, idx):
        ids = self.tokens[idx]
        if len(ids) > MAX_CTX + 1:
            start = random.randint(0, len(ids) - MAX_CTX - 1)
            ids = ids[start:start + MAX_CTX + 1]
        elif len(ids) < MAX_CTX + 1:
            ids = torch.cat([ids, torch.zeros(MAX_CTX + 1 - len(ids), dtype=torch.long)])
        return ids[:MAX_CTX], ids[1:MAX_CTX + 1]

# ─── 评估 ───
@torch.no_grad()
def eval_ppl(model, tokenizer, val_ds, ctx_len, num_batches=20):
    model.eval()
    losses = []
    for _ in range(num_batches):
        texts = [v['text'] for v in random.choices(val_ds, k=4) if v['text'].strip()]
        if not texts:
            continue
        enc = tokenizer(texts, truncation=True, max_length=ctx_len + 1,
                        padding='max_length', return_tensors='pt')
        x, y = enc['input_ids'][:, :ctx_len], enc['input_ids'][:, 1:ctx_len + 1]
        x, y = x.cuda(), y.cuda()
        _, loss = model(x, y, ctx_len=ctx_len)
        losses.append(loss.item())
    model.train()
    return math.exp(sum(losses) / (len(losses) or 1))

# ─── 对照 ───
@torch.no_grad()
def compare_hf_gpt2(tokenizer, val_ds, num_batches=10):
    from transformers import GPT2LMHeadModel
    print('Loading pretrained GPT-2...')
    hf = GPT2LMHeadModel.from_pretrained('gpt2').cuda().eval()
    losses = []
    for _ in range(num_batches):
        texts = [v['text'] for v in random.choices(val_ds, k=4) if v['text'].strip()]
        if not texts: continue
        enc = tokenizer(texts, truncation=True, max_length=MAX_CTX + 1,
                        padding='max_length', return_tensors='pt')
        x, y = enc['input_ids'][:, :MAX_CTX].cuda(), enc['input_ids'][:, 1:MAX_CTX + 1].cuda()
        out = hf(x, labels=y)
        losses.append(out.loss.item())
    ppl = math.exp(sum(losses) / len(losses))
    print(f'HF GPT-2 PPL@{MAX_CTX}: {ppl:.2f}')
    return ppl

# ─── 训练 ───
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--norm', type=str, default='learned')
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--ctx', type=int, default=1024)
    parser.add_argument('--steps', type=int, default=50000)
    parser.add_argument('--run_all', type=int, default=0)
    args = parser.parse_args()

    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.0f}GB')
    print(f'Norm: {args.norm}')

    # Install deps if needed
    import subprocess, importlib
    for pkg in ['transformers', 'datasets']:
        try:
            importlib.import_module(pkg)
        except:
            print(f'Installing {pkg}...')
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', pkg, '-q'])

    tokenizer = load_tokenizer()
    print(f'Tokenizer vocab: {tokenizer.vocab_size}')

    hf_ds, val_ds = load_dataset(tokenizer)
    train_ds = TokenizedDataset(hf_ds, tokenizer, max_samples=100000)
    loader = torch.utils.data.DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2)

    results = {}
    norms = ['learned', 'softmax'] if args.run_all else [args.norm]

    for norm in norms:
        print(f'\n{"="*50}')
        print(f'Training: {norm}')
        print(f'{"="*50}')

        model = GPT2(norm_type=norm).cuda()
        print(f'Params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M')

        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
        scaler = torch.amp.GradScaler('cuda')
        data_iter = iter(loader)

        tau_log = []
        t0 = time.time()
        curr_ctx = 512
        curriculum_step = 10000

        for step in range(args.steps):
            if step == curriculum_step:
                curr_ctx = args.ctx
                print(f'\n[step {step}] Curriculum: ctx={curr_ctx}')

            # Get batch
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.cuda(), y.cuda()

            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                _, loss = model(x, y, ctx_len=curr_ctx)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()

            if step % 100 == 0:
                speed = 100.0 / (time.time() - t0 + 1e-6) * 100
                t0 = time.time()
                tau_str = ''
                if norm == 'learned':
                    tau_vals = [F.softplus(l.attn.per_head_log_tau.detach()).mean().item() + 1.0
                                for l in model.layers]
                    avg_tau = sum(tau_vals) / len(tau_vals)
                    tau_str = f'  τ={avg_tau:.3f}'
                    tau_log.append({'step': step, 'tau': avg_tau, 'loss': loss.item()})
                print(f'  step {step:>6d}  loss={loss.item():.4f}{tau_str}  speed={speed:.0f}st/s')
                sys.stdout.flush()

            if step % 5000 == 0 and step > 0:
                print(f'\n  ── Eval @ {step} ──')
                for ec in [128, 512, args.ctx]:
                    ppl = eval_ppl(model, tokenizer, val_ds, ec, num_batches=5)
                    print(f'    ctx={ec:>4}: PPL={ppl:.2f}')

        total_time = time.time() - t0 + 1
        print(f'\nTraining done: {total_time:.0f}s ({total_time/60:.1f}min)')

        # Final eval
        print(f'\n  ── Final Eval ──')
        evals = {}
        for ec in EVAL_CTXS:
            ppl = eval_ppl(model, tokenizer, val_ds, ec, num_batches=5)
            evals[ec] = ppl
            print(f'    ctx={ec:>4}: PPL={ppl:.2f}')

        results[norm] = {
            'steps': args.steps, 'time_s': total_time,
            'final_loss': loss.item(), 'evals': evals, 'tau_trace': tau_log,
        }
        json.dump(results[norm], open(os.path.join(RESULTS_DIR, f'{norm}_real.json'), 'w'))

    # HF comparison
    print(f'\n{"="*50}\nHF GPT-2 Comparison\n{"="*50}')
    hf_ppl = compare_hf_gpt2(tokenizer, val_ds)
    results['hf_gpt2'] = hf_ppl
    json.dump(results, open(os.path.join(RESULTS_DIR, 'all_real.json'), 'w'))
    print(f'\nDONE — results in {RESULTS_DIR}/')

if __name__ == '__main__':
    main()
