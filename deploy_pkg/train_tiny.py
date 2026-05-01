"""
train_tiny.py — 75M ATTH 串行训练: softmax → s^τ

数据: TinyStories (hf-mirror.com), char-level encode
模型: ATTH 768d×10L×12H ≈ 73M
训练: ctx=512, BF16 Amp, Cosine + warmup

用法:
    python3 -u train_tiny.py --epochs 30
"""
import sys, os, time, math, json, argparse, random

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TOKENIZERS_PARALLELISM'] = 'false'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch, torch.nn.functional as F
from model_tiny import build_model, make_optimizer, count_params, get_tau_values, get_avg_tau

RESULTS_DIR = '/root/epx/tiny_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
STATUS_FILE = '/root/epx/status_tiny.txt'

CTX = 512
RESERVED_CHARS = 256
MAX_SEQ = 2048


def load_tiny_texts(max_chars=15_000_000):
    from datasets import load_dataset
    print('Loading TinyStories from hf-mirror...')
    try:
        ds = load_dataset('roneneldan/TinyStories', split='train')
    except Exception:
        print('TinyStories 失败, 尝试 tiny-textbooks...')
        ds = load_dataset('nampdn-ai/tiny-textbooks', split='train')
    text = ''
    for i, ex in enumerate(ds):
        t = ex.get('text', ex.get('content', ''))
        text += t + '\n'
        if len(text) >= max_chars:
            break
        if (i + 1) % 50000 == 0:
            print(f'  loaded {i+1} stories, {len(text)//1000}k chars')
    print(f'Corpus: {len(text)//1000}k chars, {len(ds)} stories sampled')
    return text


def encode_text(text):
    chars = sorted(set(text))
    reserved = RESERVED_CHARS
    if len(chars) > reserved - 2:
        chars = chars[:reserved - 2]
    stoi = {ch: i + 1 for i, ch in enumerate(chars)}
    stoi['<unk>'] = len(stoi) + 1
    vocab = len(stoi) + 1
    data = torch.tensor([stoi.get(ch, stoi['<unk>']) for ch in text], dtype=torch.long)
    print(f'Vocab={vocab} chars, Tokens={len(data)//1000}k')
    return data, vocab


def get_batch(data, bs, ctx, rng):
    ix = [rng.randint(0, len(data) - ctx - 1) for _ in range(bs)]
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    return x.cuda(), y.cuda()


@torch.no_grad()
def eval_ppl(model, data, vocab, ctx, n_batches=10):
    model.eval()
    rng = random.Random(999)
    total_loss = 0.0
    n = 0
    for _ in range(n_batches):
        x, y = get_batch(data, min(16, len(data)//ctx//4), ctx, rng)
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
        total_loss += loss.item()
        n += 1
    model.train()
    return math.exp(total_loss / max(n, 1))


def write_status(msg):
    with open(STATUS_FILE, 'w') as f:
        f.write(msg)


def train_one(model, norm, data_train, data_val, vocab, args):
    tag = 's^τ' if norm == 'learned' else 'softmax'
    print(f'\n{"="*55}\n  TRAIN: {tag} ({count_params(model)/1e6:.0f}M params)\n{"="*55}')

    model.cuda()
    opt = make_optimizer(model, norm=norm, lr=args.lr)
    total_steps = args.epochs * args.steps_per_epoch
    warmup = min(500, total_steps // 10)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr, total_steps=total_steps,
        pct_start=warmup / max(total_steps, 1), anneal_strategy='cos')
    scaler = torch.amp.GradScaler('cuda')
    rng = random.Random(42)

    t0 = time.time()
    step = 0
    tau_log = []
    best_ppl = 1e9

    for ep in range(args.epochs):
        for _ in range(args.steps_per_epoch):
            x, y = get_batch(data_train, args.bs, CTX, rng)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(x, return_logits=True)
                loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            step += 1

        ppl = eval_ppl(model, data_val, vocab, CTX, n_batches=8)
        elapsed = time.time() - t0
        st_per_s = step / max(elapsed, 0.1)

        tau_str = ''
        if norm == 'learned':
            avg_tau = get_avg_tau(model)
            tau_log.append({'ep': ep + 1, 'tau': round(avg_tau, 4), 'ppl': round(ppl, 2)})
            tau_str = f'  τ_avg={avg_tau:.3f}'

        print(f'  ep {ep+1:>4d}/{args.epochs}  loss={loss.item():.4f}  '
              f'ppl={ppl:.2f}{tau_str}  {st_per_s:.1f}st/s  {elapsed:.0f}s')

        status_lines = [
            f'PHASE:{tag}',
            f'GPU:{torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"}',
            f'MODEL:{tag} ep{ep+1}/{args.epochs} ppl={ppl:.2f}{tau_str} st/s={st_per_s:.1f}',
            f'MEM:{torch.cuda.max_memory_allocated()/1e9:.1f}GB',
        ]
        write_status('\n'.join(status_lines))

        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(),
                       os.path.join(RESULTS_DIR, f'model_{tag}_best.pt'))

    total_time = time.time() - t0
    print(f'  DONE {tag}: {total_time:.0f}s ({total_time/60:.1f}min) best_ppl={best_ppl:.2f}')

    write_status(f'PHASE:{tag}\nGPU:{torch.cuda.get_device_name(0)}\nDONE:{tag} ppl={best_ppl:.2f}\nTIME:{total_time:.0f}s')

    result = {
        'norm': norm, 'epochs': args.epochs, 'params': count_params(model),
        'best_ppl': round(best_ppl, 3), 'time_s': round(total_time, 1),
        'tau_log': tau_log,
        'final_tau_layers': get_tau_values(model) if norm == 'learned' else {},
    }
    with open(os.path.join(RESULTS_DIR, f'result_{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--steps_per_epoch', type=int, default=200)
    parser.add_argument('--max_chars', type=int, default=15_000_000)
    args = parser.parse_args()

    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}')

    text = load_tiny_texts(args.max_chars)
    data, vocab = encode_text(text)
    n_val = min(len(data) // 10, 500_000)
    data_train, data_val = data[:-n_val], data[-n_val:]

    dim, n_layers, n_heads = 768, 10, 12

    results = {}

    for norm in ['softmax', 'learned']:
        model = build_model(vocab, norm=norm, max_seq=MAX_SEQ,
                            dim=dim, n_layers=n_layers, n_heads=n_heads)
        n_params = count_params(model)
        print(f'\nModel ({norm}): {n_params/1e6:.1f}M params, '
              f'{n_layers}L×{dim}d×{n_heads}H')

        r = train_one(model, norm, data_train, data_val, vocab, args)
        results[norm] = r

        del model
        torch.cuda.empty_cache()

    with open(os.path.join(RESULTS_DIR, 'results_all.json'), 'w') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f'\n{"="*55}')
    for norm, r in results.items():
        tag = 's^τ' if norm == 'learned' else 'softmax'
        print(f'  {tag:<10s} best_ppl={r["best_ppl"]:.2f}  {r["time_s"]:.0f}s')
    print(f'{"="*55}\nDONE — results in {RESULTS_DIR}/\n')


if __name__ == '__main__':
    main()
