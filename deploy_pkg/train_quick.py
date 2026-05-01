"""
train_quick.py — 合成数据 + 80M ATTH 训练

数据: 合成随机文本 (零依赖, 零下载, 8M chars)
模型: ATTH 768d x 11L x 12H = 80M
流程: softmax (对照) -> s^tau+fused
"""
import sys, os, time, math, json, random, gc, io
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from model_tiny import build_model, make_optimizer, count_params, get_tau_values, get_avg_tau

RESULTS_DIR = '/root/epx/tiny_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
STATUS_FILE = '/root/epx/status_tiny.txt'
CTX = 512
EPOCHS = 15
STEPS_PER_EP = 100
BS = 16
LR = 6e-4
MAX_CHARS = 8_000_000


def log(msg):
    print(msg, flush=True)


def load_data(max_chars=MAX_CHARS):
    """Download smoltalk-chinese parquet files via modelscope hub, read with pyarrow."""
    import pyarrow.parquet as pq
    log('Installing deps...')
    os.system('/root/miniconda3/bin/pip install pyarrow pandas modelscope -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null')
    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import dataset_file_download
    log('Fetching dataset file list from ModelScope...')
    api = HubApi()
    files = api.get_dataset_files('thomas/smoltalk-chinese')
    fnames = [f['Path'] for f in files if f['Path'].endswith('.parquet')]
    log(f'Found {len(fnames)} parquet files')
    text = ''
    cnt = 0
    for fn in fnames:
        local_path = dataset_file_download('thomas/smoltalk-chinese', fn, revision='master')
        log(f'  Cached: {fn.split(chr(47))[-1]}')
        df = pq.read_table(local_path).to_pandas()
        for _, row in df.iterrows():
            msgs = row.get('messages') or row.get('conversations')
            if msgs is not None:
                if isinstance(msgs, str):
                    msgs = json.loads(msgs)
                if hasattr(msgs, 'tolist'):
                    msgs = msgs.tolist()
                if isinstance(msgs, list):
                    for m in msgs:
                        if isinstance(m, dict):
                            c = m.get('content') or m.get('value') or ''
                            if c:
                                text += str(c) + '\n'
            else:
                for k in ['text', 'content', 'instruction', 'output']:
                    v = row.get(k)
                    if v:
                        text += str(v) + '\n'
            cnt += 1
            if len(text) >= max_chars:
                break
        log(f'  {fn.split(chr(47))[-1]}: {len(df)} rows, corpus {len(text)//1000}k chars')
        if len(text) >= max_chars:
            break
    log(f'Corpus: {len(text)//1000}k chars, {cnt} entries sampled')
    return text


def encode(text):
    chars = sorted(set(text))
    n = min(len(chars), 254)
    chars = chars[:n]
    stoi = {c: i + 1 for i, c in enumerate(chars)}
    stoi['<unk>'] = len(stoi) + 1
    vocab = len(stoi) + 1
    data = torch.tensor([stoi.get(c, stoi['<unk>']) for c in text], dtype=torch.long)
    log(f'Vocab={vocab} chars, Tokens={len(data)//1000}k')
    return data, vocab


def get_batch(data, bs, ctx, rng):
    ix = [rng.randint(0, len(data) - ctx - 1) for _ in range(bs)]
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    return x.cuda(), y.cuda()


@torch.no_grad()
def eval_ppl(model, data, vocab, ctx, n_batches=8):
    model.eval()
    rng = random.Random(999)
    tl, n = 0.0, 0
    for _ in range(n_batches):
        x, y = get_batch(data, min(8, len(data)//ctx//4), ctx, rng)
        tl += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
        n += 1
    model.train()
    return math.exp(tl / max(n, 1))


def write_status(**kw):
    with open(STATUS_FILE, 'w') as f:
        f.write('\n'.join(f'{k}:{v}' for k, v in kw.items()))


def train_one(norm, data_train, data_val, vocab):
    tag = 's^tau' if norm == 'learned' else 'softmax'
    log(f'\n{"="*60}\n  TRAIN: {tag}\n{"="*60}')

    model = build_model(vocab, norm=norm, dim=768, n_layers=11, n_heads=12)
    n_p = count_params(model)
    log(f'  Params: {n_p/1e6:.1f}M')

    model.cuda()
    opt = make_optimizer(model, norm=norm, lr=LR)
    total = EPOCHS * STEPS_PER_EP
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=LR, total_steps=total, pct_start=0.05)
    scaler = torch.amp.GradScaler('cuda')
    rng = random.Random(42)
    t0, step, best_ppl, tau_log = time.time(), 0, 1e9, []

    for ep in range(EPOCHS):
        for _ in range(STEPS_PER_EP):
            x, y = get_batch(data_train, BS, CTX, rng)
            opt.zero_grad()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                loss = F.cross_entropy(model(x, return_logits=True).reshape(-1, vocab), y.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update(); sched.step(); step += 1

        ppl = eval_ppl(model, data_val, vocab, CTX)
        elapsed = time.time() - t0
        sps = step / max(elapsed, 0.1)
        mem = torch.cuda.max_memory_allocated() / 1e9
        tau_str = ''
        if norm == 'learned':
            avg_tau = get_avg_tau(model)
            tau_log.append({'ep': ep + 1, 'tau': round(avg_tau, 4), 'ppl': round(ppl, 2)})
            tau_str = f'  tau={avg_tau:.3f}'
        log(f'  ep{ep+1:>3d}/{EPOCHS}  loss={loss.item():.4f}  ppl={ppl:.2f}{tau_str}  {sps:.1f}st/s  mem={mem:.1f}G')
        write_status(PHASE=tag, GPU=torch.cuda.get_device_name(0),
                     EPOCH=f'{ep+1}/{EPOCHS}', PPL=f'{ppl:.2f}', TAU=tau_str.strip(),
                     SPEED=f'{sps:.0f}st/s', MEM=f'{mem:.1f}G',
                     ELAPSED=f'{elapsed:.0f}s')
        if ppl < best_ppl:
            best_ppl = ppl
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, f'model_{tag}_best.pt'))

    total_time = time.time() - t0
    log(f'  DONE {tag}: {total_time:.0f}s ({total_time/60:.1f}min) best_ppl={best_ppl:.2f}')
    write_status(PHASE='DONE', TAG=tag, BEST_PPL=f'{best_ppl:.2f}', TIME=f'{total_time:.0f}s')
    result = dict(norm=norm, params=n_p, best_ppl=round(best_ppl, 3),
                  time_s=round(total_time, 1), tau_log=tau_log)
    with open(os.path.join(RESULTS_DIR, f'result_{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


text = load_data()
data, vocab = encode(text)
n_val = min(len(data)//10, 500_000)
data_train, data_val = data[:-n_val], data[-n_val:]

results = {}
for norm in ['softmax', 'learned']:
    r = train_one(norm, data_train=data_train, data_val=data_val, vocab=vocab)
    results[norm] = r
    del r; torch.cuda.empty_cache(); gc.collect()

with open(os.path.join(RESULTS_DIR, 'results_all.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log(f'\n{"="*60}')
for norm, r in results.items():
    t = {'learned': 's^tau', 'softmax': 'softmax'}[norm]
    log(f'  {t:<10s} best_ppl={r["best_ppl"]:.2f}  {r["time_s"]:.0f}s')
log(f'{"="*60}\nALL DONE')
