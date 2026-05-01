"""
train_200m.py — 200M ATTH + Qwen2 分词 + 全量中文对话

模型: ATTH 896d x 12L x 14H ≈ 200M (tie_weights)
数据: smoltalk-chinese 全量, Qwen2.5 tokenizer (152k vocab, 原生中文)
优化: 分块编码, BF16 autocast, tie_weights
流程: softmax (对照) → s^tau fused v4
"""
import sys, os, time, math, json, random, gc
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch, torch.nn.functional as F
from model_tiny import build_model, make_optimizer, count_params, get_avg_tau

RESULTS_DIR = '/root/epx/tiny_results'
os.makedirs(RESULTS_DIR, exist_ok=True)
STATUS_FILE = '/root/epx/status_tiny.txt'
CTX = 1024
EPOCHS = 5
STEPS_PER_EP = 500
BS = 8
LR = 3e-4
MAX_TOKENS = 50_000_000
TOKENIZER_PATH = 'Qwen/Qwen2.5-0.5B'

def log(msg):
    print(msg, flush=True)


def load_and_tokenize():
    """Stream-tokenize smoltalk-chinese with Qwen2 tiktoken tokenizer (fast, native Chinese)."""
    os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
    log('Installing deps...')
    os.system(f'/root/miniconda3/bin/pip install pyarrow pandas transformers modelscope -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>/dev/null')
    import pyarrow.parquet as pq
    from modelscope.hub.api import HubApi
    from modelscope.hub.file_download import dataset_file_download
    from transformers import AutoTokenizer

    log(f'Loading Qwen2 tokenizer from {TOKENIZER_PATH}...')
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    vocab_size = len(tok)  # use len() to include all special tokens
    log(f'Vocab: {vocab_size} (tokenizer reports {tok.vocab_size})')

    log('Fetching dataset file list from ModelScope...')
    api = HubApi()
    files = api.get_dataset_files('thomas/smoltalk-chinese')
    fnames = [f['Path'] for f in files if f['Path'].endswith('.parquet')]
    log(f'Found {len(fnames)} parquet files')

    all_tokens = []
    total_rows = 0
    CHUNK_SIZE = 10000

    for fn in fnames:
        local_path = dataset_file_download('thomas/smoltalk-chinese', fn, revision='master')
        df = pq.read_table(local_path).to_pandas()
        file_text = ''
        for _, row in df.iterrows():
            msgs = row.get('messages') or row.get('conversations')
            if msgs is not None:
                if isinstance(msgs, str):
                    msgs = json.loads(msgs)
                if hasattr(msgs, 'tolist'):
                    msgs = msgs.tolist()
                if isinstance(msgs, list):
                    for m in msgs:
                        c = m.get('content') or m.get('value') or '' if isinstance(m, dict) else str(m)
                        if c:
                            file_text += str(c) + '\n'
            else:
                for k in ['text', 'content', 'instruction', 'output']:
                    v = row.get(k)
                    if v:
                        file_text += str(v) + '\n'
            total_rows += 1

        try:
            log(f'  Encoding {fn.split(chr(47))[-1]} ({len(file_text)//1000}k chars) (tiktoken fast)...')
            chunks = [file_text[i:i+CHUNK_SIZE] for i in range(0, len(file_text), CHUNK_SIZE)]
            file_tokens = []
            for c_idx, c in enumerate(chunks):
                if c_idx % 50 == 0:
                    log(f'    chunk {c_idx+1}/{len(chunks)}...')
                file_tokens.extend(tok.encode(c, add_special_tokens=False))
            all_tokens.extend(file_tokens)
            del file_text, chunks, file_tokens
            gc.collect()
            log(f'  Total tokens: {len(all_tokens)//1000}k')
        except Exception as e:
            log(f'  SKIPPED (error: {e})')
            del file_text
            continue

        if len(all_tokens) >= MAX_TOKENS:
            all_tokens = all_tokens[:MAX_TOKENS]
            break

    log(f'Total: {total_rows} entries, {len(all_tokens)//1000}k tokens')
    data = torch.tensor(all_tokens, dtype=torch.long)
    del all_tokens; gc.collect()
    log(f'Data tensor: {len(data)//1000}k tokens')
    return data, vocab_size


def get_batch(data, bs, ctx, rng):
    ix = [rng.randint(0, len(data) - ctx - 1) for _ in range(bs)]
    x = torch.stack([data[i:i + ctx] for i in ix])
    y = torch.stack([data[i + 1:i + ctx + 1] for i in ix])
    return x.cuda(), y.cuda()


@torch.no_grad()
def eval_ppl(model, data, vocab, ctx, n_batches=4):
    model.eval()
    rng = random.Random(999)
    tl, n = 0.0, 0
    for _ in range(n_batches):
        x, y = get_batch(data, min(4, len(data)//ctx//4), ctx, rng)
        tl += F.cross_entropy(model(x).reshape(-1, vocab), y.reshape(-1)).item()
        n += 1
    model.train()
    return math.exp(tl / max(n, 1))


def write_status(**kw):
    with open(STATUS_FILE, 'w') as f:
        f.write('\n'.join(f'{k}:{v}' for k, v in kw.items()))


def train_one(norm, data_train, data_val, vocab, load_path=None):
    tag = 's^tau' if norm == 'learned' else 'softmax'
    log(f'\n{"="*60}\n  TRAIN: {tag}\n{"="*60}')

    model = build_model(vocab, norm=norm, dim=896, n_layers=12, n_heads=14,
                        max_seq=2048, tie_weights=True)
    n_p = count_params(model)
    log(f'  Params: {n_p/1e6:.1f}M')

    model.cuda()

    if load_path:
        state = torch.load(load_path, map_location='cuda', weights_only=True)
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            log(f'  Loaded weights, missing keys (expected for tau head): {len(missing)}')
        if unexpected:
            log(f'  Unexpected keys: {len(unexpected)}')

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
            save_name = f'model_{tag}_best.pt'
            torch.save(model.state_dict(), os.path.join(RESULTS_DIR, save_name))
            log(f'  -> saved {save_name}')

    total_time = time.time() - t0
    log(f'  DONE {tag}: {total_time:.0f}s ({total_time/60:.1f}min) best_ppl={best_ppl:.2f}')
    write_status(PHASE='DONE', TAG=tag, BEST_PPL=f'{best_ppl:.2f}', TIME=f'{total_time:.0f}s')
    result = dict(norm=norm, params=n_p, best_ppl=round(best_ppl, 3),
                  time_s=round(total_time, 1), tau_log=tau_log)
    with open(os.path.join(RESULTS_DIR, f'result_{tag}.json'), 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    return result


data, vocab = load_and_tokenize()
n_val = min(len(data)//10, 1_000_000)
data_train, data_val = data[:-n_val], data[-n_val:]
log(f'Train: {len(data_train)//1000}k tokens  Val: {len(data_val)//1000}k tokens')

softmax_done = os.path.exists(os.path.join(RESULTS_DIR, 'model_softmax_best.pt'))

results = {}
if not softmax_done:
    r = train_one('softmax', data_train=data_train, data_val=data_val, vocab=vocab)
    results['softmax'] = r
    del r; torch.cuda.empty_cache(); gc.collect()
else:
    log('Softmax already done, skipping.')
    best_ppl = 0.0
    if os.path.exists(os.path.join(RESULTS_DIR, 'result_softmax.json')):
        with open(os.path.join(RESULTS_DIR, 'result_softmax.json')) as f:
            best_ppl = json.load(f).get('best_ppl', 0.0)
    results['softmax'] = dict(norm='softmax', best_ppl=best_ppl, skipped=True)

r = train_one('learned', data_train=data_train, data_val=data_val, vocab=vocab,
              load_path=os.path.join(RESULTS_DIR, 'model_softmax_best.pt'))
results['learned'] = r
del r; torch.cuda.empty_cache(); gc.collect()

with open(os.path.join(RESULTS_DIR, 'results_all.json'), 'w') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
log(f'\n{"="*60}')
for norm, r in results.items():
    t = {'learned': 's^tau', 'softmax': 'softmax'}[norm]
    log(f'  {t:<10s} best_ppl={r["best_ppl"]:.2f}  {r["time_s"]:.0f}s')
log(f'{"="*60}\nALL DONE')
