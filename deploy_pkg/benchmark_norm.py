"""
█ benchmark_norm.py — 恒源云归一化方法全面对比测试
█
█ 阶段:
█   PHASE0: 速度基准（s^τ vs tempered vs softmax）
█   PHASE1: 训练对比（dh64+RoPE L=128, 3 norm × 4 seeds = 12模型，并行）
█   PHASE2: 梯度分析（追踪τ轨迹 + 梯度范数）
█
█ 用法: /usr/local/bin/python3 -u benchmark_norm.py
"""
import sys, os, time, json, random, math, warnings, multiprocessing as mp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

mp.set_start_method('spawn', force=True)

import torch, torch.nn.functional as F
import numpy as np
from attention_mechanisms.model import ATTHModel

SD = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SD, 'benchmark_results')
os.makedirs(RESULT_DIR, exist_ok=True)
LOG_DIR = os.path.join(SD, 'benchmark_logs')
os.makedirs(LOG_DIR, exist_ok=True)

# Load data from local file (pre-packaged)
DATA_PATH = os.path.join(SD, 'shakespeare.txt')
if not os.path.exists(DATA_PATH):
    import urllib.request
    print('Downloading shakespeare.txt...', flush=True)
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        DATA_PATH)
    print('Downloaded', flush=True)

text = open(DATA_PATH, encoding='utf-8').read()
chars = sorted(set(text))
V = len(chars)
cid = {ch: i for i, ch in enumerate(chars)}
data_t = torch.tensor([cid[ch] for ch in text], dtype=torch.long)
train_t, val_t = data_t[:int(len(data_t) * 0.9)], data_t[int(len(data_t) * 0.9):]
print(f'Vocab={V} Train={len(train_t)//1000}k Val={len(val_t)//1000}k', flush=True)

device = torch.device('cuda')

def get_batch(split, blk, rng):
    d = train_t if split == 'train' else val_t
    bs = 64 if blk <= 1024 else 32
    ix = [rng.randint(0, len(d) - blk - 1) for _ in range(bs)]
    return torch.stack([d[i:i+blk] for i in ix]), torch.stack([d[i+1:i+blk+1] for i in ix])

def get_tau(m):
    vs = []
    for layer in m.layers:
        if hasattr(layer.attn, 'per_head_log_tau') and layer.attn.per_head_log_tau is not None:
            offset = 0.0 if layer.attn.norm_type == 'tempered' else 1.0
            vs.append(F.softplus(layer.attn.per_head_log_tau.detach()) + offset)
    return torch.stack([v.mean() for v in vs]).mean().item() if vs else 0

def get_tau_per_layer(m):
    r = {}
    for i, layer in enumerate(m.layers):
        if hasattr(layer.attn, 'per_head_log_tau') and layer.attn.per_head_log_tau is not None:
            offset = 0.0 if layer.attn.norm_type == 'tempered' else 1.0
            t = F.softplus(layer.attn.per_head_log_tau.detach()) + offset
            r[f'L{i}'] = {'mean': t.mean().item(), 'std': t.std().item()}
    return r

@torch.no_grad()
def eval_model(m, lens, seed=999):
    rng = random.Random(seed)
    res = {}
    m.eval()
    for blk in lens:
        ls = []
        for _ in range(8):
            vx, vy = get_batch('val', blk, rng)
            vx, vy = vx.to(device), vy.to(device)
            ls.append(F.cross_entropy(m(vx).reshape(-1, V), vy.reshape(-1)).item())
        res[blk] = math.exp(np.mean(ls))
    return res

# ============================================================
# PHASE 0: Speed Benchmark
# ============================================================
def speed_benchmark():
    print('\n' + '=' * 60, flush=True)
    print('PHASE 0: Speed Benchmark', flush=True)
    print('=' * 60, flush=True)

    results = {}
    for norm_type in ['softmax', 'learned', 'tempered']:
        torch.cuda.reset_peak_memory_stats()
        m = ATTHModel(V, 256, 4, 4, 512, 0.1, 4.0, False,
                      attn_type='standard', norm_type=norm_type,
                      use_rope=True, use_pos_emb=False, angle_emb=False)
        m = m.to(device)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler()
        rng = random.Random(42)

        # Warmup
        for _ in range(10):
            tx, ty = get_batch('train', 128, rng)
            tx, ty = tx.to(device), ty.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                loss = F.cross_entropy(m(tx).reshape(-1, V), ty.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

        # Benchmark
        torch.cuda.synchronize()
        starts = time.time()
        n_steps = 100
        for _ in range(n_steps):
            tx, ty = get_batch('train', 128, rng)
            tx, ty = tx.to(device), ty.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                loss = F.cross_entropy(m(tx).reshape(-1, V), ty.reshape(-1))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        torch.cuda.synchronize()
        elapsed = time.time() - starts
        ms_per_step = elapsed / n_steps * 1000
        mem = torch.cuda.max_memory_allocated() / 1e6
        results[norm_type] = {'ms_per_step': round(ms_per_step, 2), 'mem_mb': round(mem, 0)}
        print(f'  {norm_type:<12s}: {ms_per_step:.2f} ms/step, {mem:.0f} MB', flush=True)
        del m, opt, scaler
        torch.cuda.empty_cache()

    if results:
        base = results.get('softmax', {}).get('ms_per_step', 1)
        for nt in results:
            ratio = results[nt]['ms_per_step'] / base if base else 0
            results[nt]['vs_softmax'] = round(ratio, 2)
            print(f'  {nt:<12s}: {ratio:.2f}× vs softmax', flush=True)

    with open(os.path.join(RESULT_DIR, 'phase0_speed.json'), 'w') as f:
        json.dump(results, f, indent=2)
    print(f'[SAVE] phase0_speed.json', flush=True)
    return results


# ============================================================
# PHASE 1: Training comparison (s^τ vs tempered vs softmax)
# ============================================================
def train_model(config):
    norm_type = config['norm']
    seed = config['seed']
    block_size = 128
    epochs = 200
    n_per_ep = 50
    tag = config['tag']
    log_path = os.path.join(LOG_DIR, tag + '.log')

    def log(msg):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
        print(msg, flush=True)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = ATTHModel(V, 256, 4, 4, 2048, 0.1, 4.0, False,
                  attn_type='standard', norm_type=norm_type,
                  use_rope=True, use_pos_emb=False, angle_emb=False)
    m = m.to(device)

    tau_ps = [p for n, p in m.named_parameters() if 'log_tau' in n]
    other_ps = [p for n, p in m.named_parameters() if 'log_tau' not in n]
    opt = torch.optim.AdamW([
        {'params': other_ps, 'lr': 1e-4, 'weight_decay': 0.01},
        {'params': tau_ps, 'lr': 1e-2, 'weight_decay': 0.0}
    ], betas=(0.9, 0.98))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')
    best_state = None
    st = time.time()
    tau_traj = []

    log(f'START {tag} | norm={norm_type} | seed={seed}')
    for ep in range(epochs):
        m.train()
        for _ in range(n_per_ep):
            tx, ty = get_batch('train', block_size, rng)
            tx, ty = tx.to(device), ty.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                loss = F.cross_entropy(m(tx).reshape(-1, V), ty.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sch.step()

        m.eval()
        with torch.no_grad():
            vx, vy = get_batch('val', block_size, rng)
            vx, vy = vx.to(device), vy.to(device)
            vl = F.cross_entropy(m(vx).reshape(-1, V), vy.reshape(-1)).item()
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}
        if (ep + 1) % 10 == 0:
            tv = get_tau(m)
            tau_traj.append({'ep': ep + 1, 'tau': tv, 'ppl': math.exp(vl)})
            log(f'  [{tag}] ep{ep+1:>4} ppl={math.exp(vl):.2f} best={math.exp(best_loss):.2f} tau={tv:.4f}')

    elapsed = time.time() - st
    torch.cuda.empty_cache()
    tau_f = get_tau(m)
    tau_layers = get_tau_per_layer(m)
    evals = eval_model(m, [128, 256, 512])

    result = {
        'tag': tag, 'seed': seed, 'norm': norm_type,
        'dh': 64, 'nh': 4, 'dim': 256, 'L': 128, 'rope': True,
        'tau_final': tau_f, 'tau_layers': tau_layers,
        'tau_trajectory': tau_traj,
        'ppl': math.exp(best_loss),
        'eval_128': evals.get(128, 0),
        'eval_256': evals.get(256, 0),
        'eval_512': evals.get(512, 0),
        'time': elapsed,
    }
    log(f'  DONE [{tag}] tau={tau_f:.4f} ppl={math.exp(best_loss):.2f} time={elapsed:.0f}s')
    return result


# ============================================================
# PHASE 2: Gradient analysis
# ============================================================
def grad_analysis(config):
    norm_type = config['norm']
    seed = config['seed']
    block_size = 128
    epochs = 100
    n_per_ep = 50
    tag = config['tag']
    log_path = os.path.join(LOG_DIR, tag + '.log')

    def log(msg):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
        print(msg, flush=True)

    torch.manual_seed(seed)
    rng = random.Random(seed)
    m = ATTHModel(V, 256, 4, 4, 2048, 0.1, 4.0, False,
                  attn_type='standard', norm_type=norm_type,
                  use_rope=True, use_pos_emb=False, angle_emb=False)
    m = m.to(device)

    tau_ps = [p for n, p in m.named_parameters() if 'log_tau' in n]
    other_ps = [p for n, p in m.named_parameters() if 'log_tau' not in n]
    opt = torch.optim.AdamW([
        {'params': other_ps, 'lr': 1e-4, 'weight_decay': 0.01},
        {'params': tau_ps, 'lr': 1e-2, 'weight_decay': 0.0}
    ], betas=(0.9, 0.98))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf')
    best_state = None
    st = time.time()
    grad_log = []

    log(f'START GRAD {tag}')
    for ep in range(epochs):
        m.train()
        for _ in range(n_per_ep):
            tx, ty = get_batch('train', block_size, rng)
            tx, ty = tx.to(device), ty.to(device)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=True):
                loss = F.cross_entropy(m(tx).reshape(-1, V), ty.reshape(-1))
            scaler.scale(loss).backward()
            scaler.unscale_(opt)

            tau_grad_norm = 0.0
            total_grad_norm = 0.0
            for n, p in m.named_parameters():
                if p.grad is not None:
                    gn = p.grad.norm().item()
                    total_grad_norm += gn ** 2
                    if 'log_tau' in n:
                        tau_grad_norm += gn ** 2
            total_grad_norm = math.sqrt(total_grad_norm)
            tau_grad_norm = math.sqrt(tau_grad_norm)

            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
        sch.step()

        m.eval()
        with torch.no_grad():
            vx, vy = get_batch('val', block_size, rng)
            vx, vy = vx.to(device), vy.to(device)
            vl = F.cross_entropy(m(vx).reshape(-1, V), vy.reshape(-1)).item()
        if vl < best_loss:
            best_loss = vl
            best_state = {k: v.cpu().clone() for k, v in m.state_dict().items()}

        if (ep + 1) % 10 == 0:
            tv = get_tau(m)
            grad_log.append({
                'ep': ep + 1, 'tau': tv,
                'tau_grad_norm': tau_grad_norm,
                'total_grad_norm': total_grad_norm,
                'ppl': math.exp(vl)
            })
            log(f'  ep{ep+1:>4} tau={tv:.4f} |∇τ|={tau_grad_norm:.6f} |∇total|={total_grad_norm:.4f}')

    elapsed = time.time() - st
    torch.cuda.empty_cache()
    tau_f = get_tau(m)
    tau_layers = get_tau_per_layer(m)
    evals = eval_model(m, [128, 256, 512])

    result = {
        'tag': tag, 'seed': seed, 'norm': norm_type,
        'dh': 64, 'nh': 4, 'dim': 256, 'L': 128, 'rope': True,
        'tau_final': tau_f, 'tau_layers': tau_layers,
        'tau_trajectory': grad_log,
        'grad_trace': grad_log,
        'ppl': math.exp(best_loss),
        'eval_128': evals.get(128, 0),
        'eval_256': evals.get(256, 0),
        'eval_512': evals.get(512, 0),
        'time': elapsed,
    }
    log(f'  GRAD DONE [{tag}] tau={tau_f:.4f}')
    return result


# ============================================================
# Main
# ============================================================
if __name__ == '__main__':
    T0 = time.time()
    print('=' * 60, flush=True)
    print('BENCHMARK: s^τ vs Tempered vs Softmax', flush=True)
    print(f'GPU: {torch.cuda.get_device_name(0)}', flush=True)
    print(f'Mem: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f} GB', flush=True)
    print(f'PyTorch: {torch.__version__}, CUDA: {torch.version.cuda}', flush=True)
    print('=' * 60, flush=True)

    # Phase 0: Speed benchmark
    speed_results = speed_benchmark()
    torch.cuda.empty_cache()

    # Phase 1: Training comparison (3 norms × 4 seeds = 12 models)
    print('\n' + '=' * 60, flush=True)
    print('PHASE 1: Training Comparison', flush=True)
    print('3 norm_types × 4 seeds = 12 models', flush=True)
    print('=' * 60, flush=True)

    PHASE1_CONFIGS = []
    seed_offset = 42
    for norm_type in ['learned', 'tempered', 'softmax']:
        for i in range(4):
            seed = seed_offset + i
            tag = f'{norm_type}_s{seed}'
            PHASE1_CONFIGS.append({'tag': tag, 'norm': norm_type, 'seed': seed})

    print(f'Starting {len(PHASE1_CONFIGS)} models with 8 workers', flush=True)
    st = time.time()
    with mp.Pool(8) as pool:
        p1_results = pool.map(train_model, PHASE1_CONFIGS)
    elapsed = time.time() - st
    print(f'Phase 1 done: {elapsed:.0f}s ({elapsed/60:.1f}min)', flush=True)

    data = {}
    for r in p1_results:
        key = f"{r['norm']}_{r['seed']}"
        data[key] = r
    data['total_time'] = elapsed
    fpath = os.path.join(RESULT_DIR, 'phase1_training.json')
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print(f'[SAVE] {fpath}', flush=True)

    # Phase 2: Gradient analysis (1 model per norm type)
    print('\n' + '=' * 60, flush=True)
    print('PHASE 2: Gradient Analysis', flush=True)
    print('3 models (1 per norm type), 100 epochs with gradient tracking', flush=True)
    print('=' * 60, flush=True)

    PHASE2_CONFIGS = [
        {'tag': 'grad_learned', 'norm': 'learned', 'seed': 100},
        {'tag': 'grad_tempered', 'norm': 'tempered', 'seed': 101},
        {'tag': 'grad_softmax', 'norm': 'softmax', 'seed': 102},
    ]
    st = time.time()
    with mp.Pool(3) as pool:
        p2_results = pool.map(grad_analysis, PHASE2_CONFIGS)
    elapsed = time.time() - st
    print(f'Phase 2 done: {elapsed:.0f}s ({elapsed/60:.1f}min)', flush=True)

    data2 = {}
    for r in p2_results:
        data2[r['tag']] = r
    data2['total_time'] = elapsed
    fpath2 = os.path.join(RESULT_DIR, 'phase2_gradients.json')
    with open(fpath2, 'w', encoding='utf-8') as f:
        json.dump(data2, f, indent=2, ensure_ascii=False, default=str)
    print(f'[SAVE] {fpath2}', flush=True)

    # Final summary
    total = time.time() - T0
    print('\n' + '=' * 60, flush=True)
    print(f'ALL DONE! Total: {total:.0f}s ({total/60:.1f}min)', flush=True)
    print(f'Results in: {RESULT_DIR}', flush=True)
    print('=' * 60, flush=True)
