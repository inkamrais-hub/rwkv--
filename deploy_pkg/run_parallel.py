"""
█ run_parallel.py — PRO 6000 大规模并行实验
█ 3 串行阶段，每阶段内部并行
█
█ 用法: /root/miniconda3/bin/python3 -u run_parallel.py
"""
import sys, os, time, json, random, math, warnings, multiprocessing as mp
warnings.filterwarnings('ignore')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# CRITICAL: Use spawn to avoid CUDA+fork issues on Linux
mp.set_start_method('spawn', force=True)

import torch, torch.nn.functional as F
import urllib.request, numpy as np
from attention_mechanisms.model import ATTHModel

SD = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(SD, 'results_pro6000')
os.makedirs(RESULT_DIR, exist_ok=True)
LOG_DIR = os.path.join(SD, 'logs_pro6000')
os.makedirs(LOG_DIR, exist_ok=True)

# Shakespeare data
DATA_PATH = os.path.join(SD, 'shakespeare.txt')
if not os.path.exists(DATA_PATH):
    urllib.request.urlretrieve(
        'https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt',
        DATA_PATH)
text = open(DATA_PATH, encoding='utf-8').read()
chars = sorted(set(text)); V = len(chars)
cid = {ch: i for i, ch in enumerate(chars)}
data_t = torch.tensor([cid[ch] for ch in text], dtype=torch.long)
train_t, val_t = data_t[:int(len(data_t)*0.9)], data_t[int(len(data_t)*0.9):]

print('Vocab={} Train={}k Val={}k'.format(V, len(train_t)//1000, len(val_t)//1000), flush=True)

device = torch.device('cuda')

def get_batch(split, blk, rng):
    d = train_t if split == 'train' else val_t
    bs = 8 if blk >= 4096 else (16 if blk >= 2048 else (32 if blk >= 1024 else 64))
    ix = [rng.randint(0, len(d)-blk-1) for _ in range(bs)]
    return torch.stack([d[i:i+blk] for i in ix]), torch.stack([d[i+1:i+blk+1] for i in ix])

def get_tau(m):
    vs = []
    for layer in m.layers:
        if hasattr(layer.attn, 'per_head_log_tau') and layer.attn.per_head_log_tau is not None:
            vs.append(F.softplus(layer.attn.per_head_log_tau.detach()) + 1.0)
    return torch.stack([v.mean() for v in vs]).mean().item() if vs else 0

def get_tau_per_layer(m):
    r = {}
    for i, layer in enumerate(m.layers):
        if hasattr(layer.attn, 'per_head_log_tau') and layer.attn.per_head_log_tau is not None:
            t = F.softplus(layer.attn.per_head_log_tau.detach()) + 1.0
            r['L{}'.format(i)] = {'mean': t.mean().item(), 'std': t.std().item()}
    return r

def eval_model(m, lens, seed=999):
    rng = random.Random(seed); res = {}
    m.eval()
    with torch.no_grad():
        for blk in lens:
            ls = []
            for _ in range(8):
                vx, vy = get_batch('val', blk, rng)
                vx, vy = vx.to(device), vy.to(device)
                ls.append(F.cross_entropy(m(vx).reshape(-1, V), vy.reshape(-1)).item())
            res[blk] = math.exp(np.mean(ls))
    return res

def train_model(config):
    seed = config.get('seed', 42)
    block_size = config['L']
    dh = config['dh']
    nh = config['nh']
    dim = dh * nh
    epochs = config.get('epochs', 200)
    n_per_ep = config.get('n_per_ep', 50)
    use_rope = config.get('rope', True)
    norm_type = config.get('norm', 'learned')
    tag = config['tag']
    log_path = os.path.join(LOG_DIR, tag + '.log')

    def log(msg):
        with open(log_path, 'a') as f:
            f.write(msg + '\n')
        print(msg, flush=True)

    torch.manual_seed(seed); rng = random.Random(seed)
    m = ATTHModel(V, dim, 4, nh, 2048, 0.1, 4.0, False,
                  attn_type='standard', norm_type=norm_type,
                  use_rope=use_rope, use_pos_emb=False, angle_emb=False)
    if norm_type == 'learned':
        for layer in m.layers:
            layer.attn.per_head_log_tau.data.zero_()
    m = m.to(device)

    tau_ps = [p for n, p in m.named_parameters() if 'log_tau' in n]
    other_ps = [p for n, p in m.named_parameters() if 'log_tau' not in n]
    tau_lr = config.get('tau_lr', 1e-2)
    base_lr = config.get('base_lr', 1e-4)
    opt = torch.optim.AdamW([
        {'params': other_ps, 'lr': base_lr, 'weight_decay': 0.01},
        {'params': tau_ps, 'lr': tau_lr, 'weight_decay': 0.0}
    ], betas=(0.9, 0.98))
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.cuda.amp.GradScaler()
    best_loss = float('inf'); best_state = None; st = time.time()

    log('START {} | {} | seed={} | dh={} nh={} dim={} RoPE={}'.format(
        tag, norm_type, seed, dh, nh, dim, use_rope))

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
        if (ep+1) % 10 == 0:
            tv = get_tau(m)
            log('  [{}] ep{:>4} ppl={:.1f} best={:.1f} tau={:.4f}'.format(
                tag, ep+1, math.exp(vl), math.exp(best_loss), tv))

    elapsed = time.time() - st
    torch.cuda.empty_cache()
    tau_f = get_tau(m)
    tau_layers = get_tau_per_layer(m)
    eval_lens = [blk for blk in [128, 256, 512, 1024, 2048] if blk <= max(512, block_size * 2)]
    evals = eval_model(m, eval_lens)

    result = {
        'tag': tag, 'seed': seed, 'dh': dh, 'nh': nh, 'dim': dim,
        'L': block_size, 'rope': use_rope, 'norm': norm_type,
        'tau_final': tau_f, 'tau_layers': tau_layers,
        'ppl': math.exp(best_loss), 'eval': evals, 'time': elapsed
    }
    log('  DONE: tau={:.4f} ppl={:.1f} time={:.0f}s'.format(tau_f, math.exp(best_loss), elapsed))
    log('  EVALS: ' + str(evals))
    return result

def run_phase(phase_name, configs, max_workers=6):
    print('\n{} {} {}'.format('='*30, phase_name, '='*30), flush=True)
    print('Starting {} models with {} workers'.format(len(configs), max_workers), flush=True)
    st = time.time()

    with mp.Pool(max_workers) as pool:
        results = pool.map(train_model, configs)

    elapsed = time.time() - st
    print('Phase done: {:.0f}s ({:.1f}min)'.format(elapsed, elapsed/60), flush=True)

    # Save results
    phase_tag = phase_name.lower().replace(' ', '_')
    data = {}
    for r in results:
        key = '{}_L{}'.format(r['tag'], r['L'])
        if r.get('norm') == 'softmax':
            key = 'soft_L{}'.format(r['L'])
        data[key] = r
    data['total_time'] = elapsed
    fpath = os.path.join(RESULT_DIR, '{}.json'.format(phase_tag))
    with open(fpath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    print('[SAVE] {}'.format(fpath), flush=True)
    return results


# ============================================================
# EXPERIMENT CONFIGS
# ============================================================

def make_tag(dh, rope, norm='learned'):
    pe = 'R' if rope else 'N'
    if norm == 'softmax': return 'soft'
    return 'dh{}{}'.format(dh, pe)

# ---- PHASE 1: Complete L-scan (softmax baselines + missing s^tau) ----
PHASE1 = []
for L in [128, 256, 512, 1024]:
    PHASE1.append({'tag': 'soft_L{}'.format(L), 'dh': 64, 'nh': 4, 'L': L,
                    'rope': True, 'norm': 'softmax', 'epochs': 200, 'n_per_ep': 50})
for L in [128, 256]:
    PHASE1.append({'tag': 'dh64R_L{}'.format(L), 'dh': 64, 'nh': 4, 'L': L,
                    'rope': True, 'norm': 'learned', 'epochs': 200, 'n_per_ep': 50})

# ---- PHASE 2: dh scan (16/32/64/128 × none/RoPE) ----
PHASE2 = []
for dh, nh in [(16, 4), (32, 4), (64, 4), (128, 4)]:
    dim_check = dh * nh
    if dim_check > 512:
        nh = max(2, 512 // dh)
    for rope in [False, True]:
        tag = make_tag(dh, rope)
        PHASE2.append({'tag': tag, 'dh': dh, 'nh': nh, 'L': 128,
                        'rope': rope, 'norm': 'learned', 'epochs': 200, 'n_per_ep': 50})

# ---- PHASE 3: Multi-seed stability (8 seeds) ----
PHASE3 = []
for seed in range(42, 50):
    PHASE3.append({'tag': 'dh64R_s{}'.format(seed), 'dh': 64, 'nh': 4, 'L': 128,
                    'rope': True, 'norm': 'learned', 'epochs': 200, 'n_per_ep': 50,
                    'seed': seed})

# ---- PHASE 4: L=2048 ----
PHASE4 = [
    {'tag': 'dh64R_L2048', 'dh': 64, 'nh': 4, 'L': 2048,
     'rope': True, 'norm': 'learned', 'epochs': 200, 'n_per_ep': 30},
    {'tag': 'soft_L2048', 'dh': 64, 'nh': 4, 'L': 2048,
     'rope': True, 'norm': 'softmax', 'epochs': 200, 'n_per_ep': 30},
]

# ---- PHASE 5: L=4096 ----
PHASE5 = [
    {'tag': 'dh64R_L4096', 'dh': 64, 'nh': 4, 'L': 4096,
     'rope': True, 'norm': 'learned', 'epochs': 100, 'n_per_ep': 15},
    {'tag': 'soft_L4096', 'dh': 64, 'nh': 4, 'L': 4096,
     'rope': True, 'norm': 'softmax', 'epochs': 100, 'n_per_ep': 15},
]

if __name__ == '__main__':
    T0 = time.time()
    print('='*60, flush=True)
    print('PRO 6000 MASSIVE PARALLEL EXPERIMENT', flush=True)
    print('GPU: {}'.format(torch.cuda.get_device_name(0)), flush=True)
    print('Mem: {:.0f} GB'.format(torch.cuda.get_device_properties(0).total_memory / 1e9), flush=True)
    print('='*60, flush=True)
    print()

    # Phase 1
    r1 = run_phase('PHASE1_L-scan_complete', PHASE1, max_workers=6)
    torch.cuda.empty_cache()

    # Phase 2
    r2 = run_phase('PHASE2_dh_scan', PHASE2, max_workers=8)
    torch.cuda.empty_cache()

    # Phase 3
    r3 = run_phase('PHASE3_multi_seed', PHASE3, max_workers=8)
    torch.cuda.empty_cache()

    # Phase 4
    r4 = run_phase('PHASE4_L2048', PHASE4, max_workers=2)
    torch.cuda.empty_cache()

    # Phase 5
    r5 = run_phase('PHASE5_L4096', PHASE5, max_workers=2)
    torch.cuda.empty_cache()

    # Final summary
    total = time.time() - T0
    print('\n' + '='*60, flush=True)
    print('ALL DONE! Total: {:.0f}s ({:.1f}min)'.format(total, total/60), flush=True)
    print('Results in: {}'.format(RESULT_DIR), flush=True)
    print('='*60, flush=True)
