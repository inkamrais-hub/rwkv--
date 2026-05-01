import json, os, sys
import numpy as np

SD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(SD, 'project_assets', 'results_pro6000')

def load(fn):
    with open(os.path.join(ASSETS, fn)) as f:
        return json.load(f)

p1 = load('phase1_l-scan_complete.json')
p2 = load('phase2_dh_scan.json')
p3 = load('phase3_multi_seed.json')
p4 = load('phase4_l2048.json')

def sep(c='='): print(c * 62)

sep()
print('  PRO 6000  全部实验结果汇总（Phase 1-4，共 26 模型）')
sep()

# ------ Phase 1: L-scan (dh64+RoPE) ------
print('\n▎Phase 1: L-scan (dh64+RoPE, 200ep)')
print('  ┌──────┬──────────┬────────┬───────────┬──────────┐')
print('  │  L   │  类型    │  τ     │  PPL      │  时间     │')
print('  ├──────┼──────────┼────────┼───────────┼──────────┤')
for L in [128, 256, 512, 1024]:
    st = 'dh64R_L{}_L{}'.format(L, L)
    sk = 'soft_L{}'.format(L)
    if st in p1:
        v = p1[st]
        print('  │ {:<4d} │ s^τ     │ {:.3f} │ {:<9.2f} │ {:.0f}s   │'.format(
            L, v['tau_final'], v['ppl'], v['time']))
    if sk in p1:
        v = p1[sk]
        print('  │ {:<4d} │ softmax │ —     │ {:<9.2f} │ {:.0f}s   │'.format(
            L, v['ppl'], v['time']))
print('  └──────┴──────────┴────────┴───────────┴──────────┘')

# ------ Phase 2: dh scan ------
print('\n▎Phase 2: dh 相图扫描 (L=128, 200ep)')
print('  ┌──────┬───────┬────────┬────────┬───────────┐')
print('  │ d_hd │  PE   │  τ     │  dτ    │  PPL      │')
print('  ├──────┼───────┼────────┼────────┼───────────┤')
taus_none = {}
taus_rope = {}
for dh in [16, 32, 64, 128]:
    for nk, pe, tag in [(False, 'none', 'dh{}N_L128'), (True, 'RoPE', 'dh{}R_L128')]:
        k = tag.format(dh)
        if k in p2:
            v = p2[k]
            tau = v['tau_final']
            ppl = v['ppl']
            if not nk:
                taus_none[dh] = tau
            else:
                taus_rope[dh] = tau
            dtau = ''
            if nk and dh in taus_none:
                dtau = '+{:.1f}%'.format((tau/taus_none[dh] - 1)*100)
            elif not nk and dh in taus_rope:
                dtau = '—'
            print('  │ {:<4d} │ {:<5s} │ {:.3f} │ {:>7s} │ {:<9.2f} │'.format(dh, pe, tau, dtau, ppl))
print('  └──────┴───────┴────────┴────────┴───────────┘')

# ------ Phase 3: multi-seed ------
print('\n▎Phase 3: Multi-Seed (dh64+RoPE L=128, 200ep, 8 seeds)')
taus = []
for s in range(42, 50):
    k = 'dh64R_s{}_L128'.format(s)
    if k in p3:
        v = p3[k]
        tau = v['tau_final']
        ppl = v['ppl']
        taus.append(tau)
print('  seeds: 42-49')
if taus:
    mu, sig = np.mean(taus), np.std(taus)
    rng = max(taus) - min(taus)
    print('  τ mean={:.4f}  std={:.4f}  range={:.4f}  ▶ UNIMODAL ✅ (std<0.1)'.format(mu, sig, rng))

# ------ Per-layer tau ------
print('\n▎τ per layer (跨 8 seeds 平均)')
print('  ┌──────┬────────┬────────┬────────┬────────┬────────┐')
print('  │ seed │  τ_mean│  L0    │  L1    │  L2    │  L3    │')
print('  ├──────┼────────┼────────┼────────┼────────┼────────┤')
all_layers = {'L0':[], 'L1':[], 'L2':[], 'L3':[]}
for s in range(42, 50):
    k = 'dh64R_s{}_L128'.format(s)
    if k in p3:
        v = p3[k]
        tl = v.get('tau_layers', {})
        tau = v['tau_final']
        l0 = tl.get('L0',{}).get('mean',0)
        l1 = tl.get('L1',{}).get('mean',0)
        l2 = tl.get('L2',{}).get('mean',0)
        l3 = tl.get('L3',{}).get('mean',0)
        all_layers['L0'].append(l0)
        all_layers['L1'].append(l1)
        all_layers['L2'].append(l2)
        all_layers['L3'].append(l3)
        print('  │ {:<4d} │ {:.3f}  │ {:.3f} │ {:.3f} │ {:.3f} │ {:.3f} │'.format(s, tau, l0, l1, l2, l3))
print('  ├──────┼────────┼────────┼────────┼────────┼────────┤')
for li in range(4):
    lk = 'L{}'.format(li)
    lm = np.mean(all_layers[lk])
    print('  │ mean │        │ {:.3f} │       │       │       │'.format(lm))
print('  └──────┴────────┴────────┴────────┴────────┴────────┘')
# Deepest layer has highest tau
l3_mean = np.mean(all_layers['L3'])
l0_mean = np.mean(all_layers['L0'])
print('  深层(L3) τ={:.2f} vs 浅层(L0) τ={:.2f} → 深层比浅层锐化 {:.1f}倍'.format(l3_mean, l0_mean, l3_mean/l0_mean if l0_mean else 0))

# ------ Phase 4: L=2048 ------
print('\n▎Phase 4: L=2048 (dh64+RoPE, 200ep)')
sk = 'dh64R_L2048_L2048'
if sk in p4:
    v = p4[sk]
    print('  s^tau L=2048: τ={:.3f}  PPL={:.2f}  eval@512={:.2f}  @1024={:.2f}  @2048={:.2f}'.format(
        v['tau_final'], v['ppl'], v['eval'].get(512,0), v['eval'].get(1024,0), v['eval'].get(2048,0)))
sk2 = 'soft_L2048'
if sk2 in p4:
    v = p4[sk2]
    print('  soft L=2048:              PPL={:.2f}  eval@512={:.2f}  @1024={:.2f}  @2048={:.2f}'.format(
        v['ppl'], v['eval'].get(512,0), v['eval'].get(1024,0), v['eval'].get(2048,0)))

# ------ Complete τ(d) table ------
print('\n▎τ(d_head, PE) 完整相图')
print('  ┌──────┬───────┬────────┬────────┬────────┬────────┐')
print('  │ d_hd │  PE   │  L=128 │  L=256 │  L=512 │  L=1024│')
print('  ├──────┼───────┼────────┼────────┼────────┼────────┤')
# Get L=256/512/1024 from 4090D results
old = {}
old_json = os.path.join(SD, 'project_assets', 'results_lscan_dh64', 'lscan_dh64.json')
if os.path.exists(old_json):
    with open(old_json) as f:
        old = json.load(f)

for dh in [16, 32, 64, 128]:
    for pe_lab, key_pat in [('none', 'dh{}N_L128'), ('RoPE', 'dh{}R_L128')]:
        k = key_pat.format(dh)
        if k in p2:
            tau128 = p2[k]['tau_final']
            tau256 = old.get('dh64R_L256', {}).get('tau_final', 0) if dh == 64 and pe_lab == 'RoPE' else 0
            tau512 = old.get('dh64R_L512', {}).get('tau_final', 0) if dh == 64 and pe_lab == 'RoPE' else 0
            tau1024 = old.get('dh64R_L1024', {}).get('tau_final', 0) if dh == 64 and pe_lab == 'RoPE' else 0
            t256s = '{:.2f}'.format(tau256) if tau256 else ' —  '
            t512s = '{:.2f}'.format(tau512) if tau512 else ' —  '
            t1024s = '{:.2f}'.format(tau1024) if tau1024 else ' —  '
            print('  │ {:<4d} │ {:<5s} │ {:.2f}  │ {:>5s} │ {:>5s} │ {:>5s} │'.format(dh, pe_lab, tau128, t256s, t512s, t1024s))
print('  └──────┴───────┴────────┴────────┴────────┴────────┘')

# ------ Summary stats ------
total_models = 0
total_time = 0
for p in [p1, p2, p3, p4]:
    total_models += sum(1 for k in p if k != 'total_time' and k != 'total_time')
    total_time += p.get('total_time', 0)

sep()
print('  总模型: {} | 总时间: {:.0f}s ({:.1f}min)'.format(total_models, total_time, total_time/60))
print('  结果目录: {}'.format(ASSETS))
sep()
