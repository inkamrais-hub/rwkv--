"""
█ report_scan.py — 本地分析 tau_scan.json
█ 合并 long 实验 baseline，输出 τ 相图
"""
import os, json, sys
import numpy as np

SD = os.path.dirname(os.path.abspath(__file__))
SCAN_JSON = os.path.join(SD, '..', 'project_assets', 'results_tau_scan', 'tau_scan.json')
LONG_JSON = os.path.join(SD, '..', 'project_assets', 'results_pillar_long', 'pillar_long.json')

if not os.path.exists(SCAN_JSON):
    print('No scan results yet! Run D:\python\python.exe scripts\harvest.py first.')
    sys.exit(1)

with open(SCAN_JSON, encoding='utf-8') as f:
    scan = json.load(f)

with open(LONG_JSON, encoding='utf-8') as f:
    long_data = json.load(f)

def sep(c='=', n=70): print(c * n)

sep()
print('TAU PHASE DIAGRAM: d_head x PE x L')
sep()

merged = {}
for k, v in scan.items():
    if k.startswith('total') or k.startswith('config'):
        continue
    merged[k] = v

pa = long_data.get('part_a', {})
if 'init0' in pa:
    merged['dh16_none_L128'] = {
        'd_head': 16, 'dim': 128, 'PE': 'none', 'L': 128,
        'tau_final': pa['init0']['tau_final'],
        'tau_layers': pa['init0']['tau_layers'],
        'ppl': pa['init0']['ppl'],
        'eval': pa['init0']['eval'],
        '_source': 'long experiment Part A init=0'
    }

pd = long_data.get('part_d', {})
for k, v in pd.items():
    if k.startswith('seed'):
        sd = v.get('seed', '?')
        merged['dh16_none_L128_s{}'.format(sd)] = {
            'd_head': 16, 'dim': 128, 'PE': 'none', 'L': 128,
            'tau_final': v['tau_final'], 'tau_layers': v.get('tau_layers', {}),
            'ppl': v['ppl'], 'eval': v.get('eval', {}),
            '_source': 'long experiment Part D seed={}'.format(sd)
        }

print('\n--- tau(d_head) @ PE=none, L=128 ---')
print('  {:>8} {:>8} {:>10} {:>8}'.format('d_head', 'dim', 'tau', 'ppl'))
dh_none = {}
for k, v in merged.items():
    if v.get('PE') == 'none' and v.get('L') == 128 and v.get('norm', 'learned') != 'softmax':
        dh = v.get('d_head', 0)
        t = v.get('tau_final', 0)
        if dh not in dh_none: dh_none[dh] = []
        dh_none[dh].append(t)

for dh in sorted(dh_none.keys()):
    tv = dh_none[dh]
    print('  {:>8} {:>8} {:>10.4f} {:>8.1f}'.format(dh, dh*4, np.mean(tv),
          merged.get('dh{}_none_L128_42'.format(dh), {}).get('ppl', 0) if len(tv)==1 else 'n/a'))
if len(dh_none) >= 2:
    dhs = sorted(dh_none.keys())
    taus = [np.mean(dh_none[d]) for d in dhs]
    log_dh = np.log(dhs)
    c = np.polyfit(log_dh, taus, 1)
    print('  fit: tau={:.4f} + {:.4f} * log(d_head/16)'.format(taus[0], c[0]))

print('\n--- tau(PE) interaction ---')
print('  {:>8} {:>8} {:>10} {:>10} {:>8}'.format('d_head', 'PE', 'tau', 'tau_none', ''))
for dh in [16, 32, 64]:
    t_none = [v.get('tau_final', 0) for k, v in merged.items()
               if v.get('d_head') == dh and v.get('PE') == 'none' and v.get('L') == 128 and v.get('norm', 'learned') == 'learned']
    t_rope = [v.get('tau_final', 0) for k, v in merged.items()
               if v.get('d_head') == dh and v.get('PE') == 'RoPE' and v.get('L') == 128 and v.get('norm', 'learned') == 'learned']
    tn = np.mean(t_none) if t_none else 0
    tr = np.mean(t_rope) if t_rope else 0
    print('  {:>8} {:>8} {:>10.4f} {:>10.4f} {:>+8.4f}'.format(dh, 'none', tn, tn, 0.0))
    if t_rope:
        print('  {:>8} {:>8} {:>10.4f} {:>10.4f} {:>+8.4f}'.format(dh, 'RoPE', tr, tn, tr-tn))

print('\n--- tau per Layer @ PE=none, L=128 ---')
print('  {:>8}'.format('d_head'), end='')
for li in range(4):
    print(' {:>10}'.format('L'+str(li)), end='')
print()
for dh in [16, 32, 64]:
    for k, v in merged.items():
        if v.get('d_head') == dh and v.get('PE') == 'none' and v.get('L') == 128 and v.get('norm', 'learned') == 'learned':
            tl = v.get('tau_layers', {})
            print('  {:>8}'.format(dh), end='')
            for li in range(4):
                lk = 'L{}'.format(li)
                t = tl.get(lk, {}).get('mean', 0)
                print(' {:>10.4f}'.format(t), end='')
            print()
            break

print('\n--- tau(L) scaling across d_head ---')
for dh in [16, 32, 64]:
    row = '  d_head={:>2}:'.format(dh)
    for L in [128, 512]:
        vals = [v.get('tau_final', 0) for k, v in merged.items()
                if v.get('d_head') == dh and v.get('PE') == 'none' and v.get('L') == L and v.get('norm', 'learned') == 'learned']
        t = np.mean(vals) if vals else 0
        row += ' L={:>4}:{:>8.4f}'.format(L, t) if t else ' L={:>4}:    n/a'.format(L)
    print(row)

total = scan.get('total_time', 0)
sep()
print('Total scan time: {:.0f}s ({:.1f}min) | Est cost: ~{:.1f} yuan'.format(total, total/60, total/3600*1.88))
print('Results: {}'.format(SCAN_JSON))
sep()
