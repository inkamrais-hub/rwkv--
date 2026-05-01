"""
█ report_long.py — 本地分析 results_pillar_long/pillar_long.json
█ 生成格式化报告 + tau收敛曲线
"""
import os, json, sys
import numpy as np

SD = os.path.dirname(os.path.abspath(__file__))
JSON_PATH = os.path.join(SD, '..', 'project_assets', 'results_pillar_long', 'pillar_long.json')

if not os.path.exists(JSON_PATH):
    print('No results yet! Run D:\python\python.exe scripts\harvest.py first.')
    sys.exit(1)

with open(JSON_PATH, encoding='utf-8') as f:
    data = json.load(f)

def sep(c='=', n=70): print(c * n)

sep()
print('EPX-B112 LONG EXPERIMENT (200ep) — FULL REPORT')
sep()

print('\n--- PART A: Tau Init Convergence ---')
pa = data.get('part_a', {})
taus_a = []
print('  {:>8} {:>10} {:>10} {:>8}'.format('init_log', 'tau_init', 'tau_final', 'ppl'))
print('  ' + '-'*42)
for k in sorted(pa.keys()):
    v = pa[k]
    t = v.get('tau_final', 0)
    taus_a.append(t)
    print('  {:>8.0f} {:>10.3f} {:>10.4f} {:>8.1f}'.format(
        v.get('tau_init_log', 0), v.get('tau_init', 0), t, v.get('ppl', 0)))
if taus_a:
    tr = max(taus_a) - min(taus_a)
    print('  tau range = {:.4f}  {}'.format(tr,
        'CONVERGED (<0.2)' if tr < 0.2 else 'STILL SPREAD (>=0.2)'))

tc = data.get('tau_convergence', {})
if 'part_a' in tc:
    print('\n  --- Tau Convergence Curves ---')
    keys = sorted(tc['part_a'].keys())
    print('  {:>5}'.format('ep'), end='')
    for k in keys:
        print(' {:>9}'.format(k), end='')
    print()
    for ep in range(10, 201, 10):
        print('  {:>5}'.format(ep), end='')
        for k in keys:
            hist = tc['part_a'][k]
            found = None
            for h in hist:
                if h.get('ep') == ep:
                    found = h.get('tau_mean', 0)
                    break
            if found is not None:
                print(' {:>9.4f}'.format(found), end='')
            else:
                print(' {:>9}'.format('-'), end='')
        print()

print('\n--- PART B: Tau(L) Monotonicity ---')
pb = data.get('part_b', {})
print('\n  --- stau models ---')
print('  {:>8} {:>10} {:>10} {:>10} {:>10} {:>10}'.format('L', 'tau', 'ppl_train', '@32', '@256', '@512'))
print('  ' + '-'*65)
tau_by_L = {}
for k in sorted(pb.keys()):
    v = pb[k]
    if k.startswith('stau'):
        blk = int(k.replace('stau_L', ''))
        tau_by_L[blk] = v.get('tau', 0)
        ev = v.get('eval', {})
        print('  {:>8} {:>10.4f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}'.format(
            blk, v.get('tau', 0), v.get('ppl', 0),
            ev.get('32', 0), ev.get('256', 0), ev.get('512', 0)))

print('\n  --- softmax baselines ---')
print('  {:>8} {:>10} {:>10} {:>10} {:>10}'.format('L', 'ppl_train', '@32', '@256', '@512'))
for k in sorted(pb.keys()):
    v = pb[k]
    if k.startswith('soft'):
        ev = v.get('eval', {})
        print('  {:>8} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}'.format(
            k, v.get('ppl', 0), ev.get('32', 0), ev.get('256', 0), ev.get('512', 0)))

Ls = sorted(tau_by_L.keys())
if len(Ls) >= 3:
    tv = [tau_by_L[l] for l in Ls]
    logL = np.log(Ls)
    c = np.polyfit(logL, tv, 1)
    r2 = 1 - np.var(tv - np.polyval(c, logL)) / np.var(tv)
    mono = all(tv[i] <= tv[i+1] for i in range(len(tv)-1))
    print('\n  tau(L) fit: {:.4f} + {:.4f} * log(L/128)'.format(
        c[1] - c[0]*np.log(128), c[0]))
    print('  alpha={:.4f}  R={:.4f}  {}'.format(c[0], r2,
        'MONOTONIC' if mono else 'NOT MONO'))

print('\n  --- Tau per Layer ---')
for k in sorted(pb.keys()):
    v = pb[k]
    if k.startswith('stau') and 'tau_layers' in v:
        blk = int(k.replace('stau_L', ''))
        row = '  L={:>4}:'.format(blk)
        for li in range(4):
            lk = 'L{}'.format(li)
            if lk in v['tau_layers']:
                row += ' {:>8.4f}'.format(v['tau_layers'][lk]['mean'])
        print(row)

print('\n--- PART D: Multi-Seed Tau Distribution ---')
pd = data.get('part_d', {})
if 'results' in pd:
    print('  {:>6} {:>10} {:>10} {:>10} {:>10} {:>10}'.format('seed', 'tau', 'ppl', '@32', '@256', '@512'))
    d_taus = []
    for k in sorted(pd['results'].keys()):
        v = pd['results'][k]
        d_taus.append(v.get('tau_final', 0))
        ev = v.get('eval', {})
        print('  {:>6} {:>10.4f} {:>10.1f} {:>10.1f} {:>10.1f} {:>10.1f}'.format(
            v.get('seed', 0), v.get('tau_final', 0), v.get('ppl', 0),
            ev.get('32', 0), ev.get('256', 0), ev.get('512', 0)))
    if d_taus:
        d_mean, d_std = float(np.mean(d_taus)), float(np.std(d_taus))
        print('  tau_mean={:.4f}  tau_std={:.4f}  {}'.format(
            d_mean, d_std, 'UNIMODAL' if d_std < 0.1 else 'SPREAD'))
        print('  tau_range=[{:.4f}, {:.4f}]'.format(min(d_taus), max(d_taus)))

total = data.get('total_time', 0)
cost = 1.88 * total / 3600
sep()
if taus_a:
    tr = max(taus_a) - min(taus_a)
    print('  Part A (tau convergence): {}  tau_range={:.4f}'.format('PASS' if tr < 0.2 else 'FAIL', tr))
if len(Ls) >= 3:
    print('  Part B (tau monotonic):      {}  alpha={:.4f}'.format('PASS' if mono else 'FAIL', c[0]))
if 'results' in pd and d_taus:
    print('  Part D (tau unimodal):      {}  tau_mean={:.4f}  tau_std={:.4f}'.format(
        'PASS' if float(np.std(d_taus)) < 0.1 else 'FAIL',
        float(np.mean(d_taus)), float(np.std(d_taus))))
print('\n  Total: {:.0f}s ({:.1f}min)  |  Cost: ~{:.2f} yuan'.format(total, total/60, cost))
sep()
