"""
█ harvest_pro6000.py — 下载 PRO 6000 全部结果 + 显示汇总
"""
import paramiko, os, re, json

HOST, PORT, PW = 'connect.westd.seetacloud.com', 12359, 'NPKKDRLuIdNS'
SD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(SD, 'project_assets', 'results_pro6000')
os.makedirs(ASSETS, exist_ok=True)

ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username='root', password=PW, timeout=15)

# 1. Kill training
_, out, _ = ssh.exec_command("ps aux | grep run_parallel | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null; echo killed")
print('Killed running process')

# 2. Download results
sftp = ssh.open_sftp()
result_files = sftp.listdir('/root/epx/results_pro6000')
print('\n=== Downloaded results ===')
for fn in result_files:
    if fn.endswith('.json'):
        sftp.get('/root/epx/results_pro6000/' + fn, os.path.join(ASSETS, fn))
        size = os.path.getsize(os.path.join(ASSETS, fn))
        print(f'  {fn} ({size/1024:.0f} KB)')
sftp.close()

# 3. Also get raw logs
_, out, _ = ssh.exec_command('cat /root/epx/run_parallel.log')
master_log = out.read().decode()
with open(os.path.join(ASSETS, 'run_parallel.log'), 'w') as f:
    f.write(master_log)
print('  run_parallel.log')

ssh.close()

# 4. Parse and display ALL results
print('\n' + '='*65)
print('  PRO 6000 实验结果总汇总')
print('='*65)

# Phase 1: L-scan
p1 = os.path.join(ASSETS, 'phase1_l-scan_complete.json')
if os.path.exists(p1):
    with open(p1) as f:
        d = json.load(f)
    print('\n--- Phase 1: L-scan (dh64+RoPE) ---')
    for L in [128, 256, 512, 1024]:
        key = 'dh64R_L{}'.format(L)
        skey = 'soft_L{}'.format(L)
        if key in d:
            v = d[key]
            print('  s^tau L={:<4d}  τ={:.4f}  PPL={:.1f}  @512={:.1f}  time={:.0f}s'.format(
                L, v.get('tau_final',0), v.get('ppl',0), v.get('eval',{}).get(512,v.get('eval_p512',0)), v.get('time',0)))
        if skey in d:
            v = d[skey]
            print('  soft  L={:<4d}               PPL={:.1f}  @512={:.1f}  time={:.0f}s'.format(
                L, v.get('ppl',0), v.get('eval',{}).get(512,0), v.get('time',0)))

# Phase 2: dh scan
p2 = os.path.join(ASSETS, 'phase2_dh_scan.json')
if os.path.exists(p2):
    with open(p2) as f:
        d = json.load(f)
    print('\n--- Phase 2: dh 相图扫描 (L=128) ---')
    print('  {:>8} {:>8} {:>10} {:>10} {:>10}'.format('d_head', 'PE', 'tau', 'PPL', 'n_heads'))
    for dh in [16, 32, 64, 128]:
        for pe, tag_base in [(False, 'dh{}N'), (True, 'dh{}R')]:
            key = tag_base.format(dh)
            if key in d:
                v = d[key]
                tau = v.get('tau_final', 0)
                ppl = v.get('ppl', 0)
                nh = v.get('nh', 4)
                pe_label = 'none' if not v.get('rope', True) else 'RoPE'
                print('  {:>8} {:>8} {:>10.4f} {:>10.1f} {:>10}'.format(dh, pe_label, tau, ppl, nh))

# Phase 3: multi-seed
p3 = os.path.join(ASSETS, 'phase3_multi_seed.json')
if os.path.exists(p3):
    with open(p3) as f:
        d = json.load(f)
    taus = []
    print('\n--- Phase 3: Multi-Seed (dh64+RoPE L=128) ---')
    for s in range(42, 50):
        key = 'dh64R_s{}'.format(s)
        if key in d:
            v = d[key]
            tau = v.get('tau_final', 0)
            taus.append(tau)
            print('  seed={:<2d}  τ={:.4f}  PPL={:.1f}'.format(s, tau, v.get('ppl',0)))
    if taus:
        import numpy as np
        print('  {} seeds: mean={:.4f}  std={:.4f}  range={:.4f}'.format(
            len(taus), np.mean(taus), np.std(taus), max(taus)-min(taus)))

# Phase 4: L=2048
p4 = os.path.join(ASSETS, 'phase4_l2048.json')
if os.path.exists(p4):
    with open(p4) as f:
        d = json.load(f)
    print('\n--- Phase 4: L=2048 ---')
    for key in ['dh64R_L2048', 'soft_L2048']:
        if key in d:
            v = d[key]
            tau = v.get('tau_final', '—')
            ppl = v.get('ppl', '?')
            ev = v.get('eval', {})
            print('  {:<15s}  τ={}  PPL={}  @512={}  @1024={}'.format(
                key, tau, ppl, ev.get(512,'?'), ev.get(1024,'?')))

print('\n' + '='*65)
print('  结果保存在: {}'.format(ASSETS))
print('='*65)
