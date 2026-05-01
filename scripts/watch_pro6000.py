"""
█ watch_pro6000.py — PRO 6000 实时可视化监控（持续版）
█ 每15秒自动刷新，断连自动重试
█
█ 用法: D:\python\python.exe -u scripts\watch_pro6000.py
"""
import paramiko, time, re, sys

HOST = 'connect.westd.seetacloud.com'
PORT = 12359
PW = 'NPKKDRLuIdNS'

def fetch():
    for attempt in range(3):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username='root', password=PW, timeout=8)
            _, out, _ = ssh.exec_command('cat /root/epx/status.txt 2>/dev/null || echo "NO_STATUS"')
            data = out.read().decode().strip()
            ssh.close()
            return data
        except:
            time.sleep(2)
    return None

def render(data):
    lines = data.split('\n')
    phase, gpu, done_phases, all_done = '', '', 0, False
    models = []
    for l in lines:
        l = l.strip()
        if l.startswith('PHASE:'): phase = l[6:]
        elif l.startswith('DONE_PHASES:'): done_phases = int(l[12:])
        elif l.startswith('GPU:'): gpu = l[4:]
        elif l.startswith('DONE:YES'): all_done = True
        elif l == 'NO_STATUS': return '等待训练启动...'
        elif l and not l.startswith('DONE:') and not l.startswith('PHASE:') and not l.startswith('DONE_PHASES:') and not l.startswith('GPU:'):
            models.append(l)

    tag_map = {
        'PHASE1_L-scan_complete': 'Phase 1: L-scan', 'PHASE2_dh_scan': 'Phase 2: dh相图',
        'PHASE3_multi_seed': 'Phase 3: 多seed', 'PHASE4_L2048': 'Phase 4: L=2048',
        'PHASE5_L4096': 'Phase 5: L=4096'}
    phase_nice = phase
    for k, v in tag_map.items():
        if k in phase: phase_nice = v

    done_count = sum(1 for m in models if m.startswith('DONE:'))
    out = []
    sep = '─' * 50
    out.append('')
    out.append('  ┌{}┐'.format(sep))
    out.append('  │ PRO6000  {}'.format(phase_nice.ljust(32)))
    out.append('  │ GPU: {}'.format(gpu))
    if done_phases: out.append('  │ 已完成 {} 阶段'.format(done_phases))
    out.append('  ├{}┤'.format(sep))

    for m in models:
        if m.startswith('DONE:'):
            name = m[5:].split()[0] if m[5:].split() else m[5:]
            out.append('  │ ✅ {:<44s}│'.format(name))
        else:
            parts = m.split()
            name = parts[0] if parts else '?'
            ep, tau, ppl = '?', '—', '—'
            for p in parts[1:]:
                if p.startswith('ep'): ep = p[2:]
                elif p.startswith('tau='): tau = p[4:]
                elif p.startswith('ppl='): ppl = p[4:]
            try:
                ep_n = int(ep)
                pct = ep_n / 200
                w = 40
                n = int(pct * w)
                bar = '█' * n + '░' * (w - n)
                out.append('  │ {} ep{:<3d}/200 {} {:>3.0f}%│'.format(name.ljust(10), ep_n, bar, pct * 100))
                if tau != '—' and tau != '0.0000':
                    out.append('  │ {:>46s}│'.format('τ={}  PPL={}'.format(tau, ppl)))
            except:
                out.append('  │ {} ep{}  │'.format(name.ljust(10), ep))

    out.append('  ├{}┤'.format(sep))
    out.append('  │ {}/{} done{:<29s}│'.format(done_count, len(models), ''))
    if all_done:
        out.append('  │ 🎉 全部实验完成!                         │')
    out.append('  └{}┘'.format(sep))
    return '\n'.join(out)

last = ''
tick = 0

print('  ' + '=' * 52, flush=True)
print('  watch_pro6000.py  实时监控（每15秒刷新）', flush=True)
print('  Ctrl+C 退出', flush=True)
print('  ' + '=' * 52, flush=True)

while True:
    tick += 1
    raw = fetch()
    if raw:
        rendered = render(raw)
        if rendered != last:
            print(rendered, flush=True)
            print('  --- [{}] tick {} ---'.format(time.strftime('%H:%M:%S'), tick), flush=True)
            last = rendered
    else:
        print('  [{}] tick {} SSH失败，等待重试...'.format(time.strftime('%H:%M:%S'), tick), flush=True)
    time.sleep(15)
