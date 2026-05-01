"""
█ watch.py — 远端训练仪表盘（每5秒刷新）
█ 用法: D:\python\python.exe -u scripts\watch_hy.py
"""
import paramiko, time, sys, os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

REMOTE_STATUS = '/root/epx/status.txt'

def fetch():
    for _ in range(2):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username='root', password=PW, timeout=4)
            _, out, _ = ssh.exec_command(f'cat {REMOTE_STATUS} 2>/dev/null || echo "NO_STATUS"')
            data = out.read().decode().strip()
            ssh.close()
            return data
        except:
            time.sleep(1)
    return None

def render(data):
    lines = data.split('\n')
    phase, gpu, models = '', '', []
    for l in lines:
        l = l.strip()
        if l.startswith('PHASE:'): phase = l[6:]
        elif l.startswith('GPU:'): gpu = l[4:]
        elif l == 'NO_STATUS': return '等待训练启动...'
        elif l and not l.startswith('DONE'):
            models.append(l)

    tag_map = {
        'PHASE0_Speed': '速度基准', 'PHASE1_Training': '训练对比',
        'PHASE2_Gradient': '梯度分析', 'ALL_DONE': '全部完成'}
    phase_nice = tag_map.get(phase, phase) if ':' in phase else phase

    done = sum(1 for m in models if m.startswith('DONE:'))
    out = []
    sep = '─' * 54
    out.append(f'  ┌{sep}┐')
    out.append(f'  │ {"云端  " + phase_nice:<29s}  GPU:{gpu:<22s}│')
    out.append(f'  ├{sep}┤')

    for m in models:
        if m.startswith('DONE:'):
            name = m[5:].split()[0] if m[5:].split() else m[5:]
            out.append(f'  │ ✅ {name:<50s}│')
            continue

        parts = m.split()
        name = parts[0] if parts else '?'
        ep, tau, ppl = '?', '-', '-'
        for p in parts[1:]:
            if p.startswith('ep'): ep = p[2:]
            elif p.startswith('tau='): tau = p[4:]
            elif p.startswith('ppl='): ppl = p[4:]

        try:
            ep_n = int(ep)
            pct = ep_n / 200
            n = int(pct * 20)
            bar = '█' * n + '░' * (20 - n)
            line = f'  │ {name:<15s} {bar} {pct*100:3.0f}%'
            # Only show tau if it's actually learning (different from init)
            if float(tau) != 0.6931 and float(tau) != 1.6931:
                line += f'  τ={float(tau):.2f}'
            line += f'  PPL={ppl}'
            line += ' ' * (54 - len(line) + 2) + '│'
            out.append(line)
        except:
            out.append(f'  │ {name:<15s} {"░"*20}   ?%  PPL=?{" ":<20s}│')

    out.append(f'  ├{sep}┤')
    out.append(f'  │ {done}/{len(models)} done{"":<42s}│')
    out.append(f'  └{sep}┘')
    return '\n'.join(out)

last = ''
tick = 0
print('  ' + '=' * 58)
print(f'  5090 @ {HOST}:{PORT}  每5秒刷新  Ctrl+C 退出')
print('  ' + '=' * 58, flush=True)

while True:
    tick += 1
    raw = fetch()
    if raw:
        r = render(raw)
        if r != last or tick % 6 == 0:  # force refresh every 3s even if same
            os.system('cls' if os.name == 'nt' else 'clear')
            print(r, flush=True)
            print(f'  [{time.strftime("%H:%M:%S")}]', flush=True)
            last = r
    else:
        print(f'  [{time.strftime("%H:%M:%S")}] 断连, {tick}', flush=True)
    time.sleep(0.5)
