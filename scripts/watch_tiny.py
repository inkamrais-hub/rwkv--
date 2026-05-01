"""
watch_tiny.py — 训练监控终端 (双行: dashboard + 实时日志)

用法: D:\python\python.exe -u scripts\watch_tiny.py
      每 5 秒刷新, 显示 dashboard + 最近日志
"""
import paramiko, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

STATUS_FILE = '/root/epx/status_tiny.txt'
LOG_FILE = '/root/tiny_train.log'


def fetch():
    for _ in range(2):
        try:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(HOST, port=PORT, username='root', password=PW, timeout=5)
            _, o1, _ = ssh.exec_command(f'cat {STATUS_FILE} 2>/dev/null || echo "NO_STATUS"')
            _, o2, _ = ssh.exec_command(f'tail -10 {LOG_FILE} 2>/dev/null || echo "NO_LOG"')
            status = o1.read().decode().strip()
            logtail = o2.read().decode().strip()
            ssh.close()
            return status, logtail
        except Exception:
            time.sleep(1)
    return None, None


def render_dashboard(status, logtail):
    lines = ['', '  ┌──────────────────────────────────────────────────────────────┐']
    if status and status != 'NO_STATUS':
        d = {}
        for l in status.split('\n'):
            if ':' in l:
                k, v = l.split(':', 1)
                d[k.strip()] = v.strip()
        phase = d.get('PHASE', '')
        if phase == 'DONE':
            lines.append(f'  │ ✅ DONE  {d.get("TAG","")}  best_ppl={d.get("BEST_PPL","")}  {d.get("TIME","")}    │')
        else:
            phase_disp = {'s^tau': 's^τ (fused+FP8)', 'softmax': 'softmax (对照)'}.get(phase, phase)
            lines.append(f'  │ 🚀 {phase_disp:<45s} │')
        lines.append(f'  ├──────────────────────────────────────────────────────────────┤')
        if d.get('GPU'): lines.append(f'  │ GPU: {d["GPU"]:<52s} │')
        if d.get('EPOCH'): lines.append(f'  │ Epoch: {d["EPOCH"]:<12s}  PPL: {d.get("PPL",""):<8s}  {d.get("TAU",""):<12s} │')
        if d.get('SPEED'): lines.append(f'  │ Speed: {d["SPEED"]:<8s}  Mem: {d.get("MEM",""):<6s}  FP8: {d.get("FP8",""):<5s}  {d.get("ELAPSED",""):<8s} │')
    else:
        lines.append(f'  │ ⏳ 等待训练启动...                                       │')
    lines.append(f'  └──────────────────────────────────────────────────────────────┘')

    if logtail and logtail != 'NO_LOG':
        lines.append(f'  ── 日志 ──')
        for l in logtail.split('\n')[-6:]:
            if l.strip():
                lines.append(f'  {l.strip()[:80]}')

    lines.append(f'  [{time.strftime("%H:%M:%S")}]')
    return '\n'.join(lines)


def main():
    print('█ 80M ATTH 训练监控', flush=True)
    print('█ Ctrl+C 退出\n', flush=True)
    last = None
    try:
        while True:
            status, logtail = fetch()
            cur = f'{status or ""}|{logtail or ""}'
            if cur != last:
                last = cur
                print('\033[2J\033[H', end='')
                print(render_dashboard(status, logtail), flush=True)
            else:
                sys.stdout.write('.'); sys.stdout.flush()
            time.sleep(5)
    except KeyboardInterrupt:
        print('\n退出', flush=True)


if __name__ == '__main__':
    main()
