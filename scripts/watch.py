"""
█ watch.py — 持续监控 AutoDL 训练进度
█ 用法:
█   持续监控: D:\python\python.exe -u scripts\watch.py
█   单次检查: D:\python\python.exe -u scripts\watch.py --once
█                        D:\python\python.exe -u scripts\watch.py --once --interval 5
"""
import paramiko, requests, re, time, argparse

T = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': T, 'Content-Type': 'application/json'}
B = 'https://api.autodl.com'

def log(msg):
    print(msg, flush=True)

def find_instance():
    r = requests.post(B + '/api/v1/dev/instance/pro/list', headers=H, json={}, timeout=10)
    insts = r.json()['data']['list']
    running = [i for i in insts if i.get('status') == 'running']
    if not running:
        return None
    uid = running[0]['uuid']
    snap = requests.get(B + '/api/v1/dev/instance/pro/snapshot?instance_uuid=' + uid, headers=H, timeout=10)
    sdi = snap.json()['data']
    m = re.search(r'ssh -p (\d+) root@([\w.]+)', sdi['ssh_command'])
    return {
        'uid': uid, 'host': m.group(2), 'port': int(m.group(1)),
        'pw': sdi['root_password'], 'gpu': sdi.get('snapshot_gpu_alias_name', '?'),
        'price': sdi.get('payg_price', 0) / 1000
    }

def read_remote(info):
    try:
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(info['host'], port=info['port'], username='root', password=info['pw'], timeout=10)

        _, stdout, _ = ssh.exec_command('wc -l /root/epx/experiments/train.log 2>/dev/null')
        wc_out = stdout.read().decode().strip()
        lines = int(wc_out.split()[0]) if wc_out else 0

        _, stdout, _ = ssh.exec_command('ps aux | grep python | grep -v grep | grep -v tensorboard | grep -v jupyter')
        python_procs = stdout.read().decode().strip()
        proc = 'run_pillar' if 'run_pillar' in python_procs else ''
        proc_detail = python_procs[:200] if python_procs else ''

        _, stdout, _ = ssh.exec_command('tail -30 /root/epx/experiments/train.log 2>/dev/null')
        tail = stdout.read().decode().strip()

        _, stdout, _ = ssh.exec_command('tail -10 /root/epx/experiments/train.err 2>/dev/null')
        err = stdout.read().decode().strip()

        _, stdout, _ = ssh.exec_command(
            "nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader 2>/dev/null"
        )
        gpu_info = stdout.read().decode().strip()

        ssh.close()
        return lines, proc, tail, err, gpu_info
    except Exception as e:
        return -1, None, '', str(e)[:80], ''

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--once', action='store_true')
    parser.add_argument('--interval', type=int, default=60)
    args = parser.parse_args()

    log('watch.py  Interval:%ds  Ctrl+C to stop' % args.interval)
    if args.once:
        log('  (once mode)')

    last_lines = 0
    dead_ticks = 0
    seen_sections = set()
    last_section = ''

    for tick in range(1, 99999):
        info = find_instance()
        if not info:
            log('[tick %d] No running instances.' % tick)
            break

        if tick == 1:
            log('%s %s %.2f/h' % (info['uid'][:24], info['gpu'], info['price']))

        lines, proc, tail, err, gpu = read_remote(info)

        if lines < 0:
            log('[tick %d] SSH error: %s' % (tick, err))
            if args.once:
                break
            time.sleep(args.interval)
            continue

        # Detect current experiment section from log tail
        current_section = ''
        if tail:
            for line in tail.split('\n'):
                line = line.strip()
                if line.startswith('L-SCAN:'):
                    current_section = line
                    if line not in seen_sections:
                        seen_sections.add(line)

        # GPU info line
        gpu_str = ' | GPU: ' + gpu if gpu else ''
        if not gpu:
            gpu_str = ' | GPU: N/A'

        # Detect crash via stderr
        crashed = False
        crash_reason = ''
        if err:
            if 'OutOfMemoryError' in err:
                crashed = True
                crash_reason = 'OOM'
            elif 'Error' in err or 'Traceback' in err:
                crashed = True
                crash_reason = 'CRASH'
            err_lines = [l for l in err.split('\n') if l.strip()]
            if err_lines:
                for el in err_lines[-3:]:
                    if 'OutOfMemoryError' in el:
                        log('  OOM! ' + el.strip()[:120])
                    elif 'Error' in el:
                        log('  ERROR: ' + el.strip()[:120])

        # Show new log lines
        new_output = False
        if lines > last_lines and tail:
            new_count = min(lines - last_lines, 30)
            tail_lines = tail.split('\n')
            for l in tail_lines[-new_count:]:
                l = l.strip()
                if l:
                    log('  ' + l)
                    new_output = True
            last_lines = lines
            dead_ticks = 0

        # Determine status
        if crashed and not proc:
            log('[tick %d] CRASHED: %s' % (tick, crash_reason))
            break

        if not proc:
            dead_ticks += 1
            finished = ('VERDICT' in tail or 'Total time' in tail or '[SAVE]' in tail)
            all_done = all('DONE:' in tail for section in seen_sections)
            if finished or all_done:
                if dead_ticks >= 2:
                    log('[tick %d] Training finished!' % tick)
                    break
                else:
                    if dead_ticks == 1:
                        log('[tick %d] Training done, waiting for save...' % tick)
            elif dead_ticks == 1:
                log('[tick %d] Process restarting (transition between experiments)...' % tick)
            elif dead_ticks >= 3:
                if err:
                    log('[tick %d] Process dead after %d ticks. stderr has errors.' % (tick, dead_ticks))
                else:
                    log('[tick %d] Process dead for %d ticks, no stderr. May be idle.' % (tick, dead_ticks))
                if args.once:
                    log('[tick %d] Status: %s | %d log lines%s' % (
                        tick, 'CRASHED' if err else 'DEAD', lines, gpu_str))
                    break
        else:
            dead_ticks = 0
            if current_section and current_section != last_section:
                log('[tick %d] RUNNING: %s | log=%d%s' % (tick, current_section, lines, gpu_str))
                last_section = current_section

        if tick % 10 == 0 and proc:
            log('[tick %d] RUNNING | log=%d lines%s' % (tick, lines, gpu_str))

        # Special: if log has stopped growing for 5+ ticks but process is still there,
        # might be stuck in OOM loop
        if not new_output and proc and tick > 3 and dead_ticks == 0:
            if err:
                log('[tick %d] WARN: Log not growing but process running. Check stderr.' % tick)

        if args.once:
            if not proc:
                if not seen_sections:
                    log('[tick %d] Status: WAITING (no sections started yet)' % tick)
                else:
                    last_sec = max(seen_sections) if seen_sections else '?'
                    log('[tick %d] Status: WAITING (last done: %s) | log=%d lines%s' % (
                        tick, last_sec, lines, gpu_str))
            else:
                state = current_section if current_section else 'running'
                log('[tick %d] Status: RUNNING %s | log=%d lines%s' % (tick, state, lines, gpu_str))
            break

        time.sleep(args.interval)

if __name__ == '__main__':
    main()
