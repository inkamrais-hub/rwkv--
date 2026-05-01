"""
█ check.py — 查看 AutoDL 远端实例状态 + 训练日志
█ 用法: D:\python\python.exe scripts\check.py
"""
import requests, paramiko, re

T = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': T, 'Content-Type': 'application/json'}
B = 'https://api.autodl.com'

r = requests.post(B + '/api/v1/dev/instance/pro/list', headers=H, json={}, timeout=10)
insts = r.json()['data']['list']
print('Instances: ' + str(len(insts)))

running = [i for i in insts if i.get('status') == 'running']
if not running:
    print('No running instances!')
    for i in insts:
        if i.get('status') == 'shutdown':
            print('  Releasing shutdown: ' + i['uuid'][:20])
            requests.post(B + '/api/v1/dev/instance/pro/release', headers=H, json={'instance_uuid': i['uuid']}, timeout=10)
    exit(0)

for inst in running:
    uid = inst['uuid']
    snap = requests.get(B + '/api/v1/dev/instance/pro/snapshot?instance_uuid=' + uid, headers=H, timeout=10)
    sdi = snap.json()['data']
    price = sdi.get('payg_price', 0) / 1000
    gpu = sdi.get('snapshot_gpu_alias_name', '?')
    print('  ' + uid[:22] + ' ' + gpu + ' ' + str(price) + 'yuan/h')

    ssh_cmd = sdi['ssh_command']
    pw = sdi['root_password']
    m = re.search(r'ssh -p (\d+) root@([\w.]+)', ssh_cmd)
    if not m: continue
    host, port = m.group(2), int(m.group(1))

    try:
        ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh.connect(host, port=port, username='root', password=pw, timeout=10)
        stdin, stdout, stderr = ssh.exec_command('cat /root/epx/experiments/train.log 2>/dev/null')
        log = stdout.read().decode().strip()
        if log:
            lines = log.split('\n')
            print('  LOG({} lines):'.format(len(lines)))
            for l in lines[-8:]:
                if l.strip(): print('    ' + l.strip())
        else:
            print('  LOG empty')
        stdin, stdout, stderr = ssh.exec_command('ps aux | grep run_pillar | grep -v grep')
        proc = stdout.read().decode().strip()
        print('  Process: ' + ('RUNNING' if proc else 'DEAD'))
        if not proc:
            stdin, stdout, stderr = ssh.exec_command('cat /root/epx/experiments/train.err 2>/dev/null')
            err = stdout.read().decode().strip()
            if err: print('  STDERR: ' + err[:300])
        ssh.close()
    except Exception as e:
        print('  SSH failed: ' + str(e)[:80])
