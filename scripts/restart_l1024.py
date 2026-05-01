"""
█ restart_l1024.py — 远端脚本已打好补丁，只需杀进程 + 重启
"""
import paramiko, re, requests, time

T = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': T, 'Content-Type': 'application/json'}

snap = requests.get('https://api.autodl.com/api/v1/dev/instance/pro/snapshot?instance_uuid=pro-77757062a9f9', headers=H, timeout=10)
sdi = snap.json()['data']
m = re.search(r'ssh -p (\d+) root@([\w.]+)', sdi['ssh_command'])
host, port = m.group(2), int(m.group(1))
pw = sdi['root_password']

ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username='root', password=pw, timeout=15)

# Kill old training
ssh.exec_command("ps aux | grep run_pillar | grep -v grep | awk '{print $2}' | xargs kill 2>/dev/null; echo done")
time.sleep(2)

# Clear stderr
ssh.exec_command('> /root/epx/experiments/train.err')

# Write run.sh
sftp = ssh.open_sftp()
run_sh = b'#!/bin/bash\ncd /root/epx/experiments\n/root/miniconda3/bin/python3 -u run_pillar.py 2>>train.err >>train.log\necho "EXIT CODE: $?" >>train.log\n'
with sftp.open('/root/epx/experiments/run.sh', 'wb') as f:
    f.write(run_sh)
sftp.close()
ssh.exec_command('chmod +x /root/epx/experiments/run.sh')

# Kill old screen, start new one
ssh.exec_command('screen -S pillar -X quit 2>/dev/null')
time.sleep(1)
ssh.exec_command('cd /root/epx/experiments && screen -dmS pillar bash run.sh')
time.sleep(4)

# Verify
_, stdout, _ = ssh.exec_command('ps aux | grep run_pillar | grep -v grep')
proc = stdout.read().decode().strip()
if proc:
    print('Process RUNNING')
else:
    print('WARNING: Process not found!')
    _, stdout, _ = ssh.exec_command('cat /root/epx/experiments/train.err')
    err = stdout.read().decode().strip()
    if err:
        print('stderr:', err[:300])

_, stdout, _ = ssh.exec_command('wc -l /root/epx/experiments/train.log')
print('Log lines:', stdout.read().decode().strip())

ssh.close()
print('Monitor: D:\\python\\python.exe -u scripts\\watch.py')
