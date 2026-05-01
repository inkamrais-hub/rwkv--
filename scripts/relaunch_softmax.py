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

# Kill everything
ssh.exec_command("ps aux | grep 'run_pillar\\|run_softmax' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null")
time.sleep(2)
ssh.exec_command('screen -S pillar -X quit 2>/dev/null')
time.sleep(1)
print('Killed old processes')

# Write run.sh via sftp
sftp = ssh.open_sftp()
run_sh = b'#!/bin/bash\ncd /root/epx/experiments\n/root/miniconda3/bin/python3 -u run_softmax.py 1>>train2.log 2>>train2.err\n'
with sftp.open('/root/epx/experiments/run2.sh', 'wb') as f:
    f.write(run_sh)
sftp.close()
ssh.exec_command('chmod +x /root/epx/experiments/run2.sh')
print('run2.sh written')

# Clear log files
ssh.exec_command('> /root/epx/experiments/train2.log')
ssh.exec_command('> /root/epx/experiments/train2.err')

# Start with nohup
ssh.exec_command('cd /root/epx/experiments && nohup bash run2.sh > /dev/null 2>&1 &')
time.sleep(3)

# Verify process started
_, out, _ = ssh.exec_command("ps aux | grep run_softmax | grep -v grep")
proc = out.read().decode().strip()
if proc:
    print('Softmax RUNNING')
    print('  PID:', proc[:100])
else:
    print('NOT RUNNING')
    _, out, _ = ssh.exec_command('cat /root/epx/experiments/train2.err')
    err = out.read().decode().strip()
    if err:
        print('  stderr:', err[:300])

# Wait and check
time.sleep(15)
_, out, _ = ssh.exec_command('wc -l /root/epx/experiments/train2.log')
print('\nLog lines:', out.read().decode().strip())
_, out, _ = ssh.exec_command('tail -5 /root/epx/experiments/train2.log')
print('Log tail:', out.read().decode().strip()[:300])
_, out, _ = ssh.exec_command('cat /root/epx/experiments/train2.err')
err = out.read().decode().strip()
if err:
    print('stderr:', err[:300])

_, out, _ = ssh.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader')
print('GPU:', out.read().decode().strip())

ssh.close()
