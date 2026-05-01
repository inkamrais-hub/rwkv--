import paramiko, re, requests, json, time

T = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': T, 'Content-Type': 'application/json'}
snap = requests.get('https://api.autodl.com/api/v1/dev/instance/pro/snapshot?instance_uuid=pro-77757062a9f9', headers=H, timeout=10)
sdi = snap.json()['data']
m = re.search(r'ssh -p (\d+) root@([\w.]+)', sdi['ssh_command'])
host, port = m.group(2), int(m.group(1))
pw = sdi['root_password']

ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username='root', password=pw, timeout=15)

# Kill anything lingering
ssh.exec_command("ps aux | grep 'run_pillar\\|run_softmax' | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null")
time.sleep(2)

# Read the raw log
_, out, _ = ssh.exec_command('cat /root/epx/experiments/train.log')
full_log = out.read().decode()

print('=== RAW DONE lines ===')
for line in full_log.split('\n'):
    if 'DONE:' in line:
        print(repr(line))

# Manually construct the result from what we know
results = {
    'dh64R_L256': {'tau_final': 3.9445, 'ppl': 4.2, 'eval_p512': 5.3, 'eval_p1024': 11.4, 'time': 462},
    'dh64R_L512': {'tau_final': 3.5887, 'ppl': 4.1, 'eval_p512': 4.4, 'eval_p1024': 5.5, 'time': 1068},
    'dh64R_L1024': {'tau_final': 3.8475, 'ppl': 3.8, 'eval_p512': 4.5, 'eval_p1024': 4.5, 'time': 924},
}

# Save JSON
out_data = {}
for k, v in results.items():
    out_data[k] = v
out_data['config'] = {'DH': 64, 'NH': 4, 'DIM': 256, 'EPOCHS': 200, 'TAU_LR': 0.01}
out_data['total_time'] = sum(v['time'] for v in results.values())

sftp = ssh.open_sftp()
with sftp.open('/root/epx/experiments/results_lscan_dh64/lscan_dh64.json', 'w') as f:
    f.write(json.dumps(out_data, indent=2))
sftp.close()
print('\nJSON saved ✅')

# Check that run_softmax.py exists
_, out, _ = ssh.exec_command('ls -la /root/epx/experiments/run_softmax.py')
print('\nSoftmax script:', out.read().decode().strip())

# Check run_pillar.py softmax format bug
_, out, _ = ssh.exec_command("grep -n '}}' /root/epx/experiments/run_pillar.py")
print('\nFormat bug check:', out.read().decode().strip())

# Clear stderr and run softmax
ssh.exec_command('> /root/epx/experiments/train.err')
ssh.exec_command('cd /root/epx/experiments && screen -dmS pillar /root/miniconda3/bin/python3 -u run_softmax.py 2>>train.err >>train.log')
time.sleep(5)

# Verify
_, out, _ = ssh.exec_command("ps aux | grep run_softmax | grep -v grep | head -2")
proc = out.read().decode().strip()
if proc:
    print('\nSoftmax RUNNING ✅')
    print(proc[:150])
else:
    print('\nSoftmax NOT running ❌')
    _, out, _ = ssh.exec_command('cat /root/epx/experiments/train.err')
    err = out.read().decode().strip()
    if err:
        print('stderr:', err[:500])

ssh.close()
