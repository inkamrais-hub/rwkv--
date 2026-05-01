import paramiko, re, requests, datetime

T = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': T, 'Content-Type': 'application/json'}

snap = requests.get('https://api.autodl.com/api/v1/dev/instance/pro/snapshot?instance_uuid=pro-77757062a9f9', headers=H, timeout=10)
sdi = snap.json()['data']
price = sdi.get('payg_price', 1880) / 1000
gpu_name = sdi.get('snapshot_gpu_alias_name', '?')
m = re.search(r'ssh -p (\d+) root@([\w.]+)', sdi['ssh_command'])
host, port = m.group(2), int(m.group(1))
pw = sdi['root_password']

started = '2026-04-30T12:48:41'
start_dt = datetime.datetime.strptime(started, '%Y-%m-%dT%H:%M:%S')
now_dt = datetime.datetime.now()
run_min = (now_dt - start_dt).total_seconds() / 60
run_cost = price * run_min / 60

ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username='root', password=pw, timeout=15)

_, out, _ = ssh.exec_command('cat /root/epx/experiments/train.log')
full_log = out.read().decode()

# parse sections
lines = full_log.strip().split('\n')
sections = {}
current = None
for line in lines:
    line = line.strip()
    if line.startswith('L-SCAN:'):
        current = line.split('L-SCAN:')[1].strip()
    elif 'DONE:' in line and current:
        for p in line.split():
            if p.endswith('s') and p[:-1].isdigit():
                sections[current] = int(p[:-1])
                break
        current = None

_, out, _ = ssh.exec_command("grep '^\\[dh64R_L' /root/epx/experiments/train.log | head -1")
first_ep = out.read().decode().strip()
_, out, _ = ssh.exec_command("grep '^\\[dh64R_L' /root/epx/experiments/train.log | awk 'END{print}'")
last_ep = out.read().decode().strip()
_, out, _ = ssh.exec_command('wc -l /root/epx/experiments/train.log')
log_lines = out.read().decode().strip()
_, out, _ = ssh.exec_command('cat /root/epx/experiments/train.err')
stderr = out.read().decode().strip()
_, out, _ = ssh.exec_command('nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader')
gpu = out.read().decode().strip()
_, out, _ = ssh.exec_command("ps aux | grep python | grep -v grep | grep -v tensorboard | grep -v jupyter")
proc = out.read().decode().strip()
ssh.close()

# extract current ep
ep_now = 0
if last_ep:
    epm = re.search(r'ep\s*(\d+)', last_ep)
    if epm:
        ep_now = int(epm.group(1))

# time estimates
l512_time = sections.get('dh64+RoPE L=512', 1068)
per_512ep = l512_time / 200
per_1024ep = per_512ep * 2.0
remaining_ep = 200 - ep_now
est_remaining = remaining_ep * per_1024ep

print('=== 实例状态 ===')
print(f'GPU: {gpu_name} | {gpu}')
print(f'运行时间: {run_min:.0f} 分钟 | 已花费: ¥{run_cost:.2f}')
print(f'费率: ¥{price}/h')
print(f'进程: {"RUNNING ✅" if proc else "DEAD ❌"}')

print('\n=== 已完成 (L-scan 实验) ===')
for k, v in sections.items():
    print(f'  {k}: {v//60}m{v%60:02d}s')

print('\n=== L=1024 进度 ===')
print(f'  最新: {last_ep}')
if ep_now > 0:
    print(f'  已跑: {ep_now}/200 ep ({ep_now/200*100:.0f}%)')
    print(f'  预计剩余: {est_remaining:.0f}s ({est_remaining/60:.1f} 分钟)')
    print(f'  预计费用: ¥{price * est_remaining / 3600:.2f}')

print(f'\n=== 日志 ===')
print(f'  行数: {log_lines}')
print(f'  stderr: {"空 ✅" if not stderr else stderr[:200]}')

print('\n=== 预算总览 ===')
print(f'  初始余额: ¥30')
print(f'  已用: ¥{run_cost + 4.77 + 3.40 + 2.0 + 0.6:.2f}')
left = 11.0 - price * est_remaining / 3600
print(f'  剩余: ~¥{left:.2f}')
