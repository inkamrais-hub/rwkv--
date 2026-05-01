"""harvest.py — 从远端下载实验结果 + 自动释放实例"""
import paramiko, requests, os, time, re, json

TOKEN = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': TOKEN, 'Content-Type': 'application/json'}
B = 'https://api.autodl.com'
SD = os.path.dirname(os.path.abspath(__file__))

r = requests.post(B + '/api/v1/dev/instance/pro/list', headers=H, json={}, timeout=10)
insts = r.json()['data']['list']
running = [i for i in insts if i.get('status') == 'running']
if not running:
    print('No running instances to harvest!')
    exit(0)

inst = running[0]
uid = inst['uuid']
snap = requests.get(B + '/api/v1/dev/instance/pro/snapshot?instance_uuid=' + uid, headers=H, timeout=10)
sdi = snap.json()['data']
m = re.search(r'ssh -p (\d+) root@([\w.]+)', sdi['ssh_command'])
host, port = m.group(2), int(m.group(1))
pw = sdi['root_password']

print('Harvesting: {} @ {}'.format(uid[:22], host))

# Find result directories on remote
ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username='root', password=pw, timeout=15)

_, stdout, _ = ssh.exec_command('ls -d /root/epx/experiments/results_* 2>/dev/null')
result_dirs = stdout.read().decode().strip().split('\n')
result_dirs = [d.strip() for d in result_dirs if d.strip()]

if not result_dirs:
    print('No result directories found!')
    ssh.close()
    exit(0)

sftp = ssh.open_sftp()
for rdir in result_dirs:
    name = os.path.basename(rdir)
    local = os.path.join(SD, name)
    os.makedirs(local, exist_ok=True)
    for fn in sftp.listdir(rdir):
        sftp.get(rdir + '/' + fn, os.path.join(local, fn))
        print('  DL: {}/{}'.format(name, fn))
sftp.close()

# Read and display
for rdir in result_dirs:
    name = os.path.basename(rdir)
    json_path = os.path.join(SD, name, name + '.json')
    alt_json = os.path.join(SD, name, name.replace('results_', '') + '.json')
    found = None
    for p in [json_path, alt_json]:
        if os.path.exists(p):
            found = p
            break
    if not found:
        # search for any json
        import glob
        jsons = glob.glob(os.path.join(SD, name, '*.json'))
        found = jsons[0] if jsons else None
    if found:
        with open(found) as f:
            data = json.load(f)
        print('\n=== {} ({}) ==='.format(name, found))
        if 'total_time' in data:
            total = data['total_time']
            cost = 1.88 * total / 3600
            print('  Time: {:.0f}s ({:.1f}min)  Cost: ~{:.2f} yuan'.format(total, total/60, cost))

ssh.close()

# Release
try:
    ssh2 = paramiko.SSHClient(); ssh2.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh2.connect(host, port=port, username='root', password=pw, timeout=10)
    ssh2.exec_command('shutdown now', timeout=10)
    ssh2.close()
    time.sleep(15)
except:
    pass
for _ in range(5):
    rr = requests.post(B + '/api/v1/dev/instance/pro/release', headers=H, json={'instance_uuid': uid}, timeout=10)
    if '成功' in rr.json().get('msg', '') or 'Success' in rr.json().get('msg', ''):
        print('\nInstance released ✅')
        break
    time.sleep(5)
