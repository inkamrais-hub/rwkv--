"""
█ harvest_and_release.py — 从旧 4090D 下载结果，然后释放实例
"""
import paramiko, requests, os, re, json, time, shutil

T = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': T, 'Content-Type': 'application/json'}
B = 'https://api.autodl.com'

SD = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ASSETS = os.path.join(SD, 'project_assets')
os.makedirs(ASSETS, exist_ok=True)

# 1. Get old instance info
r = requests.post(B + '/api/v1/dev/instance/pro/list', headers=H, json={}, timeout=10)
insts = r.json()['data']['list']
running = [i for i in insts if i.get('status') == 'running']

if not running:
    print('No running instances to harvest!')
else:
    inst = running[0]
    uid = inst['uuid']
    snap = requests.get(B + '/api/v1/dev/instance/pro/snapshot?instance_uuid=' + uid, headers=H, timeout=10)
    sdi = snap.json()['data']
    m = re.search(r'ssh -p (\d+) root@([\w.]+)', sdi['ssh_command'])
    host, port = m.group(2), int(m.group(1))
    pw = sdi['root_password']

    print('Harvesting: {}'.format(uid[:22]))

    ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(host, port=port, username='root', password=pw, timeout=15)

    # 2. Download results
    result_dirs = ['results_pillar_long', 'results_tau_scan', 'results_lscan_dh64']
    sftp = ssh.open_sftp()
    for rdir in result_dirs:
        remote = '/root/epx/experiments/' + rdir
        local = os.path.join(ASSETS, rdir)
        os.makedirs(local, exist_ok=True)
        try:
            for fn in sftp.listdir(remote):
                sftp.get(remote + '/' + fn, os.path.join(local, fn))
                print('  DL: {}/{}'.format(rdir, fn))
        except:
            print('  {}: not found'.format(rdir))
    sftp.close()

    # 3. Also get the full raw logs
    _, out, _ = ssh.exec_command('cat /root/epx/experiments/train.log')
    log1 = out.read().decode()
    with open(os.path.join(ASSETS, 'train_lscan.log'), 'w') as f:
        f.write(log1)
    print('  DL: train_lscan.log ({} chars)'.format(len(log1)))

    _, out, _ = ssh.exec_command('cat /root/epx/experiments/train2.log')
    log2 = out.read().decode()
    if log2.strip():
        with open(os.path.join(ASSETS, 'train_softmax.log'), 'w') as f:
            f.write(log2)
        print('  DL: train_softmax.log ({} chars)'.format(len(log2)))

    # 4. Display summary
    lscan_json = os.path.join(ASSETS, 'results_lscan_dh64', 'lscan_dh64.json')
    if os.path.exists(lscan_json):
        with open(lscan_json) as f:
            data = json.load(f)
        print('\n=== L-SCAN RESULTS ===')
        for k, v in data.items():
            if k in ('config', 'total_time', 'tau_history'):
                if k == 'total_time': print('  total_time: {:.0f}s'.format(v))
                continue
            if isinstance(v, dict):
                tf = v.get('tau_final', v.get('ppl', '?'))
                ppl = v.get('ppl', '?')
                eval512 = v.get('eval', v.get('eval_p512', v.get('eval', {}).get(512, '?')))
                if isinstance(eval512, dict): eval512 = eval512.get(512, '?')
                print('  {:15s}  ppl={}  tau={}  @512={}'.format(k, ppl, v.get('tau_final','?'), eval512))

    ssh.close()

    # 5. Release old instance
    print('\n--- Releasing old instance ---')
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
        msg = rr.json().get('msg', '')
        if '成功' in msg or 'Success' in msg:
            print('Old instance released')
            break
        time.sleep(5)

print('\nDone! Results in: {}'.format(ASSETS))
