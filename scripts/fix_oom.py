"""
█ fix_oom.py — SSH 进远端实例，修 OOM
█ 改动1: get_batch 自适应 batch（L≥1024→16, L≥768→32, else→64）
█ 改动2: 模型间 torch.cuda.empty_cache()
"""
import paramiko, re, requests

T = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': T, 'Content-Type': 'application/json'}

snap = requests.get('https://api.autodl.com/api/v1/dev/instance/pro/snapshot?instance_uuid=pro-77757062a9f9', headers=H, timeout=10)
sdi = snap.json()['data']
m = re.search(r'ssh -p (\d+) root@([\w.]+)', sdi['ssh_command'])
host, port = m.group(2), int(m.group(1))
pw = sdi['root_password']

ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(host, port=port, username='root', password=pw, timeout=15)

_, stdout, _ = ssh.exec_command('cat /root/epx/experiments/run_pillar.py')
script = stdout.read().decode()

# Patch 1: get_batch → adaptive batch
old_batch = '''def get_batch(split, blk, rng):
    d = train_t if split == 'train' else val_t
    ix = [rng.randint(0, len(d)-blk-1) for _ in range(64)]
    return torch.stack([d[i:i+blk] for i in ix]), torch.stack([d[i+1:i+blk+1] for i in ix])'''

new_batch = '''def get_batch(split, blk, rng):
    d = train_t if split == 'train' else val_t
    bs = 16 if blk >= 1024 else (32 if blk >= 768 else 64)
    ix = [rng.randint(0, len(d)-blk-1) for _ in range(bs)]
    return torch.stack([d[i:i+blk] for i in ix]), torch.stack([d[i+1:i+blk+1] for i in ix])'''

assert old_batch in script, 'get_batch not found!'
script = script.replace(old_batch, new_batch)

# Patch 2: add cache clear between s^tau models in main loop
old_loop = '''    # s^tau models
    for blk in scan_len:
        tag = 'dh64R_L{}'.format(blk)
        sep(); print('L-SCAN: dh64+RoPE L={}'.format(blk)); sep()
        r = train_one(42, blk, tag)
        results[tag] = r
        print('  DONE: tau={:.4f} ppl={:.1f} (@512={:.1f} @1024={:.1f}) {:.0f}s'.format(
            r['tau_final'], r['ppl'], r['eval_p512'], r['eval_p1024'], r['time']), flush=True)'''

new_loop = '''    # s^tau models
    for blk in scan_len:
        tag = 'dh64R_L{}'.format(blk)
        sep(); print('L-SCAN: dh64+RoPE L={}'.format(blk)); sep()
        r = train_one(42, blk, tag)
        results[tag] = r
        print('  DONE: tau={:.4f} ppl={:.1f} (@512={:.1f} @1024={:.1f}) {:.0f}s'.format(
            r['tau_final'], r['ppl'], r['eval_p512'], r['eval_p1024'], r['time']), flush=True)
        torch.cuda.empty_cache()'''

assert old_loop in script, 'main loop not found!'
script = script.replace(old_loop, new_loop)

# Patch 3: add cache clear between softmax models
old_soft_loop = '''    # softmax baselines
    print()
    for blk in scan_len:
        tag = 'soft_L{}'.format(blk)
        sep('-'); print('SOFTMAX baseline L={}'.format(blk)); sep('-')'''

new_soft_loop = '''    # softmax baselines
    print()
    for blk in scan_len:
        torch.cuda.empty_cache()
        tag = 'soft_L{}'.format(blk)
        sep('-'); print('SOFTMAX baseline L={}'.format(blk)); sep('-')'''

assert old_soft_loop in script, 'softmax loop not found!'
script = script.replace(old_soft_loop, new_soft_loop)

# Write back
sftp = ssh.open_sftp()
with sftp.open('/root/epx/experiments/run_pillar.py', 'w') as f:
    f.write(script)
sftp.close()
print('run_pillar.py patched ✅')

# Verify
_, stdout, _ = ssh.exec_command('grep -n "bs = " /root/epx/experiments/run_pillar.py')
print('Batch lines:', stdout.read().decode().strip())

_, stdout, _ = ssh.exec_command('grep -n "empty_cache" /root/epx/experiments/run_pillar.py')
print('Cache-clears:', stdout.read().decode().strip())

# Clear stderr
ssh.exec_command('> /root/epx/experiments/train.err')
print('Stderr cleared ✅')

# Check log current state
_, stdout, _ = ssh.exec_command('wc -l /root/epx/experiments/train.log')
print('Log lines:', stdout.read().decode().strip())

_, stdout, _ = ssh.exec_command('tail -3 /root/epx/experiments/train.log')
print('Log tail:', stdout.read().decode().strip())

ssh.close()
print('\nDone! Now deploy restart script.')
