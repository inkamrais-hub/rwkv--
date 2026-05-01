"""
█ harvest_hy.py — 恒源云结果下载 + 汇总显示
█
█ 用法: D:\python\python.exe scripts\harvest_hy.py
"""
import paramiko, os, json, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW, REMOTE_EPX_DIR, LOCAL_ASSETS

REMOTE_RESULTS = os.path.join(REMOTE_EPX_DIR, 'results_pro6000')
REMOTE_LOG = os.path.join(REMOTE_EPX_DIR, 'run_parallel.log')
LOCAL_RESULTS = os.path.join(LOCAL_ASSETS, 'results_pro6000')
os.makedirs(LOCAL_RESULTS, exist_ok=True)

def connect():
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
    return ssh

ssh = connect()
print(f'Connected to {HOST}:{PORT}')

# 1. Kill any running training
_, out, _ = ssh.exec_command(
    "ps aux | grep run_parallel | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null; echo killed")
print('Killed running process')

# 2. Download results
sftp = ssh.open_sftp()
try:
    result_files = sftp.listdir(REMOTE_RESULTS)
except:
    print(f'No {REMOTE_RESULTS} directory')
    result_files = []

print('\n=== Downloaded results ===')
for fn in result_files:
    if fn.endswith('.json'):
        sftp.get(os.path.join(REMOTE_RESULTS, fn), os.path.join(LOCAL_RESULTS, fn))
        size = os.path.getsize(os.path.join(LOCAL_RESULTS, fn))
        print(f'  {fn} ({size/1024:.0f} KB)')

# 3. Download master log
try:
    sftp.get(REMOTE_LOG, os.path.join(LOCAL_RESULTS, 'run_parallel.log'))
    print('  run_parallel.log')
except:
    print('  (no run_parallel.log)')
sftp.close()
ssh.close()

# 4. Parse and display
print('\n' + '=' * 65)
print('  恒源云 实验结果汇总')
print('=' * 65)

json_files = [f for f in os.listdir(LOCAL_RESULTS) if f.endswith('.json')]
for jf in sorted(json_files):
    with open(os.path.join(LOCAL_RESULTS, jf)) as f:
        d = json.load(f)
    n_models = sum(1 for k in d if k != 'total_time')
    elapsed = d.get('total_time', 0)
    print(f'\n--- {jf} ({n_models} models, {elapsed:.0f}s) ---')
    for key, v in sorted(d.items()):
        if key == 'total_time': continue
        tau = v.get('tau_final', '—')
        if tau == 0: tau = '—'
        ppl = v.get('ppl', '?')
        if isinstance(ppl, float): ppl = f'{ppl:.2f}'
        tag = v.get('tag', key)
        print(f'  {tag:<20s} τ={str(tau):>6s}  PPL={ppl}')
