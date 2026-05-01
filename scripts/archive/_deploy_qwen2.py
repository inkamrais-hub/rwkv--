"""收割当前结果 + 部署 Qwen2 新版本"""
import paramiko, time, sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW, REMOTE_PYTHON

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
def run(cmd):
    _, o, _ = s.exec_command(cmd)
    return o.read().decode().strip()

# ===== 1. Wait for completion =====
print('Waiting for training to finish...')
while True:
    o = run('tail -2 /root/tiny_train.log | cat')
    if 'ALL DONE' in o:
        print('  ✅ Training complete!')
        break
    if run('ps aux|grep train_200|grep -v grep|head -1'):
        time.sleep(30)
    else:
        print('  ⚠️ Process died, checking log...')
        break

# ===== 2. Download results =====
print('\nDownloading results...')
LOCAL_RESULTS = os.path.join(os.path.dirname(__file__), '..', 'project_assets', 'tiny_results')
os.makedirs(LOCAL_RESULTS, exist_ok=True)

sf = s.open_sftp()
files = ['result_softmax.json', 'result_s^tau.json', 'results_all.json',
         'model_softmax_best.pt', 'model_s^tau_best.pt']
for f in files:
    try:
        sf.get(f'/root/epx/tiny_results/{f}', os.path.join(LOCAL_RESULTS, f))
        sz = os.path.getsize(os.path.join(LOCAL_RESULTS, f)) // 1024
        print(f'  ✅ {f} ({sz}K)')
    except Exception as e:
        print(f'  ⚠️ {f}: {e}')
sf.close()

# Show results
try:
    with open(os.path.join(LOCAL_RESULTS, 'results_all.json')) as f:
        r = json.load(f)
    for k, v in r.items():
        t = {'learned': 's^tau', 'softmax': 'softmax'}.get(k, k)
        print(f'\n  {t}: best_ppl={v["best_ppl"]:.2f}  {v["time_s"]:.0f}s  params={v["params"]/1e6:.1f}M')
except: pass

# ===== 3. Deploy Qwen2 version =====
print('\n=== Deploying Qwen2 tokenizer version ===')
print('  Uploading...')
sf = s.open_sftp()
sf.put(os.path.join(os.path.dirname(__file__),'..','deploy_pkg.tar.gz'), '/root/deploy_pkg.tar.gz')
sf.close()
run('cd /root && rm -rf attention_mechanisms && tar xzf deploy_pkg.tar.gz')

print('  Installing deps...')
run('export HF_ENDPOINT=https://hf-mirror.com && /root/miniconda3/bin/pip install pyarrow pandas transformers modelscope -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1')

print('  Launching train_200m.py (Qwen2)...')
run('pkill -9 -f train_200 2>/dev/null; pkill -9 -f train_quick 2>/dev/null')
run('cd /root && nohup /root/miniconda3/bin/python -u train_200m.py > tiny_train.log 2>&1 &')
time.sleep(20)

p = run('ps aux|grep train_200|grep -v grep|grep python|head -1')
print(f'  Process: {p[:100] if p else "NONE!"}')
l = run('tail -4 /root/tiny_train.log')
print(f'  Log:\n{l[:500]}')

s.close()
print('\n✅ DONE')
