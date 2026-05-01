import paramiko, io, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW, REMOTE_PYTHON

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
def run(cmd):
    _, o, e = s.exec_command(cmd)
    return o.read().decode().strip(), e.read().decode().strip()

# Upload package
print('Uploading...')
sf = s.open_sftp()
sf.put(os.path.join(os.path.dirname(__file__), '..', 'deploy_pkg.tar.gz'), '/root/deploy_pkg.tar.gz')
sf.close()
run('cd /root && rm -rf attention_mechanisms && tar xzf deploy_pkg.tar.gz')

# Kill old, Start training
run('pkill -9 -f train_quick 2>/dev/null')
time.sleep(1)
run('cd /root && nohup /root/miniconda3/bin/python -u train_quick.py > tiny_train.log 2>&1 &')
time.sleep(25)

_, o, _ = s.exec_command('ps aux | grep train_quick | grep -v grep | head -2')
p = o.read().decode().strip()
print(f'Proc: {p[:150] if p else "NONE"}')
_, o, _ = s.exec_command('tail -8 /root/tiny_train.log')
print(f'Log:\n{o.read().decode()[:800]}')
_, o, _ = s.exec_command('cat /root/status_tiny.txt 2>/dev/null || echo NS')
print(f'Status: {o.read().decode()[:200]}')
s.close()
