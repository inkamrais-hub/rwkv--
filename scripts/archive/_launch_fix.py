"""Deploy fp8 fix and relaunch training via SFTP"""
import paramiko, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

pkg_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'deploy_pkg.tar.gz')
remote = '/root/deploy_pkg.tar.gz'

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)

# 1. kill old training
_, o, _ = s.exec_command('pkill -f train_200m.py 2>/dev/null; sleep 1')
print('Killed old process.')

# 2. SFTP upload
sftp = s.open_sftp()
print(f'Uploading {pkg_path}...')
sftp.put(pkg_path, remote)
print('Upload done.')
sftp.close()

# 3. extract + verify
_, o, _ = s.exec_command(f'tar -xzf {remote} -C /root && rm {remote}')
time.sleep(1)
_, o, _ = s.exec_command('ls -la /root/fp8_utils.py /root/train_200m.py')
print(o.read().decode().strip())

# 4. restart
_, o, _ = s.exec_command('cd /root && nohup /root/miniconda3/bin/python -u train_200m.py > tiny_train.log 2>&1 & echo LAUNCHED:$!')
time.sleep(2)
_, o, _ = s.exec_command('ps aux | grep train_200m | grep -v grep')
print('Running:', o.read().decode().strip() or 'None')

s.close()
print('Done.')
