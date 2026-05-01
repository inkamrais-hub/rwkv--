"""上传 + 启动 Qwen2 版训练 (不等待)"""
import paramiko, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)

print('Upload...')
sf = s.open_sftp()
sf.put(os.path.join(os.path.dirname(__file__),'..','deploy_pkg.tar.gz'), '/root/deploy_pkg.tar.gz')
sf.close()
s.exec_command('cd /root && rm -rf attention_mechanisms && tar xzf deploy_pkg.tar.gz')

print('Launch...')
s.exec_command('pkill -9 -f train_200 2>/dev/null; pkill -9 -f train_quick 2>/dev/null')
s.exec_command('cd /root && nohup /root/miniconda3/bin/python -u train_200m.py > tiny_train.log 2>&1 &')
time.sleep(15)

_, o, _ = s.exec_command('ps aux|grep train_200|grep -v grep|grep python|head -1')
print('Proc:', o.read().decode()[:150])
_, o, _ = s.exec_command('tail -3 /root/tiny_train.log')
print('Log:', o.read().decode()[:500])

s.close()
print('DONE')
