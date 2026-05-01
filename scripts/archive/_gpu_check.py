"""Check GPU + process status"""
import paramiko, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW
s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
def run(cmd):
    _, o, _ = s.exec_command(cmd)
    return o.read().decode().strip()
print('GPU:', run('nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader'))
print('Proc:', run('ps aux|grep train_200|grep -v grep|grep python|head -1')[:100])
print('Log:', run('tail -3 /root/tiny_train.log')[:400])
print('LN:', run('wc -l /root/tiny_train.log'))
s.close()
