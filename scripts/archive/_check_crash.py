import paramiko, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
_, o, _ = s.exec_command('tail -30 /root/tiny_train.log')
print(o.read().decode())
s.close()
