"""直接流式读取远端训练日志"""
import paramiko, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username='root', password=PW, timeout=10)
_, _, _ = ssh.exec_command('')  # prime connection

last = ''
try:
    while True:
        _, o, _ = ssh.exec_command('cat /root/epx/tiny_train.log 2>/dev/null')
        cur = o.read().decode()
        if cur and cur != last:
            new = cur[len(last):] if cur.startswith(last) else cur
            print(new, end='', flush=True)
            last = cur
        time.sleep(2)
except KeyboardInterrupt:
    print('\n退出')
    ssh.close()
