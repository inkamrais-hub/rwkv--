import paramiko, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)

def run(cmd):
    _, o, e = s.exec_command(cmd)
    time.sleep(0.5)
    return o.read().decode().strip(), e.read().decode().strip()

o, _ = run('ps -p 9170 -o pid,state,pcpu,rss --no-headers 2>/dev/null || echo DEAD')
print(f'PID 9170: {o}')
o, _ = run('wc -l /root/tiny_train.log')
print(f'Log size: {o}')
o, _ = run('tail -6 /root/tiny_train.log')
print(f'---\n{o}')

s.close()
