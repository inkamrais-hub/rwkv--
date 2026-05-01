import paramiko, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW, REMOTE_PYTHON
s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
def run(c):
    _, o, _ = s.exec_command(c)
    return o.read().decode().strip()
print('tiny_results:', run('ls /root/tiny_results/ 2>/dev/null || echo EMPTY'))
print('---')
# Check result files with glob
print(run('cat /root/tiny_results/result_softmax.json 2>/dev/null || echo NO_SOFTMAX'))
print('---')
print(run('find /root/tiny_results -name \"result*\" 2>/dev/null | head -5'))
print('---')
print(run('ls -la /root/tiny_results/ 2>&1'))
s.close()
