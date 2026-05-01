import paramiko, os
from hy_config import HOST, PORT, PW
s=paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST,port=PORT,username='root',password=PW,timeout=10)
def run(c):
    _,o,_=s.exec_command(c)
    return o.read().decode().strip()
print('model_weights:',run('find /root -maxdepth 3 -name \"*.pt\" 2>/dev/null||echo none'))
print('result_jsons:',run('find /root -maxdepth 3 -name \"*.json\" -not -path \"*/miniconda3/*\" 2>/dev/null||echo none'))
print('epx_dir:',run('ls /root/epx/ 2>/dev/null||echo no_epx'))
s.close()
