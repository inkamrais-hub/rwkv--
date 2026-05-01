"""一键部署 + 启动训练 (新实例)"""
import paramiko, time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW, REMOTE_PYTHON

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
def run(cmd):
    _, o, e = s.exec_command(cmd)
    return o.read().decode().strip(), e.read().decode().strip()

print('1/5 上传 deploy_pkg...')
sf = s.open_sftp()
sf.put(os.path.join(os.path.dirname(__file__),'..','deploy_pkg.tar.gz'), '/root/deploy_pkg.tar.gz')
sf.close()
print('2/5 解压...')
run('cd /root && rm -rf attention_mechanisms && tar xzf deploy_pkg.tar.gz')
run('ls /root/train_quick.py /root/model_tiny.py /root/fp8_utils.py')
print('3/5 安装 modelscope + 降级 datasets...')
o,_ = run(f'{REMOTE_PYTHON} -m pip install datasets==2.21.0 modelscope -q -i https://pypi.tuna.tsinghua.edu.cn/simple 2>&1')
print(f'  pip done')
print('4/5 启动训练...')
run('pkill -9 -f train_quick 2>/dev/null')
run('cd /root && nohup /root/miniconda3/bin/python -u train_quick.py > tiny_train.log 2>&1 &')
time.sleep(12)
o,_ = run('pgrep -af train_quick | grep python')
print(f'  进程: {o[:200] if o else "NONE"}')
o,_ = run('tail -8 /root/tiny_train.log')
print(f'  日志:\n{o[:600]}')
o,_ = run('cat /root/status_tiny.txt 2>/dev/null || echo NS')
print(f'  状态: {o[:200]}')
s.close()
print('\n5/5 ✅ 部署完成! 启动监控: D:\\python\\python.exe -u scripts\\watch_tiny.py')
