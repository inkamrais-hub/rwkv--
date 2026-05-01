"""
█ deploy_hy.py — 打包代码 + 上传恒源云 + 启动训练
█
█ 用法: D:\python\python.exe scripts\deploy_hy.py
█       默认启动 run_parallel.py
█       可指定启动命令: D:\python\python.exe scripts\deploy_hy.py "python3 xxx.py"
"""
import paramiko, os, tarfile, time, sys
from hy_config import *

def make_package():
    print('Creating tar.gz...')
    if os.path.exists(LOCAL_PAYLOAD):
        os.remove(LOCAL_PAYLOAD)
    with tarfile.open(LOCAL_PAYLOAD, 'w:gz') as tar:
        tar.add(LOCAL_PKG_DIR, arcname='epx')
    pkg_size = os.path.getsize(LOCAL_PAYLOAD) // 1024
    print(f'Package: {pkg_size} KB')
    return pkg_size

def connect():
    print(f'Connecting to {HOST}:{PORT} ...')
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
    print('SSH OK')
    return ssh

def upload(ssh):
    print('Uploading deploy_pkg.tar.gz ...')
    sftp = ssh.open_sftp()
    sftp.put(LOCAL_PAYLOAD, f'{REMOTE_HOME}/epx.tar.gz')
    sftp.close()
    print('Upload done')

def extract(ssh):
    print('Extracting...')
    _, out, _ = ssh.exec_command(
        f'cd {REMOTE_HOME} && rm -rf epx && tar xzf epx.tar.gz && chmod +x {REMOTE_EPX_DIR}/run_parallel.py')
    time.sleep(2)
    _, out, _ = ssh.exec_command(f'ls {REMOTE_EPX_DIR}/run_parallel.py {REMOTE_EPX_DIR}/attention_mechanisms/attention.py')
    files = out.read().decode().strip()
    print('Files:', files[:200])

def verify_torch(ssh):
    _, out, _ = ssh.exec_command(f'{REMOTE_PYTHON} -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"')
    torch_test = out.read().decode().strip()
    print('Torch:', torch_test)
    return torch_test

def start_training(ssh, cmd=None):
    if cmd is None:
        cmd = f'cd {REMOTE_EPX_DIR} && nohup {REMOTE_PYTHON} -u run_parallel.py > run_parallel.log 2>&1 &'
    print(f'Starting: {cmd[:80]}...')
    ssh.exec_command(cmd)
    time.sleep(5)
    _, out, _ = ssh.exec_command('ps aux | grep run_parallel | grep -v grep')
    proc = out.read().decode().strip()
    if proc:
        print('RUNNING ✅ | PID:', proc[:100])
    else:
        print('NOT RUNNING ❌')
        _, out, _ = ssh.exec_command(f'cat {REMOTE_EPX_DIR}/run_parallel.log | tail -20')
        log = out.read().decode().strip()
        if log:
            print('Last log lines:', log[-500:])

def deploy_status_writer(ssh):
    print('Deploying status writer...')
    sftp = ssh.open_sftp()
    local_sw = os.path.join(SD, 'deploy_pkg', 'write_status.py')
    sftp.put(local_sw, f'{REMOTE_EPX_DIR}/write_status.py')
    sftp.close()
    ssh.exec_command(
        f"ps aux | grep write_status | grep -v grep | awk '{{print $2}}' | xargs kill -9 2>/dev/null")
    time.sleep(1)
    ssh.exec_command(
        f'cd {REMOTE_EPX_DIR} && nohup {REMOTE_PYTHON} -u write_status.py > /dev/null 2>&1 &')
    time.sleep(3)
    _, out, _ = ssh.exec_command('ps aux | grep write_status | grep -v grep')
    proc = out.read().decode().strip()
    print('Status writer:', 'RUNNING ✅' if proc else 'NOT RUNNING ❌')

if __name__ == '__main__':
    train_cmd = sys.argv[1] if len(sys.argv) > 1 else None

    make_package()
    ssh = connect()

    try:
        upload(ssh)
        extract(ssh)
        verify_torch(ssh)
        deploy_status_writer(ssh)
        start_training(ssh, train_cmd)
        print(f'\nMonitor: ssh -p {PORT} root@{HOST}')
        print(f'Then: tail -f {REMOTE_EPX_DIR}/run_parallel.log')
        print(f'Locally: D:\\python\\python.exe scripts\\watch_hy.py')
    finally:
        ssh.close()
