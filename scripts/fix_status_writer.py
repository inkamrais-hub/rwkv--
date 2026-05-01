import paramiko, time

HOST, PORT, PW = 'connect.westd.seetacloud.com', 12359, 'NPKKDRLuIdNS'
ssh = paramiko.SSHClient(); ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
ssh.connect(HOST, port=PORT, username='root', password=PW, timeout=15)

# Kill old status writer
ssh.exec_command("ps aux | grep write_status | grep -v grep | awk '{print $2}' | xargs kill -9 2>/dev/null")
time.sleep(1)

# Upload new version
sftp = ssh.open_sftp()
sftp.put('f:\\τ\\deploy_pkg\\write_status.py', '/root/epx/write_status.py')
sftp.close()
print('Uploaded new write_status.py')

# Restart
ssh.exec_command('cd /root/epx && nohup /root/miniconda3/bin/python3 -u write_status.py > /dev/null 2>&1 &')
time.sleep(3)

# Verify
_, out, _ = ssh.exec_command('cat /root/epx/status.txt')
print('\nStatus:')
print(out.read().decode().strip())

ssh.close()
