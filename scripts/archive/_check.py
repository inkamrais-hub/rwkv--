import paramiko, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=10)

# Kill training
s.exec_command('pkill -9 -f train_quick 2>/dev/null')

# Check env
_, o, _ = s.exec_command('python3 --version 2>&1 || /root/miniconda3/bin/python --version')
print('Python:', o.read().decode().strip())

_, o, _ = s.exec_command('/root/miniconda3/bin/python -c "import datasets; print(\'datasets:\', datasets.__version__)" 2>&1')
print(o.read().decode().strip()[:200])

_, o, _ = s.exec_command('/root/miniconda3/bin/python -c "import torch; print(\'torch:\', torch.__version__, \'cuda:\', torch.version.cuda)" 2>&1')
print(o.read().decode().strip()[:200])

# What data files are available?
_, o, _ = s.exec_command('ls -la /root/tiny_results/ 2>/dev/null')
print('Results:', o.read().decode().strip()[:300])

_, o, _ = s.exec_command('cat /root/tiny_results/result_softmax.json 2>/dev/null | head -20')
print('Last result:', o.read().decode().strip()[:300])

s.close()
