import paramiko, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW
d = os.path.join(os.path.dirname(__file__), '..', 'project_assets', 'tiny_results')
os.makedirs(d, exist_ok=True)
s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)
sf = s.open_sftp()
files = ['results_all.json','result_softmax.json','result_s^tau.json','model_softmax_best.pt','model_s^tau_best.pt']
for f in files:
    sf.get(f'/root/epx/tiny_results/{f}', os.path.join(d, f))
    sz = os.path.getsize(os.path.join(d, f)) // 1024
    print(f'  {f}: {sz}K')
sf.close()
s.close()
print('OK')
