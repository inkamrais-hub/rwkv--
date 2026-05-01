"""Check conversations column format"""
import paramiko, io
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW, REMOTE_PYTHON

s = paramiko.SSHClient()
s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
s.connect(HOST, port=PORT, username='root', password=PW, timeout=15)

# Find a cached parquet file
_, o, _ = s.exec_command('ls /root/.cache/modelscope/hub/datasets/thomas/smoltalk-chinese/data/*.parquet 2>/dev/null | head -1')
fp = o.read().decode().strip()
print(f'File: {fp}')

test = f'''
import json, pyarrow.parquet as pq
df = pq.read_table("{fp}").to_pandas()
print(f"Cols: {{list(df.columns)}}")
conv = df.iloc[0]["conversations"]
print(f"Type: {{type(conv).__name__}}")
print(f"Repr: {{str(conv)[:500]}}")
if isinstance(conv, str):
    print("IS_STR")
    parsed = json.loads(conv)
    print(f"Parsed: {{type(parsed).__name__}} {{len(parsed)}} items")
    if isinstance(parsed, list):
        print(f"Item0 type: {{type(parsed[0]).__name__}}")
        print(f"Item0 keys: {{list(parsed[0].keys()) if isinstance(parsed[0], dict) else 'N/A'}}")
elif isinstance(conv, list):
    print("IS_LIST")
    print(f"Len: {{len(conv)}}")
    print(f"Item0 type: {{type(conv[0]).__name__}}")
    if isinstance(conv[0], dict):
        print(f"Item0 keys: {{list(conv[0].keys())}}")
    elif isinstance(conv[0], list):
        print(f"Item0 inner: {{str(conv[0])[:200]}}")
'''

sf = s.open_sftp()
sf.putfo(io.StringIO(test), '/root/_t_fmt.py')
sf.close()
_, o, _ = s.exec_command(f'{REMOTE_PYTHON} /root/_t_fmt.py 2>&1')
print(o.read().decode()[:1000])
s.close()
