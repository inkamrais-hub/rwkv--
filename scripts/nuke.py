"""nuke.py — 释放所有 AutoDL 实例（关机+清理）"""
import requests, time

T = 'eyJhbGciOiJFUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjk4Nzk5NiwidXVpZCI6IjlmMTQ0YjNmNmYyZTBiMDAiLCJ0ZW5hbnQiOiJhdXRvZGwiLCJhdWQiOiJkZXZlbG9wX2FwaSJ9.uDBu0H2YsGIvmf9uXes-n6cigCGhxBYZX_9pBvTpgOEnbODfynJdk3W5jM0PxGU5hDcnKXwRvGrjZMbM44_iFQ'
H = {'Authorization': T, 'Content-Type': 'application/json'}
B = 'https://api.autodl.com'

r = requests.post(B + '/api/v1/dev/instance/pro/list', headers=H, json={}, timeout=10)
insts = r.json()['data']['list']
print('Instances: ' + str(len(insts)))

for i in insts:
    uid = i['uuid']; st = i['status']
    print('  ' + uid[:20] + ' ' + st)
    if st == 'shutdown':
        for _ in range(5):
            rr = requests.post(B + '/api/v1/dev/instance/pro/release', headers=H, json={'instance_uuid': uid}, timeout=10)
            print('  release: ' + rr.json().get('msg', ''))
            if '成功' in rr.json().get('msg', '') or 'Success' in rr.json().get('msg', ''):
                break
            time.sleep(5)

r2 = requests.post(B + '/api/v1/dev/instance/pro/list', headers=H, json={}, timeout=10)
print('Remaining: ' + str(len(r2.json()['data']['list'])))
if len(r2.json()['data']['list']) == 0:
    print('ALL CLEAN')
