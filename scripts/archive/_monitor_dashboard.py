"""Local web dashboard for training monitoring"""
import json, os, sys, threading, time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime

import paramiko
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hy_config import HOST, PORT, PW

STATUS = {"updated": "waiting...", "gpu": "", "log": "", "error": ""}

def poll_remote():
    global STATUS
    while True:
        try:
            s = paramiko.SSHClient()
            s.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            s.connect(HOST, port=PORT, username="root", password=PW, timeout=10)

            _, o, _ = s.exec_command(
                'gpu=$(nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits 2>/dev/null)'
                '; echo "GPU:$gpu"'
                '; echo "LOGTAIL:"; tail -8 /root/tiny_train.log 2>/dev/null'
                '; echo "LN:"; wc -l /root/tiny_train.log 2>/dev/null'
                '; echo "PS:"; ps aux | grep python | grep -v grep | head -3'
            )
            time.sleep(1)
            out = o.read().decode().strip()
            s.close()

            STATUS = {
                "updated": datetime.now().strftime("%H:%M:%S"),
                "gpu": "",
                "log": out,
                "error": ""
            }
        except Exception as e:
            STATUS = {**STATUS, "updated": datetime.now().strftime("%H:%M:%S"), "error": str(e)}
        time.sleep(5)

class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            s = STATUS
            html = f"""<!DOCTYPE html><html><head>
<meta charset="utf-8"><meta http-equiv="refresh" content="5">
<title>τ Training Monitor</title>
<style>
* {{ margin:0; padding:0; box-sizing:border-box }}
body {{ font-family:'Cascadia Code','Fira Code','Consolas',monospace; background:#0d1117; color:#c9d1d9; padding:20px }}
h1 {{ color:#58a6ff; font-size:20px; margin-bottom:12px }}
.box {{ background:#161b22; border:1px solid #30363d; border-radius:8px; padding:16px; margin-bottom:12px; white-space:pre-wrap; font-size:14px; line-height:1.6 }}
.label {{ color:#8b949e; font-size:12px }}
.updated {{ color:#58a6ff; font-size:12px; margin-bottom:8px }}
.error {{ color:#f85149; background:#161b22; border:1px solid #f85149; border-radius:8px; padding:12px; margin-bottom:12px }}
.green {{ color:#3fb950 }}
.yellow {{ color:#d29922 }}
</style></head><body>
<h1>τ · Training Monitor</h1>
<div class="updated">updated: {s['updated']} | auto-refresh 5s</div>
"""
            if s["error"]:
                html += f'<div class="error">SSH Error: {s["error"]}</div>'
            html += f'<div class="box">{s["log"]}</div>'
            html += "</body></html>"
            self.wfile.write(html.encode())
        else:
            self.send_response(404)
            self.end_headers()

if __name__ == "__main__":
    t = threading.Thread(target=poll_remote, daemon=True)
    t.start()
    port = 8899
    print(f"Dashboard: http://localhost:{port}")
    HTTPServer(("0.0.0.0", port), Handler).serve_forever()
