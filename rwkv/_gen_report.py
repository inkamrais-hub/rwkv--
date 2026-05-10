import os, base64
b64 = open(r"F:T¦Ówkv\RWKV_STAU_TECHNICAL_REPORT.md.b64", "r").read()
data = base64.b64decode(b64).decode("utf-8")
with open(r"F:T¦Ówkv\RWKV_STAU_TECHNICAL_REPORT.md", "w", encoding="utf-8") as f:
    f.write(data)
print(f"Written {len(data)} chars")
