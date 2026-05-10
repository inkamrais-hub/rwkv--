import os, base64
tau_dir = None
for d in os.listdir(chr(70)+chr(58)+chr(47)):
    if len(d) == 1 and ord(d) == 964:
        tau_dir = d
        break
target = os.path.join(chr(70)+chr(58)+chr(47), tau_dir, chr(114)+chr(119)+chr(107)+chr(118), chr(114)+chr(117)+chr(110)+chr(95)+chr(116)+chr(101)+chr(115)+chr(116)+chr(95)+chr(99)+chr(108)+chr(111)+chr(117)+chr(100)+chr(46)+chr(112)+chr(121))
# The b64 content will be appended
B64 = chr(34) + chr(34)  # placeholder
