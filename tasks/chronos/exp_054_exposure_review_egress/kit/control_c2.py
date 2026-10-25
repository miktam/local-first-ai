#!/usr/bin/env python3
"""exp_054 control C2: acts the recorder must see, or the audit is void.
A name resolution and an HTTP request to example.com (IANA's reserved example domain), and a spawned dig for it."""
import subprocess
import urllib.request

try:
    urllib.request.urlopen("http://example.com/", timeout=10).read(200)
except Exception as e:
    print("http:", e)
subprocess.run(["/usr/bin/dig", "+short", "example.com"], capture_output=True, timeout=15)
print("control done")
