"""exp_054: record every network-relevant act of this Python process. Active only when AUDIT54_LOG is set.

Python loads sitecustomize from PYTHONPATH at start-up (not under -I or -S), so a child Python that inherits the
environment records itself too. Recorded, one JSON line each:
  audit        the interpreter's own audit events for sockets, HTTP, URL requests and spawned programs
  getaddrinfo  every name resolution, with the addresses it returned (what the resolver was asked)
"""
import json
import os
import socket
import sys
import time

_LOG = os.environ.get("AUDIT54_LOG")
if _LOG:
    _fh = open(_LOG, "a", buffering=1, encoding="utf-8")
    _PID = os.getpid()
    _EVENTS = {"socket.connect", "socket.sendto", "socket.sendmsg", "socket.bind", "socket.getaddrinfo",
               "socket.gethostbyname", "socket.gethostbyaddr", "socket.getnameinfo", "urllib.Request",
               "http.client.connect", "http.client.send", "subprocess.Popen", "os.system", "os.posix_spawn",
               "os.exec", "os.spawn", "os.startfile", "ftplib.connect", "smtplib.connect", "imaplib.open",
               "poplib.connect", "nntplib.connect", "telnetlib.Telnet.open", "webbrowser.open"}

    def _write(kind, **kw):
        _fh.write(json.dumps({"t": time.time(), "pid": _PID, "argv0": sys.argv[0] if sys.argv else "",
                              "kind": kind, **kw}, default=repr) + "\n")

    def _hook(event, args):
        if event in _EVENTS:
            if event == "http.client.send":          # payload bytes: record only the request line
                first = bytes(args[1][:200]).split(b"\r\n", 1)[0] if len(args) > 1 else b""
                _write("audit", event=event, args=[first.decode("latin-1")])
            else:
                _write("audit", event=event, args=[repr(a)[:800] for a in args])

    sys.addaudithook(_hook)
    _orig_gai = socket.getaddrinfo

    def _gai(host, port, *a, **k):
        try:
            res = _orig_gai(host, port, *a, **k)
        except Exception as e:
            _write("getaddrinfo", host=host, port=port, error=repr(e))
            raise
        _write("getaddrinfo", host=host, port=port, addrs=sorted({r[4][0] for r in res}))
        return res

    socket.getaddrinfo = _gai
