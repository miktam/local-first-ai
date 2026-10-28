#!/usr/bin/env python3
"""exp_057 positive control: make exactly one outbound connection to a non-loopback host, so the watcher must see it.

If the Ollama-tree watcher is the thing running, this control is run as a *separate* known PID that the watcher is
pointed at for the control window. It proves a real phone-home would not slip past the sampler. Benign: it opens a
TCP socket to example.com:80, sends nothing of substance, and closes.
"""
import socket
import sys
import time


def main():
    host = sys.argv[1] if len(sys.argv) > 1 else "example.com"
    time.sleep(0.5)                       # give the watcher a sample before and after
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(5)
    try:
        s.connect((host, 80))
        print(f"connected to {host}:80 (control)")
    except OSError as e:
        print(f"connect to {host}:80 failed: {e}")
    finally:
        time.sleep(0.5)
        s.close()


if __name__ == "__main__":
    main()
