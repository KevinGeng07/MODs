"""Start and stop the modsdb binary as a child process."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional

REPO = Path(__file__).resolve().parents[2]


def find_binary() -> Path:
    """$MODSDB_BIN, else <repo>/bin/modsdb (built with `go build` if missing)."""
    env = os.environ.get("MODSDB_BIN")
    if env:
        return Path(env)
    binary = REPO / "bin" / "modsdb"
    if not binary.exists():
        if shutil.which("go") is None:
            raise FileNotFoundError(f"{binary} not found and Go is not installed to build it")
        binary.parent.mkdir(exist_ok=True)
        subprocess.run(["go", "build", "-o", str(binary), "./cmd/modsdb"], cwd=REPO / "db", check=True)
    return binary


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DBProcess:
    """A running modsdb. Use as a context manager or call stop()."""

    def __init__(self, data_dir: str | os.PathLike, port: Optional[int] = None, extra_args: tuple = (),
                 startup_timeout: float = 20.0, quiet: bool = True):
        self.port = port or free_port()
        self.addr = f"127.0.0.1:{self.port}"
        cmd = [str(find_binary()), "--data-dir", str(data_dir), "--addr", self.addr, *extra_args]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.log: list[str] = []
        ready = threading.Event()

        def pump():
            for line in self.proc.stdout:
                self.log.append(line.rstrip())
                if not quiet:
                    print("[modsdb]", line.rstrip(), flush=True)
                if "listening on" in line:
                    ready.set()
            ready.set()

        threading.Thread(target=pump, daemon=True).start()
        if not ready.wait(startup_timeout) or self.proc.poll() is not None:
            self.stop()
            raise RuntimeError("modsdb failed to start:\n" + "\n".join(self.log))

    def stop(self, timeout: float = 10.0) -> None:
        if self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def kill(self) -> None:
        """SIGKILL: simulate a crash (no final snapshot)."""
        self.proc.kill()
        self.proc.wait()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()


def wait_ready(client, timeout: float = 10.0) -> None:
    deadline = time.time() + timeout
    while True:
        try:
            client.stats(timeout=1.0)
            return
        except Exception:
            if time.time() > deadline:
                raise
            time.sleep(0.1)
