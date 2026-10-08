"""Start and stop the modsdb binary as a child process."""

from __future__ import annotations

import os
import shutil
import signal
import socket
import subprocess
import threading
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def find_binary() -> Path:
    """$MODSDB_BIN, else <repo>/bin/modsdb, (re)built with Go when missing or older than its sources."""
    if os.environ.get("MODSDB_BIN"):
        return Path(os.environ["MODSDB_BIN"])
    binary = REPO / "bin" / "modsdb"
    built = binary.stat().st_mtime if binary.exists() else 0
    if any(src.stat().st_mtime > built for src in (REPO / "db").rglob("*.go")):
        if shutil.which("go") is None:
            raise FileNotFoundError(f"building {binary} needs Go (https://go.dev/dl)")
        binary.parent.mkdir(exist_ok=True)
        subprocess.run(["go", "build", "-o", str(binary), "./cmd/modsdb"], cwd=REPO / "db", check=True)
    return binary


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class DBProcess:
    """A running modsdb on a free local port. Use as a context manager or call stop()."""

    def __init__(self, data_dir: str | os.PathLike, startup_timeout: float = 20.0):
        self.addr = f"127.0.0.1:{free_port()}"
        cmd = [str(find_binary()), "--data-dir", str(data_dir), "--addr", self.addr]
        self.proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        self.log: list[str] = []
        ready = threading.Event()

        def pump():  # keep the pipe drained; modsdb prints "listening on" once recovery is done
            for line in self.proc.stdout:
                self.log.append(line.rstrip())
                if "listening on" in line:
                    ready.set()
            ready.set()

        threading.Thread(target=pump, daemon=True).start()
        if not ready.wait(startup_timeout) or self.proc.poll() is not None:
            self.stop()
            raise RuntimeError("modsdb failed to start:\n" + "\n".join(self.log))

    def stop(self, timeout: float = 10.0) -> None:
        """SIGTERM: modsdb writes a final snapshot and exits."""
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
