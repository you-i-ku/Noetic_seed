"""Stdio transport for MCP (実装済み)。

claw-code 参照: rust/crates/runtime/src/mcp_stdio.rs

subprocess.Popen で外部 MCP server を起動し、
stdin/stdout で Content-Length framed JSON-RPC をやり取り。
"""
import os
import queue
import subprocess
import threading
import time
from typing import Optional

from core.runtime.mcp.client import BaseTransport, TransportType
from core.runtime.mcp.protocol import encode_message, parse_all_messages


class StdioTransport(BaseTransport):
    """外部 MCP server を subprocess として起動する transport。"""

    transport_type = TransportType.STDIO

    def __init__(self, command: str, args: Optional[list] = None,
                 env: Optional[dict] = None, cwd: Optional[str] = None):
        self.command = command
        self.args = list(args or [])
        self.env = dict(env or {})
        self.cwd = cwd

        self._proc: Optional[subprocess.Popen] = None
        self._recv_queue: queue.Queue = queue.Queue()
        self._reader_thread: Optional[threading.Thread] = None
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_log: list = []
        self._stop_flag = threading.Event()

    def start(self) -> None:
        if self._proc is not None:
            return
        merged_env = dict(os.environ)
        merged_env.update(self.env)

        self._proc = subprocess.Popen(
            [self.command] + self.args,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            env=merged_env,
            cwd=self.cwd,
            bufsize=0,
        )
        self._reader_thread = threading.Thread(
            target=self._read_loop, daemon=True,
        )
        self._reader_thread.start()
        self._stderr_thread = threading.Thread(
            target=self._stderr_loop, daemon=True,
        )
        self._stderr_thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._proc is not None:
            try:
                if self._proc.stdin:
                    self._proc.stdin.close()
            except Exception:
                pass
            try:
                self._proc.terminate()
                try:
                    self._proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self._proc.kill()
            except Exception:
                pass
            self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def send(self, message) -> None:
        if not self.is_running():
            raise RuntimeError("stdio transport not running")
        data = encode_message(message)
        assert self._proc.stdin is not None
        try:
            self._proc.stdin.write(data)
            self._proc.stdin.flush()
        except BrokenPipeError as e:
            raise RuntimeError(f"stdio pipe closed: {e}")

    def recv(self, timeout: float = 30.0) -> Optional[dict]:
        try:
            return self._recv_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def get_stderr_log(self, max_lines: int = 50) -> list:
        return list(self._stderr_log[-max_lines:])

    # ---- internals ----

    def _read_loop(self) -> None:
        """stdout を読み続けて framed message を queue に push。"""
        buf = b""
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        while not self._stop_flag.is_set() and proc.poll() is None:
            try:
                chunk = proc.stdout.read(4096)
            except Exception:
                break
            if not chunk:
                time.sleep(0.01)
                continue
            buf += chunk
            messages, buf = parse_all_messages(buf)
            for m in messages:
                self._recv_queue.put(m)

    def _stderr_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stderr is None:
            return
        while not self._stop_flag.is_set() and proc.poll() is None:
            try:
                line = proc.stderr.readline()
            except Exception:
                break
            if not line:
                time.sleep(0.01)
                continue
            try:
                self._stderr_log.append(
                    line.decode("utf-8", errors="replace").rstrip()
                )
            except Exception:
                pass
