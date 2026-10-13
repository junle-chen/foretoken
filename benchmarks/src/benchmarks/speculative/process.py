# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Own the launch/readiness/drain lifecycle of one experiment-local vLLM process group."""

from __future__ import annotations

import os
import json
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import contextmanager
from pathlib import Path


def _live_group_members(group: int) -> list[int]:
    """Find this server group's live Linux processes while allowing already-exited zombies."""
    members = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            fields = (entry / "stat").read_text().rsplit(")", 1)[1].split()
        except (FileNotFoundError, ProcessLookupError):
            continue
        if int(fields[2]) == group and fields[0] != "Z":
            members.append(int(entry.name))
    return members


@contextmanager
def native_server(command: list[str], env: dict[str, str], log_path: Path, url: str, cwd: Path):
    """Yield a ready task-owned server and always terminate its own process group on exit."""
    address = urllib.parse.urlparse(url)
    with socket.socket() as probe:
        if probe.connect_ex((address.hostname, address.port)) == 0:
            raise RuntimeError(f"Experiment port {address.port} is already occupied")
    with log_path.open("x") as log:
        started, started_utc = time.perf_counter(), time.time()
        ready_seconds = None
        server = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT,
                                  start_new_session=True, cwd=cwd)
        try:
            deadline = time.monotonic() + 1200
            while True:
                if server.poll() is not None:
                    raise RuntimeError(f"Native server exited {server.returncode}; inspect {log_path}")
                try:
                    with urllib.request.urlopen(url + "/health", timeout=2):
                        ready_seconds = time.perf_counter() - started
                        break
                except (urllib.error.URLError, TimeoutError):
                    if time.monotonic() > deadline:
                        raise RuntimeError("Native server startup timed out")
                    time.sleep(2)
            yield server
        finally:
            if server.poll() is None:
                server.send_signal(signal.SIGINT)
                try:
                    server.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(server.pid, signal.SIGKILL)
                    server.wait()
            try:
                os.killpg(server.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass  # The API server already reaped its workers normally.
            # API exit alone does not prove its CUDA workers released the card.
            drain_started = time.monotonic()
            members = _live_group_members(server.pid)
            for stop_signal in (signal.SIGTERM, signal.SIGKILL):
                if not members:
                    break
                try:
                    os.killpg(server.pid, stop_signal)
                except ProcessLookupError:
                    members = []
                    break
                deadline = time.monotonic() + 30
                while members and time.monotonic() < deadline:
                    time.sleep(0.5)
                    members = _live_group_members(server.pid)
            elapsed = time.perf_counter() - started
            gpus = len(env["CUDA_VISIBLE_DEVICES"].split(","))
            log_path.with_suffix(".lifecycle.json").write_text(json.dumps({
                "started_unix_seconds": started_utc, "ready_after_seconds": ready_seconds,
                "process_lifecycle_seconds": elapsed, "reserved_gpu_seconds": elapsed * gpus,
                "worker_group_drain_seconds": time.monotonic() - drain_started,
                "remaining_worker_pids": members,
                "gpus": gpus, "scope": "reserved process lifecycle including startup and drain; not kernel busy time"},
                indent=2) + "\n")
            if members:
                raise RuntimeError(f"Owned server process group {server.pid} did not drain: {members}")
