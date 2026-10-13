# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Prepare pinned official sources and separate Python overlays without modifying base environments."""

from __future__ import annotations

import json
import hashlib
import subprocess
from pathlib import Path


def setup(args, suite: dict) -> dict:
    """Create selected checkouts and reusable environments; publish unavailable prerequisites honestly."""
    args.root.mkdir(parents=True, exist_ok=True)
    status_path = args.root / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    for name in args.methods or list(suite["upstream"]):
        entry = suite["upstream"][name]
        if "unavailable" in entry:
            status[name] = {"state": "unavailable", "reason": entry["unavailable"],
                            "repository": entry["repository"], "revision": entry["revision"]}
            continue
        checkout = args.root / name
        if not checkout.exists():
            subprocess.run(["git", "clone", "--no-checkout", "--depth", "1", entry["repository"], str(checkout)], check=True)
            subprocess.run(["git", "-C", str(checkout), "fetch", "--depth", "1", "origin", entry["revision"]], check=True)
            subprocess.run(["git", "-C", str(checkout), "checkout", "--detach", entry["revision"]], check=True)
        actual = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
        if actual != entry["revision"]:
            raise ValueError(f"{checkout} uses {actual}; expected {entry['revision']}")
        if "patch" in entry:
            patch = Path(entry["patch"]).absolute()
            applied = subprocess.run(["git", "-C", str(checkout), "apply", "--reverse", "--check", str(patch)],
                                     capture_output=True)
            if applied.returncode:
                subprocess.run(["git", "-C", str(checkout), "apply", str(patch)], check=True)
            (checkout / "source-patch.json").write_text(json.dumps({"file": entry["patch"],
                "sha256": hashlib.sha256(patch.read_bytes()).hexdigest()}, indent=2) + "\n")
        python = checkout / ".venv/bin/python"
        if not python.exists():
            subprocess.run([str(args.engine_python), "-m", "venv", "--system-site-packages",
                            str(checkout / ".venv")], check=True)
        # pip recognizes inherited CUDA packages; keep the base PyTorch installation.
        subprocess.run([str(python), "-m", "pip", "install", *entry["dependencies"]], check=True)
        subprocess.run([str(python), "-m", "pip", "install", "--no-deps", "-e", "."], check=True)
        # Record versions rather than pip URLs, which can contain credentials or local paths.
        versions = subprocess.check_output([str(python), "-c",
            "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name']:m.version(d.metadata['Name']) for d in m.distributions()}))"], text=True)
        (checkout / "environment.json").write_text(versions)
        status[name] = {"state": "prepared", "revision": actual, "python": str(python.absolute())}
        if name == "d2sd":
            status[name]["missing_prerequisite"] = "DTA checkpoint is required; it is not published by this source repository."
        print(f"Prepared {name}", flush=True)
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    return status
