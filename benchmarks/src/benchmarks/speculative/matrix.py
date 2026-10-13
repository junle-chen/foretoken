# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Schedule independent task/backend groups while keeping each matched Target on one GPU."""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .data import DATASETS
from .records import prepare_run, write_json
from .report import summarize


def run(args, suite: dict) -> dict:
    """Execute a persistent local matrix; individual runners own engines, checks and checkpoints."""
    tasks = args.tasks or list(DATASETS)
    unknown = set(tasks) - set(DATASETS)
    if unknown:
        raise ValueError(f"Unknown tasks: {sorted(unknown)}")
    if len(set(args.gpus)) != len(args.gpus):
        raise ValueError("Each GPU must appear only once in the worker pool")
    jobs = [{"task": task, "backend": backend} for task in tasks for backend in args.backends]
    random.Random(0).shuffle(jobs)
    for index, job in enumerate(jobs):
        job["gpu"] = args.gpus[index % len(args.gpus)]
    manifest = {"suite": suite, "jobs": jobs, "gpus": args.gpus,
                "input": str(args.input.absolute()), "models": str(args.models_dir.absolute()),
                "engine_python": str(args.engine_python.absolute()),
                "reference_python": str(args.reference_python.absolute()),
                "upstream_root": str(args.upstream_root.absolute()),
                "max_tokens": args.max_tokens, "repetitions": args.repetitions, "limit": args.limit,
                "native_modes": {"native": "BF16 performance/FLASH_ATTN", "native-strict": "BF16 strict/FLEX_ATTENTION",
                                 "native-fp32": "FP32 performance/FLEX_ATTENTION; highest matmul precision; NVIDIA_TF32_OVERRIDE=0; model compilation NONE; FULL graphs"},
                "official_dtype": "bfloat16 (treegraft-fp32 and ddtree-fp32: float32)", "official_budget": 63}
    runners = {"matrix"}
    for backend in args.backends:
        if backend.startswith("native"):
            runners.add("native")
        elif backend == "reference":
            runners.add("reference")
        else:
            runners.add("official")
    prepare_run(args.output, manifest, args.input / "manifest.json", args.resume,
                runners=tuple(sorted(runners)))
    status_path = args.output / "status.json"
    states = json.loads(status_path.read_text()) if status_path.exists() else {}
    states["unavailable"] = {"whiflash": suite["upstream"]["whiflash"]["unavailable"],
                             "d2sd": "Second-Draft DTA checkpoint has not been published or supplied."}
    pending = []
    for job in jobs:
        key = f"{job['task']}/{job['backend']}"
        if states.get(key, {}).get("state") != "completed":
            pending.append((key, job))
            states.setdefault(key, {"state": "pending"})
    lock = threading.Lock()

    def publish(key=None, value=None):
        """Atomically publish worker state for readers and the next matrix invocation."""
        with lock:
            if key is not None:
                states[key] = value
            write_json(status_path, states)

    publish()
    if args.plan_only:
        return states

    def worker(index, gpu):
        """Run groups sequentially on one explicit device; failures leave other groups running."""
        for key, job in pending:
            if job["gpu"] != gpu:
                continue
            backend, task = job["backend"], job["task"]
            output = args.output / task / backend
            output.parent.mkdir(exist_ok=True)
            python = args.reference_python if backend == "reference" else sys.executable
            command = backend
            extra = []
            if backend.startswith("native"):
                command = "native"
                mode = "strict" if backend == "native-strict" else "performance"
                attention = "FLEX_ATTENTION" if backend != "native" else "FLASH_ATTN"
                extra = ["--engine-python", str(args.engine_python.absolute()), "--mode", mode,
                         "--attention-backend", attention,
                         "--port", str(args.port + index)]
                if backend == "native-fp32":
                    extra += ["--dtype", "float32", "--compilation-mode", "none"]
            elif backend.startswith(("treegraft", "ddtree")):
                method = backend.removesuffix("-fp32")
                checkout = args.upstream_root / method
                python = checkout / ".venv/bin/python"
                command = "official"
                extra = ["--method", method, "--checkout", str(checkout.absolute()),
                         "--dtype", "float32" if backend.endswith("-fp32") else "bfloat16"]
            argv = [str(python), "-m", "benchmarks.speculative", "--suite", str(args.suite.absolute()), command,
                    "--gpu", gpu, "--input", str((args.input / f"{task}.jsonl").absolute()),
                    "--output", str(output.absolute()), "--models-dir", str(args.models_dir.absolute()),
                    "--max-tokens", str(args.max_tokens), "--repetitions", str(args.repetitions), *extra]
            if args.limit is not None:
                argv += ["--limit", str(args.limit)]
            if (output / "manifest.json").exists():
                argv.append("--resume")
            log_path = output.parent / f"{backend}.log"
            publish(key, {"state": "running", "gpu": gpu, "command": argv})
            print(f"START {key} GPU{gpu}", flush=True)
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).parents[2]) + os.pathsep + env.get("PYTHONPATH", "")
            with log_path.open("a") as stream:
                result = subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, env=env)
            publish(key, {"state": "completed" if result.returncode == 0 else "failed",
                          "gpu": gpu, "returncode": result.returncode,
                          "result": str(output.relative_to(args.output))})
            print(f"DONE {key}: exit {result.returncode}", flush=True)

    with ThreadPoolExecutor(max_workers=len(args.gpus)) as executor:
        list(executor.map(worker, range(len(args.gpus)), args.gpus))
    summarize(args.output)
    return states
