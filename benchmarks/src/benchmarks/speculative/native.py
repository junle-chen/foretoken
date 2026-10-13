# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Run matched vLLM baselines using Foretoken's service client and performance runner."""

from __future__ import annotations

import json
import os
import random
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import httpx

from benchmarks.config.benchmark import ModelServiceSource
from benchmarks.integrations.distributions import CompletionComparisonClient
from benchmarks.model_service import resolve_model_service

from .data import load_inputs, model_path
from .process import native_server
from .records import collect, compare, prepare_run, write_json


def capture(client: CompletionComparisonClient, rows: list[dict], path: Path,
            maximum: int) -> list[list[int]]:
    """Save native generated IDs for fixed prefixes; callers compare these without retokenization."""
    return [r["token_ids"] for r in collect(path, rows,
        lambda row: client.generate(row["prompt_token_ids"], maximum))]


def counters(body: str) -> dict[str, float]:
    """Sum exact Prometheus metric names for one experiment's engine snapshots."""
    values = defaultdict(float)
    for line in body.splitlines():
        match = re.fullmatch(r'([^\s{#]+)(?:\{.*\})?\s+([^\s]+)(?:\s+[^\s]+)?', line)
        if match:
            values[match[1]] += float(match[2])
    return dict(values)


def measure(url: str, rows: list[dict], directory: Path, maximum: int,
            concurrency: int, repetitions: int) -> list[dict]:
    """Execute existing Foretoken perf and retain its raw metrics plus engine counter deltas."""
    workload = directory / "workload.jsonl"
    workload.write_text("".join(json.dumps({"prompt": r["prompt_token_ids"]}) + "\n" for r in rows))
    results = []
    for repetition in range(repetitions):
        output = directory / f"repeat-{repetition}"
        saved = output / "performance.json"
        if saved.exists():
            results.append(json.loads(saved.read_text()))
            continue
        if output.exists():
            output.rename(directory / f"interrupted-repeat-{repetition}-{time.time_ns()}")
        output.mkdir()
        command = [str(Path(sys.executable).with_name("foretoken")), "perf", "--url", url + "/v1/completions",
                   "--model", "reference-target", "--dataset", str(workload.absolute()),
                   "--temperature", "0", "--max-tokens", str(maximum),
                   "--max-concurrency", str(concurrency), "--num-prompts", str(len(rows)),
                   "--warmup-requests", "0", "--timeout", "600s", "--output", "local,quiet",
                   "--output-dir", str(output.absolute())]
        with httpx.Client(timeout=30) as observer:
            time.sleep(2)
            before = observer.get(url + "/metrics")
            before.raise_for_status()
            (output / "metrics-before.txt").write_text(before.text)
            with (output / "perf.log").open("x") as log:
                subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
            # vLLM's logger publishes engine statistics asynchronously.
            time.sleep(2)
            after = observer.get(url + "/metrics")
            after.raise_for_status()
            (output / "metrics-after.txt").write_text(after.text)
        paths = list(output.rglob("metrics.json"))
        if len(paths) != 1:
            raise RuntimeError(f"Expected one Foretoken metrics file in {output}, found {len(paths)}")
        result = json.loads(paths[0].read_text())
        if result["success_num"] != len(rows) or result["failed_num"]:
            raise RuntimeError(f"Incomplete requests: inspect {paths[0]}")
        old, new = counters(before.text), counters(after.text)
        delta = {name: value - old.get(name, 0) for name, value in new.items()}
        point = {"metrics": str(paths[0].relative_to(directory)),
                 "tpot_ms": result["tpot"]["mean"] * 1000,
                 "p95_e2e_s": result["latency"]["p95"],
                 "output_tokens_per_second": result["throughput"]["generation_tokens_per_second"]}
        rounds = delta.get("vllm:spec_decode_num_drafts_total", 0)
        if rounds:
            point.update(proposal_rounds=rounds,
                accepted_per_proposal_round=delta["vllm:spec_decode_num_accepted_tokens_total"] / rounds,
                proposed_per_proposal_round=delta["vllm:spec_decode_num_draft_tokens_total"] / rounds)
        results.append(point)
        saved.write_text(json.dumps(point, indent=2) + "\n")
        (directory / "performance.json").write_text(json.dumps(results, indent=2) + "\n")
    return results


def run(args, suite: dict) -> dict:
    """Own each native condition's server, checks and measurements in a new resource group."""
    rows = load_inputs(args.input, args.limit)
    selected = args.methods or list(suite["native"])
    unknown = set(selected) - set(suite["native"]) - {"target"}
    if unknown:
        raise ValueError(f"Unknown native methods: {sorted(unknown)}")
    conditions = {"target": None}
    for name in selected:
        if name == "target":
            continue
        entry = suite["native"][name]
        for gamma in entry["gamma"]:
            spec = {"method": entry["method"], "num_speculative_tokens": gamma}
            if "model" in entry:
                spec["model"] = model_path(entry["model"], args.models_dir)
            if entry["method"] == "ngram":
                spec.update(prompt_lookup_min=1, prompt_lookup_max=5)
            conditions[f"{name}-{gamma}"] = spec
    candidates = list(conditions)[1:]
    random.Random(args.seed).shuffle(candidates)
    order = ["target", *candidates]
    manifest = {"backend": "vllm", "mode": args.mode, "suite": suite, "order": order,
                "input": str(args.input.absolute()), "rows": len(rows), "gpu": args.gpu,
                "max_tokens": args.max_tokens, "concurrency": args.concurrency,
                "seed": args.seed, "memory": args.memory,
                "dtype": args.dtype, "compilation_mode": args.compilation_mode,
                "float32_matmul_precision": "highest", "nvidia_tf32_override": "0",
                "repetitions": args.repetitions, "attention_backend": args.attention_backend,
                "warmup_requests": min(8, len(rows)), "metrics_publish_wait_s": 2,
                "prefix_warmup_new_tokens": 1,
                "scope": "screening" if args.limit is not None else "evaluation",
                "acceptance_scope": "vLLM proposal-counted rounds; includes no inferred bonus tokens"}
    manifest["engine_packages"] = json.loads(subprocess.check_output([
        str(args.engine_python.absolute()), "-c",
        "import importlib.metadata as m,json; print(json.dumps({p:m.version(p) for p in ['vllm','torch','transformers']}))"], text=True))
    prepare_run(args.output, manifest, args.input, args.resume, runners=("native",))
    status_path = args.output / "status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    reference = None
    url = f"http://127.0.0.1:{args.port}"
    maximum_length = max(len(r["prompt_token_ids"]) for r in rows) + args.max_tokens + max(
        (spec["num_speculative_tokens"] for spec in conditions.values() if spec), default=0) + 1
    for name in order:
        directory = args.output / name
        previous = status.get(name, {})
        if previous.get("state") == "completed" or (previous.get("state") == "failed" and "sequence_check" in previous):
            if name == "target":
                reference = [json.loads(line)["token_ids"] for line in
                             (directory / "capture.jsonl").read_text().splitlines()]
            continue
        directory.mkdir(exist_ok=args.resume)
        spec = conditions[name]
        command = [str(args.engine_python.absolute()), "-m", "vllm.entrypoints.openai.api_server",
                   "--model", model_path(suite["target"], args.models_dir),
                   "--served-model-name", "reference-target", "--host", "127.0.0.1",
                   "--port", str(args.port), "--dtype", args.dtype, "--no-async-scheduling",
                   "--max-model-len", str(maximum_length), "--max-num-seqs", str(args.concurrency),
                   "--gpu-memory-utilization", str(args.memory), "--enable-prefix-caching",
                   "--attention-backend", args.attention_backend]
        if spec is not None:
            command += ["--speculative-config", json.dumps(spec)]
        if args.compilation_mode == "none":
            command += ["--compilation-config", '{"mode":0,"cudagraph_mode":"FULL"}']
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu, "OMP_NUM_THREADS": "1",
               "VLLM_BATCH_INVARIANT": "1" if args.mode == "strict" else "0",
               "VLLM_USE_V2_MODEL_RUNNER": "0",
               "NVIDIA_TF32_OVERRIDE": "0",
               "VLLM_FLOAT32_MATMUL_PRECISION": "highest",
               "VLLM_CACHE_ROOT": str((directory / "runtime-cache").absolute())}
        (directory / "command.json").write_text(json.dumps({"argv": command,
            "env": {key: env[key] for key in ("CUDA_VISIBLE_DEVICES", "VLLM_BATCH_INVARIANT",
                                            "VLLM_USE_V2_MODEL_RUNNER", "VLLM_CACHE_ROOT", "NVIDIA_TF32_OVERRIDE",
                                            "VLLM_FLOAT32_MATMUL_PRECISION")}}, indent=2) + "\n")
        print(f"START {name} ({args.mode})", flush=True)
        status[name] = {"state": "running"}
        write_json(args.output / "status.json", status)
        log_path = directory / "server.log"
        if log_path.exists():
            log_path = directory / f"server-resume-{time.time_ns()}.log"
        try:
            with native_server(command, env, log_path, url, Path.cwd()):
                source = ModelServiceSource(url=url + "/v1/completions", model="reference-target")
                with resolve_model_service(source) as service, CompletionComparisonClient(service, timeout=600) as client:
                    for row in rows:
                        client.generate(row["prompt_token_ids"], 1)
                    samples = capture(client, rows, directory / "capture.jsonl", args.max_tokens)
                    if name == "target":
                        reference = samples
                        repeat = capture(client, rows, directory / "self-repeat.jsonl", args.max_tokens)
                        check = compare(reference, repeat)
                        if not check["passed"]:
                            raise RuntimeError("Target self-repeat differs; this resource group cannot be compared")
                    else:
                        check = compare(reference, samples)
                    result = {"state": "completed", "sequence_check": check}
                    # Sequence checks and normal serving measurements have separate execution modes.
                    if args.mode == "performance":
                        result["comparison_scope"] = "matched-output" if check["passed"] else "numerical-diagnostic"
                        for row in rows[:min(8, len(rows))]:
                            client.generate(row["prompt_token_ids"], args.max_tokens)
                        result["performance"] = measure(url, rows, directory, args.max_tokens,
                                                        args.concurrency, args.repetitions)
                    elif not check["passed"]:
                        result["state"] = "failed"
                    status[name] = result
        except (RuntimeError, httpx.HTTPError, subprocess.CalledProcessError) as error:
            status[name] = {"state": "failed", "error": str(error)}
            if name == "target":
                write_json(args.output / "status.json", status)
                raise
        write_json(args.output / "status.json", status)
        print(f"DONE {name}: {status[name]['state']}", flush=True)
    return status
