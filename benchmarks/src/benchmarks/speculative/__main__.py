# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Prepare data and run experimental speculative-decoding baseline groups."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import yaml


def positive(value: str) -> int:
    """Parse positive workload sizes once at the experiment command boundary."""
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def main() -> None:
    """Dispatch data preparation, native/reference measurement and pinned upstream setup."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, default=Path("benchmarks/experiments/multi-draft/suite.yaml"))
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Freeze the eight-task inputs and source revisions")
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--tokenizer", default="Qwen/Qwen3-8B")
    prepare.add_argument("--tasks", nargs="+", default=None)
    prepare.add_argument("--seed", type=int, default=0)
    prepare.add_argument("--limit", type=positive, help="Freeze a small per-task screening set")
    setup = commands.add_parser("setup", help="Prepare official sources and separate environments")
    setup.add_argument("--root", type=Path, default=Path("reference/multi-draft"))
    setup.add_argument("--engine-python", type=Path, required=True, help="Existing CUDA PyTorch interpreter")
    setup.add_argument("--methods", nargs="+", choices=["treegraft", "ddtree", "d2sd", "whiflash"])
    analyze = commands.add_parser("analyze", help="Measure admitted-source coverage in reference traces")
    analyze.add_argument("--input", type=Path, required=True)
    analyze.add_argument("--output", type=Path, required=True)
    report = commands.add_parser("report", help="Summarize completed and pending matrix groups")
    report.add_argument("--input", type=Path, required=True, help="Matrix output directory")
    matrix = commands.add_parser("matrix", help="Run the eight-task matrix on a fixed local GPU pool")
    matrix.add_argument("--input", type=Path, required=True, help="Frozen data directory")
    matrix.add_argument("--output", type=Path, required=True)
    matrix.add_argument("--gpus", nargs="+", required=True)
    matrix.add_argument("--engine-python", type=Path, required=True)
    matrix.add_argument("--reference-python", type=Path, required=True)
    matrix.add_argument("--upstream-root", type=Path, default=Path("reference/multi-draft"))
    matrix.add_argument("--models-dir", type=Path, default=Path("data/models"))
    matrix.add_argument("--tasks", nargs="+")
    matrix.add_argument("--backends", nargs="+",
                        choices=["native", "native-strict", "native-fp32", "reference", "treegraft", "treegraft-fp32", "ddtree", "ddtree-fp32"],
                        default=["native", "native-fp32", "reference", "treegraft", "treegraft-fp32", "ddtree", "ddtree-fp32"])
    matrix.add_argument("--max-tokens", type=positive, default=2048)
    matrix.add_argument("--repetitions", type=positive, default=3)
    matrix.add_argument("--limit", type=positive)
    matrix.add_argument("--port", type=int, default=18500, help="First port; one port per GPU worker")
    matrix.add_argument("--resume", action="store_true")
    matrix.add_argument("--plan-only", action="store_true")
    for command in ("native", "reference", "official"):
        child = commands.add_parser(command)
        child.add_argument("--input", type=Path, required=True)
        child.add_argument("--output", type=Path, required=True, help="New result directory")
        child.add_argument("--resume", action="store_true", help="Resume an identical interrupted run")
        child.add_argument("--models-dir", type=Path, default=Path("data/models"))
        child.add_argument("--gpu", required=True)
        child.add_argument("--max-tokens", type=positive, default=2048)
        child.add_argument("--repetitions", type=positive, default=3)
        child.add_argument("--seed", type=int, default=0)
        child.add_argument("--limit", type=positive, help="Use a prefix of the inputs for screening only")
        if command == "official":
            child.add_argument("--method", required=True, choices=["treegraft", "ddtree", "d2sd", "whiflash"])
            child.add_argument("--checkout", type=Path, required=True)
            child.add_argument("--budget", type=positive, default=63)
            child.add_argument("--draft-attention", choices=["sdpa", "flash_attention_2"], default="sdpa")
            child.add_argument("--dta", type=Path, help="D²SD's second-Draft checkpoint")
            child.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
            child.add_argument("--target-head-dtype", choices=["model", "float32"], default="model",
                               help="Use FP32 Target output projection for a separate mixed-precision control")
        else:
            child.add_argument("--methods", nargs="+")
        if command == "native":
            child.add_argument("--engine-python", type=Path, required=True)
            child.add_argument("--mode", choices=["strict", "performance"], required=True)
            child.add_argument("--attention-backend", default="FLASH_ATTN")
            child.add_argument("--concurrency", type=positive, default=1)
            child.add_argument("--port", type=int, default=18320)
            child.add_argument("--memory", type=float, default=0.70)
            child.add_argument("--dtype", choices=["bfloat16", "float32"], default="bfloat16")
            child.add_argument("--compilation-mode", choices=["default", "none"], default="default",
                               help="Disable model torch.compile while retaining FULL CUDA graphs for a precision control")
    args = parser.parse_args()
    suite = yaml.safe_load(args.suite.read_text())
    if args.command == "prepare":
        from .data import DATASETS, prepare as freeze
        tasks = args.tasks or list(DATASETS)
        unknown = set(tasks) - set(DATASETS)
        if unknown:
            parser.error(f"unknown datasets: {sorted(unknown)}")
        result = freeze(args.output, args.tokenizer, tasks, args.seed, args.limit)
        result = {"output": str(args.output), "datasets": {
            task: entry["rows"] for task, entry in result["datasets"].items()}}
    elif args.command == "setup":
        from .upstream import setup as bootstrap
        result = bootstrap(args, suite)
    elif args.command == "analyze":
        from .analysis import coverage
        result = coverage(args.input)
        with args.output.open("x") as stream:
            stream.write(json.dumps(result, indent=2) + "\n")
    elif args.command == "matrix":
        from .matrix import run
        from .records import run_lock
        with run_lock(args.output):
            result = run(args, suite)
    elif args.command == "report":
        from .report import summarize
        result = summarize(args.input)
    else:
        # Select the device before the backend imports torch or initializes CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
        if args.command == "native":
            if args.mode == "strict" and args.attention_backend != "FLEX_ATTENTION":
                parser.error("strict mode requires --attention-backend FLEX_ATTENTION")
            from .native import run
        elif args.command == "reference":
            from .reference import run
        else:
            from .adapters import run
        from .records import run_lock
        with run_lock(args.output):
            result = run(args, suite)
    print(json.dumps(result, indent=2))
    if args.command in {"native", "reference", "matrix"} and any(
        value.get("state") == "failed" for value in result.values()
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
