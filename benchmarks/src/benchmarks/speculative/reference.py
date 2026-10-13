# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Run greedy multi-Draft and equal-budget branch controls in the reference backend."""

from __future__ import annotations

import json
import random
import statistics

from .data import load_inputs, model_path
from .records import collect, compare, prepare_run, write_json


def run(args, suite: dict) -> dict:
    """Load only the selected resident pool and save checks, raw round traces and repetitions.

    Reference timings include cold request prefill and use one request
    at a time. They are independent of native Foretoken service measurements.
    """
    import torch
    import transformers

    from .tree import Model, TreeRuntime

    rows = load_inputs(args.input, args.limit)
    conditions = suite["reference"]["conditions"]
    names = ["target", *(name for name in (args.methods or list(conditions)) if name != "target")]
    pool = {"T"} | {draft for name in names for draft in conditions[name]["drafts"]}
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    models = {name: Model(name, model_path(entry["model"], args.models_dir), entry["dtype"])
              for name, entry in suite["reference"]["models"].items() if name in pool}
    runtime = TreeRuntime(models, "T", {})
    manifest = {"backend": "transformers-reference", "torch": torch.__version__,
                "transformers": transformers.__version__, "suite": suite["reference"],
                "gpu": torch.cuda.get_device_name(), "cuda_graph": False,
                "device": args.gpu, "seed": args.seed, "rows": len(rows), "conditions": names,
                "max_tokens": args.max_tokens, "repetitions": args.repetitions,
                "scope": "screening" if args.limit is not None else "evaluation",
                "timing": "request wall time including cold prefill with request-local GPU KV; C=1"}
    prepare_run(args.output, manifest, args.input, args.resume, runners=("reference",))

    def execute(row, trace=False):
        """Run one cold-prefix request; the runtime owns its KV until generation ends."""
        list(runtime.generate(row["prompt_token_ids"], args.max_tokens, trace=trace))
        return runtime.last_record

    status, reference = {}, None
    for name in names:
        runtime.config = conditions[name]
        samples = [r["tokens"] for r in collect(args.output / f"{name}-check.jsonl", rows,
                                               lambda row: execute(row, trace=True))]
        if name == "target":
            reference = samples
            repeat = [r["tokens"] for r in collect(args.output / "target-self-repeat.jsonl", rows, execute)]
            check = compare(reference, repeat)
            if not check["passed"]:
                raise RuntimeError("Reference Target self-repeat failed")
        else:
            check = compare(reference, samples)
        status[name] = {"state": "completed" if check["passed"] else "failed", "sequence_check": check,
                        "performance": []}
        write_json(args.output / "status.json", status)
    eligible = [name for name in names if status[name]["sequence_check"]["passed"]]
    order = []
    for repetition in range(args.repetitions):
        shuffled = eligible.copy()
        random.Random(args.seed + repetition).shuffle(shuffled)
        order.append(shuffled)
        for name in shuffled:
            runtime.config = conditions[name]
            records = collect(args.output / f"{name}-repeat-{repetition}.jsonl", rows, execute)
            if [r["tokens"] for r in records] != reference:
                raise RuntimeError(f"{name}: measured outputs differ from checked Target")
            elapsed = sum(r["elapsed_s"] for r in records)
            tokens = sum(len(r["tokens"]) for r in records)
            rounds = [step for r in records for step in r["rounds"]]
            status[name]["performance"].append({"e2e_ms_per_token": 1000 * elapsed / tokens,
                "committed_per_round": sum(step["committed"] for step in rounds) / len(rounds),
                "mean_nodes": statistics.mean(step["nodes"] for step in rounds)})
            write_json(args.output / "status.json", status)
    (args.output / "order.json").write_text(json.dumps(order, indent=2) + "\n")
    write_json(args.output / "status.json", status)
    return status
