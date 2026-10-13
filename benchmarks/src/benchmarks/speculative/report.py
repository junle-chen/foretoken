# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Summarize the matrix without pooling unlike timing or precision groups."""

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path

from .records import write_json


def summarize(directory: Path) -> dict:
    """Write a readable report and machine-readable ledger from available child evidence."""
    manifest = json.loads((directory / "manifest.json").read_text())
    states = json.loads((directory / "status.json").read_text())
    records = []
    for job in manifest["jobs"]:
        task, backend = job["task"], job["backend"]
        path = directory / task / backend
        status_path = path / "status.json"
        if not status_path.exists():
            continue
        status = json.loads(status_path.read_text())
        official = backend.startswith(("treegraft", "ddtree"))
        if official:
            candidates = {}
            for name, values in status["performance"].items():
                scope = status["comparison_scope"]
                if name == "target":
                    scope = "matched-output" if all(c["passed"] for c in
                        status.get("repeat_checks", {}).get(name, [])) else "numerical-diagnostic"
                elif any(not c["passed"] for c in status.get("repeat_checks", {}).get(name, [])):
                    scope = "numerical-diagnostic"
                candidates[name] = {
                    "sequence_check": status["target_self_repeat"] if name == "target" else status["sequence_check"],
                    "performance": values, "comparison_scope": scope}
        else:
            candidates = status
        metric = "tpot_ms" if backend.startswith("native") else "e2e_ms_per_token"
        target = candidates.get("target", {})
        target_values = target.get("performance", [])
        target_numbers = target_values if official else [p[metric] for p in target_values]
        target_consistent = target.get("sequence_check", {}).get("passed", False) and \
            target.get("comparison_scope") != "numerical-diagnostic"
        required = 0 if backend == "native-strict" else manifest["repetitions"]
        placement = states[f"{task}/{backend}"]
        for name, result in candidates.items():
            values = result.get("performance", [])
            numbers = values if official else [p[metric] for p in values]
            check = result.get("sequence_check", {})
            scope = result.get("comparison_scope", ("matched-output" if check.get("passed") else
                "numerical-diagnostic") if check else "pending")
            speedup = None
            if required and len(numbers) == len(target_numbers) == required and target_consistent and scope == "matched-output":
                speedup = statistics.mean(target_numbers) / statistics.mean(numbers)
            row = {"task": task, "backend": backend, "method": name,
                   "host": placement.get("host", job.get("host", "local")),
                   "gpu": placement.get("gpu", job.get("gpu")),
                   "state": result.get("state", placement["state"]), "error": result.get("error"),
                   "check_matches": check.get("matches"), "check_samples": check.get("samples"),
                   "repetitions": len(numbers), "metric": metric, "comparison_scope": scope,
                   "mean_ms": statistics.mean(numbers) if numbers else None,
                   "std_ms": statistics.stdev(numbers) if len(numbers) > 1 else None,
                   "repeat_values_ms": numbers, "speedup": speedup,
                   "evidence": str(status_path.relative_to(directory))}
            if not official:
                for metric_name in ("accepted_per_proposal_round", "proposed_per_proposal_round",
                                    "committed_per_round", "mean_nodes"):
                    observed = [p[metric_name] for p in values if metric_name in p]
                    if observed:
                        row[metric_name] = statistics.mean(observed)
            records.append(row)
    job_states = {f"{job['task']}/{job['backend']}": states[f"{job['task']}/{job['backend']}"]["state"]
                  for job in manifest["jobs"]}
    grouped = defaultdict(list)
    for row in records:
        grouped[row["backend"], row["method"]].append(row)
    consistency = []
    for (backend, method), rows in grouped.items():
        checked = [r for r in rows if r["check_samples"] is not None]
        eligible = [r for r in rows if r["speedup"] is not None]
        total = len({j["task"] for j in manifest["jobs"] if j["backend"] == backend})
        consistency.append({"backend": backend, "method": method, "tasks": total,
            "tasks_checked": len(checked), "matches": sum(r["check_matches"] for r in checked),
            "samples": sum(r["check_samples"] for r in checked), "eligible_tasks": len(eligible),
            "geomean_speedup": statistics.geometric_mean(r["speedup"] for r in eligible)
            if len(eligible) == total else None})
    summary = {"max_tokens": manifest["max_tokens"], "repetitions": manifest["repetitions"],
               "scope": "screening" if manifest["limit"] is not None else "evaluation",
               "complete": all(state == "completed" for state in job_states.values()),
               "jobs": job_states, "unavailable": states["unavailable"], "consistency": consistency,
               "records": records}
    write_json(directory / "summary.json", summary)
    columns = ["task", "backend", "method", "host", "gpu", "state", "comparison_scope",
               "check_matches", "check_samples", "repetitions", "metric", "mean_ms", "std_ms", "speedup", "evidence"]
    with (directory / "summary.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    counts = {state: list(job_states.values()).count(state) for state in sorted(set(job_states.values()))}
    lines = ["# Multi-Draft baseline matrix", "", f"Job states: {counts}.", "",
             f"Output cap: {manifest['max_tokens']}; requested repetitions: {manifest['repetitions']}; scope: {summary['scope']}.",
             "", "Native TPOT and Transformers request time per output token are separate metrics. "
             "Speedups use only the same task/backend Target. Numerical-diagnostic rows are excluded from speedup claims. "
             "A completed process does not imply token equality. Strict groups only check outputs. "
             "Native equality is checked in separate non-streaming captures; Foretoken perf retains timing and usage, "
             "not native token IDs from its streaming repetitions.", "",
             "## Execution progress", "",
             "All planned groups appear here, including those that have not published checks yet. "
             "The measurement tables below include only published evidence.", "",
             "| Task | Backend | Host/GPU | State | Exit code |",
             "| --- | --- | --- | --- | --- |"]
    for job in manifest["jobs"]:
        placement = states[f"{job['task']}/{job['backend']}"]
        host = placement.get("host", job.get("host", "local"))
        gpu = placement.get("gpu", job.get("gpu"))
        device = f"{host}/{gpu}" if gpu is not None else "unassigned"
        lines.append(f"| {job['task']} | {job['backend']} | {device} | "
                     f"{placement['state']} | {placement.get('returncode', '—')} |")
    lines += ["", "## Consistency across tasks", "",
             "Aggregate speedup is shown only when every task has complete, matching measurements. "
             "It is the geometric mean of within-group ratios; latencies from different tasks are not pooled.", "",
             "| Backend | Method | Tasks checked | Exact sequences | Eligible tasks | Aggregate speedup |",
             "| --- | --- | --- | --- | --- | --- |"]
    for row in consistency:
        speed = f"{row['geomean_speedup']:.3f}x" if row["geomean_speedup"] is not None else "—"
        lines.append(f"| {row['backend']} | {row['method']} | {row['tasks_checked']}/{row['tasks']} | "
                     f"{row['matches']}/{row['samples']} | {row['eligible_tasks']}/{row['tasks']} | {speed} |")
    lines += ["", "## Per-task measurements", "",
              "| Task | Backend | Method | Host/GPU | Exact sequences | Repeats | Metric | Mean ± SD ms | Speedup | Scope |",
              "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for row in records:
        exact = f"{row['check_matches']}/{row['check_samples']}" if row['check_samples'] is not None else "pending"
        mean = f"{row['mean_ms']:.3f}" if row['mean_ms'] is not None else "—"
        if row['std_ms'] is not None:
            mean += f" ± {row['std_ms']:.3f}"
        speedup = f"{row['speedup']:.3f}x" if row['speedup'] is not None else "—"
        lines.append(f"| {row['task']} | {row['backend']} | {row['method']} | {row['host']}/{row['gpu']} | {exact} | {row['repetitions']} | "
                     f"{row['metric']} | {mean} | {speedup} | {row['comparison_scope']} |")
    lines += ["", "Unavailable baselines:", ""]
    lines.extend(f"- {name}: {reason}" for name, reason in states["unavailable"].items())
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return {"jobs": counts, "records": len(records), "report": str(directory / "report.md")}
