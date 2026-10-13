# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Summarize observed per-source coverage on shared, Target-committed prefixes."""

from __future__ import annotations

import json
import statistics
from pathlib import Path


def coverage(path: Path) -> dict:
    """Return round and prompt oracle coverage from reference check traces, not a speed forecast.

    Only candidates actually admitted under the recorded node budget are visible.
    EOS and output truncation censor acceptance; unexecuted Drafts are not inferred.
    """
    records = [json.loads(line) for line in path.read_text().splitlines()]
    names = sorted({name for record in records for step in record["rounds"]
                    for sources in step["tree"]["sources"] for name in sources})
    if not names:
        raise ValueError("Coverage requires a reference candidate check trace with source provenance")
    lengths, prompt_lengths = [], []
    for record in records:
        prompt = []
        for step in record["rounds"]:
            tree = step["tree"]
            children = {(parent, token): index for index, (parent, token) in enumerate(
                zip(tree["parents"], tree["tokens"], strict=True))}
            accepted, parent = [], -1
            for token in step["output"]:
                node = children.get((parent, token))
                if node is None:
                    break
                accepted.append(node)
                parent = node
            source_lengths = {}
            for name in names:
                count = 0
                for node in accepted:
                    if name not in tree["sources"][node]:
                        break
                    count += 1
                source_lengths[name] = count
            prompt.append(source_lengths)
        lengths.extend(prompt)
        prompt_lengths.append({name: statistics.mean(r[name] for r in prompt) for name in names})
    means = {name: statistics.mean(r[name] for r in lengths) for name in names}
    best = max(means.values())
    oracle = statistics.mean(max(r.values()) for r in lengths)
    return {"source": str(path), "prompts": len(records), "rounds": len(lengths),
            "mean_accepted_by_source": means, "best_fixed_source_mean": best,
            "round_oracle_mean": oracle,
            "prompt_oracle_mean": statistics.mean(max(r.values()) for r in prompt_lengths),
            "complementary_accepted_tokens_per_round": oracle - best,
            "relative_coverage_gain": oracle / best - 1 if best else None,
            "scope": "observed admitted candidates on shared committed prefixes; censored by EOS/output cap; no speed oracle"}
