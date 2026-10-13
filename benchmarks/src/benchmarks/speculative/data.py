# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Freeze public evaluation inputs once for native, reference and upstream runners."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


# Explicit source/schema mappings follow DFlash and DDTree's dataset loaders.
DATASETS = {
    "gsm8k": ("openai/gsm8k", "main", "test", 128, "question"),
    "math500": ("HuggingFaceH4/MATH-500", None, "test", 128, "problem"),
    "aime25": ("MathArena/aime_2025", None, "train", 30, "problem"),
    "humaneval": ("openai/openai_humaneval", None, "test", 164, "prompt"),
    "mbpp": ("google-research-datasets/mbpp", "sanitized", "test", 128, "prompt"),
    "livecodebench": ("livecodebench/code_generation_lite", None, "test", 128, "question_content"),
    "mt-bench": ("HuggingFaceH4/mt_bench_prompts", None, "train", 80, "prompt"),
    "alpaca": ("tatsu-lab/alpaca", None, "train", 128, "instruction"),
}


def model_path(identifier: str, models_dir: Path) -> str:
    """Resolve a runner's model ID through existing local weights before using the Hub."""
    local = models_dir / identifier
    if local.is_dir():
        return str(local.absolute())
    from huggingface_hub import try_to_load_from_cache
    config = try_to_load_from_cache(identifier, "config.json", cache_dir=str(models_dir / "hub"))
    if isinstance(config, str):
        return str(Path(config).parent.absolute())
    # Existing offline downloads can contain one pinned snapshot without a refs/main file.
    snapshots = list((models_dir / "hub" / ("models--" + identifier.replace("/", "--")) /
                      "snapshots").glob("*/config.json"))
    return str(snapshots[0].parent.absolute()) if len(snapshots) == 1 else identifier


def load_inputs(path: Path, limit: int | None = None) -> list[dict]:
    """Read nonempty frozen rows shared by all runners, optionally for a screening subset."""
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    rows = rows if limit is None else rows[:limit]
    if not rows:
        raise ValueError(f"No inputs in {path}")
    return rows


def format_turns(task: str, row: dict, field: str) -> list[str]:
    """Convert one public source row to question turns using the baseline prompt convention."""
    text = row[field]
    if task in {"gsm8k", "math500", "aime25"}:
        text += "\nPlease reason step by step, and put your final answer within \\boxed{}."
    elif task == "humaneval":
        text = f"Write a solution to the following problem and make sure that it passes the tests:\n```python\n{text}\n```"
    elif task == "alpaca" and row["input"]:
        text += "\n\nInput:\n" + row["input"]
    elif task == "livecodebench":
        code = row["starter_code"] or "# YOUR CODE HERE"
        text = ("You are an expert Python programmer. Generate a correct Python program "
                "that matches the specification and passes all tests. Return only the program.\n\n"
                f"### Question:\n{text}\n\n### Format:\n```python\n{code}\n```\n\n### Answer:")
    return text if isinstance(text, list) else [text]


def prepare(output: Path, tokenizer_name: str, tasks: list[str], seed: int,
            limit: int | None = None) -> dict:
    """Download pinned source revisions and publish token IDs plus provenance in a new directory.

    MT-Bench uses its first turn; later turns remain in each row for separate
    conversation studies. This protocol gives every method the same fixed prefix.
    """
    from datasets import load_dataset
    from huggingface_hub import HfApi, hf_hub_download
    from transformers import AutoTokenizer

    output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    api = HfApi()
    manifest = {"tokenizer": tokenizer_name, "tokenizer_revision": tokenizer.init_kwargs.get("_commit_hash"),
                "chat_template": tokenizer.chat_template, "enable_thinking": False,
                "add_generation_prompt": True, "seed": seed, "turn_protocol": "first user turn",
                "datasets": {}}
    for task in tasks:
        repository, config, split, count, field = DATASETS[task]
        revision = api.dataset_info(repository).sha
        if task == "livecodebench":
            files = [hf_hub_download(repository, f"test{suffix}.jsonl", repo_type="dataset", revision=revision)
                     for suffix in ("", "2", "3", "4", "5", "6")]
            dataset = load_dataset("json", data_files={split: files}, split=split)
        else:
            dataset = load_dataset(repository, config, split=split, revision=revision)
        count = min(count if limit is None else limit, len(dataset))
        indices = sorted(random.Random(seed).sample(range(len(dataset)), count))
        path = output / f"{task}.jsonl"
        with path.open("x") as stream:
            for index in indices:
                source = dataset[index]
                turns = format_turns(task, source, field)
                messages = [{"role": "user", "content": turns[0]}]
                tokens = tokenizer.apply_chat_template(messages, tokenize=True,
                    add_generation_prompt=True, enable_thinking=False, return_dict=False)
                row = {"id": f"{task}:{index}", "task": task, "source_index": index,
                       "messages": messages, "turns": turns, "prompt_token_ids": tokens,
                       "prompt": tokens}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["datasets"][task] = {"repository": repository, "revision": revision,
            "config": config, "split": split, "source_rows": len(dataset), "rows": count,
            "indices": indices, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
        (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"Frozen {task}: {count} rows", flush=True)
    return manifest
