# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Call official tree runtimes with frozen IDs, retaining their model and verification algorithms."""

from __future__ import annotations

import json
import random
import statistics
import subprocess
import sys
import time

from .data import load_inputs, model_path
from .records import collect, compare, prepare_run, write_json


def configure_target_head(target, dtype: str) -> None:
    """Set an explicit shared output-head precision for causal and tree verification paths."""
    if dtype == "float32":
        target.lm_head.float()
        target.lm_head.register_forward_pre_hook(lambda module, inputs: (inputs[0].float(),))


def load_runtime(args, suite: dict, rows: list[dict]):
    """Load the selected official runtime and return its Target plus a fixed-prefix generator."""
    import torch
    from transformers import AutoModelForCausalLM

    dtype = getattr(torch, args.dtype)
    target_path = model_path(suite["target"], args.models_dir)
    max_length = max(len(r["prompt_token_ids"]) for r in rows) + args.max_tokens + args.budget + 16
    if args.method == "treegraft":
        from graft.combine_tree import CombineDraftGenerator, CombinePropagator

        depth, top_k = 5, 10
        # Upstream temporarily expands frontiers before pruning the final candidate budget.
        max_length += depth * top_k ** 2
        draft = CombineDraftGenerator(draft_strategy="combine", scheduler_family="treegraft_scheduler",
            scheduler_budget_b0=5, scheduler_pair_id="target-small", scheduler_task_id=rows[0]["task"],
            total_tokens=args.budget, depth=depth, top_k=top_k, max_matching_ngram_size=5,
            stop_on_relax_zero=True, legacy_total_token_stop=False, reselect_frontier_by_middle_score=True,
            reselect_frontier_lookback_layers=5, combine_layers=[],
            middle_model_path=model_path("Qwen/Qwen3-0.6B", args.models_dir),
            middle_max_length=max_length, middle_device_map="cuda:0", middle_dtype=dtype)
        runtime = CombinePropagator(base_model_path=target_path, draft_generator=draft,
            max_length=max_length, device_map="cuda:0", dtype=dtype)
        target = runtime.base_model
        configure_target_head(target, args.target_head_dtype)

        @torch.inference_mode()
        def generate(ids):
            """Return official committed output and step count, clipping only the requested output boundary."""
            draft.set_scheduler_prompt_len_tokens(ids.shape[1])
            output, _, step = runtime.generate(input_ids=ids, attention_mask=torch.ones_like(ids),
                temperature=0, max_new_tokens=args.max_tokens, log=True)
            return output[0, ids.shape[1]:], {"decode_rounds": step + 1}

        # Keep the official three-round scheduler calibration, using the frozen first prefix.
        ids = torch.tensor([rows[0]["prompt_token_ids"]], device=target.device)
        draft.set_scheduler_force_warmup_mode(True)
        draft.set_scheduler_force_warmup_collection(target_timing=True, middle_timing=True)
        draft.set_scheduler_forced_layers(list(range(1, depth + 1)))
        try:
            for _ in range(3):
                generate(ids)
            target_ms = statistics.mean(r["elapsed_ms"] for r in runtime.last_target_forward_records if r["elapsed_ms"] > 0)
            middle_ms = statistics.mean(r["elapsed_ms"] for r in runtime.last_middle_forward_records if r["elapsed_ms"] > 0)
        finally:
            draft.set_scheduler_force_warmup_mode(False)
            draft.set_scheduler_force_warmup_collection(target_timing=False, middle_timing=False)
            draft.set_scheduler_forced_layers(None)
        draft.finalize_scheduler_after_warmup(A_t=None, T_tgt=target_ms, T_mid=middle_ms)
        return target, generate

    from model import DFlashDraftModel

    target = AutoModelForCausalLM.from_pretrained(target_path, dtype=dtype,
                                               attn_implementation="sdpa").to("cuda:0").eval()
    configure_target_head(target, args.target_head_dtype)
    draft = DFlashDraftModel.from_pretrained(model_path("z-lab/Qwen3-8B-DFlash-b16", args.models_dir),
        dtype=dtype, attn_implementation=args.draft_attention).to("cuda:0").eval()
    eos = target.generation_config.eos_token_id
    stop_tokens = eos if isinstance(eos, list) else [eos]
    if args.method == "ddtree":
        from ddtree import ddtree_generate

        def generate(ids):
            """Use DDTree's public generator with its checkpoint block size and explicit node budget."""
            result = ddtree_generate(model=draft, target=target, input_ids=ids,
                mask_token_id=draft.mask_token_id, max_new_tokens=args.max_tokens,
                block_size=draft.block_size, stop_token_ids=stop_tokens,
                temperature=0, tree_budget=args.budget)
            return result.output_ids[0, ids.shape[1]:], {"decode_rounds": result.decode_rounds,
                "acceptance_lengths": result.acceptance_lengths, "stage_times": result.stage_times}
    else:
        from benchmark import _build_cascade_runners
        from generation import D3Generator

        dta = DFlashDraftModel.from_pretrained(str(args.dta), dtype=dtype,
            attn_implementation=args.draft_attention).to("cuda:0").eval()
        dta_runner, verify_runner = _build_cascade_runners("d3", 32, dta, target)
        runtime = D3Generator(target=target, draft=draft, dta=dta, block_size=draft.block_size,
                              block_size_2=32, dta_cascade_runner=dta_runner, verify_cascade_runner=verify_runner)

        def generate(ids):
            """Use D²SD's official cascade and retain the returned committed-length observations."""
            result = runtime.generate(input_ids=ids, max_new_tokens=args.max_tokens,
                stop_token_ids=stop_tokens, temperature=0)
            return result.output_ids[0, ids.shape[1]:], {"acceptance_lengths": result.acceptance_lengths,
                                                       "decode_rounds": len(result.acceptance_lengths)}
    return target, generate


def run(args, suite: dict) -> dict:
    """Measure an official runtime against same-model causal decoding, reporting numerical mismatches.

    These are C=1 Transformers measurements, including prefill. SDPA Draft
    execution is explicitly labelled and is not the upstream FlashAttention preset.
    """
    if args.method == "whiflash":
        raise ValueError(suite["upstream"]["whiflash"]["unavailable"])
    if args.method == "d2sd" and args.dta is None:
        raise ValueError("D²SD requires --dta with the unpublished second-Draft checkpoint")
    revision = subprocess.check_output(["git", "-C", str(args.checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != suite["upstream"][args.method]["revision"]:
        raise ValueError("Official checkout revision differs from suite.yaml")
    sys.path.insert(0, str(args.checkout.absolute()))
    import torch
    import transformers
    from transformers import DynamicCache

    rows = load_inputs(args.input, args.limit)
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    target, candidate = load_runtime(args, suite, rows)
    eos = target.generation_config.eos_token_id
    stop_tokens = set(eos if isinstance(eos, list) else [eos])
    baseline_cache = None
    if args.method == "treegraft":
        from graft.model.kv_cache import initialize_past_key_values
        baseline_cache = initialize_past_key_values(target,
            max(len(row["prompt_token_ids"]) for row in rows) + args.max_tokens + 1)

    @torch.inference_mode()
    def baseline(ids):
        """Decode causally with the very same Target weights, attention and model-specific KV."""
        if args.method == "treegraft":
            target.model.tree_mask = None
            target.model.tree_mode = None
            cache, _, lengths = baseline_cache
            lengths.zero_()
        else:
            cache = DynamicCache()
        tokens = []
        for _ in range(args.max_tokens):
            out = target(input_ids=ids, past_key_values=cache, use_cache=True)
            token = out.logits[0, -1].argmax()
            tokens.append(int(token))
            if int(token) in stop_tokens:
                break
            ids = token.reshape(1, 1)
        return tokens, {}

    @torch.inference_mode()
    def execute(generate, row):
        """Return synchronized end-to-end time and native output IDs for a frozen input."""
        ids = torch.tensor([row["prompt_token_ids"]], device=target.device)
        torch.cuda.synchronize()
        started = time.perf_counter()
        output, details = generate(ids)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - started
        tokens = output.tolist() if isinstance(output, torch.Tensor) else output
        tokens = tokens[:args.max_tokens]
        first_eos = next((i for i, token in enumerate(tokens) if token in stop_tokens), None)
        if first_eos is not None:
            tokens = tokens[:first_eos + 1]
        return {"id": row["id"], "tokens": tokens, "elapsed_s": elapsed, **details}

    manifest = {"backend": "official-transformers", "method": args.method, "revision": revision,
                "torch": torch.__version__, "transformers": transformers.__version__, "dtype": args.dtype,
                "gpu": torch.cuda.get_device_name(), "target_head_dtype": args.target_head_dtype,
                "target_attention": "eager" if args.method == "treegraft" else "sdpa",
                "draft_attention": "eager" if args.method == "treegraft" else args.draft_attention,
                "budget": args.budget, "max_tokens": args.max_tokens, "repetitions": args.repetitions,
                "rows": len(rows), "seed": args.seed, "device": args.gpu,
                "scope": "screening" if args.limit is not None else "evaluation",
                "timing": "synchronized request wall time including prefill; C=1",
                "input": str(args.input.absolute())}
    patch_record = args.checkout / "source-patch.json"
    if patch_record.exists():
        manifest["source_patch"] = json.loads(patch_record.read_text())
    prepare_run(args.output, manifest, args.input, args.resume, runners=("official",))
    generators = {"target": baseline, args.method: candidate}
    checked = {}
    for name, generate in generators.items():
        records = collect(args.output / f"{name}-check.jsonl", rows, lambda row: execute(generate, row))
        checked[name] = [r["tokens"] for r in records]
    self_repeat = [r["tokens"] for r in collect(args.output / "target-self-repeat.jsonl", rows,
                                               lambda row: execute(baseline, row))]
    target_check = compare(checked["target"], self_repeat)
    if not target_check["passed"]:
        raise RuntimeError("Official Target self-repeat failed")
    status = {"sequence_check": compare(checked["target"], checked[args.method]),
              "target_self_repeat": target_check, "performance": {}}
    status["comparison_scope"] = "matched-output" if status["sequence_check"]["passed"] else "numerical-diagnostic"
    for name, generate in generators.items():
        execute(generate, rows[0])
        status["performance"][name] = []
    for repetition in range(args.repetitions):
        order = list(generators)
        random.Random(args.seed + repetition).shuffle(order)
        for name in order:
            records = collect(args.output / f"{name}-repeat-{repetition}.jsonl", rows,
                              lambda row: execute(generators[name], row))
            status["performance"][name].append(1000 * sum(r["elapsed_s"] for r in records) /
                                               sum(len(r["tokens"]) for r in records))
            status.setdefault("repeat_checks", {}).setdefault(name, []).append(
                compare(checked["target"], [r["tokens"] for r in records]))
            write_json(args.output / "status.json", status)
    if any(not check["passed"] for checks in status["repeat_checks"].values() for check in checks):
        status["comparison_scope"] = "numerical-diagnostic"
    status["metric"] = "e2e_ms_per_token"
    status["state"] = "completed"
    write_json(args.output / "status.json", status)
    return status
