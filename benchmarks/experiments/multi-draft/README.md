<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# Multi-Draft baselines

English | [简体中文](README_zh.md)

Freeze one workload and compare speculative-decoding baselines with a matched Target-only run. This experimental platform has three separate execution groups:

| Group | Implemented baselines | Measurement |
| --- | --- | --- |
| Native vLLM 0.26 | Target-only, Qwen3-0.6B, n-gram, EAGLE-3, DFlash | Existing `foretoken perf`; supports concurrency |
| Transformers reference | Target-only, single S/M Draft, two branches from S, static S+M tree merge | C=1, greedy, explicit ancestor masks, no CUDA graph |
| Official Transformers | TreeGraft, DDTree, D²SD adapter | C=1, frozen token IDs, matched causal Target decoding |

Run from the repository root after `pip install -e .`. Use a CUDA environment for inference and choose available GPUs with `nvidia-smi`. The reference runtime uses Transformers 4.55.4 and PyTorch 2.6; native execution uses an existing vLLM 0.26 environment.

## Freeze the data

```bash
python -m benchmarks.speculative prepare --output results/multi-draft/core8
```

This prepares 914 prompts: GSM8K 128, MATH-500 128, AIME25 30, HumanEval 164, MBPP 128, LiveCodeBench 128, MT-Bench 80 and Alpaca 128. `--tokenizer data/models/Qwen/Qwen3-8B` reuses a local tokenizer. `--tasks gsm8k mt-bench --limit 2` prepares a small screening set in a different directory.

Each task has a JSONL file containing source IDs, messages, original turns and frozen token IDs. `manifest.json` records source revisions, splits, sampling indices, seed, template and file digests. All methods consume those same IDs with thinking disabled. MT-Bench uses the first user turn; subsequent turns are retained but are not measured by this protocol. These are inference inputs, not a model-quality scorer.

## Run the matrix

After preparing the official environments below, set `FORETOKEN_VLLM_PYTHON` and `FORETOKEN_REFERENCE_PYTHON` to the corresponding interpreters and choose unused GPUs:

```bash
python -m benchmarks.speculative matrix \
  --input results/multi-draft/core8 --output results/multi-draft/full \
  --gpus 0 1 --engine-python "$FORETOKEN_VLLM_PYTHON" \
  --reference-python "$FORETOKEN_REFERENCE_PYTHON"
```

The default matrix has 56 task/backend groups: eight tasks with native BF16, native FP32, reference, TreeGraft BF16, TreeGraft FP32, DDTree BF16 and DDTree FP32. Each native group includes all 18 conditions in the suite, and each reference group includes five. Official groups include their own causal Target. Every group stays on its assigned GPU. Defaults are C=1, a 2048-token output cap and three repetitions; EOS may end a request earlier. This is a long GPU workload. Use a persistent terminal session; `--tasks gsm8k --backends reference ddtree --limit 2 --max-tokens 32 --repetitions 1` checks execution first.

Native FP32 uses FlexAttention, requests highest PyTorch matmul precision and `NVIDIA_TF32_OVERRIDE=0`, disables model torch.compile, and requests FULL CUDA graphs. Kernel-specific arithmetic still follows the backend implementation. These are separate numerical controls, not BF16 deployment results. `native-strict` is an optional check-only backend; batch invariance does not guarantee equality between causal and speculative execution in this suite. WhiFlash and D²SD remain explicitly unavailable until their missing artifacts are supplied.

Add `--plan-only` to save the job plan without launching GPU work, then add `--resume` to the same command to run it. `--resume` also continues interrupted runs with identical data, settings and execution sources for the selected backends. Full source snapshots are retained, but changes to reporting or unselected backends do not prevent resumption. Completed request rows and native performance repetitions are reused; interrupted native repetitions are archived and rerun. A changed experiment requires a new output directory. Matrix exit is nonzero if any child fails.

The runner writes `status.json` while running and `summary.json` / `report.md` at the end. Refresh the report at any time with `python -m benchmarks.speculative report --input results/multi-draft/full`. The execution table lists every planned group; measurement tables show only published evidence. Reports separate metrics and precision groups and omit speedup claims for numerical-diagnostic rows. Raw checks, timings, source snapshots and logs remain in each task/backend directory. `summary.csv` provides the same measurements with host/device placement, repeat counts and standard deviations. Reports calculate speedups from these measurements only when all requested repetitions and output checks pass for both the candidate and its Target; individual runners save measurements and checks without derived speedups.

`matrix` schedules a local GPU pool. When distributing whole groups across hosts, give every group one writer and a separate output directory; local disk and NFS views may not share a file-lock domain. Keep each candidate and its Target on the same host/device and preserve running source snapshots when updating reports.

## Native baselines

Set `FORETOKEN_VLLM_PYTHON` to your vLLM environment's Python executable. The runner resolves model IDs through `data/models/` and its existing Hub cache before using the configured Hugging Face cache.

First check strict sequence equality in a new result directory:

```bash
python -m benchmarks.speculative native \
  --engine-python "$FORETOKEN_VLLM_PYTHON" --gpu 0 \
  --input results/multi-draft/core8/gsm8k.jsonl \
  --output results/multi-draft/gsm8k-strict \
  --mode strict --attention-backend FLEX_ATTENTION
```

Then measure the normal execution mode in its own group:

```bash
python -m benchmarks.speculative native \
  --engine-python "$FORETOKEN_VLLM_PYTHON" --gpu 0 \
  --input results/multi-draft/core8/gsm8k.jsonl \
  --output results/multi-draft/gsm8k-performance --mode performance
```

Target-only is always included. `--methods small ngram eagle3 dflash` selects families; their gamma grids and checkpoints live in [suite.yaml](suite.yaml). Change that file or pass `--suite PATH` before the subcommand for a different checkpoint or budget. The default EAGLE-3 checkpoint preserves the existing community-checkpoint baseline; it is not asserted to be the strongest available checkpoint.

The default measurement is BF16, greedy, 2048 output tokens and three repetitions. Use `--limit 2 --max-tokens 16 --repetitions 1` for execution screening, or `--concurrency 4` for a service load comparison. Each condition owns its server and drains its process group at exit. Startup and compilation are outside the performance window. The runner saves raw Foretoken metrics, engine snapshots, native output IDs and first-divergence positions.

Strict mode performs sequence checks only. Normal mode saves sequence differences as numerical diagnostics and measures the actual deployment path; it does not claim strict sequence equality when differences occur. Failed Target self-repeat or request execution stops the affected comparison. Compare speed only with the Target-only run in the same group. The modes follow vLLM's distinction between [algorithmic losslessness and numerical stability](https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding).

For the separate FP32 control, add `--dtype float32 --attention-backend FLEX_ATTENTION --compilation-mode none` to a performance run. Native runs set `VLLM_FLOAT32_MATMUL_PRECISION=highest` and `NVIDIA_TF32_OVERRIDE=0`; this does not certify every custom kernel as IEEE FP32. Strict checks report empirical sequence equality, not a hardware-independent numerical guarantee.

## Multi-Draft reference controls

Run with the reference environment's Python:

```bash
python -m benchmarks.speculative reference --gpu 0 \
  --input results/multi-draft/core8/gsm8k.jsonl \
  --output results/multi-draft/gsm8k-reference
```

The default pool uses FP32 Target weights and BF16 Draft weights. `branched` and `merge` each admit at most eight candidate nodes; the branch control uses one model, while merge uses two. Target alone commits tokens, and every model owns its KV. Only passing sequence checks enter repeated measurements; raw round traces retain node provenance and stage costs. Each request prefills fresh KV on the model's GPU and releases it when generation ends. Times include cold prefill and are reported as end-to-end milliseconds per output token, not service TPOT. Historical runs that included CPU prompt-cache transfers have a different timing protocol and must remain in separate result groups.

Reference timing does not establish native multi-Draft performance. Production paged-KV multi-Draft integration and trained multi-head Drafts are separate work. [SGLang's speculative-decoding guide](https://docs.sglang.io/docs/advanced_features/speculative_decoding) describes a mature single-Draft tree path for future engine comparisons.

`python -m benchmarks.speculative analyze --input results/multi-draft/gsm8k-reference/merge-check.jsonl --output results/multi-draft/coverage.json` reports each source's acceptance on the same committed prefixes and observed round/prompt oracle coverage. It sees only admitted nodes and includes EOS/output-cap censoring; it does not predict speed or unexecuted candidates.

## Official baselines

Set `FORETOKEN_TORCH_PYTHON` to an existing CUDA PyTorch 2.6 environment. Prepare pinned official checkouts and isolated Python overlays:

```bash
python -m benchmarks.speculative setup --engine-python "$FORETOKEN_TORCH_PYTHON"
```

Sources, environment versions and status are saved under ignored `reference/multi-draft/`. Base CUDA packages are inherited; the base environment is not modified. The adapter calls official model/generation APIs without changing their inference algorithms and bypasses upstream text-template loaders to use the frozen IDs.

TreeGraft receives a recorded import-only compatibility patch: each model loader imports its selected architecture, avoiding a broken doc decorator in the unused Llama module when running Qwen3.

```bash
reference/multi-draft/treegraft/.venv/bin/python -m benchmarks.speculative official \
  --method treegraft --checkout reference/multi-draft/treegraft --gpu 0 \
  --input results/multi-draft/core8/gsm8k.jsonl \
  --output results/multi-draft/gsm8k-treegraft --budget 63

reference/multi-draft/ddtree/.venv/bin/python -m benchmarks.speculative official \
  --method ddtree --checkout reference/multi-draft/ddtree --gpu 0 \
  --input results/multi-draft/core8/gsm8k.jsonl \
  --output results/multi-draft/gsm8k-ddtree --budget 63
```

Official adapters use BF16 and report synchronized end-to-end time including prefill. DDTree defaults to SDPA for both models; this differs from its upstream FlashAttention Draft preset. Use `--draft-attention flash_attention_2` after installing a compatible FlashAttention package for that preset. TreeGraft's Qwen3 models use upstream eager attention regardless of that flag. It retains its scheduler's three-round calibration on the first frozen user prefix. EOS IDs and output-token limits terminate generation.

Use `--dtype float32` in a separate output group to diagnose numerical sequence differences. It changes both Target and Draft precision, disables TF32 matmul and is not a BF16 deployment performance result. Nonmatching sequences are labelled `numerical-diagnostic`.

`--target-head-dtype float32` changes only the Target output projection, shared by causal and speculative paths. It can remove BF16 output-logit ties, but does not guarantee equality because earlier layers also depend on query shape. Keep it separate from full FP32 and the upstream BF16 configuration.

WhiFlash's pinned [public repository](https://github.com/SamsungLabs/WhiFlash) only contains a placeholder README. Its setup status is `unavailable`. [D²SD](https://github.com/catnanami/D2-SD) has code but has not released the second Draft checkpoint; its environment is prepared and execution requires `--method d2sd --dta PATH`. Neither is reported as an executed baseline until its missing prerequisite is supplied.

Output directories must be new unless `--resume` is specified. Downloads, environments and raw results stay excluded from Git. Use the saved `status.json`, sequence checks and repetition files to distinguish completed execution, failed checks and unavailable baselines; small screening runs are not full benchmark results.
