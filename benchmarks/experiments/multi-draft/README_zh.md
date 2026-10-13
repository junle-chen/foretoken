<!-- SPDX-License-Identifier: Apache-2.0 -->
<!-- SPDX-FileCopyrightText: Copyright contributors to the Foretoken project -->

# 多 Draft 基线

[English](README.md) | 简体中文

先冻结一份输入，再用同组 Target-only 比较各推测解码基线。实验平台分成三个执行组：

| 执行组 | 已实现基线 | 测量方式 |
| --- | --- | --- |
| 原生 vLLM 0.26 | Target-only、Qwen3-0.6B、n-gram、EAGLE-3、DFlash | 复用 `foretoken perf`，支持并发 |
| Transformers 参考组 | Target-only、S/M 单 Draft、S 两分支、S+M 静态合并树 | C=1、greedy、显式祖先 mask、无 CUDA graph |
| 官方 Transformers 组 | TreeGraft、DDTree、D²SD adapter | C=1、冻结 token IDs、同模型因果解码对照 |

在仓库根目录执行 `pip install -e .` 后运行。推理需要 CUDA 环境，先用 `nvidia-smi` 选择可用 GPU。参考组使用 Transformers 4.55.4 和 PyTorch 2.6；原生组复用已有 vLLM 0.26 环境。

## 冻结数据

```bash
python -m benchmarks.speculative prepare --output results/multi-draft/core8
```

共准备 914 条：GSM8K 128、MATH-500 128、AIME25 30、HumanEval 164、MBPP 128、LiveCodeBench 128、MT-Bench 80、Alpaca 128。`--tokenizer data/models/Qwen/Qwen3-8B` 可复用本地 tokenizer。用 `--tasks gsm8k mt-bench --limit 2` 在新目录准备小规模筛查集。

每个任务生成一个 JSONL，包含来源 ID、messages、原始 turns 和冻结 token IDs。`manifest.json` 记录源 revision、split、抽样索引、seed、模板和文件摘要。所有方法使用同一 token 输入并关闭思考。MT-Bench 只测首轮用户问题，后续 turns 保留但不参与此协议。这些是推理输入，平台不承担任务质量评分。

## 运行完整矩阵

按下文准备官方环境，将 `FORETOKEN_VLLM_PYTHON` 和 `FORETOKEN_REFERENCE_PYTHON` 分别设置为对应环境的 Python，再选择空闲 GPU：

```bash
python -m benchmarks.speculative matrix \
  --input results/multi-draft/core8 --output results/multi-draft/full \
  --gpus 0 1 --engine-python "$FORETOKEN_VLLM_PYTHON" \
  --reference-python "$FORETOKEN_REFERENCE_PYTHON"
```

默认包含 56 个任务/后端组：八个任务分别运行原生 BF16、原生 FP32、参考组、TreeGraft BF16、TreeGraft FP32、DDTree BF16、DDTree FP32。每个原生组覆盖 suite 中全部 18 个条件，参考组覆盖五个条件，官方组包含各自的因果 Target。每组固定在分配的 GPU 上执行。默认 C=1、输出上限 2048、重复三次；EOS 可以提前结束请求。这是长时间 GPU 实验，请使用持久终端会话。先加 `--tasks gsm8k --backends reference ddtree --limit 2 --max-tokens 32 --repetitions 1` 可检查执行链路。

原生 FP32 使用 FlexAttention，设置最高 PyTorch matmul 精度与 `NVIDIA_TF32_OVERRIDE=0`，关闭模型 torch.compile，申请 FULL CUDA graphs；各自定义 kernel 的算术仍由后端实现决定。它是独立数值对照，不代表 BF16 部署性能。可选 `native-strict` 只检查输出；本套件中 batch invariance 并不保证因果解码与推测验证的序列相同。WhiFlash、D²SD 在补齐缺失产物前明确记为 unavailable。

加 `--plan-only` 可只保存任务计划，再用相同命令加 `--resume` 执行。`--resume` 也能恢复中断的实验，但数据、配置和所选后端的执行源码必须一致。完整源码快照仍会保留，修改报告或未选用的后端不会阻止恢复。已完成的请求和原生性能重复会被复用，中断的原生性能重复归档后重跑；更改实验必须使用新目录。任一子任务失败时，矩阵命令返回非零退出码。

运行中写入 `status.json`，结束后生成 `summary.json` 和 `report.md`。也可随时执行 `python -m benchmarks.speculative report --input results/multi-draft/full` 更新报告。执行状态表列出全部计划组，测量表只展示已发布的证据。报告分别呈现各指标、精度组，数值诊断行不显示加速结论。原始检查、时间、源码快照和日志保留在任务/后端子目录。`summary.csv` 同时保存主机/GPU、重复次数与标准差。加速比由报告统一计算，只有候选及其 Target 都通过检查并完成全部重复时才展示；各执行器只保存测量值和检查结果。

`matrix` 调度本机 GPU 池。跨主机分派整组实验时，每组必须有唯一写入者和独立输出目录；本地磁盘与 NFS 路径不一定共享文件锁。候选与 Target 保持在同主机、同 GPU 上比较，更新报告时保留正在执行的源码快照。

## 原生基线

把 `FORETOKEN_VLLM_PYTHON` 设置为 vLLM 环境的 Python。模型先从 `data/models/` 及其已有 Hub 缓存解析，再使用既定 Hugging Face 缓存。

先在新目录检查严格逐 token 一致性：

```bash
python -m benchmarks.speculative native \
  --engine-python "$FORETOKEN_VLLM_PYTHON" --gpu 0 \
  --input results/multi-draft/core8/gsm8k.jsonl \
  --output results/multi-draft/gsm8k-strict \
  --mode strict --attention-backend FLEX_ATTENTION
```

然后在独立组测正常执行模式：

```bash
python -m benchmarks.speculative native \
  --engine-python "$FORETOKEN_VLLM_PYTHON" --gpu 0 \
  --input results/multi-draft/core8/gsm8k.jsonl \
  --output results/multi-draft/gsm8k-performance --mode performance
```

每次自动加入 Target-only。`--methods small ngram eagle3 dflash` 选择方法；gamma 网格和 checkpoint 集中在 [suite.yaml](suite.yaml)。换 checkpoint 或预算时修改该文件，或在子命令前传 `--suite PATH`。默认 EAGLE-3 保留已有社区 checkpoint 对照，没有认定它是可用权重中最强的版本。

默认 BF16、greedy、输出上限 2048、重复三次。`--limit 2 --max-tokens 16 --repetitions 1` 用于运行筛查；`--concurrency 4` 用于服务并发比较。每个条件独立管理服务，并在退出时清空自己的进程组。启动和编译不计入性能窗口。保存原始 Foretoken 指标、引擎快照、输出 token IDs 和首次分歧位置。

严格模式只检查序列。正常模式把序列差异保留为数值诊断并测实际部署路径；存在差异时不主张严格序列一致。Target 自重复或请求执行失败会停止受影响的比较。加速比只能使用同组 Target-only。两种模式遵循 vLLM 对[算法无损与数值稳定性](https://docs.vllm.ai/en/latest/features/speculative_decoding/#lossless-guarantees-of-speculative-decoding)的区分。

独立 FP32 对照在性能命令中添加 `--dtype float32 --attention-backend FLEX_ATTENTION --compilation-mode none`。原生运行设置 `VLLM_FLOAT32_MATMUL_PRECISION=highest` 和 `NVIDIA_TF32_OVERRIDE=0`，不据此宣称所有自定义 kernel 都采用 IEEE FP32。严格检查给出实测序列是否相等，不构成跨硬件数值一致性保证。

## 多 Draft 参考对照

使用参考环境的 Python：

```bash
python -m benchmarks.speculative reference --gpu 0 \
  --input results/multi-draft/core8/gsm8k.jsonl \
  --output results/multi-draft/gsm8k-reference
```

默认 Target 为 FP32，Draft 为 BF16。`branched` 与 `merge` 都最多接纳八个候选节点，前者使用一个模型，后者使用两个。只有 Target 提交输出，各模型独立拥有 KV。通过序列检查的条件才进入重复测量；逐轮记录保留节点来源与阶段成本。每个请求直接在模型 GPU 上 prefill 并持有独立 KV，生成结束后释放。时间包含冷 prefill，报告端到端毫秒/输出 token，不称为服务 TPOT。历史运行包含 CPU prompt 缓存传输，计时协议不同，须保留在独立结果组中。

参考时间不建立原生多 Draft 性能结论。原生 paged-KV 多 Draft 集成、训练型多头 Draft 是独立工作。[SGLang 推测解码指南](https://docs.sglang.io/docs/advanced_features/speculative_decoding)提供成熟的单 Draft 树路径，可供后续引擎对照。

`python -m benchmarks.speculative analyze --input results/multi-draft/gsm8k-reference/merge-check.jsonl --output results/multi-draft/coverage.json` 计算同一确认前缀上各来源的接受长度及已观察到的逐轮/逐 prompt oracle 覆盖收益。只使用预算内接纳的节点，受 EOS/输出上限截断，不预测速度或未执行候选。

## 官方基线

把 `FORETOKEN_TORCH_PYTHON` 设置为已有 CUDA PyTorch 2.6 环境。准备固定 revision 的官方代码和独立 Python overlay：

```bash
python -m benchmarks.speculative setup --engine-python "$FORETOKEN_TORCH_PYTHON"
```

源码、环境版本和状态写入被 Git 忽略的 `reference/multi-draft/`。CUDA 包从基础环境继承，不修改基础环境。adapter 调用官方模型与生成 API，保留推理算法，绕过官方文本模板加载器，直接传冻结 IDs。

TreeGraft 应用并记录一个只调整 import 的兼容补丁：加载器仅导入选中架构，避免 Qwen3 运行被未使用的 Llama 模块文档装饰器阻断。

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

官方 adapter 使用 BF16，报告同步后的整请求时间，包含 prefill。DDTree 默认两个模型均使用 SDPA，与其官方 FlashAttention Draft 配置不同。安装兼容的 FlashAttention 后，用 `--draft-attention flash_attention_2` 运行该配置。TreeGraft 的 Qwen3 模型使用官方 eager attention，不受该参数影响；它在第一条冻结首轮前缀上保留三轮 scheduler 校准。输出按模型 EOS IDs 与 token 上限结束。

用 `--dtype float32` 在独立目录诊断数值序列分歧。它改变 Target 和 Draft 的精度并关闭 TF32 matmul，不属于 BF16 部署性能结果。序列不一致时标为 `numerical-diagnostic`。

`--target-head-dtype float32` 只改变 Target 输出投影精度，因果和推测路径使用同一个输出头。它能消除部分 BF16 logits 并列，但前面各层也会受 query 形状影响，因此不保证序列相等；与完整 FP32、官方 BF16 配置分别记录。

WhiFlash 固定版本的[公开仓库](https://github.com/SamsungLabs/WhiFlash)只有占位 README，setup 状态为 `unavailable`。[D²SD](https://github.com/catnanami/D2-SD) 有代码，但没有发布第二个 Draft checkpoint；环境可以准备，运行需传 `--method d2sd --dta PATH`。补齐缺失条件并实际执行之前，不把它们记为已运行的基线。

除非指定 `--resume`，输出目录必须是新目录。下载、环境和原始结果不进入 Git。通过保存的 `status.json`、序列检查与重复记录区分执行完成、检查失败和不可用基线；小规模筛查不等于完整 benchmark 结果。
