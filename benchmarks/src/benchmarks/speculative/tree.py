# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Greedy reference tree executor using Transformers 4.55 model and cache APIs.

This is an experimental reference backend, not a vLLM adapter. Each request owns
model-specific KV. Target alone commits tokens; proposal trees never commit output.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterator

import torch
from transformers import AutoModelForCausalLM, DynamicCache


@dataclass
class TokenTree:
    """Topologically ordered candidate nodes with model provenance, excluding the prefix."""

    tokens: list[int] = field(default_factory=list)
    parents: list[int] = field(default_factory=list)
    depths: list[int] = field(default_factory=list)
    sources: list[list[str]] = field(default_factory=list)

    def add(self, token: int, parent: int, source: str) -> int:
        """Append a candidate whose parent is already present, returning its node index."""
        index = len(self.tokens)
        self.tokens.append(token)
        self.parents.append(parent)
        self.depths.append(1 if parent < 0 else self.depths[parent] + 1)
        self.sources.append([source])
        return index

    def ancestors(self, index: int) -> list[int]:
        """Return the node's inclusive path, used for mask construction and KV selection."""
        path = []
        while index >= 0:
            path.append(index)
            index = self.parents[index]
        return list(reversed(path))

    def match(self, tokens: list[int]) -> list[int]:
        """Find the longest candidate path matching the Target's committed suffix."""
        children = {(parent, token): i for i, (parent, token) in enumerate(
            zip(self.parents, self.tokens, strict=True))}
        parent, path = -1, []
        for token in tokens:
            index = children.get((parent, token))
            if index is None:
                break
            path.append(index)
            parent = index
        return path


def merge_trees(trees: list[TokenTree], budget: int | None = None) -> TokenTree:
    """Deduplicate parent/token pairs in breadth-first budget order across fixed model order.

    Allocation proceeds by depth and then round-robin source; no Target output or
    uncalibrated cross-model probabilities influence admission.
    """
    merged = TokenTree()
    maps = [{-1: -1} for _ in trees]
    pairs = {}
    maximum = max((max(tree.depths, default=0) for tree in trees), default=0)
    for depth in range(1, maximum + 1):
        levels = [[i for i, d in enumerate(tree.depths) if d == depth] for tree in trees]
        for offset in range(max((len(level) for level in levels), default=0)):
            for source, (tree, level) in enumerate(zip(trees, levels, strict=True)):
                if offset >= len(level):
                    continue
                index = level[offset]
                parent = maps[source].get(tree.parents[index])
                if parent is None:
                    continue
                key = parent, tree.tokens[index]
                if key in pairs:
                    node = pairs[key]
                    merged.sources[node] = sorted(set(merged.sources[node] + tree.sources[index]))
                elif budget is None or len(merged.tokens) < budget:
                    node = merged.add(key[1], parent, tree.sources[index][0])
                    pairs[key] = node
                else:
                    continue
                maps[source][index] = node
    return merged


class Model:
    """Own resident weights; each session prefills and owns its independent GPU KV."""

    def __init__(self, name: str, path: str, dtype: str, device: str = "cuda:0") -> None:
        self.name = name
        self.device = torch.device(device)
        self.net = AutoModelForCausalLM.from_pretrained(
            path, torch_dtype=getattr(torch, dtype), attn_implementation="sdpa",
            local_files_only=True).to(self.device).eval()
        self.dtype = self.net.dtype
        eos = self.net.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    @torch.inference_mode()
    def session(self, prompt: list[int]) -> Session:
        """Prefill a fresh session for the caller, keeping KV on the model's device."""
        session = Session(self)
        session.ensure(prompt)
        return session


class Session:
    """Own one request's model KV, computed prefix and any uncommitted candidate tree."""

    def __init__(self, model: Model) -> None:
        self.model = model
        self.cache = DynamicCache()
        self.tokens: list[int] = []
        self.next_logits: torch.Tensor | None = None
        self.tree: TokenTree | None = None
        self.tree_logits: list[torch.Tensor] = []
        self.catchup_tokens = 0
        self.forward_events: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @torch.inference_mode()
    def ensure(self, committed: list[int]) -> int:
        """Compute a missing committed suffix using this model's own weights, returning its size."""
        common = 0
        for left, right in zip(self.tokens, committed):
            if left != right:
                break
            common += 1
        if common == len(self.tokens) == len(committed):
            return 0
        if common < len(self.tokens):
            # Recompute the boundary token to restore its next-token logits as well as KV.
            common = max(0, common - 1)
            self.cache.crop(common)
            self.tokens = self.tokens[:common]
        suffix = committed[common:]
        ids = torch.tensor([suffix], device=self.model.device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.model.net(input_ids=ids, past_key_values=self.cache,
                             use_cache=True, logits_to_keep=1)
        end.record()
        self.forward_events.append(("catchup", start, end))
        self.next_logits = out.logits[0, -1].float()
        self.tokens = list(committed)
        self.catchup_tokens += len(suffix)
        return len(suffix)

    @torch.inference_mode()
    def tree_inputs(self, tree: TokenTree, first: int = 0) -> dict:
        """Prepare explicit ancestor visibility and logical RoPE positions for new tree nodes."""
        length = len(self.tokens)
        count = len(tree.tokens)
        rows = count - first
        mask = torch.full((rows, length + count), torch.finfo(self.model.dtype).min,
                          dtype=self.model.dtype, device=self.model.device)
        mask[:, :length] = 0
        for row, node in enumerate(range(first, count)):
            positions = [length + index for index in tree.ancestors(node)]
            mask[row, positions] = 0
        ids = torch.tensor([tree.tokens[first:]], device=self.model.device)
        positions = torch.tensor([[length + depth - 1 for depth in tree.depths[first:]]],
                                 device=self.model.device)
        physical = torch.arange(length + first, length + count, device=self.model.device)
        return {"input_ids": ids, "attention_mask": {"full_attention": mask[None, None]},
                "position_ids": positions, "cache_position": physical, "use_cache": True}

    @torch.inference_mode()
    def forward_tree(self, tree: TokenTree, first: int = 0) -> torch.Tensor:
        """Append new tree nodes in one model forward, recording deferred CUDA-event timing."""
        inputs = self.tree_inputs(tree, first)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        out = self.model.net(**inputs, past_key_values=self.cache)
        end.record()
        self.forward_events.append(("tree", start, end))
        logits = out.logits[0].float()
        self.tree = tree
        self.tree_logits.extend(logits.unbind(0))
        return logits

    @torch.inference_mode()
    def commit(self, committed: list[int]) -> None:
        """Select computed KV on the accepted path; missing correction/suffix stays pending."""
        if self.tree is None:
            return
        length = len(self.tokens)
        path = self.tree.match(committed[length:])
        path = [index for index in path if index < len(self.tree_logits)]
        if path:
            indices = torch.tensor(list(range(length)) + [length + i for i in path],
                                   device=self.model.device)
            for layer in self.cache.layers:
                layer.keys = layer.keys.index_select(-2, indices)
                layer.values = layer.values.index_select(-2, indices)
            self.tokens.extend(self.tree.tokens[i] for i in path)
            self.next_logits = self.tree_logits[path[-1]]
        else:
            self.cache.crop(length)
        self.tree = None
        self.tree_logits = []


@torch.inference_mode()
def propose_trees(sessions: list[Session], depth: int, width: int = 1,
                  branch_at: int = 1, budget: int | None = None) -> list[TokenTree]:
    """Expand the forest by depth and stop GPU work once the merged node budget is full.

    All candidates in the final level are selected from existing logits. Their
    uncomputed KV is deferred until committed-prefix catch-up, including bonus.
    """
    trees = [TokenTree() for _ in sessions]
    frontiers = [[(-1, session.next_logits)] for session in sessions]
    for level in range(1, depth + 1):
        first = [len(tree.tokens) for tree in trees]
        for session, tree, frontier in zip(sessions, trees, frontiers, strict=True):
            for parent, logits in frontier:
                if parent >= 0 and tree.tokens[parent] in session.model.eos:
                    continue
                count = width if level == branch_at else 1
                tokens = (logits.argmax().view(1) if count == 1 else
                          torch.argsort(logits, descending=True, stable=True)[:count]).tolist()
                for token in tokens:
                    tree.add(token, parent, session.model.name)
        if all(len(tree.tokens) == offset for tree, offset in zip(trees, first, strict=True)):
            break
        stop = level == depth or (budget is not None and len(merge_trees(trees).tokens) >= budget)
        for index, (session, tree, offset) in enumerate(zip(sessions, trees, first, strict=True)):
            session.tree = tree
            if stop or len(tree.tokens) == offset:
                frontiers[index] = []
            else:
                logits = session.forward_tree(tree, offset)
                frontiers[index] = [(offset + row, value) for row, value in enumerate(logits)]
        if stop:
            break
    return trees


@torch.inference_mode()
def verify(session: Session, tree: TokenTree, remaining: int,
           prefix: list[int]) -> list[int]:
    """Verify pending prefix tokens and candidates together, returning only newly committed output."""
    pending = prefix[len(session.tokens):]
    combined = tree
    if pending:
        combined, parent = TokenTree(), -1
        for token in pending:
            parent = combined.add(token, parent, session.model.name)
        offset = len(pending)
        for index, token in enumerate(tree.tokens):
            parent = offset - 1 if tree.parents[index] < 0 else offset + tree.parents[index]
            node = combined.add(token, parent, tree.sources[index][0])
            combined.sources[node] = tree.sources[index]
    logits = session.forward_tree(combined) if combined.tokens else None
    predictions = logits.argmax(-1).tolist() if logits is not None else []
    root = predictions[len(pending) - 1] if pending else int(session.next_logits.argmax())
    predictions = predictions[len(pending):]
    children = {(parent, token): i for i, (parent, token) in enumerate(
        zip(tree.parents, tree.tokens, strict=True))}
    parent, token, output = -1, root, []
    while len(output) < remaining:
        output.append(token)
        if token in session.model.eos:
            break
        node = children.get((parent, token))
        if node is None:
            break
        parent, token = node, predictions[node]
    return output


def cascade_propose(small: Session, middle: Session, prefix: list[int], outer: int,
                    inner: int) -> tuple[TokenTree, int]:
    """Generate M's temporary continuation using S→M verification; only outer T may commit it."""
    temporary = list(prefix)
    count = 0
    while len(temporary) - len(prefix) < outer:
        small.ensure(temporary)
        candidate = propose_trees([small], inner)[0]
        chunk = verify(middle, candidate, outer - len(temporary) + len(prefix), temporary)
        temporary.extend(chunk)
        small.commit(temporary)
        middle.commit(temporary)
        count += 1
        if chunk[-1] in middle.model.eos:
            break
    tree, parent = TokenTree(), -1
    for token in temporary[len(prefix):]:
        parent = tree.add(token, parent, middle.model.name)
    return tree, count


class TreeRuntime:
    """Own resident weights and execute one greedy request at a time for the reference server."""

    def __init__(self, models: dict[str, Model], target: str, config: dict) -> None:
        self.models = models
        self.target = target
        self.config = config
        self.last_record: dict = {}

    def generate(self, prompt: list[int], maximum: int, trace: bool = False) -> Iterator[list[int]]:
        """Yield committed chunks and publish request evidence, releasing all request KV on exit."""
        started = time.perf_counter()
        target = self.models[self.target].session(prompt)
        names = self.config.get("drafts", [])
        drafts = [self.models[name].session(prompt) for name in names]
        committed = list(prompt)
        rounds = []
        timings = []
        try:
            while len(committed) - len(prompt) < maximum:
                record = {"round": len(rounds), "prefix_length": len(committed)}
                sessions = [target, *drafts]
                offsets = [len(session.forward_events) for session in sessions]
                start = time.perf_counter()
                # Target's pending bonus is fused into its next tree forward below.
                catchup = {self.target: 0}
                record["target_pending_tokens"] = len(committed) - len(target.tokens)
                for name, session in zip(names, drafts, strict=True):
                    catchup[name] = session.ensure(committed)
                record["catchup_ms"] = 1000 * (time.perf_counter() - start)
                record["catchup_tokens"] = catchup
                start = time.perf_counter()
                if "inner_depth" in self.config:
                    proposal, inner_rounds = cascade_propose(drafts[0], drafts[1], committed,
                        self.config["depth"], self.config["inner_depth"])
                    trees = [proposal]
                    record["inner_rounds"] = inner_rounds
                else:
                    trees = propose_trees(drafts, self.config["depth"], self.config.get("width", 1),
                                          self.config.get("branch_at", 1), self.config.get("budget")) if drafts else []
                record["proposal_ms"] = 1000 * (time.perf_counter() - start)
                start = time.perf_counter()
                merged = merge_trees(trees, self.config.get("budget"))
                record["merge_ms"] = 1000 * (time.perf_counter() - start)
                start = time.perf_counter()
                chunk = verify(target, merged, maximum - (len(committed) - len(prompt)), committed)
                record["verify_ms"] = 1000 * (time.perf_counter() - start)
                committed.extend(chunk)
                start = time.perf_counter()
                target.commit(committed)
                for session in drafts:
                    session.commit(committed)
                record["commit_enqueue_ms"] = 1000 * (time.perf_counter() - start)
                record.update(nodes=len(merged.tokens), proposed=sum(len(t.tokens) for t in trees),
                              committed=len(chunk),
                              target_forward_rows=len(merged.tokens) + record["target_pending_tokens"])
                if trace:
                    record.update(tree={"tokens": merged.tokens, "parents": merged.parents,
                                        "sources": merged.sources}, output=chunk)
                rounds.append(record)
                timings.append({session.model.name: session.forward_events[offset:]
                                for session, offset in zip(sessions, offsets, strict=True)})
                yield chunk
                if chunk[-1] in target.model.eos:
                    break
            torch.cuda.synchronize()
            for record, spans in zip(rounds, timings, strict=True):
                record["model_gpu_ms"] = {name: {
                    kind: sum(start.elapsed_time(end) for stage, start, end in events if stage == kind)
                    for kind in ("catchup", "tree")} for name, events in spans.items()}
                record["model_forward_calls"] = {name: {
                    kind: sum(stage == kind for stage, _, _ in events)
                    for kind in ("catchup", "tree")} for name, events in spans.items()}
            self.last_record = {"tokens": committed[len(prompt):], "rounds": rounds,
                                "elapsed_s": time.perf_counter() - started,
                                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                "timing_kind": "host stages with natural token-read synchronization"}
        finally:
            # Request state is never retained after completion or client cancellation.
            del target, drafts
