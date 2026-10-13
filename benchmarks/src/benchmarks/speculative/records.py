# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the Foretoken project
"""Compare native token IDs across baseline backends without decoding or retokenizing."""

import hashlib
import fcntl
import json
import shutil
from contextlib import contextmanager
from pathlib import Path


_RUNNER_SOURCES = {
    "native": ("native.py", "process.py"),
    "reference": ("reference.py", "tree.py"),
    "official": ("adapters.py",),
    "matrix": ("matrix.py",),
}


@contextmanager
def run_lock(directory: Path):
    """Exclude overlapping local runners; distributed schedulers must assign separate outputs.

    Local disk and NFS views need not share a flock domain. Cross-host ownership
    belongs to the scheduler, not this process-local launch boundary.
    """
    directory.parent.mkdir(parents=True, exist_ok=True)
    with (directory.parent / f".{directory.name}.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"A runner already owns {directory}") from error
        yield


def write_json(path: Path, value: dict) -> None:
    """Publish a complete status snapshot for concurrent progress/report readers."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def snapshot_sources(directory: Path) -> None:
    """Preserve the experiment implementation used by a runner in its new result group."""
    destination = directory / "source"
    destination.mkdir()
    for source in Path(__file__).parent.glob("*.py"):
        shutil.copy2(source, destination / source.name)


def prepare_run(directory: Path, manifest: dict, inputs: Path, resume: bool,
                *, runners: tuple[str, ...]) -> None:
    """Snapshot a new run, or resume with matching inputs, settings and selected execution code.

    Keep all experiment sources for provenance; reporting and unrelated backends
    do not constrain resumption. Matrix callers include their selected child runners.
    """
    manifest["input_sha256"] = hashlib.sha256(inputs.read_bytes()).hexdigest()
    path = directory / "manifest.json"
    if resume:
        if json.loads(path.read_text()) != manifest:
            raise ValueError(f"Run settings or inputs changed; use a new output directory: {directory}")
        sources = {"__main__.py", "records.py", "data.py"}
        for runner in runners:
            sources.update(_RUNNER_SOURCES[runner])
        for name in sorted(sources):
            source = Path(__file__).with_name(name)
            saved = directory / "source" / source.name
            if not saved.exists() or source.read_bytes() != saved.read_bytes():
                raise ValueError(f"Experiment source changed; use a new output directory: {source.name}")
    else:
        directory.mkdir(parents=True, exist_ok=False)
        snapshot_sources(directory)
        path.write_text(json.dumps(manifest, indent=2) + "\n")


def collect(path: Path, rows: list[dict], execute) -> list[dict]:
    """Append completed requests durably and resume the remaining ordered frozen inputs."""
    raw = path.read_bytes() if path.exists() else b""
    if raw and not raw.endswith(b"\n"):
        # Only newline-terminated rows are committed if a process dies during a large trace write.
        end = raw.rfind(b"\n") + 1
        path.with_suffix(path.suffix + ".partial").write_bytes(raw[end:])
        with path.open("r+b") as stream:
            stream.truncate(end)
        raw = raw[:end]
    records = [json.loads(line) for line in raw.splitlines()]
    if [r["id"] for r in records] != [r["id"] for r in rows[:len(records)]]:
        raise ValueError(f"Saved request order differs from inputs: {path}")
    with path.open("a") as stream:
        for row in rows[len(records):]:
            record = {"id": row["id"], **execute(row)}
            stream.write(json.dumps(record) + "\n")
            stream.flush()
            records.append(record)
            if len(records) % 8 == 0 or len(records) == len(rows):
                print(f"{path.name}: {len(records)}/{len(rows)}", flush=True)
    return records


def compare(reference: list[list[int]], samples: list[list[int]]) -> dict:
    """Report sequence matches and first divergence positions for numerical diagnostics."""
    mismatches = []
    for index, (expected, actual) in enumerate(zip(reference, samples, strict=True)):
        if expected != actual:
            position = next((i for i, (a, b) in enumerate(zip(expected, actual)) if a != b),
                            min(len(expected), len(actual)))
            mismatches.append({"index": index, "position": position,
                               "reference": expected[position:position + 1],
                               "candidate": actual[position:position + 1]})
    return {"matches": len(samples) - len(mismatches), "samples": len(samples),
            "passed": not mismatches, "first_divergences": mismatches}
