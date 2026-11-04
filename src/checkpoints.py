"""Immutable epoch snapshots, published together through one atomic index."""
import json
import os
from pathlib import Path
from uuid import uuid4
from src.dataset_spec import read_json, sha256

FORMAT_VERSION = 1
INDEX_NAME = "checkpoint_index.json"
FILES = ("model.pt", "paper_metadata.json", "training_state.pt")
IDENTITY_FIELDS = ("checkpoint_id", "epoch", "step")


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + "." + uuid4().hex + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def _read_index(path):
    index = read_json(path)
    if index.get("format_version") != FORMAT_VERSION or any(not isinstance(index.get(k), str) for k in ("best", "last")):
        raise ValueError("Invalid checkpoint index")
    return index


def _indexed_path(root, value):
    candidate = (root / value).resolve()
    if candidate.parent != (root / "checkpoints").resolve():
        raise ValueError("Checkpoint index must refer to a snapshot inside this run")
    return candidate


def checkpoint_metadata(directory, *, require_committed=False):
    """Resolve best/last once and validate the immutable snapshot before use.

    Legacy weight-only exports can still be read for evaluation, but cannot be
    used as training recovery points without a complete, committed snapshot.
    """
    path = Path(directory).resolve()
    index_path = path.parent / INDEX_NAME
    if path.name in ("best", "last") and index_path.is_file():
        path = _indexed_path(path.parent, _read_index(index_path)[path.name])
        require_committed = True
    commit_path = path / "commit.json"
    if not commit_path.is_file():
        if require_committed or path.parent.name == "checkpoints":
            raise ValueError("Checkpoint is not a committed training snapshot; legacy/partial recovery is refused")
        return path, read_json(path / "paper_metadata.json")
    commit = read_json(commit_path)
    if commit.get("format_version") != FORMAT_VERSION or set(commit.get("files", {})) != set(FILES):
        raise ValueError("Invalid checkpoint commit manifest")
    for name, expected in commit["files"].items():
        file = path / name
        if not file.is_file() or sha256(file) != expected:
            raise ValueError(f"Checkpoint integrity mismatch: {name}")
    metadata = read_json(path / "paper_metadata.json")
    _check_identity(commit, metadata.get("metadata", {}))
    return path, metadata


def _check_identity(expected, actual):
    if (not isinstance(expected.get("checkpoint_id"), str) or not expected["checkpoint_id"]
            or type(expected.get("epoch")) is not int or expected["epoch"] < 1
            or type(expected.get("step")) is not int or expected["step"] < 0
            or any(expected.get(k) != actual.get(k) for k in IDENTITY_FIELDS)):
        raise ValueError("Checkpoint identity/epoch/step mismatch")


def _check_state(state):
    required = {"optimizer", "scheduler", "best", "bad_epochs", "random_state", "torch_rng", "cuda_rng"}
    if not required <= state.keys():
        raise ValueError("Incomplete training state")


def load_training_checkpoint(directory):
    import torch
    path, metadata = checkpoint_metadata(directory, require_committed=True)
    state = torch.load(path / "training_state.pt", map_location="cpu", weights_only=False)
    _check_identity(read_json(path / "commit.json"), state)
    _check_state(state)
    return path, metadata, state


def save_training_checkpoint(output, model, metadata, state, *, is_best):
    """Publish only after weights, metadata, optimizer and RNG are all complete.

    Unpublished snapshots left by interruption are harmless. The last and best
    references change in the same replace operation; old snapshots are retained.
    """
    import torch
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    previous = _read_index(root / INDEX_NAME) if (root / INDEX_NAME).exists() else None
    if previous is None and not is_best:
        raise ValueError("The first committed checkpoint must establish a best model")
    if any(metadata.get(k) != state.get(k) for k in ("epoch", "step")):
        raise ValueError("Checkpoint metadata and training state epoch/step mismatch")
    identity = {"checkpoint_id": uuid4().hex, "epoch": state["epoch"], "step": state["step"]}
    _check_identity(identity, identity)
    _check_state(state)
    path = root / "checkpoints" / f"epoch-{state['epoch']:04d}-step-{state['step']:08d}-{identity['checkpoint_id']}"
    path.mkdir(parents=True, exist_ok=False)
    model.save_checkpoint(path, {**metadata, **identity})
    with (path / "training_state.pt").open("wb") as stream:
        torch.save({**state, **identity}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    for name in ("model.pt", "paper_metadata.json"):
        with (path / name).open("r+b") as stream:
            os.fsync(stream.fileno())
    _atomic_json(path / "commit.json", {"format_version": FORMAT_VERSION, **identity,
                                       "files": {name: sha256(path / name) for name in FILES}})
    relative = path.relative_to(root).as_posix()
    index = {"format_version": FORMAT_VERSION, "last": relative,
             "best": relative if is_best else previous["best"]}
    # This is the single publication point. A failed save cannot mix epochs.
    _atomic_json(root / INDEX_NAME, index)
    return path
