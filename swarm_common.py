"""Shared primitives for Hub-backed, round-based DiLoCo training."""

from __future__ import annotations

import json
import math
import os
import re
from collections.abc import Iterable, Iterator, Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import TracebackType
from typing import Any, Self

import torch
from huggingface_hub import HfApi, snapshot_download
from safetensors import safe_open
from safetensors.torch import save_file

PROTOCOL_VERSION = 1
STATE_PATH = "swarm/state.json"
MERGE_MARKER_PATH = "swarm/merge-result.json"
UPDATE_ROOT = "swarm/updates"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_id(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    if not value:
        raise ValueError("Identifier must contain at least one letter or digit.")
    return value[:80]


def atomic_write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary_path, path)


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_hub_json(
    api: HfApi,
    repo_id: str,
    path_in_repo: str,
    *,
    revision: str = "main",
    token: str | None = None,
) -> dict[str, Any]:
    local_path = api.hf_hub_download(repo_id, path_in_repo, revision=revision, token=token)
    return read_json(Path(local_path))


def load_swarm_state(api: HfApi, repo_id: str, *, token: str | None = None) -> dict[str, Any]:
    state = load_hub_json(api, repo_id, STATE_PATH, token=token)
    if state.get("protocol_version") != PROTOCOL_VERSION:
        raise RuntimeError(f"Unsupported swarm protocol {state.get('protocol_version')!r}; expected {PROTOCOL_VERSION}.")
    return state


def download_model_snapshot(
    repo_id: str,
    revision: str,
    *,
    token: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    return Path(
        snapshot_download(
            repo_id,
            revision=revision,
            token=token,
            cache_dir=cache_dir,
            ignore_patterns=["swarm/*"],
        )
    )


def torch_dtype_from_name(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    try:
        return mapping[name]
    except KeyError as error:
        raise ValueError(f"Unsupported dtype {name!r}; choose from {', '.join(mapping)}.") from error


class ShardedTensorReader:
    """Random-access reader over one or more safetensors shards."""

    def __init__(self, root: Path, files: Iterable[str | Path] | None = None) -> None:
        self.root = Path(root)
        if files is None:
            index_path = self.root / "model.safetensors.index.json"
            if index_path.exists():
                index = read_json(index_path)
                files = sorted(set(index["weight_map"].values()))
            elif (self.root / "model.safetensors").exists():
                files = ["model.safetensors"]
            else:
                files = sorted(path.name for path in self.root.glob("*.safetensors"))
        self.files = [Path(path) if Path(path).is_absolute() else self.root / path for path in files]
        if not self.files:
            raise FileNotFoundError(f"No safetensors files found under {self.root}.")
        self._stack: ExitStack | None = None
        self._handles: list[Any] = []
        self._key_to_handle: dict[str, Any] = {}

    def __enter__(self) -> Self:
        self._stack = ExitStack()
        for path in self.files:
            handle = self._stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            self._handles.append(handle)
            for key in handle.keys():  # noqa: SIM118 - safe_open handles are not mappings.
                if key in self._key_to_handle:
                    raise ValueError(f"Duplicate tensor {key!r} across shards.")
                self._key_to_handle[key] = handle
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self._stack is not None
        self._stack.close()
        self._stack = None
        self._handles.clear()
        self._key_to_handle.clear()

    def keys(self) -> set[str]:
        return set(self._key_to_handle)

    def get_tensor(self, name: str) -> torch.Tensor:
        try:
            return self._key_to_handle[name].get_tensor(name)
        except KeyError as error:
            raise KeyError(f"Tensor {name!r} is absent from {self.files}.") from error


def write_sharded_tensors(
    tensors: Iterable[tuple[str, torch.Tensor]],
    output_dir: Path,
    *,
    prefix: str,
    max_shard_bytes: int,
    metadata: dict[str, str] | None = None,
) -> list[Path]:
    """Writes an iterable without retaining more than one shard in RAM."""

    output_dir.mkdir(parents=True, exist_ok=True)
    temporary_paths: list[Path] = []
    current: dict[str, torch.Tensor] = {}
    current_bytes = 0

    def flush() -> None:
        nonlocal current, current_bytes
        if not current:
            return
        path = output_dir / f"{prefix}-{len(temporary_paths) + 1:05d}.safetensors"
        save_file(current, path, metadata=metadata or {"format": "pt"})
        temporary_paths.append(path)
        current = {}
        current_bytes = 0

    for name, tensor in tensors:
        tensor = tensor.detach().cpu().contiguous()
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current and current_bytes + tensor_bytes > max_shard_bytes:
            flush()
        current[name] = tensor
        current_bytes += tensor_bytes
    flush()

    if not temporary_paths:
        raise ValueError("Cannot write an empty tensor collection.")

    final_paths: list[Path] = []
    shard_count = len(temporary_paths)
    for shard_index, temporary_path in enumerate(temporary_paths, start=1):
        final_path = output_dir / f"{prefix}-{shard_index:05d}-of-{shard_count:05d}.safetensors"
        temporary_path.rename(final_path)
        final_paths.append(final_path)
    return final_paths


def named_trainable_parameters(model: torch.nn.Module) -> Iterator[tuple[str, torch.nn.Parameter]]:
    yield from ((name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad)


def build_delta_shards(
    model: torch.nn.Module,
    base_checkpoint: Path,
    output_dir: Path,
    *,
    dtype: torch.dtype,
    max_shard_bytes: int,
) -> tuple[list[Path], float, int]:
    """Stores pseudo-gradients (`base - local`) and returns their true L2 norm."""

    squared_norm = 0.0
    tensor_count = 0
    with ShardedTensorReader(base_checkpoint) as base_reader:
        parameter_names = {name for name, _ in named_trainable_parameters(model)}
        missing = parameter_names - base_reader.keys()
        if missing:
            raise ValueError(f"Base checkpoint is missing trainable tensors: {sorted(missing)[:5]}")

        def deltas() -> Iterator[tuple[str, torch.Tensor]]:
            nonlocal squared_norm, tensor_count
            for name, parameter in named_trainable_parameters(model):
                base = base_reader.get_tensor(name).to(torch.float32)
                local = parameter.detach().to(device="cpu", dtype=torch.float32)
                delta = base - local
                if not torch.isfinite(delta).all():
                    raise FloatingPointError(f"Non-finite delta in {name}.")
                squared_norm += float(torch.sum(delta * delta, dtype=torch.float64).item())
                tensor_count += 1
                yield name, delta.to(dtype)

        paths = write_sharded_tensors(
            deltas(),
            output_dir,
            prefix="delta",
            max_shard_bytes=max_shard_bytes,
            metadata={"format": "pt", "delta_sign": "base-minus-local"},
        )
    return paths, math.sqrt(squared_norm), tensor_count


@dataclass
class UpdatePackage:
    pr_num: int
    revision: str
    author: str
    metadata_path: str
    metadata: dict[str, Any]
    local_dir: Path | None = None
    local_files: list[Path] | None = None
    verified_norm: float | None = None

    @property
    def identity(self) -> str:
        return f"{self.author}/{self.metadata['worker_id']}"


def validate_update_metadata(metadata: Mapping[str, Any], state: Mapping[str, Any]) -> None:
    required = {
        "protocol_version",
        "update_id",
        "worker_id",
        "round",
        "base_model_revision",
        "delta_files",
        "tensor_count",
        "local_steps",
        "num_tokens",
    }
    missing = required - metadata.keys()
    if missing:
        raise ValueError(f"Update metadata is missing: {sorted(missing)}")
    if metadata["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("Update uses a different swarm protocol version.")
    if metadata["round"] != state["round"]:
        raise ValueError(f"Update round {metadata['round']} does not match current round {state['round']}.")
    if metadata["base_model_revision"] != state["base_model_revision"]:
        raise ValueError("Update was trained from a different base model revision.")
    if not isinstance(metadata["delta_files"], list) or not metadata["delta_files"]:
        raise ValueError("Update has no delta files.")
    if metadata["local_steps"] <= 0 or metadata["num_tokens"] <= 0:
        raise ValueError("Update local_steps and num_tokens must be positive.")


def verify_update(
    update: UpdatePackage,
    expected_shapes: Mapping[str, torch.Size],
) -> float:
    if not update.local_dir or not update.local_files:
        raise ValueError("Update files have not been downloaded.")
    squared_norm = 0.0
    with ShardedTensorReader(update.local_dir, update.local_files) as reader:
        expected_keys = set(expected_shapes)
        if reader.keys() != expected_keys:
            missing = expected_keys - reader.keys()
            unexpected = reader.keys() - expected_keys
            raise ValueError(f"Delta keys differ from model; missing={sorted(missing)[:5]}, unexpected={sorted(unexpected)[:5]}.")
        for name, shape in expected_shapes.items():
            tensor = reader.get_tensor(name)
            if tensor.shape != shape:
                raise ValueError(f"Delta {name} has shape {tuple(tensor.shape)}, expected {tuple(shape)}.")
            if not torch.isfinite(tensor).all():
                raise FloatingPointError(f"Delta {name} contains non-finite values.")
            value = tensor.to(torch.float32)
            squared_norm += float(torch.sum(value * value, dtype=torch.float64).item())
    update.verified_norm = math.sqrt(squared_norm)
    return update.verified_norm


def apply_diloco_merge(
    model: torch.nn.Module,
    updates: list[UpdatePackage],
    *,
    momentum_reader: ShardedTensorReader | None,
    momentum_output_dir: Path,
    outer_lr: float,
    outer_momentum: float,
    nesterov: bool,
    weighting: str,
    max_delta_norm: float,
    momentum_dtype: torch.dtype,
    max_shard_bytes: int,
) -> list[Path]:
    """Applies one DiLoCo outer step and writes the next momentum state."""

    if not updates:
        raise ValueError("At least one update is required.")
    readers = ExitStack()
    update_readers = [readers.enter_context(ShardedTensorReader(update.local_dir, update.local_files)) for update in updates]
    if momentum_reader is not None:
        readers.enter_context(momentum_reader)
    momentum_keys = momentum_reader.keys() if momentum_reader is not None else set()

    try:
        raw_weights = [1.0 if weighting == "uniform" else float(update.metadata["num_tokens"]) for update in updates]
        weight_sum = sum(raw_weights)
        if weight_sum <= 0:
            raise ValueError("Aggregate update weight must be positive.")

        def next_momentum() -> Iterator[tuple[str, torch.Tensor]]:
            with torch.no_grad():
                for name, parameter in named_trainable_parameters(model):
                    average_delta = torch.zeros(parameter.shape, dtype=torch.float32, device="cpu")
                    for update, reader, weight in zip(updates, update_readers, raw_weights):
                        assert update.verified_norm is not None
                        clip_scale = 1.0
                        if max_delta_norm > 0 and update.verified_norm > max_delta_norm:
                            clip_scale = max_delta_norm / update.verified_norm
                        average_delta.add_(reader.get_tensor(name).to(torch.float32), alpha=weight * clip_scale)
                    average_delta.div_(weight_sum)

                    if momentum_reader is not None and name in momentum_keys:
                        momentum = momentum_reader.get_tensor(name).to(torch.float32)
                        momentum.mul_(outer_momentum).add_(average_delta)
                    else:
                        momentum = average_delta.clone()

                    direction = average_delta + outer_momentum * momentum if nesterov else momentum
                    parameter.add_(direction.to(parameter.device, parameter.dtype), alpha=-outer_lr)
                    yield name, momentum.to(momentum_dtype)

        return write_sharded_tensors(
            next_momentum(),
            momentum_output_dir,
            prefix="outer-momentum",
            max_shard_bytes=max_shard_bytes,
            metadata={"format": "pt", "optimizer": "nesterov" if nesterov else "momentum"},
        )
    finally:
        readers.close()


def parse_size(value: str) -> int:
    match = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([KMG]B)?\s*", value.upper())
    if not match:
        raise ValueError(f"Invalid byte size {value!r}; examples: 500MB, 2GB.")
    number = float(match.group(1))
    unit = match.group(2) or "B"
    multiplier = {"B": 1, "KB": 1000, "MB": 1000**2, "GB": 1000**3}[unit]
    return int(number * multiplier)
