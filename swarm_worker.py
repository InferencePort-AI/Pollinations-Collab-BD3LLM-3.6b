"""Volunteer worker for Hub-backed, round-based DiLoCo training."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import socket
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import CommitOperationAdd, HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, default_data_collator, set_seed

from swarm_common import (
    PROTOCOL_VERSION,
    UPDATE_ROOT,
    atomic_write_json,
    build_delta_shards,
    download_model_snapshot,
    load_swarm_state,
    parse_size,
    sanitize_id,
    torch_dtype_from_name,
    utc_now,
)
from train import build_lm_dataset

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a local BD3LM replica and submit a DiLoCo pseudo-gradient as a Hub Pull Request.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--hub_repo_id", required=True, help="Model repository containing swarm/state.json.")
    parser.add_argument("--token", default=None, help="HF token; defaults to HF_TOKEN or `hf auth login`.")
    parser.add_argument("--worker_id", default=socket.gethostname(), help="Stable identifier for this GPU/worker.")
    parser.add_argument("--repeat", action="store_true", help="Keep joining new rounds until interrupted.")
    parser.add_argument("--poll_interval", type=float, default=60.0)
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--state_dir", default="~/.cache/bd3lm-swarm", help="Persists the last submitted round.")
    parser.add_argument("--force_resubmit", action="store_true", help="Allow another PR for an already submitted round.")

    data = parser.add_argument_group("data")
    data.add_argument("--dataset_name", default="Salesforce/wikitext")
    data.add_argument("--dataset_config_name", default="wikitext-2-raw-v1")
    data.add_argument("--train_text_column", default="text")
    data.add_argument("--block_size", type=int, default=2048)
    data.add_argument("--preprocessing_num_workers", type=int, default=4)

    training = parser.add_argument_group("local training")
    training.add_argument("--local_steps", type=int, default=None, help="Defaults to the coordinator policy.")
    training.add_argument("--per_device_train_batch_size", type=int, default=1)
    training.add_argument("--gradient_accumulation_steps", type=int, default=8)
    training.add_argument("--learning_rate", type=float, default=3e-4)
    training.add_argument("--weight_decay", type=float, default=0.1)
    training.add_argument("--warmup_steps", type=int, default=20)
    training.add_argument("--max_grad_norm", type=float, default=1.0)
    training.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    training.add_argument("--mixed_precision", choices=["auto", "bf16", "fp16", "no"], default="auto")
    training.add_argument("--seed", type=int, default=42)
    training.add_argument("--logging_steps", type=int, default=10)
    training.add_argument("--dataloader_num_workers", type=int, default=2)

    update = parser.add_argument_group("update upload")
    update.add_argument("--update_dtype", choices=["bfloat16", "float16", "float32"], default=None)
    update.add_argument("--max_shard_size", default="2GB")
    return parser


def resolve_precision(mode: str) -> tuple[torch.dtype, bool, bool]:
    if mode == "auto":
        mode = "bf16" if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else "no"
    if mode == "bf16":
        return torch.bfloat16, True, False
    if mode == "fp16":
        if not torch.cuda.is_available():
            raise ValueError("fp16 local training requires CUDA.")
        return torch.float16, False, True
    return torch.float32, False, False


def train_one_round(args: argparse.Namespace, api: HfApi, state: dict[str, Any]) -> str:
    if state["status"] != "open":
        raise RuntimeError(f"Round {state['round']} is currently {state['status']!r}, not open.")

    round_number = int(state["round"])
    local_steps = args.local_steps or int(state["policy"]["local_steps"])
    update_dtype_name = args.update_dtype or state["policy"].get("update_dtype", "bfloat16")
    model_dtype, use_bf16, use_fp16 = resolve_precision(args.mixed_precision)
    update_dtype = torch_dtype_from_name(update_dtype_name)
    worker_id = sanitize_id(args.worker_id)
    worker_seed_offset = int(hashlib.sha256(worker_id.encode()).hexdigest()[:8], 16)
    worker_seed = (args.seed + round_number * 1_000_003 + worker_seed_offset) % (2**31)
    set_seed(worker_seed)

    logger.info(
        "Joining round %d from model revision %s for %d local steps",
        round_number,
        state["base_model_revision"][:12],
        local_steps,
    )
    base_snapshot = download_model_snapshot(
        args.hub_repo_id,
        state["base_model_revision"],
        token=args.token,
        cache_dir=args.cache_dir,
    )
    tokenizer = AutoTokenizer.from_pretrained(base_snapshot, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        base_snapshot,
        trust_remote_code=True,
        dtype=model_dtype,
    )
    model.config.use_cache = False

    lm_datasets = build_lm_dataset(args, tokenizer)
    train_dataset = lm_datasets["train"]
    if not len(train_dataset):
        raise RuntimeError("Tokenization produced no full training blocks; reduce --block_size.")

    with tempfile.TemporaryDirectory(prefix=f"bd3lm-worker-r{round_number}-") as temporary_directory:
        temporary_path = Path(temporary_directory)
        training_args = TrainingArguments(
            output_dir=str(temporary_path / "trainer"),
            per_device_train_batch_size=args.per_device_train_batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            max_grad_norm=args.max_grad_norm,
            max_steps=local_steps,
            logging_steps=args.logging_steps,
            save_strategy="no",
            eval_strategy="no",
            gradient_checkpointing=args.gradient_checkpointing,
            gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
            bf16=use_bf16,
            fp16=use_fp16,
            report_to="none",
            seed=worker_seed,
            dataloader_num_workers=args.dataloader_num_workers,
            remove_unused_columns=False,
        )
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            data_collator=default_data_collator,
        )
        train_result = trainer.train()

        delta_dir = temporary_path / "delta"
        delta_paths, delta_norm, tensor_count = build_delta_shards(
            model,
            base_snapshot,
            delta_dir,
            dtype=update_dtype,
            max_shard_bytes=parse_size(args.max_shard_size),
        )

        # Do not upload expensive stale work if the coordinator advanced while we trained.
        latest_state = load_swarm_state(api, args.hub_repo_id, token=args.token)
        if (
            latest_state["status"] != "open"
            or latest_state["round"] != round_number
            or latest_state["base_model_revision"] != state["base_model_revision"]
        ):
            raise RuntimeError("The swarm advanced while this worker trained; discarding the stale local update.")

        update_id = uuid.uuid4().hex
        update_prefix = f"{UPDATE_ROOT}/round-{round_number:06d}/{worker_id}/{update_id}"
        delta_files_in_repo = [f"{update_prefix}/{path.name}" for path in delta_paths]
        num_tokens = local_steps * args.per_device_train_batch_size * args.gradient_accumulation_steps * args.block_size
        metadata = {
            "protocol_version": PROTOCOL_VERSION,
            "update_id": update_id,
            "worker_id": worker_id,
            "round": round_number,
            "base_model_revision": state["base_model_revision"],
            "delta_sign": "base-minus-local",
            "delta_dtype": update_dtype_name,
            "delta_files": delta_files_in_repo,
            "delta_norm_reported": delta_norm,
            "tensor_count": tensor_count,
            "local_steps": local_steps,
            "num_tokens": num_tokens,
            "train_loss": train_result.metrics.get("train_loss"),
            "dataset_name": args.dataset_name,
            "dataset_config_name": args.dataset_config_name,
            "block_size": args.block_size,
            "submitted_at": utc_now(),
        }
        metadata_path = delta_dir / "metadata.json"
        atomic_write_json(metadata_path, metadata)

        operations = [
            CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=local_path)
            for path_in_repo, local_path in zip(delta_files_in_repo, delta_paths)
        ]
        operations.append(CommitOperationAdd(path_in_repo=f"{update_prefix}/metadata.json", path_or_fileobj=metadata_path))
        commit = api.create_commit(
            repo_id=args.hub_repo_id,
            operations=operations,
            commit_message=f"[swarm] round {round_number} update from {worker_id}",
            commit_description=json.dumps(
                {"round": round_number, "worker_id": worker_id, "update_id": update_id}, sort_keys=True
            ),
            revision="main",
            create_pr=True,
            token=args.token,
        )
        logger.info("Submitted update %s as %s", update_id, commit.pr_url or commit.commit_url)
        return update_id


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = build_parser().parse_args()
    args.token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=args.token)
    worker_id = sanitize_id(args.worker_id)
    state_file = Path(args.state_dir).expanduser() / f"{sanitize_id(args.hub_repo_id.replace('/', '--'))}--{worker_id}.json"

    last_submitted_round: int | None = None
    if state_file.exists() and not args.force_resubmit:
        with state_file.open(encoding="utf-8") as handle:
            last_submitted_round = json.load(handle).get("last_submitted_round")
    while True:
        state = load_swarm_state(api, args.hub_repo_id, token=args.token)
        if state["status"] != "open" or state["round"] == last_submitted_round:
            if not args.repeat:
                reason = state["status"] if state["status"] != "open" else "already submitted"
                raise RuntimeError(f"Cannot join round {state['round']}: {reason}.")
            time.sleep(args.poll_interval)
            continue
        try:
            update_id = train_one_round(args, api, state)
            last_submitted_round = int(state["round"])
            atomic_write_json(
                state_file,
                {
                    "hub_repo_id": args.hub_repo_id,
                    "worker_id": worker_id,
                    "last_submitted_round": last_submitted_round,
                    "update_id": update_id,
                },
            )
        except RuntimeError as error:
            if not args.repeat or "advanced" not in str(error):
                raise
            logger.warning("%s", error)
        if not args.repeat:
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
