"""Coordinator for Hub-backed, round-based DiLoCo training."""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from huggingface_hub import CommitOperationAdd, CommitOperationDelete, HfApi
from transformers import AutoModelForCausalLM

from swarm_common import (
    MERGE_MARKER_PATH,
    PROTOCOL_VERSION,
    STATE_PATH,
    UPDATE_ROOT,
    ShardedTensorReader,
    UpdatePackage,
    apply_diloco_merge,
    atomic_write_json,
    download_model_snapshot,
    load_hub_json,
    load_swarm_state,
    named_trainable_parameters,
    parse_size,
    read_json,
    torch_dtype_from_name,
    utc_now,
    validate_update_metadata,
    verify_update,
)
from train import build_model, load_tokenizer

logger = logging.getLogger(__name__)


def json_bytes(value: dict[str, Any]) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def add_common_hub_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--hub_repo_id", required=True)
    parser.add_argument("--token", default=None, help="Owner write token; defaults to HF_TOKEN or `hf auth login`.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Initialize and coordinate a churn-tolerant BD3LM DiLoCo swarm.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    commands = parser.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Publish/identify a base checkpoint and create swarm/state.json.")
    add_common_hub_arguments(init)
    source = init.add_mutually_exclusive_group()
    source.add_argument("--checkpoint", help="Local save_pretrained checkpoint to upload.")
    source.add_argument("--use_existing_model", action="store_true", help="Use model files already at repo main.")
    init.add_argument("--create_repo", action="store_true")
    init.add_argument("--private", action="store_true")
    init.add_argument("--force", action="store_true", help="Replace an existing swarm state.")
    init.add_argument("--tokenizer_name_or_path", default="gpt2")
    init.add_argument("--debug_tiny_model", action="store_true")
    init.add_argument("--max_model_shard_size", default="4GB")
    init.add_argument("--min_updates", type=int, default=2)
    init.add_argument("--max_updates", type=int, default=8)
    init.add_argument("--local_steps", type=int, default=500)
    init.add_argument("--outer_lr", type=float, default=0.7)
    init.add_argument("--outer_momentum", type=float, default=0.9)
    init.add_argument("--no_nesterov", action="store_true")
    init.add_argument("--update_dtype", choices=["bfloat16", "float16", "float32"], default="bfloat16")

    run = commands.add_parser("run", help="Poll update PRs and commit outer steps indefinitely.")
    add_common_hub_arguments(run)
    run.add_argument("--state_dir", default="./swarm-coordinator-state")
    run.add_argument("--cache_dir", default=None)
    run.add_argument("--poll_interval", type=float, default=60.0)
    run.add_argument("--once", action="store_true", help="Poll/merge once, then exit.")
    run.add_argument("--coordinator_dtype", choices=["float32", "bfloat16"], default="float32")
    run.add_argument("--momentum_dtype", choices=["float32", "bfloat16"], default="bfloat16")
    run.add_argument("--weighting", choices=["uniform", "tokens"], default="uniform")
    run.add_argument("--max_delta_norm", type=float, default=0.0, help="Clip each update globally; 0 disables.")
    run.add_argument("--max_updates_per_author", type=int, default=1)
    run.add_argument("--allowed_authors", nargs="*", default=None, help="Optional HF username allowlist.")
    run.add_argument("--max_shard_size", default="2GB")
    run.add_argument("--max_model_shard_size", default="4GB")

    status = commands.add_parser("status", help="Print current state and open swarm PRs.")
    add_common_hub_arguments(status)
    return parser


def upload_directory(api: HfApi, repo_id: str, directory: Path, token: str | None, message: str) -> str:
    operations = [
        CommitOperationAdd(path_in_repo=path.name, path_or_fileobj=path) for path in sorted(directory.iterdir()) if path.is_file()
    ]
    if not operations:
        raise ValueError(f"No files found in {directory}.")
    return api.create_commit(
        repo_id=repo_id,
        operations=operations,
        commit_message=message,
        token=token,
    ).oid


def initialize_swarm(args: argparse.Namespace, api: HfApi) -> None:
    if args.create_repo:
        api.create_repo(args.hub_repo_id, private=args.private, exist_ok=True, token=args.token)
    existing_files = set(api.list_repo_files(args.hub_repo_id, token=args.token))
    if STATE_PATH in existing_files and not args.force:
        raise RuntimeError(f"{STATE_PATH} already exists; pass --force to replace it.")
    if args.min_updates <= 0 or args.max_updates < args.min_updates:
        raise ValueError("Require 0 < min_updates <= max_updates.")

    if args.use_existing_model:
        has_weights = "model.safetensors" in existing_files or "model.safetensors.index.json" in existing_files
        if (
            "config.json" not in existing_files
            or "tokenizer_config.json" not in existing_files
            or "modeling_bd3lm.py" not in existing_files
            or not has_weights
        ):
            raise RuntimeError(
                "Existing repo must contain config.json, tokenizer_config.json, modeling_bd3lm.py, and safetensors model weights."
            )
        model_revision = api.repo_info(args.hub_repo_id, token=args.token).sha
    else:
        with tempfile.TemporaryDirectory(prefix="bd3lm-swarm-init-") as temporary_directory:
            checkpoint_dir = Path(temporary_directory) / "checkpoint"
            if args.checkpoint:
                source = Path(args.checkpoint).resolve()
                if not source.is_dir():
                    raise FileNotFoundError(source)
                shutil.copytree(source, checkpoint_dir)
            else:
                checkpoint_dir.mkdir()
                tokenizer = load_tokenizer(args.tokenizer_name_or_path)
                model = build_model(tokenizer, args.debug_tiny_model)
                model.config.register_for_auto_class()
                model.register_for_auto_class("AutoModelForCausalLM")
                model.save_pretrained(checkpoint_dir, max_shard_size=args.max_model_shard_size)
                tokenizer.save_pretrained(checkpoint_dir)
            checkpoint_files = {path.name for path in checkpoint_dir.iterdir() if path.is_file()}
            has_weights = "model.safetensors" in checkpoint_files or "model.safetensors.index.json" in checkpoint_files
            if (
                "config.json" not in checkpoint_files
                or "tokenizer_config.json" not in checkpoint_files
                or "modeling_bd3lm.py" not in checkpoint_files
                or not has_weights
            ):
                raise RuntimeError(
                    "Checkpoint must contain config.json, tokenizer_config.json, modeling_bd3lm.py, "
                    "and safetensors model weights."
                )
            model_revision = upload_directory(
                api,
                args.hub_repo_id,
                checkpoint_dir,
                args.token,
                "Initialize BD3LM swarm model",
            )

    state = {
        "protocol_version": PROTOCOL_VERSION,
        "status": "open",
        "round": 0,
        "base_model_revision": model_revision,
        "pending_merge": None,
        "initialized_at": utc_now(),
        "updated_at": utc_now(),
        "policy": {
            "min_updates": args.min_updates,
            "max_updates": args.max_updates,
            "local_steps": args.local_steps,
            "outer_lr": args.outer_lr,
            "outer_momentum": args.outer_momentum,
            "nesterov": not args.no_nesterov,
            "update_dtype": args.update_dtype,
        },
    }
    api.create_commit(
        repo_id=args.hub_repo_id,
        operations=[CommitOperationAdd(path_in_repo=STATE_PATH, path_or_fileobj=json_bytes(state))],
        commit_message="Open BD3LM swarm round 0",
        parent_commit=model_revision,
        token=args.token,
    )
    logger.info("Initialized %s at model revision %s", args.hub_repo_id, model_revision)


def latest_pr_revision(api: HfApi, repo_id: str, pr_num: int, token: str | None) -> tuple[Any, str]:
    details = api.get_discussion_details(repo_id, pr_num, token=token)
    revisions = [getattr(event, "oid", None) for event in details.events]
    revisions = [revision for revision in revisions if revision]
    if not revisions:
        raise ValueError(f"PR #{pr_num} contains no commits.")
    return details, revisions[-1]


def package_from_pr(
    api: HfApi,
    repo_id: str,
    pr_num: int,
    state: dict[str, Any],
    token: str | None,
) -> UpdatePackage:
    details, revision = latest_pr_revision(api, repo_id, pr_num, token)
    prefix = f"{UPDATE_ROOT}/round-{state['round']:06d}/"
    files = api.list_repo_files(repo_id, revision=revision, token=token)
    metadata_paths = [path for path in files if path.startswith(prefix) and path.endswith("/metadata.json")]
    if len(metadata_paths) != 1:
        raise ValueError(f"Expected exactly one round-{state['round']} metadata file, found {len(metadata_paths)}.")
    metadata_path = metadata_paths[0]
    metadata = load_hub_json(api, repo_id, metadata_path, revision=revision, token=token)
    validate_update_metadata(metadata, state)
    update_directory = metadata_path.rsplit("/", 1)[0] + "/"
    for path in metadata["delta_files"]:
        if not isinstance(path, str) or not path.startswith(update_directory) or not path.endswith(".safetensors"):
            raise ValueError(f"Unsafe delta path {path!r} in PR #{pr_num}.")
        if path not in files:
            raise ValueError(f"PR #{pr_num} metadata references missing file {path!r}.")
    return UpdatePackage(
        pr_num=pr_num,
        revision=revision,
        author=details.author,
        metadata_path=metadata_path,
        metadata=metadata,
    )


def discover_updates(
    api: HfApi,
    repo_id: str,
    state: dict[str, Any],
    token: str | None,
    max_updates_per_author: int,
    allowed_authors: list[str] | None,
) -> list[UpdatePackage]:
    discussions = sorted(
        (
            discussion
            for discussion in api.get_repo_discussions(repo_id, discussion_type="pull_request", token=token)
            if discussion.status in {"open", "draft"} and discussion.title.startswith("[swarm]")
        ),
        key=lambda discussion: discussion.created_at,
    )
    accepted: list[UpdatePackage] = []
    identities: set[str] = set()
    author_counts: dict[str, int] = {}
    for discussion in discussions:
        try:
            if allowed_authors is not None and discussion.author not in allowed_authors:
                raise ValueError(f"Author {discussion.author} is not allowlisted.")
            update = package_from_pr(api, repo_id, discussion.num, state, token)
            if update.identity in identities:
                raise ValueError(f"Duplicate participant identity {update.identity}.")
            if author_counts.get(update.author, 0) >= max_updates_per_author:
                raise ValueError(f"Author {update.author} already reached this round's update limit.")
            identities.add(update.identity)
            author_counts[update.author] = author_counts.get(update.author, 0) + 1
            accepted.append(update)
        except Exception as error:  # noqa: BLE001 - malformed/unreadable volunteer PRs are isolated.
            logger.warning("Ignoring PR #%d: %s", discussion.num, error)
    return accepted


def download_update_files(
    api: HfApi,
    repo_id: str,
    update: UpdatePackage,
    directory: Path,
    token: str | None,
) -> None:
    update.local_dir = directory / f"pr-{update.pr_num}"
    update.local_dir.mkdir(parents=True, exist_ok=True)
    update.local_files = []
    for path_in_repo in update.metadata["delta_files"]:
        local_path = api.hf_hub_download(
            repo_id,
            path_in_repo,
            revision=update.revision,
            local_dir=update.local_dir,
            token=token,
        )
        update.local_files.append(Path(local_path))


def active_momentum_reader(state_dir: Path) -> ShardedTensorReader | None:
    momentum_dir = state_dir / "outer-momentum"
    manifest_path = momentum_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    return ShardedTensorReader(momentum_dir, manifest["files"])


def promote_momentum(state_dir: Path, source_round: int) -> None:
    pending = state_dir / f"pending-momentum-round-{source_round:06d}"
    active = state_dir / "outer-momentum"
    active_manifest = active / "manifest.json"
    if active_manifest.exists() and read_json(active_manifest).get("completed_round") == source_round:
        return
    if not pending.exists():
        logger.warning("Pending momentum is missing; outer momentum will reset on the next round.")
        if active.exists():
            shutil.rmtree(active)
        return
    backup = state_dir / "outer-momentum.previous"
    if backup.exists():
        shutil.rmtree(backup)
    if active.exists():
        active.rename(backup)
    pending.rename(active)
    if backup.exists():
        shutil.rmtree(backup)


def transition_to_merging(
    api: HfApi,
    args: argparse.Namespace,
    state: dict[str, Any],
    updates: list[UpdatePackage],
) -> dict[str, Any]:
    parent_commit = api.repo_info(args.hub_repo_id, token=args.token).sha
    pending = [
        {
            "pr_num": update.pr_num,
            "revision": update.revision,
            "author": update.author,
            "metadata_path": update.metadata_path,
            "metadata": update.metadata,
        }
        for update in updates
    ]
    merging_state = {**state, "status": "merging", "pending_merge": pending, "updated_at": utc_now()}
    api.create_commit(
        repo_id=args.hub_repo_id,
        operations=[CommitOperationAdd(path_in_repo=STATE_PATH, path_or_fileobj=json_bytes(merging_state))],
        commit_message=f"Lock BD3LM swarm round {state['round']} merge",
        parent_commit=parent_commit,
        token=args.token,
    )
    return merging_state


def updates_from_pending(state: dict[str, Any]) -> list[UpdatePackage]:
    return [
        UpdatePackage(
            pr_num=item["pr_num"],
            revision=item["revision"],
            author=item["author"],
            metadata_path=item["metadata_path"],
            metadata=item["metadata"],
        )
        for item in state["pending_merge"]
    ]


def matching_merge_marker(api: HfApi, args: argparse.Namespace, state: dict[str, Any]) -> dict[str, Any] | None:
    try:
        marker = load_hub_json(api, args.hub_repo_id, MERGE_MARKER_PATH, token=args.token)
    except Exception:  # noqa: BLE001 - an absent/unreadable marker means no completed merge.
        return None
    expected_ids = sorted(item["metadata"]["update_id"] for item in state["pending_merge"])
    if marker.get("source_round") == state["round"] and sorted(marker.get("update_ids", [])) == expected_ids:
        return marker
    return None


def finalize_merge(
    api: HfApi,
    args: argparse.Namespace,
    state: dict[str, Any],
    model_revision: str,
    marker: dict[str, Any],
) -> None:
    source_round = int(state["round"])
    promote_momentum(Path(args.state_dir), source_round)
    parent_commit = api.repo_info(args.hub_repo_id, token=args.token).sha
    next_state = {
        **state,
        "status": "open",
        "round": source_round + 1,
        "base_model_revision": model_revision,
        "pending_merge": None,
        "updated_at": utc_now(),
        "last_merge": marker,
    }
    api.create_commit(
        repo_id=args.hub_repo_id,
        operations=[CommitOperationAdd(path_in_repo=STATE_PATH, path_or_fileobj=json_bytes(next_state))],
        commit_message=f"Open BD3LM swarm round {source_round + 1}",
        parent_commit=parent_commit,
        token=args.token,
    )
    for item in state["pending_merge"]:
        try:
            api.change_discussion_status(
                args.hub_repo_id,
                item["pr_num"],
                "closed",
                token=args.token,
                comment=f"Accepted into swarm round {source_round}; raw delta is not merged into main.",
            )
        except Exception as error:  # noqa: BLE001 - closing a PR is best-effort after commit.
            logger.warning("Could not close accepted PR #%d: %s", item["pr_num"], error)
    logger.info("Committed round %d; round %d is now open", source_round, source_round + 1)


def commit_merged_model(
    api: HfApi,
    args: argparse.Namespace,
    output_dir: Path,
    state: dict[str, Any],
    updates: list[UpdatePackage],
) -> tuple[str, dict[str, Any]]:
    parent_commit = api.repo_info(args.hub_repo_id, token=args.token).sha
    current_files = set(api.list_repo_files(args.hub_repo_id, token=args.token))
    new_files = {str(path.relative_to(output_dir)): path for path in output_dir.rglob("*") if path.is_file()}
    operations: list[Any] = [
        CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=path) for path_in_repo, path in sorted(new_files.items())
    ]
    old_weight_files = {
        path for path in current_files if path.startswith("model") and path.endswith((".safetensors", ".safetensors.index.json"))
    }
    for path in sorted(old_weight_files - new_files.keys()):
        operations.append(CommitOperationDelete(path_in_repo=path))

    marker = {
        "protocol_version": PROTOCOL_VERSION,
        "source_round": state["round"],
        "source_model_revision": state["base_model_revision"],
        "update_ids": [update.metadata["update_id"] for update in updates],
        "participants": [update.identity for update in updates],
        "verified_delta_norms": {update.identity: update.verified_norm for update in updates},
        "outer_lr": state["policy"]["outer_lr"],
        "outer_momentum": state["policy"]["outer_momentum"],
        "nesterov": state["policy"]["nesterov"],
        "merged_at": utc_now(),
    }
    operations.append(CommitOperationAdd(path_in_repo=MERGE_MARKER_PATH, path_or_fileobj=json_bytes(marker)))
    commit = api.create_commit(
        repo_id=args.hub_repo_id,
        operations=operations,
        commit_message=f"Merge BD3LM swarm round {state['round']}",
        parent_commit=parent_commit,
        token=args.token,
    )
    return commit.oid, marker


def execute_merge(
    api: HfApi,
    args: argparse.Namespace,
    state: dict[str, Any],
    updates: list[UpdatePackage],
) -> None:
    state_dir = Path(args.state_dir).resolve()
    state_dir.mkdir(parents=True, exist_ok=True)
    source_round = int(state["round"])
    pending_momentum = state_dir / f"pending-momentum-round-{source_round:06d}"
    if pending_momentum.exists():
        shutil.rmtree(pending_momentum)

    with tempfile.TemporaryDirectory(prefix=f"bd3lm-coordinator-r{source_round}-") as temporary_directory:
        work_dir = Path(temporary_directory)
        base_snapshot = download_model_snapshot(
            args.hub_repo_id,
            state["base_model_revision"],
            token=args.token,
            cache_dir=args.cache_dir,
        )
        model = AutoModelForCausalLM.from_pretrained(
            base_snapshot,
            trust_remote_code=True,
            dtype=torch_dtype_from_name(args.coordinator_dtype),
        )
        expected_shapes = {name: parameter.shape for name, parameter in named_trainable_parameters(model)}
        for update in updates:
            download_update_files(api, args.hub_repo_id, update, work_dir / "updates", args.token)
            norm = verify_update(update, expected_shapes)
            logger.info("Verified %s from PR #%d: delta norm %.6g", update.identity, update.pr_num, norm)

        if state["status"] == "open":
            state = transition_to_merging(api, args, state, updates)

        momentum_paths = apply_diloco_merge(
            model,
            updates,
            momentum_reader=active_momentum_reader(state_dir),
            momentum_output_dir=pending_momentum,
            outer_lr=float(state["policy"]["outer_lr"]),
            outer_momentum=float(state["policy"]["outer_momentum"]),
            nesterov=bool(state["policy"]["nesterov"]),
            weighting=args.weighting,
            max_delta_norm=args.max_delta_norm,
            momentum_dtype=torch_dtype_from_name(args.momentum_dtype),
            max_shard_bytes=parse_size(args.max_shard_size),
        )
        atomic_write_json(
            pending_momentum / "manifest.json",
            {"completed_round": source_round, "files": [path.name for path in momentum_paths]},
        )

        model_output = work_dir / "model"
        model.save_pretrained(model_output, max_shard_size=args.max_model_shard_size, safe_serialization=True)
        model_revision, marker = commit_merged_model(api, args, model_output, state, updates)
        finalize_merge(api, args, state, model_revision, marker)


def coordinator_iteration(args: argparse.Namespace, api: HfApi) -> bool:
    state = load_swarm_state(api, args.hub_repo_id, token=args.token)
    if state["status"] == "merging":
        marker = matching_merge_marker(api, args, state)
        if marker is not None:
            model_revision = api.repo_info(args.hub_repo_id, token=args.token).sha
            finalize_merge(api, args, state, model_revision, marker)
            return True
        logger.info("Resuming interrupted round %d merge", state["round"])
        execute_merge(api, args, state, updates_from_pending(state))
        return True
    if state["status"] != "open":
        logger.info("Swarm status is %s; waiting", state["status"])
        return False

    updates = discover_updates(
        api,
        args.hub_repo_id,
        state,
        args.token,
        args.max_updates_per_author,
        args.allowed_authors,
    )
    minimum = int(state["policy"]["min_updates"])
    maximum = int(state["policy"]["max_updates"])
    logger.info("Round %d has %d/%d valid updates", state["round"], len(updates), minimum)
    if len(updates) < minimum:
        return False
    execute_merge(api, args, state, updates[:maximum])
    return True


def show_status(args: argparse.Namespace, api: HfApi) -> None:
    state = load_swarm_state(api, args.hub_repo_id, token=args.token)
    print(json.dumps(state, indent=2, sort_keys=True))
    discussions = [
        discussion
        for discussion in api.get_repo_discussions(
            args.hub_repo_id, discussion_type="pull_request", discussion_status="open", token=args.token
        )
        if discussion.title.startswith("[swarm]")
    ]
    print(f"\nOpen swarm pull requests: {len(discussions)}")
    for discussion in discussions:
        print(f"  #{discussion.num}: {discussion.title} ({discussion.author})")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = build_parser().parse_args()
    args.token = args.token or os.environ.get("HF_TOKEN")
    api = HfApi(token=args.token)

    if args.command == "init":
        initialize_swarm(args, api)
        return
    if args.command == "status":
        show_status(args, api)
        return
    while True:
        coordinator_iteration(args, api)
        if args.once:
            return
        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
