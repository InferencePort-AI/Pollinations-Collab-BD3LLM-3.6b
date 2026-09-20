from __future__ import annotations

import copy
from pathlib import Path

import torch

from modeling_bd3lm import BD3LMConfig, BD3LMForCausalLM
from swarm_common import (
    ShardedTensorReader,
    UpdatePackage,
    apply_diloco_merge,
    build_delta_shards,
    named_trainable_parameters,
    verify_update,
)


def tiny_model() -> BD3LMForCausalLM:
    config = BD3LMConfig(
        vocab_size=64,
        hidden_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=1,
        head_dim=8,
        rope_dim=4,
        shared_expert_intermediate_size=64,
        num_routed_experts=2,
        routed_expert_intermediate_size=32,
        num_experts_per_tok=1,
    )
    return BD3LMForCausalLM(config)


def make_update(base_dir: Path, base: BD3LMForCausalLM, offset: float, worker: str, root: Path) -> UpdatePackage:
    local = copy.deepcopy(base)
    with torch.no_grad():
        for _, parameter in named_trainable_parameters(local):
            parameter.sub_(offset)
    output = root / worker
    files, norm, tensor_count = build_delta_shards(
        local,
        base_dir,
        output,
        dtype=torch.float32,
        max_shard_bytes=1024,
    )
    return UpdatePackage(
        pr_num=1,
        revision="test",
        author=worker,
        metadata_path="metadata.json",
        metadata={
            "worker_id": worker,
            "num_tokens": 1,
            "tensor_count": tensor_count,
            "delta_norm_reported": norm,
        },
        local_dir=output,
        local_files=files,
    )


def test_outer_sgd_is_fedavg_when_lr_is_one(tmp_path: Path) -> None:
    torch.manual_seed(0)
    base = tiny_model()
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    update_a = make_update(base_dir, base, 0.1, "a", tmp_path / "updates")
    update_b = make_update(base_dir, base, 0.3, "b", tmp_path / "updates")
    expected_shapes = {name: parameter.shape for name, parameter in named_trainable_parameters(base)}
    verify_update(update_a, expected_shapes)
    verify_update(update_b, expected_shapes)

    merged = copy.deepcopy(base)
    momentum_files = apply_diloco_merge(
        merged,
        [update_a, update_b],
        momentum_reader=None,
        momentum_output_dir=tmp_path / "momentum",
        outer_lr=1.0,
        outer_momentum=0.0,
        nesterov=True,
        weighting="uniform",
        max_delta_norm=0.0,
        momentum_dtype=torch.float32,
        max_shard_bytes=1024,
    )
    assert len(momentum_files) > 1
    for (base_name, base_parameter), (merged_name, merged_parameter) in zip(
        named_trainable_parameters(base), named_trainable_parameters(merged)
    ):
        assert base_name == merged_name
        assert torch.allclose(merged_parameter, base_parameter - 0.2, atol=1e-6)


def test_delta_clipping_limits_worker_influence(tmp_path: Path) -> None:
    torch.manual_seed(0)
    base = tiny_model()
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    update = make_update(base_dir, base, 10.0, "a", tmp_path / "updates")
    expected_shapes = {name: parameter.shape for name, parameter in named_trainable_parameters(base)}
    norm = verify_update(update, expected_shapes)

    merged = copy.deepcopy(base)
    apply_diloco_merge(
        merged,
        [update],
        momentum_reader=None,
        momentum_output_dir=tmp_path / "momentum",
        outer_lr=1.0,
        outer_momentum=0.0,
        nesterov=False,
        weighting="uniform",
        max_delta_norm=norm / 10.0,
        momentum_dtype=torch.float32,
        max_shard_bytes=1024,
    )
    for (base_name, base_parameter), (merged_name, merged_parameter) in zip(
        named_trainable_parameters(base), named_trainable_parameters(merged)
    ):
        assert base_name == merged_name
        assert torch.allclose(merged_parameter, base_parameter - 1.0, atol=1e-5)


def test_nesterov_momentum_persists_across_rounds(tmp_path: Path) -> None:
    torch.manual_seed(0)
    base = tiny_model()
    base_dir = tmp_path / "base-0"
    base.save_pretrained(base_dir)
    first = make_update(base_dir, base, 0.1, "a", tmp_path / "updates-0")
    shapes = {name: parameter.shape for name, parameter in named_trainable_parameters(base)}
    verify_update(first, shapes)

    merged = copy.deepcopy(base)
    first_momentum = tmp_path / "momentum-0"
    first_momentum_files = apply_diloco_merge(
        merged,
        [first],
        momentum_reader=None,
        momentum_output_dir=first_momentum,
        outer_lr=0.7,
        outer_momentum=0.9,
        nesterov=True,
        weighting="uniform",
        max_delta_norm=0.0,
        momentum_dtype=torch.float32,
        max_shard_bytes=1024,
    )
    base_dir_1 = tmp_path / "base-1"
    merged.save_pretrained(base_dir_1)
    second = make_update(base_dir_1, merged, 0.2, "a", tmp_path / "updates-1")
    verify_update(second, shapes)

    before_second = copy.deepcopy(merged)
    apply_diloco_merge(
        merged,
        [second],
        momentum_reader=ShardedTensorReader(first_momentum, first_momentum_files),
        momentum_output_dir=tmp_path / "momentum-1",
        outer_lr=0.7,
        outer_momentum=0.9,
        nesterov=True,
        weighting="uniform",
        max_delta_norm=0.0,
        momentum_dtype=torch.float32,
        max_shard_bytes=1024,
    )
    # m_1 = 0.9 * 0.1 + 0.2 = 0.29; Nesterov direction = 0.2 + 0.9 * 0.29 = 0.461.
    expected_step = 0.7 * 0.461
    for (_, before), (_, after) in zip(named_trainable_parameters(before_second), named_trainable_parameters(merged)):
        assert torch.allclose(after, before - expected_step, atol=1e-5)


def test_real_local_steps_merge_and_reload(tmp_path: Path) -> None:
    torch.manual_seed(0)
    base = tiny_model()
    base_dir = tmp_path / "base"
    base.save_pretrained(base_dir)
    updates = []
    for index in range(2):
        local = copy.deepcopy(base).train()
        optimizer = torch.optim.AdamW(local.parameters(), lr=1e-3)
        input_ids = torch.randint(0, local.config.vocab_size, (2, 12))
        loss = local(input_ids=input_ids, labels=input_ids).loss
        loss.backward()
        optimizer.step()
        output = tmp_path / f"worker-{index}"
        files, norm, tensor_count = build_delta_shards(
            local,
            base_dir,
            output,
            dtype=torch.bfloat16,
            max_shard_bytes=2048,
        )
        update = UpdatePackage(
            pr_num=index + 1,
            revision="test",
            author=f"worker-{index}",
            metadata_path="metadata.json",
            metadata={
                "worker_id": f"worker-{index}",
                "num_tokens": input_ids.numel(),
                "tensor_count": tensor_count,
                "delta_norm_reported": norm,
            },
            local_dir=output,
            local_files=files,
        )
        updates.append(update)

    shapes = {name: parameter.shape for name, parameter in named_trainable_parameters(base)}
    for update in updates:
        assert verify_update(update, shapes) > 0
    merged = copy.deepcopy(base)
    apply_diloco_merge(
        merged,
        updates,
        momentum_reader=None,
        momentum_output_dir=tmp_path / "momentum",
        outer_lr=0.7,
        outer_momentum=0.9,
        nesterov=True,
        weighting="tokens",
        max_delta_norm=0.0,
        momentum_dtype=torch.bfloat16,
        max_shard_bytes=2048,
    )
    merged_dir = tmp_path / "merged"
    merged.save_pretrained(merged_dir)
    reloaded = BD3LMForCausalLM.from_pretrained(merged_dir)
    output = reloaded(input_ids=torch.randint(0, reloaded.config.vocab_size, (1, 8))).logits
    assert torch.isfinite(output).all()
