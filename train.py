"""
One-click training entry point for BD3LM, built entirely on the standard
Hugging Face `transformers.Trainer`.

Quick start
-----------
    python train.py --debug_tiny_model --max_steps 20

Real run on a single GPU:
    python train.py \\
        --tokenizer_name_or_path gpt2 \\
        --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \\
        --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \\
        --output_dir ./bd3lm-checkpoints

Multi-GPU (DeepSpeed ZeRO-2), via `accelerate`:
    accelerate launch --config_file configs/fsdp_config.yaml train.py \\
        --deepspeed configs/deepspeed_zero2.json ...

This script deliberately uses only the standard `Trainer` training loop —
no custom step function, no custom loss computation — because
`BD3LMForCausalLM` already returns a proper `CausalLMOutputWithPast` with
`.loss` populated whenever `labels` is passed. Anything that works with a
native `transformers` causal-LM (DeepSpeed, FSDP, LoRA/PEFT, TRL's
`GRPOTrainer`/`PPOTrainer`, `torch.compile`, ...) works here unmodified.
"""

from __future__ import annotations

import argparse
import logging
import math
from functools import partial
from itertools import chain
from typing import Any, Dict, List, Optional

import torch
from datasets import DatasetDict, load_dataset
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
    default_data_collator,
    set_seed,
)

from modeling_bd3lm import BD3LMConfig, BD3LMForCausalLM

logger = logging.getLogger(__name__)


# =============================================================================
# CLI
# =============================================================================


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One-click Trainer-based training script for BD3LM.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    data_group = parser.add_argument_group("data")
    data_group.add_argument("--dataset_name", type=str, default="wikitext", help="A `datasets` Hub dataset name.")
    data_group.add_argument("--dataset_config_name", type=str, default="wikitext-2-raw-v1")
    data_group.add_argument("--train_text_column", type=str, default="text")
    data_group.add_argument(
        "--tokenizer_name_or_path",
        type=str,
        default="gpt2",
        help="Any `AutoTokenizer`-compatible name or local path. BD3LM's embedding table is "
        "resized to match this tokenizer automatically, regardless of BD3LMConfig's default "
        "128,000-token vocabulary — so any tokenizer you point at works out of the box.",
    )
    data_group.add_argument("--block_size", type=int, default=2048, help="Fixed sequence length each training example is chunked into.")
    data_group.add_argument("--preprocessing_num_workers", type=int, default=4)

    model_group = parser.add_argument_group("model")
    model_group.add_argument(
        "--debug_tiny_model",
        action="store_true",
        help="Swap in a tiny BD3LMConfig for a fast end-to-end smoke test instead of the "
        "full ~3.6B-parameter spec configuration.",
    )
    model_group.add_argument(
        "--resume_from_checkpoint", type=str, default=None, help="Path to a BD3LM checkpoint directory to continue training from."
    )

    optim_group = parser.add_argument_group("optimization")
    optim_group.add_argument("--per_device_train_batch_size", type=int, default=4)
    optim_group.add_argument("--per_device_eval_batch_size", type=int, default=4)
    optim_group.add_argument("--gradient_accumulation_steps", type=int, default=8)
    optim_group.add_argument("--learning_rate", type=float, default=3e-4)
    optim_group.add_argument("--weight_decay", type=float, default=0.1)
    optim_group.add_argument("--adam_beta1", type=float, default=0.9)
    optim_group.add_argument("--adam_beta2", type=float, default=0.95)
    optim_group.add_argument("--max_grad_norm", type=float, default=1.0)
    optim_group.add_argument("--warmup_steps", type=int, default=200)
    optim_group.add_argument("--lr_scheduler_type", type=str, default="cosine")
    optim_group.add_argument("--num_train_epochs", type=float, default=1.0)
    optim_group.add_argument("--max_steps", type=int, default=-1, help="If > 0, overrides --num_train_epochs.")

    runtime_group = parser.add_argument_group("runtime")
    runtime_group.add_argument("--output_dir", type=str, default="./bd3lm-checkpoints")
    runtime_group.add_argument("--logging_steps", type=int, default=10)
    runtime_group.add_argument("--save_steps", type=int, default=500)
    runtime_group.add_argument("--eval_steps", type=int, default=500)
    runtime_group.add_argument("--save_total_limit", type=int, default=3)
    runtime_group.add_argument("--gradient_checkpointing", action="store_true", default=True)
    runtime_group.add_argument("--no_gradient_checkpointing", dest="gradient_checkpointing", action="store_false")
    runtime_group.add_argument("--bf16", action="store_true", default=torch.cuda.is_available())
    runtime_group.add_argument("--deepspeed", type=str, default=None, help="Path to a DeepSpeed JSON config, e.g. configs/deepspeed_zero2.json.")
    runtime_group.add_argument("--fsdp", type=str, default="", help='e.g. "full_shard auto_wrap"; typically set via an `accelerate` FSDP config instead (see configs/fsdp_config.yaml).')
    runtime_group.add_argument("--report_to", type=str, default="none", help='e.g. "wandb", "tensorboard", or "none".')
    runtime_group.add_argument("--seed", type=int, default=42)
    runtime_group.add_argument("--dataloader_num_workers", type=int, default=4)

    hub_group = parser.add_argument_group("hub")
    hub_group.add_argument("--push_to_hub", action="store_true")
    hub_group.add_argument("--hub_model_id", type=str, default=None)

    return parser


# =============================================================================
# Tokenizer & model construction
# =============================================================================


def load_tokenizer(tokenizer_name_or_path: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.eos_token is not None else tokenizer.unk_token
    return tokenizer


def build_model_config(tokenizer: PreTrainedTokenizerBase, debug_tiny_model: bool) -> BD3LMConfig:
    """
    Builds a `BD3LMConfig` sized to the given tokenizer's actual vocabulary.
    `train()` additionally calls `model.resize_token_embeddings(len(tokenizer))`
    right after model construction as a second, belt-and-suspenders guarantee
    that the embedding table and the tokenizer never drift out of sync.
    """
    common_kwargs: Dict[str, Any] = dict(
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
        bos_token_id=tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 0,
        eos_token_id=tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0,
    )
    if debug_tiny_model:
        return BD3LMConfig(
            hidden_size=128,
            num_hidden_layers=2,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=16,
            rope_dim=8,
            max_position_embeddings=512,
            shared_expert_intermediate_size=256,
            num_routed_experts=4,
            routed_expert_intermediate_size=128,
            num_experts_per_tok=2,
            **common_kwargs,
        )
    return BD3LMConfig(**common_kwargs)


def build_model(tokenizer: PreTrainedTokenizerBase, debug_tiny_model: bool) -> BD3LMForCausalLM:
    config = build_model_config(tokenizer, debug_tiny_model)
    model = BD3LMForCausalLM(config)
    model.resize_token_embeddings(len(tokenizer))
    return model


# =============================================================================
# Dataset construction (tokenize -> concatenate -> chunk into fixed blocks)
# =============================================================================


def tokenize_function(
    examples: Dict[str, List[str]], tokenizer: PreTrainedTokenizerBase, text_column: str
) -> Dict[str, List[List[int]]]:
    return tokenizer(examples[text_column])


def group_texts(examples: Dict[str, List[List[int]]], block_size: int) -> Dict[str, List[List[int]]]:
    """
    Concatenates every tokenized example end-to-end and re-chunks the
    result into fixed-length `block_size` blocks (dropping the final,
    shorter-than-`block_size` remainder). This is the standard
    `run_clm.py`-style grouping used across the `transformers` causal-LM
    examples: with every training example already exactly `block_size`
    tokens long, no padding is needed, so `labels` can simply be a copy of
    `input_ids` and batching only needs `default_data_collator` to stack
    tensors.
    """
    concatenated_examples = {key: list(chain(*examples[key])) for key in examples.keys()}
    total_length = len(concatenated_examples[next(iter(examples.keys()))])
    total_length = (total_length // block_size) * block_size
    result = {
        key: [values[i : i + block_size] for i in range(0, total_length, block_size)]
        for key, values in concatenated_examples.items()
    }
    result["labels"] = result["input_ids"].copy()
    return result


def build_lm_dataset(args: argparse.Namespace, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    raw_datasets = load_dataset(args.dataset_name, args.dataset_config_name)
    column_names = raw_datasets["train"].column_names

    tokenized_datasets = raw_datasets.map(
        partial(tokenize_function, tokenizer=tokenizer, text_column=args.train_text_column),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        remove_columns=column_names,
        desc="Tokenizing dataset",
    )
    lm_datasets = tokenized_datasets.map(
        partial(group_texts, block_size=args.block_size),
        batched=True,
        num_proc=args.preprocessing_num_workers,
        desc=f"Grouping text into {args.block_size}-token blocks",
    )
    return lm_datasets


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    args = build_argument_parser().parse_args()
    set_seed(args.seed)

    tokenizer = load_tokenizer(args.tokenizer_name_or_path)
    model = build_model(tokenizer, args.debug_tiny_model)

    total_parameters = sum(p.numel() for p in model.parameters())
    logger.info("BD3LM parameter count: %s (%.3fB)", f"{total_parameters:,}", total_parameters / 1e9)
    logger.info("Tokenizer vocabulary size: %d", len(tokenizer))

    lm_datasets = build_lm_dataset(args, tokenizer)
    train_dataset = lm_datasets["train"]
    eval_dataset: Optional[Any] = lm_datasets["validation"] if "validation" in lm_datasets else None
    logger.info("Train examples (post-chunking): %d", len(train_dataset))
    if eval_dataset is not None:
        logger.info("Eval examples (post-chunking): %d", len(eval_dataset))

    if args.gradient_checkpointing:
        model.config.use_cache = False

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        max_grad_norm=args.max_grad_norm,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type=args.lr_scheduler_type,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if args.gradient_checkpointing else None,
        bf16=args.bf16,
        deepspeed=args.deepspeed,
        fsdp=args.fsdp if args.fsdp else "",
        report_to=args.report_to,
        seed=args.seed,
        push_to_hub=args.push_to_hub,
        hub_model_id=args.hub_model_id,
        dataloader_num_workers=args.dataloader_num_workers,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=default_data_collator,
    )

    train_result = trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    # Register the auto classes one more time on these exact instances, then
    # save — this guarantees `config.json` carries `auto_map` and that
    # `modeling_bd3lm.py` is copied alongside the checkpoint, so the saved
    # directory can be loaded anywhere with
    # `AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)`.
    model.config.register_for_auto_class()
    model.register_for_auto_class("AutoModelForCausalLM")

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()

    metrics = train_result.metrics
    metrics["train_total_parameters"] = total_parameters
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)

    if eval_dataset is not None:
        eval_metrics = trainer.evaluate()
        try:
            eval_metrics["perplexity"] = math.exp(eval_metrics["eval_loss"])
        except OverflowError:
            eval_metrics["perplexity"] = float("inf")
        trainer.log_metrics("eval", eval_metrics)
        trainer.save_metrics("eval", eval_metrics)

    logger.info("Training complete. Checkpoint saved to %s", args.output_dir)


if __name__ == "__main__":
    main()
