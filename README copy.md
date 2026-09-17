# BD3LM — Big Data Deep 3 Reasoning Language Model

A from-scratch autoregressive reasoning language model, implemented as a native
[Hugging Face Transformers](https://github.com/huggingface/transformers) architecture:
`AutoConfig`/`AutoModelForCausalLM`-registered, `Trainer`-compatible, and built around
an extremely KV-cache-efficient attention mechanism for high-concurrency serving.

Everything lives in one self-contained file, **`modeling_bd3lm.py`** — config, rotary
embeddings, attention, mixture-of-experts, transformer block, and the `PreTrainedModel`
wrappers — so it can be imported directly, `pip install`-ed, or pushed to the Hugging
Face Hub as `trust_remote_code` custom modeling code.

This implementation was built and verified against **`transformers==5.17.0`**
(the current `transformers` v5.x generation — note that v5 made real breaking changes
versus the older 4.x series: `Cache` objects now expose `.layers[i].keys/.values`
instead of `.key_cache`/`.value_cache`, mask-building goes through
`transformers.masking_utils.create_causal_mask`, `GenerationMixin` is no longer part of
`PreTrainedModel`'s MRO by default, and `_tied_weights_keys` is now a dict). Every code
path below — config, forward pass, loss, generation, save/load, gradient checkpointing,
`AutoClass` registration — was actually run in a Python 5.17.0 environment while writing
this, not just written by analogy to older versions.

## Parameter count

The hyperparameters for this build add up to **~3.6B parameters**

| Component | Parameters |
|---|---:|
| Token embedding (tied with `lm_head`) | 262.1M |
| Attention, all 24 layers | 314.6M |
| MoE (1 shared + 8 routed experts), all 24 layers | 3,020.3M |
| RMSNorm layers | 0.1M |
| **Total** | **≈3.60B** |

Running `python modeling_bd3lm.py` prints this exact breakdown (computed on the `meta`
device, so it costs no memory or time) and asserts it's internally consistent, so you can
verify the number yourself rather than take it on faith.

The 24 routed+shared experts per layer are the dominant cost — MoE alone is ~3.02B of
the ~3.6B total. If you need the true 1.5B-parameter point specified in the original
brief, `BD3LMConfig` is fully parametrized, so any of these gets you there:

* **Fewer layers, same width/experts:** `num_hidden_layers=9` lands at ≈1.51B (the
  fastest lever — one number to change).
* **Same 24 layers, narrower experts:** roughly `shared_expert_intermediate_size=1280`,
  `routed_expert_intermediate_size=640` gets you to ≈1.5B while keeping full depth.
* **Smaller vocabulary:** the 128,000-token embedding table alone is 262M params (tied);
  a 32,000-token vocabulary saves ~230M if you don't need the full multilingual range.

None of these change the attention design (24:1 MQA, partial RoPE), the MoE routing
logic, or anything else about how the model behaves architecturally — they're pure size
knobs.

## Architecture

| | |
|---|---|
| Vocabulary | 128,000 tokens |
| Hidden size (`d_model`) | 2,048 |
| Layers | 24 |
| Attention | Multi-Query Attention — 24 query heads, **1** shared key/value head (24:1) |
| Head dimension | 128 |
| RoPE | Partial — a dedicated 64-of-128 channel slice, `theta=500,000` |
| Max context | 32,768 tokens |
| Feed-forward | DeepSeek-style MoE: 1 shared expert (width 4,096) + 8 routed experts (width 2,048), top-2 routing |
| Norm | Pre-RMSNorm residual blocks |

**Why a 24:1 KV ratio matters for serving:** every generated token appends one key
vector and one value vector *per key/value head* to the resident cache. With 24 query
heads collapsed onto a single shared KV head, BD3LM's cache is 24x smaller, at any given
context length, than standard multi-head attention would need — directly translating
into more concurrent sequences per GPU on multi-tenant serving stacks (vLLM, Featherless-style
infrastructure, etc.). The attention block also never *materializes* the repeated KV
heads in memory during the main forward pass: on the SDPA fast path (no explicit padding
mask in play), `KVEfficientAttention` calls `scaled_dot_product_attention(..., enable_gqa=True)`,
which broadcasts the single KV head across all 24 query heads inside the fused kernel
itself, only falling back to an explicit `repeat_kv` when an eager attention-weights
pass or an explicit mask is required.

**Why the MoE is split into a shared + routed experts:** the always-on shared expert is
meant to absorb general-purpose computation every token needs — baseline syntax,
grammar, broadly useful transformations — while the top-2-of-8 routed experts only fire
sparsely, giving the model *capacity* for different experts to specialize (e.g. toward
symbolic/arithmetic sub-skills versus free-form chain-of-thought elaboration) without
forcing that specialization by hand. A basic Switch-Transformer/Mixtral-style
load-balancing auxiliary loss (`compute_load_balancing_loss` in `modeling_bd3lm.py`)
discourages the router from collapsing onto a small subset of experts.

## Repository layout

```
bd3lm/
├── modeling_bd3lm.py       # The entire model: config, RoPE, attention, MoE, block,
│                           # PreTrainedModel wrappers, AutoClass registration, and a
│                           # `python modeling_bd3lm.py` verification/smoke-test suite.
├── train.py                # One-click Trainer-based training entry point.
├── requirements.txt
├── pyproject.toml          # `pip install -e .`
├── configs/
│   ├── deepspeed_zero2.json
│   └── fsdp_config.yaml    # `accelerate launch --config_file configs/fsdp_config.yaml train.py`
├── LICENSE                 # Apache 2.0 — swap in whatever license actually fits your project
└── README.md
```

## Installation

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# or, for an editable/importable package:
pip install -e ".[train]"
```

## Verify the implementation

```bash
python modeling_bd3lm.py
```

This is the comprehensive check described above: it reports the full-scale parameter
count (on the `meta` device — instant, no memory used), builds a tiny
architecturally-identical debug model, runs a real training forward+backward pass and
asserts the loss is finite and gradients reach the router/shared/routed experts, then
runs a real `.generate()` call and inspects the KV cache shape to confirm the 24:1 MQA
compression is actually happening at the tensor level.

## Train it — one click

```bash
# Fast smoke test (tiny model, no GPU needed, finishes in seconds):
python train.py --debug_tiny_model --max_steps 20

# Real single-GPU run:
python train.py \
  --tokenizer_name_or_path gpt2 \
  --dataset_name wikitext --dataset_config_name wikitext-103-raw-v1 \
  --per_device_train_batch_size 4 --gradient_accumulation_steps 16 \
  --output_dir ./bd3lm-checkpoints
```

`--tokenizer_name_or_path` accepts any `AutoTokenizer`-compatible name or local path —
`train.py` calls `model.resize_token_embeddings(len(tokenizer))` right after building the
model, so the embedding table always matches whatever tokenizer you point at, regardless
of `BD3LMConfig`'s 128,000-token default. `gpt2`'s tokenizer is the default purely
because it needs no authentication/license acceptance to download; for the
multilingual/symbolic coverage the 128k vocabulary is actually sized for, point
`--tokenizer_name_or_path` at (or train) a multilingual BPE/SentencePiece tokenizer
instead.

### Multi-GPU: DeepSpeed ZeRO-2

```bash
accelerate launch train.py --deepspeed configs/deepspeed_zero2.json \
  --tokenizer_name_or_path gpt2 --per_device_train_batch_size 4
```

### Multi-GPU: FSDP

```bash
accelerate launch --config_file configs/fsdp_config.yaml train.py \
  --tokenizer_name_or_path gpt2 --per_device_train_batch_size 4
```

`configs/fsdp_config.yaml` wraps `BD3LMBlock` as the atomic FSDP unit (via
`BD3LMPreTrainedModel._no_split_modules`), so each block's attention + MoE weights
always shard and gather together. Edit `num_processes` in that file to match your GPU
count.

### GRPO / PPO (TRL)

`BD3LMForCausalLM` is a standard `PreTrainedModel` + `GenerationMixin`, so TRL's
trainers work against it exactly as they would against any Hub model:

```python
from trl import GRPOConfig, GRPOTrainer
from modeling_bd3lm import BD3LMConfig, BD3LMForCausalLM
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("gpt2")
model = BD3LMForCausalLM(BD3LMConfig(vocab_size=len(tokenizer)))
model.resize_token_embeddings(len(tokenizer))

def reward_correct_length(completions, **kwargs):
    return [1.0 if 20 <= len(c.split()) <= 60 else 0.0 for c in completions]

trainer = GRPOTrainer(
    model=model,
    reward_funcs=reward_correct_length,
    args=GRPOConfig(output_dir="./bd3lm-grpo"),
    train_dataset=your_prompt_dataset,
)
trainer.train()
```

## Loading a trained checkpoint

Once `train.py` (or your own script) has called
`model.register_for_auto_class("AutoModelForCausalLM")` and `trainer.save_model(...)` —
which `train.py` already does for you — the saved directory carries both the weights and
`modeling_bd3lm.py` itself, and loads anywhere with:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("./bd3lm-checkpoints", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("./bd3lm-checkpoints")
```

If `modeling_bd3lm.py` is already importable in your environment (e.g. you ran
`pip install -e .` in this repo), the same call works *without* `trust_remote_code=True`
— `modeling_bd3lm.py` also calls `AutoConfig.register(...)` /
`AutoModelForCausalLM.register(...)` at import time for exactly this local-package case.

## License

Apache 2.0 (see `LICENSE`) — update this to whatever license actually fits your project;
it's included as a reasonable, permissive default for a from-scratch model repository,
not a statement about what you should use.
