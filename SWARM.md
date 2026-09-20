# BD3LM Volunteer GPU Swarm

This workflow trains through independent local replicas rather than a synchronous
FSDP/DeepSpeed collective. A worker may join any open round, train locally, submit an
update, and disconnect. The coordinator only needs a quorum; no particular worker is
required to remain online.

It is round-based partial-participation DiLoCo, not fully asynchronous SGD:

1. Every worker in round `t` pulls the exact same committed model `theta_t`.
2. It runs `H` local AdamW steps and uploads `g_i = theta_t - theta_local_i`.
3. The coordinator validates and averages a quorum of updates.
4. It applies an outer SGD/Nesterov step and opens round `t + 1`.
5. Late updates are rejected because applying a delta to the wrong base corrupts training.

Workers upload through Hugging Face Pull Requests. They use their own HF identities and
do not need your repository's write token. Only the coordinator writes `main`.

## Requirements

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
hf auth login
```

Use a Hugging Face **model** repository. For a private repo, grant each contributor read
access so they can download the model and open a PR. The coordinator account needs write
access. Keep its write token off volunteer machines.

Replace `ORG/BD3LM-swarm` below with your repo ID.

## 1. Initialize the repository

Run a cheap pilot first:

```bash
python swarm_coordinator.py init \
  --hub_repo_id ORG/BD3LM-swarm \
  --create_repo --private --debug_tiny_model \
  --min_updates 2 --max_updates 8 --local_steps 20
```

For a real run, initialize from a tested `save_pretrained` checkpoint:

```bash
python swarm_coordinator.py init \
  --hub_repo_id ORG/BD3LM-swarm \
  --checkpoint ./bd3lm-seed \
  --min_updates 2 --max_updates 8 --local_steps 500 \
  --outer_lr 0.7 --outer_momentum 0.9
```

If the repository root already contains `config.json`, tokenizer files, and safetensors
weights:

```bash
python swarm_coordinator.py init \
  --hub_repo_id ORG/BD3LM-swarm --use_existing_model \
  --min_updates 2 --max_updates 8 --local_steps 500
```

Without `--checkpoint`, `--use_existing_model`, or `--debug_tiny_model`, `init` constructs
the full random 3.6B model locally. That requires enough CPU RAM to create and save it.

Initialization writes `swarm/state.json`. Its policy is the authoritative round number,
base model commit, quorum, local step count, and DiLoCo outer optimizer settings.

## 2. Run the coordinator

Run exactly one coordinator against a persistent local directory:

```bash
python swarm_coordinator.py run \
  --hub_repo_id ORG/BD3LM-swarm \
  --state_dir /persistent/bd3lm-coordinator \
  --poll_interval 60
```

The coordinator:

- scans open `[swarm]` Pull Requests;
- accepts updates only from the current model commit and round;
- accepts one worker per HF author by default;
- checks exact parameter keys, shapes, and finite values;
- independently recomputes each global delta norm;
- averages accepted updates uniformly by default;
- applies outer Nesterov (`lr=0.7`, `momentum=0.9` by default);
- commits the next model and closes accepted PRs without merging raw deltas.

For trusted contributors with multiple GPUs under one HF account:

```bash
python swarm_coordinator.py run ... --max_updates_per_author 4
```

For a public repository, allowlist your contributors before the coordinator downloads
their update files:

```bash
python swarm_coordinator.py run ... --allowed_authors alice bob carol
```

For unequal local batch/step counts, weight by estimated processed tokens:

```bash
python swarm_coordinator.py run ... --weighting tokens
```

For untrusted or experimental workers, set an absolute global L2 bound after observing
normal update norms in a pilot:

```bash
python swarm_coordinator.py run ... --max_delta_norm 250.0
```

`0` disables clipping. Clipping helps limit one update's magnitude but does not make
arbitrary untrusted training code or poisoned data safe.

The outer momentum is persisted under `--state_dir`, not uploaded publicly. Keep this
directory durable. The Hub state/merge marker makes publication crash-safe; if the local
momentum directory is lost, the model can continue but momentum restarts from zero.

## 3. Join as a worker

Each contributor authenticates with their own HF token and runs:

```bash
python swarm_worker.py \
  --hub_repo_id ORG/BD3LM-swarm \
  --worker_id alice-rtx4090 \
  --repeat \
  --dataset_name Salesforce/wikitext \
  --dataset_config_name wikitext-103-raw-v1 \
  --per_device_train_batch_size 1 \
  --gradient_accumulation_steps 8
```

Without `--repeat`, the process trains and submits one update. With `--repeat`, it waits
for the next open round and keeps contributing until interrupted.

The worker records its last submitted round under `~/.cache/bd3lm-swarm` to avoid
re-uploading after a restart. Use `--force_resubmit` only when the previous PR was lost
or explicitly rejected.

Stopping a worker is safe:

- after upload, its PR remains available to the coordinator;
- during local training, only that worker's unfinished cycle is lost;
- if the round advances before upload, the worker discards the stale delta;
- on restart, it pulls the newest committed model and rejoins.

One machine with multiple GPUs can run independent workers instead of local DDP:

```bash
CUDA_VISIBLE_DEVICES=0 python swarm_worker.py ... --worker_id host-gpu0 --repeat
CUDA_VISIBLE_DEVICES=1 python swarm_worker.py ... --worker_id host-gpu1 --repeat
```

Increase the coordinator's `--max_updates_per_author` for this setup.

## Inspect status

```bash
python swarm_coordinator.py status --hub_repo_id ORG/BD3LM-swarm
```

The Hub repository also exposes `swarm/state.json`, `swarm/merge-result.json`, and open
worker PRs directly in the web UI.

## Resource reality

The swarm removes the requirement for a low-latency network and permanently available
workers. It does **not** shard one model across unrelated machines: every worker trains a
complete model replica.

For the default 3.6B model:

- bf16 weights are about 7.2 GB;
- each bf16 delta upload is about 7.2 GB per outer round;
- AdamW weights, gradients, and optimizer state require substantially more VRAM/RAM;
- the coordinator needs the model, outer momentum, and one-parameter-at-a-time fp32
  scratch space (32 GB system RAM is a practical minimum; more is safer);
- `--coordinator_dtype bfloat16 --momentum_dtype bfloat16` reduces coordinator RAM at
  the cost of precision.

If contributors cannot train a complete 3.6B replica, initialize a smaller BD3LM config
(the README lists 1.5B-size options). LoRA is useful for post-training an already capable
base model, but LoRA alone is not a substitute for pretraining this random-init model.

Updates are sharded (`--max_shard_size 2GB`) and uploaded only once per hundreds of local
steps, which is why this works over normal internet links. Closed PRs can still consume
Hub/Xet storage history; monitor repository storage on a long run. An object store or
ephemeral bucket is a better update transport at large scale, while the model and round
state can remain on the Hub.

## Failure and trust model

- Run only one coordinator. Hub commits use optimistic parent checks so a second one
  fails instead of silently applying the same round twice.
- The coordinator locks a selected quorum before publishing, records a merge marker in
  the model commit, and resumes interrupted publication safely.
- Workers never receive the coordinator's write token.
- A malicious update can still steer training while remaining finite and norm-bounded.
  For an open public swarm, add allowlisted HF identities, held-out evaluation gates,
  robust aggregation, and signed worker metadata before treating resulting weights as
  trustworthy.
