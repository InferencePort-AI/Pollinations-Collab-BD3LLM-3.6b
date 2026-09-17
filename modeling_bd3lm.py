"""
BD3LM — Big Data Deep 3 Reasoning Language Model
==================================================

A from-scratch autoregressive reasoning language model implemented as a
native Hugging Face Transformers architecture, built against the
`transformers` 5.x API (Cache-object `past_key_values`, `masking_utils`,
`GradientCheckpointingLayer`).

IMPORTANT — parameter-count note
---------------------------------
The literal hyperparameters requested for this build (128,000-token
vocabulary, 24 layers, 2048 hidden size, a 4096-wide shared expert plus
eight 2048-wide routed experts) multiply out to roughly **3.6B**
parameters see the worked breakdown that the verification
suite below prints, and the README for exactly which knobs to turn if you
need to land on 1.5B on the nose. Every dimension is controlled by
`BD3LMConfig`, so nothing here is hardcoded around the 3.6B figure.

Architecture summary
---------------------
* Vocabulary: 128,000 tokens (generous multilingual + symbolic coverage,
  so common shorthand, digit groupings, and non-English subwords are
  less likely to fragment into many tokens).
* 24 transformer blocks, Pre-RMSNorm residual structure.
* Attention: Ultra-Compressed Multi-Query Attention — 24 query heads
  share a single key/value head (24:1), so the resident KV cache per
  token is 24x smaller than standard multi-head attention. A partial
  rotary embedding (a dedicated 64-of-128 channel slice) extends cleanly
  to 32,768-token contexts.
* Feed-forward: DeepSeek-style fine-grained Mixture-of-Experts — one
  always-on shared expert plus eight sparsely-routed experts (top-2 per
  token), with a basic load-balancing auxiliary loss.
* Fully native to `transformers`: `AutoConfig` / `AutoModelForCausalLM`
  registration, standard `Cache`-based `past_key_values`, and a `forward`
  signature that `Trainer`, `accelerate` (DeepSpeed/FSDP), and `trl`
  (GRPO/PPO) all understand without any custom training loop.

This single file intentionally contains the entire model implementation
— configuration, rotary embeddings, attention, mixture-of-experts,
transformer block, and the `PreTrainedModel` wrappers — so it can be
imported directly or pushed to the Hugging Face Hub as `trust_remote_code`
custom modeling code.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import GenerationMixin, PretrainedConfig, PreTrainedModel
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache, DynamicCache
from transformers.masking_utils import create_causal_mask
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.utils import logging

logger = logging.get_logger(__name__)


# =============================================================================
# 1. BD3LMConfig — native Hugging Face configuration class
# =============================================================================


class BD3LMConfig(PretrainedConfig):
    """
    Configuration class storing every architectural hyperparameter of a
    BD3LM model. Mirrors the pattern used by native `transformers`
    configuration classes (e.g. `LlamaConfig`): instantiating a
    `BD3LMConfig` does not build any tensors, it only records the shape of
    the model that `BD3LMForCausalLM(config)` will build.

    Args:
        vocab_size: Number of unique tokens the model can embed / predict.
        hidden_size: Width of the residual stream (`d_model`).
        num_hidden_layers: Number of stacked `BD3LMBlock` transformer blocks.
        num_attention_heads: Number of *query* heads.
        num_key_value_heads: Number of *key/value* heads. BD3LM defaults
            to a single shared key/value head (Multi-Query Attention), so
            the KV cache stored per token is
            `num_attention_heads // num_key_value_heads` times smaller
            than standard multi-head attention.
        head_dim: Dimensionality of each attention head.
        rope_dim: Number of channels, out of `head_dim`, that receive
            rotary position embeddings. The remaining
            `head_dim - rope_dim` channels of every head pass straight
            through unrotated (rotary applied to a dedicated slice).
        rope_theta: Base for the rotary embedding's geometric frequency
            progression. Larger bases extrapolate more gracefully to long
            contexts.
        max_position_embeddings: Maximum sequence length the rotary
            embeddings (and any position-dependent logic) are expected to
            support.
        shared_expert_intermediate_size: Hidden width of the single
            always-active shared expert MLP.
        num_routed_experts: Number of sparsely-gated, fine-grained routed
            experts.
        routed_expert_intermediate_size: Hidden width of each routed
            expert MLP.
        num_experts_per_tok: Number of routed experts activated per token
            ("top-k" routing).
        norm_topk_prob: Whether the top-k routing weights are renormalized
            to sum to 1 before being used to combine expert outputs.
        router_aux_loss_coef: Coefficient multiplying the load-balancing
            auxiliary loss before it is added to the primary
            cross-entropy loss.
        hidden_act: Activation function used inside every SwiGLU MLP
            (shared expert, routed experts).
        rms_norm_eps: Epsilon used inside every `BD3LMRMSNorm`.
        attention_dropout: Dropout probability applied to attention
            weights (only exercised on the eager / `output_attentions`
            code path).
        attention_bias: Whether the Q/K/V/O projections include a bias
            term.
        initializer_range: Standard deviation used for the plain-normal
            initialization of newly created weights.
        use_cache: Whether the model returns `past_key_values` by default.
        tie_word_embeddings: Whether the input embedding matrix and the
            output `lm_head` projection share the same weight tensor.
    """

    model_type = "bd3lm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size: int = 128_000,
        hidden_size: int = 2048,
        num_hidden_layers: int = 24,
        num_attention_heads: int = 24,
        num_key_value_heads: int = 1,
        head_dim: int = 128,
        rope_dim: int = 64,
        rope_theta: float = 500_000.0,
        max_position_embeddings: int = 32_768,
        shared_expert_intermediate_size: int = 4096,
        num_routed_experts: int = 8,
        routed_expert_intermediate_size: int = 2048,
        num_experts_per_tok: int = 2,
        norm_topk_prob: bool = True,
        router_aux_loss_coef: float = 0.001,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-6,
        attention_dropout: float = 0.0,
        attention_bias: bool = False,
        initializer_range: float = 0.02,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        pad_token_id: int = 0,
        bos_token_id: int = 1,
        eos_token_id: int = 2,
        **kwargs: Any,
    ) -> None:
        if num_attention_heads % num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads ({num_attention_heads}) must be divisible by "
                f"num_key_value_heads ({num_key_value_heads}) for grouped/multi-query attention."
            )
        if rope_dim > head_dim:
            raise ValueError(f"rope_dim ({rope_dim}) cannot exceed head_dim ({head_dim}).")
        if rope_dim % 2 != 0:
            raise ValueError(f"rope_dim ({rope_dim}) must be even (rotary embeddings rotate channel pairs).")
        if num_experts_per_tok > num_routed_experts:
            raise ValueError(
                f"num_experts_per_tok ({num_experts_per_tok}) cannot exceed num_routed_experts ({num_routed_experts})."
            )

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.rope_dim = rope_dim
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.shared_expert_intermediate_size = shared_expert_intermediate_size
        self.num_routed_experts = num_routed_experts
        self.routed_expert_intermediate_size = routed_expert_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.router_aux_loss_coef = router_aux_loss_coef
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.attention_dropout = attention_dropout
        self.attention_bias = attention_bias
        self.initializer_range = initializer_range
        self.use_cache = use_cache

        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )


# =============================================================================
# 2. FeatherweightRoPE & apply_rope — partial rotary position embeddings
# =============================================================================


class FeatherweightRoPE(nn.Module):
    """
    Lightweight rotary position embedding generator.

    Precomputes only `rope_dim / 2` inverse frequencies (32 floats for the
    default configuration) and derives `cos`/`sin` tensors on demand from
    whatever `position_ids` are requested. Because nothing beyond that
    tiny frequency vector is ever cached, this module supports arbitrarily
    long contexts — including the full 32,768-token design target — with
    a fixed, negligible memory footprint.
    """

    def __init__(self, config: BD3LMConfig, device: Optional[torch.device] = None) -> None:
        super().__init__()
        self.rope_dim = config.rope_dim
        self.rope_theta = config.rope_theta
        self.max_position_embeddings = config.max_position_embeddings

        inverse_frequency = 1.0 / (
            self.rope_theta
            ** (
                torch.arange(0, self.rope_dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float32)
                / self.rope_dim
            )
        )
        self.register_buffer("inverse_frequency", inverse_frequency, persistent=False)

    @torch.no_grad()
    def forward(
        self, hidden_states_for_dtype: torch.Tensor, position_ids: torch.LongTensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            hidden_states_for_dtype: Any tensor already on the target
                device/dtype; only its `.device`/`.dtype` are read.
            position_ids: `(batch_size, sequence_length)` absolute token
                positions to compute rotary angles for.

        Returns:
            `(cos, sin)`, each shaped `(batch_size, sequence_length,
            rope_dim)`, ready to be passed straight into `apply_rope`.
        """
        expanded_inverse_frequency = (
            self.inverse_frequency[None, :, None]
            .float()
            .expand(position_ids.shape[0], -1, 1)
            .to(position_ids.device)
        )
        expanded_position_ids = position_ids[:, None, :].float()

        rotary_angles = (expanded_inverse_frequency @ expanded_position_ids).transpose(1, 2)
        combined_angles = torch.cat((rotary_angles, rotary_angles), dim=-1)

        cos = combined_angles.cos()
        sin = combined_angles.sin()
        return cos.to(dtype=hidden_states_for_dtype.dtype), sin.to(dtype=hidden_states_for_dtype.dtype)


def rotate_half(hidden_states_slice: torch.Tensor) -> torch.Tensor:
    """Splits the last dimension in half and swaps the halves with a sign flip — the standard rotary-embedding helper (`[-x2, x1]`)."""
    first_half = hidden_states_slice[..., : hidden_states_slice.shape[-1] // 2]
    second_half = hidden_states_slice[..., hidden_states_slice.shape[-1] // 2 :]
    return torch.cat((-second_half, first_half), dim=-1)


def apply_rope(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    rope_dim: int,
    unsqueeze_dim: int = 1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Applies rotary position embeddings to only the first `rope_dim`
    channels of `query_states`/`key_states`, leaving the remaining
    `head_dim - rope_dim` channels of every head untouched. This
    "dedicated slice" design keeps the rotary computation cheap while
    still letting the un-rotated channels carry pure content information.

    Args:
        query_states: `(batch, num_attention_heads, seq_len, head_dim)`.
        key_states: `(batch, num_key_value_heads, seq_len, head_dim)`.
        cos, sin: `(batch, seq_len, rope_dim)`, from `FeatherweightRoPE`.
        rope_dim: Number of leading channels to rotate.
        unsqueeze_dim: Dimension at which to insert a broadcastable head
            axis into `cos`/`sin` (default `1`, the head dimension).

    Returns:
        The rotated `(query_states, key_states)`.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)

    query_rotary_part, query_pass_through_part = query_states[..., :rope_dim], query_states[..., rope_dim:]
    key_rotary_part, key_pass_through_part = key_states[..., :rope_dim], key_states[..., rope_dim:]

    query_rotary_part = (query_rotary_part * cos) + (rotate_half(query_rotary_part) * sin)
    key_rotary_part = (key_rotary_part * cos) + (rotate_half(key_rotary_part) * sin)

    rotated_query_states = torch.cat((query_rotary_part, query_pass_through_part), dim=-1)
    rotated_key_states = torch.cat((key_rotary_part, key_pass_through_part), dim=-1)
    return rotated_query_states, rotated_key_states


# =============================================================================
# 3. KVEfficientAttention — Ultra-Compressed Multi-Query Attention
# =============================================================================


def repeat_kv(hidden_states: torch.Tensor, num_repeats: int) -> torch.Tensor:
    """
    Expands a `(batch, num_key_value_heads, seq_len, head_dim)` tensor to
    `(batch, num_key_value_heads * num_repeats, seq_len, head_dim)` by
    repeating each key/value head `num_repeats` times.

    This is only used on the eager (`output_attentions=True`) fallback
    path and whenever the fused SDPA `enable_gqa` broadcast trick isn't
    applicable (see `KVEfficientAttention.forward`). The compact,
    un-repeated tensors are what actually get written into
    `past_key_values` — that's the whole point of Multi-Query Attention:
    the resident KV cache stays `num_repeats`x smaller than full
    multi-head attention would need, regardless of how attention scores
    happen to be computed on any given step.
    """
    batch_size, num_key_value_heads, sequence_length, head_dim = hidden_states.shape
    if num_repeats == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch_size, num_key_value_heads, num_repeats, sequence_length, head_dim
    )
    return hidden_states.reshape(batch_size, num_key_value_heads * num_repeats, sequence_length, head_dim)


class KVEfficientAttention(nn.Module):
    """
    Ultra-Compressed Multi-Query Attention (MQA).

    `num_attention_heads` query heads (24 by default) all read from a
    single shared key/value head (`num_key_value_heads = 1` by default),
    a 24:1 compression ratio. Every generated token therefore only needs
    to append one key vector and one value vector to the cache — instead
    of one per query head — which is what makes this block so
    KV-RAM-efficient at serving time: the resident cache for a given
    context length is `num_attention_heads / num_key_value_heads` times
    smaller than standard multi-head attention, directly translating into
    more concurrent sequences per GPU.

    On the main (non-`output_attentions`) path, this also avoids ever
    materializing a repeated key/value tensor: whenever no explicit
    additive mask is in play, `torch.nn.functional.scaled_dot_product_attention`
    is called with `enable_gqa=True`, which broadcasts the single KV head
    across all 24 query heads inside the fused kernel itself.

    `past_key_values` follows the standard Hugging Face `Cache` protocol
    (e.g. `DynamicCache`), so this block works unmodified with
    `PreTrainedModel.generate()`, `Trainer`, and gradient checkpointing.
    """

    def __init__(self, config: BD3LMConfig, layer_idx: int) -> None:
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_attention_heads // self.num_key_value_heads
        self.head_dim = config.head_dim
        self.rope_dim = config.rope_dim
        self.attention_dropout = config.attention_dropout
        self.scaling = self.head_dim**-0.5
        self.is_causal = True

        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.num_key_value_heads * self.head_dim, bias=config.attention_bias)
        self.o_proj = nn.Linear(self.num_attention_heads * self.head_dim, self.hidden_size, bias=config.attention_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, sequence_length, _ = hidden_states.shape
        query_shape = (batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        key_value_shape = (batch_size, sequence_length, self.num_key_value_heads, self.head_dim)

        query_states = self.q_proj(hidden_states).view(query_shape).transpose(1, 2)
        key_states = self.k_proj(hidden_states).view(key_value_shape).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rope(query_states, key_states, cos, sin, self.rope_dim)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        if output_attentions:
            # Eager fallback: materializes the broadcast KV heads so attention
            # weights can be returned to the caller for inspection.
            key_states_for_scores = repeat_kv(key_states, self.num_key_value_groups)
            value_states_for_scores = repeat_kv(value_states, self.num_key_value_groups)

            attention_scores = torch.matmul(query_states, key_states_for_scores.transpose(2, 3)) * self.scaling
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask
            attention_probabilities = F.softmax(attention_scores, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attention_probabilities = F.dropout(attention_probabilities, p=self.attention_dropout, training=self.training)
            attention_output = torch.matmul(attention_probabilities, value_states_for_scores)
            attention_output = attention_output.transpose(1, 2).contiguous()
        else:
            attention_probabilities = None
            sdpa_kwargs: Dict[str, Any] = {}
            if self.num_key_value_groups > 1 and attention_mask is None and self.head_dim <= 256:
                # Let the fused kernel broadcast the single KV head across all
                # query heads instead of physically repeating it in memory.
                sdpa_kwargs["enable_gqa"] = True
            else:
                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

            query_length = query_states.shape[2]
            is_causal = attention_mask is None and query_length > 1 and self.is_causal

            attention_output = F.scaled_dot_product_attention(
                query_states,
                key_states,
                value_states,
                attn_mask=attention_mask,
                dropout_p=self.attention_dropout if self.training else 0.0,
                scale=self.scaling,
                is_causal=is_causal,
                **sdpa_kwargs,
            )
            attention_output = attention_output.transpose(1, 2).contiguous()

        attention_output = attention_output.reshape(batch_size, sequence_length, -1)
        attention_output = self.o_proj(attention_output)
        return attention_output, attention_probabilities


# =============================================================================
# 4. DeepSeekMoE — shared + fine-grained routed expert mixture-of-experts
# =============================================================================


class BD3LMMLP(nn.Module):
    """A single SwiGLU feed-forward expert (used for both the shared expert and every routed expert)."""

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.activation_fn = ACT2FN[hidden_act]

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.activation_fn(self.gate_proj(hidden_states)) * self.up_proj(hidden_states))


def compute_load_balancing_loss(router_logits: torch.Tensor, num_routed_experts: int, num_experts_per_tok: int) -> torch.Tensor:
    """
    Basic Switch-Transformer / Mixtral-style load-balancing auxiliary loss.

    Encourages the router to spread tokens evenly across experts by
    penalizing the correlation between (a) the fraction of top-k routing
    slots each expert actually receives and (b) the average softmax
    probability the router assigns that expert. A perfectly balanced
    routing solution minimizes this loss; a router that collapses onto a
    small subset of experts is penalized.

    This is intentionally the *basic* formulation: it treats every token
    in the flattened batch uniformly and does not down-weight padding
    positions. A production system training on heavily padded batches
    could extend this with `attention_mask`-aware weighting, following
    the same pattern Mixtral's `load_balancing_loss_func` uses.

    Args:
        router_logits: `(num_tokens, num_routed_experts)` raw router
            logits (pre-softmax), flattened over batch and sequence.
        num_routed_experts: Total number of routed experts.
        num_experts_per_tok: Number of experts activated per token
            ("top-k").

    Returns:
        A scalar tensor.
    """
    routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
    _, selected_experts = torch.topk(routing_weights, num_experts_per_tok, dim=-1)
    selection_mask = F.one_hot(selected_experts, num_classes=num_routed_experts).float()  # (tokens, top_k, experts)

    fraction_of_slots_per_expert = selection_mask.mean(dim=0)  # (top_k, experts)
    average_router_probability_per_expert = routing_weights.mean(dim=0)  # (experts,)

    load_balancing_loss = torch.sum(fraction_of_slots_per_expert * average_router_probability_per_expert.unsqueeze(0))
    return load_balancing_loss * num_routed_experts


class DeepSeekMoE(nn.Module):
    """
    DeepSeek-style fine-grained Mixture-of-Experts feed-forward layer.

    Every token is processed by two kinds of experts simultaneously:

    * A single **shared expert** (always active, wide intermediate size)
      intended to absorb general-purpose, always-needed computation —
      baseline language syntax, grammar, broadly useful transformations
      that every token benefits from.
    * `num_routed_experts` **fine-grained routed experts** (narrower
      intermediate size each), of which only the top-`num_experts_per_tok`
      are activated per token by a learned router. Because only a sparse
      subset of routed experts fire for any given token, this gives the
      routing mechanism the *capacity* to let different experts specialize
      — for example toward symbolic/arithmetic sub-skills versus
      free-form chain-of-thought elaboration — although which expert ends
      up specializing in what is not hand-designed; it emerges (or
      doesn't) from training data and objective.

    A basic load-balancing auxiliary loss (`compute_load_balancing_loss`)
    is always computed and returned alongside the output so the caller can
    add `router_aux_loss_coef * aux_loss` to the primary training loss.
    """

    def __init__(self, config: BD3LMConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_routed_experts = config.num_routed_experts
        self.num_experts_per_tok = config.num_experts_per_tok
        self.norm_topk_prob = config.norm_topk_prob

        self.shared_expert = BD3LMMLP(config.hidden_size, config.shared_expert_intermediate_size, config.hidden_act)
        self.routed_experts = nn.ModuleList(
            [
                BD3LMMLP(config.hidden_size, config.routed_expert_intermediate_size, config.hidden_act)
                for _ in range(self.num_routed_experts)
            ]
        )
        self.router = nn.Linear(config.hidden_size, self.num_routed_experts, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size, sequence_length, hidden_size = hidden_states.shape
        flattened_hidden_states = hidden_states.view(-1, hidden_size)

        router_logits = self.router(flattened_hidden_states)
        routing_weights = F.softmax(router_logits, dim=-1, dtype=torch.float32)
        top_k_weights, top_k_expert_indices = torch.topk(routing_weights, self.num_experts_per_tok, dim=-1)
        if self.norm_topk_prob:
            top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)
        top_k_weights = top_k_weights.to(flattened_hidden_states.dtype)

        routed_output = torch.zeros_like(flattened_hidden_states)
        expert_assignment_mask = F.one_hot(top_k_expert_indices, num_classes=self.num_routed_experts).permute(2, 1, 0)
        # expert_assignment_mask: (num_routed_experts, num_experts_per_tok, num_tokens)

        for expert_index in range(self.num_routed_experts):
            top_k_slot, token_index = torch.where(expert_assignment_mask[expert_index])
            if token_index.numel() == 0:
                continue

            expert_module = self.routed_experts[expert_index]
            tokens_for_this_expert = flattened_hidden_states[token_index]
            expert_output = expert_module(tokens_for_this_expert)

            per_token_weight = top_k_weights[token_index, top_k_slot].unsqueeze(-1)
            routed_output.index_add_(0, token_index, expert_output * per_token_weight)

        shared_output = self.shared_expert(flattened_hidden_states)
        combined_output = (routed_output + shared_output).view(batch_size, sequence_length, hidden_size)

        aux_loss = compute_load_balancing_loss(router_logits, self.num_routed_experts, self.num_experts_per_tok)
        return combined_output, aux_loss


# =============================================================================
# 5. BD3LMBlock — Pre-RMSNorm transformer block
# =============================================================================


class BD3LMRMSNorm(nn.Module):
    """Root-Mean-Square LayerNorm (no mean-centering, no bias), computed in float32 for numerical stability."""

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        normalized_hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * normalized_hidden_states.to(input_dtype)

    def extra_repr(self) -> str:
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class BD3LMBlock(GradientCheckpointingLayer):
    """
    One complete BD3LM transformer block:

        residual -> RMSNorm -> KVEfficientAttention -> (+residual)
                 -> RMSNorm -> DeepSeekMoE           -> (+residual)

    i.e. a standard Pre-Norm residual structure, with the position-wise
    feed-forward sub-layer replaced by the shared+routed mixture-of-experts.

    Subclassing `GradientCheckpointingLayer` (rather than plain `nn.Module`)
    means gradient checkpointing "just works": when the owning model calls
    `gradient_checkpointing_enable()`, every `BD3LMBlock.__call__` is
    transparently wrapped with `torch.utils.checkpoint.checkpoint`, and
    `use_cache`/`past_key_values` are automatically disabled during
    checkpointed training so the KV cache is never written to twice
    (once on the forward pass, once on the checkpoint's backward replay).
    """

    def __init__(self, config: BD3LMConfig, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = KVEfficientAttention(config, layer_idx)
        self.moe = DeepSeekMoE(config)
        self.input_layernorm = BD3LMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = BD3LMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, attention_probabilities = self.self_attn(
            hidden_states=hidden_states,
            position_embeddings=position_embeddings,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            output_attentions=output_attentions,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states, aux_loss = self.moe(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states, attention_probabilities, aux_loss


# =============================================================================
# 6. BD3LMForCausalLM — PreTrainedModel wrappers
# =============================================================================


@dataclass
class BD3LMModelOutput(ModelOutput):
    """Output of the BD3LM backbone (`BD3LMModel`) — the standard `BaseModelOutputWithPast` shape, plus the averaged MoE auxiliary loss."""

    last_hidden_state: torch.FloatTensor = None
    past_key_values: Optional[Cache] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    aux_loss: Optional[torch.FloatTensor] = None


class BD3LMPreTrainedModel(PreTrainedModel):
    """Shared base class providing weight initialization and the standard `transformers` integration hooks used by every BD3LM model class."""

    config_class = BD3LMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["BD3LMBlock"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_sdpa = True
    _can_compile_fullgraph = False

    def _init_weights(self, module: nn.Module) -> None:
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, BD3LMRMSNorm):
            module.weight.data.fill_(1.0)


class BD3LMModel(BD3LMPreTrainedModel):
    """The BD3LM decoder backbone: token embedding -> `num_hidden_layers` `BD3LMBlock`s -> final RMSNorm. Returns hidden states, not logits (see `BD3LMForCausalLM` for the language-modeling head)."""

    def __init__(self, config: BD3LMConfig) -> None:
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList([BD3LMBlock(config, layer_idx) for layer_idx in range(config.num_hidden_layers)])
        self.norm = BD3LMRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = FeatherweightRoPE(config)
        self.gradient_checkpointing = False

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, ...], BD3LMModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Exactly one of `input_ids` or `inputs_embeds` must be provided.")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states: Optional[Tuple[torch.Tensor, ...]] = () if output_hidden_states else None
        all_self_attentions: Optional[Tuple[torch.Tensor, ...]] = () if output_attentions else None
        total_aux_loss = torch.zeros((), device=hidden_states.device, dtype=torch.float32)

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            hidden_states, attention_probabilities, layer_aux_loss = decoder_layer(
                hidden_states,
                position_embeddings=position_embeddings,
                attention_mask=causal_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
            )

            total_aux_loss = total_aux_loss + layer_aux_loss.to(total_aux_loss.device)

            if output_attentions:
                all_self_attentions = all_self_attentions + (attention_probabilities,)

        hidden_states = self.norm(hidden_states)
        total_aux_loss = total_aux_loss / len(self.layers)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(
                value
                for value in [hidden_states, past_key_values if use_cache else None, all_hidden_states, all_self_attentions, total_aux_loss]
                if value is not None
            )

        return BD3LMModelOutput(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            aux_loss=total_aux_loss,
        )


class BD3LMForCausalLM(BD3LMPreTrainedModel, GenerationMixin):
    """
    BD3LM with a causal-language-modeling head on top of the `BD3LMModel`
    backbone: a linear projection from `hidden_size` back to `vocab_size`
    (tied to the input embedding by default), plus the shifted
    cross-entropy loss + MoE load-balancing loss that `labels` trigger.

    Being a standard `PreTrainedModel` + `GenerationMixin` subclass with a
    conventional `forward(input_ids, attention_mask, labels,
    past_key_values, ...)` signature, this class is directly usable with
    `transformers.Trainer`, `accelerate`-launched DeepSpeed/FSDP training,
    `.generate()`, and TRL's `GRPOTrainer` / `PPOTrainer` without any
    adapter code.
    """

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: BD3LMConfig) -> None:
        super().__init__(config)
        self.model = BD3LMModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        self.post_init()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def set_decoder(self, decoder: BD3LMModel) -> None:
        self.model = decoder

    def get_decoder(self) -> BD3LMModel:
        return self.model

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Any,
    ) -> Union[Tuple[torch.Tensor, ...], CausalLMOutputWithPast]:
        """
        Standard Hugging Face causal-LM forward pass.

        If `labels` is provided, `logits` are shifted by one position
        against `labels` (`logits[:, :-1]` predicts `labels[:, 1:]`) and a
        causal cross-entropy loss is computed (`ignore_index=-100`,
        matching the padding convention used by every standard
        `transformers` data collator), with the MoE load-balancing
        auxiliary loss added on top, scaled by
        `config.router_aux_loss_coef`.
        """
        return_dict = return_dict if return_dict is not None else self.config.return_dict
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        outputs: BD3LMModelOutput = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        logits = logits.float()

        loss: Optional[torch.Tensor] = None
        if labels is not None:
            shifted_logits = logits[..., :-1, :].contiguous()
            shifted_labels = labels[..., 1:].contiguous()

            loss_fct = nn.CrossEntropyLoss()
            flattened_logits = shifted_logits.view(-1, self.config.vocab_size)
            flattened_labels = shifted_labels.view(-1).to(flattened_logits.device)
            causal_lm_loss = loss_fct(flattened_logits, flattened_labels)

            aux_loss = (
                outputs.aux_loss.to(causal_lm_loss.device)
                if outputs.aux_loss is not None
                else torch.zeros((), device=causal_lm_loss.device)
            )
            loss = causal_lm_loss + self.config.router_aux_loss_coef * aux_loss

        if not return_dict:
            output = (logits, outputs.past_key_values, outputs.hidden_states, outputs.attentions)
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Cache] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Standard `GenerationMixin` hook: trims `input_ids` to just the newest tokens once a `past_key_values` cache exists, and derives `position_ids` from `attention_mask` so left-padded batches stay correctly aligned."""
        if past_key_values is not None and cache_position is not None and input_ids.shape[1] != cache_position.shape[0]:
            input_ids = input_ids[:, cache_position]

        position_ids = kwargs.get("position_ids")
        if attention_mask is not None and position_ids is None:
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        model_inputs: Dict[str, Any] = {"input_ids": input_ids.contiguous()}
        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache", True),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs


# =============================================================================
# 7. Auto-class registration
# =============================================================================

BD3LMConfig.register_for_auto_class()
BD3LMModel.register_for_auto_class("AutoModel")
BD3LMForCausalLM.register_for_auto_class("AutoModelForCausalLM")

try:
    from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

    AutoConfig.register("bd3lm", BD3LMConfig)
    AutoModel.register(BD3LMConfig, BD3LMModel)
    AutoModelForCausalLM.register(BD3LMConfig, BD3LMForCausalLM)
except ValueError:
    # Already registered (e.g. this module was imported more than once in
    # the same interpreter session) — safe to ignore.
    pass


# =============================================================================
# 8. Verification block — smoke tests for both training and inference paths
# =============================================================================

if __name__ == "__main__":

    def format_param_count(num_params: int) -> str:
        return f"{num_params:,} ({num_params / 1e9:.3f}B)"

    print("=" * 88)
    print("BD3LM verification suite")
    print("=" * 88)

    # --- 1. Report the full, spec-matching parameter count ------------------
    # Built on the `meta` device so this is instant and uses ~0 bytes of
    # real memory, regardless of how large the configured model is.
    print("\n[1] Full-scale (spec) configuration parameter count")
    print("-" * 88)
    production_config = BD3LMConfig()
    with torch.device("meta"):
        meta_scale_model = BD3LMForCausalLM(production_config)

    embedding_param_count = sum(p.numel() for p in meta_scale_model.model.embed_tokens.parameters())
    attention_param_count = sum(sum(p.numel() for p in layer.self_attn.parameters()) for layer in meta_scale_model.model.layers)
    moe_param_count = sum(sum(p.numel() for p in layer.moe.parameters()) for layer in meta_scale_model.model.layers)
    norm_param_count = sum(
        sum(p.numel() for p in layer.input_layernorm.parameters()) + sum(p.numel() for p in layer.post_attention_layernorm.parameters())
        for layer in meta_scale_model.model.layers
    ) + sum(p.numel() for p in meta_scale_model.model.norm.parameters())
    total_param_count = sum(p.numel() for p in meta_scale_model.parameters())

    component_sum = embedding_param_count + attention_param_count + moe_param_count + norm_param_count
    assert component_sum == total_param_count, (
        f"Component breakdown ({component_sum:,}) does not match the total parameter count "
        f"({total_param_count:,}); `lm_head` is tied to `embed_tokens` so it must not be counted twice."
    )

    print(f"  Embedding / tied lm_head  : {format_param_count(embedding_param_count)}")
    print(f"  Attention (all {production_config.num_hidden_layers} layers) : {format_param_count(attention_param_count)}")
    print(f"  MoE (all {production_config.num_hidden_layers} layers)       : {format_param_count(moe_param_count)}")
    print(f"  RMSNorm layers            : {format_param_count(norm_param_count)}")
    print(f"  TOTAL                     : {format_param_count(total_param_count)}")
    print(
        "\n  Note: the exact hyperparameters requested for this build (24 layers, 2048 hidden,\n"
        "  a 4096-wide shared expert + eight 2048-wide routed experts, 128,000-token vocab)\n"
        "  land at roughly 3.6B parameters rather than 1.5B. Every number above is controlled\n"
        "  by BD3LMConfig, so trimming num_hidden_layers and/or the expert intermediate sizes\n"
        "  is all that's needed to hit a smaller target exactly — see the README."
    )
    del meta_scale_model

    # --- 2. Build a tiny, architecturally-identical debug model -------------
    # Same MQA ratio, same partial-RoPE slice, same shared+routed MoE
    # shape — just far smaller, so the checks below run in seconds on a
    # laptop CPU with no GPU required.
    print("\n[2] Building a small debug-scale model for the functional checks below")
    print("-" * 88)
    debug_config = BD3LMConfig(
        vocab_size=1_000,
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
    )
    model = BD3LMForCausalLM(debug_config)
    debug_total_params = sum(p.numel() for p in model.parameters())
    print(f"  Debug config total parameters: {format_param_count(debug_total_params)}")

    # --- 3. Mock TRAINING forward pass: random tokens + labels -> loss ------
    print("\n[3] Mock training forward pass (loss computation + backward)")
    print("-" * 88)
    torch.manual_seed(0)
    batch_size, sequence_length = 2, 24
    random_input_ids = torch.randint(low=0, high=debug_config.vocab_size, size=(batch_size, sequence_length))
    random_attention_mask = torch.ones_like(random_input_ids)
    random_labels = random_input_ids.clone()

    model.train()
    training_outputs = model(input_ids=random_input_ids, attention_mask=random_attention_mask, labels=random_labels)
    assert training_outputs.loss is not None, "Training forward pass did not return a loss."
    assert torch.isfinite(training_outputs.loss), f"Loss is not finite: {training_outputs.loss}"
    assert training_outputs.logits.shape == (batch_size, sequence_length, debug_config.vocab_size)

    training_outputs.loss.backward()
    shared_expert_grad = model.model.layers[0].moe.shared_expert.gate_proj.weight.grad
    router_grad = model.model.layers[0].moe.router.weight.grad
    assert shared_expert_grad is not None and torch.isfinite(shared_expert_grad).all(), "Shared-expert gradients are missing or non-finite."
    assert router_grad is not None and torch.isfinite(router_grad).all(), "Router gradients are missing or non-finite."

    print(f"  loss = {training_outputs.loss.item():.4f}")
    print(f"  logits shape = {tuple(training_outputs.logits.shape)}")
    print("  Gradients confirmed finite for the router and the shared expert (and, by extension, every routed expert this batch reached).")
    model.zero_grad()

    # --- 4. Mock autoregressive INFERENCE pass, exercising the KV cache -----
    print("\n[4] Mock autoregressive inference pass (.generate() with KV caching)")
    print("-" * 88)
    model.eval()
    prompt_ids = torch.randint(low=0, high=debug_config.vocab_size, size=(1, 6))
    num_new_tokens = 12

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=prompt_ids,
            attention_mask=torch.ones_like(prompt_ids),
            max_new_tokens=num_new_tokens,
            min_new_tokens=num_new_tokens,  # guarantees a fixed-length output for this smoke test
            do_sample=False,
            use_cache=True,
            pad_token_id=debug_config.pad_token_id,
        )
    assert generated_ids.shape == (1, prompt_ids.shape[1] + num_new_tokens)
    print(f"  prompt length = {prompt_ids.shape[1]} tokens -> generated length = {generated_ids.shape[1]} tokens")

    with torch.no_grad():
        prefill_outputs = model(input_ids=prompt_ids, use_cache=True)
    try:
        layer_zero_cached_keys = prefill_outputs.past_key_values.layers[0].keys
        expected_kv_heads = debug_config.num_key_value_heads
        print(
            f"  layer-0 cached key tensor shape = {tuple(layer_zero_cached_keys.shape)} "
            f"(dim 1 == num_key_value_heads == {expected_kv_heads}, confirming the MQA cache compression)"
        )
        assert layer_zero_cached_keys.shape[1] == expected_kv_heads
    except (AttributeError, IndexError):
        print("  (Skipping raw cache-shape inspection: this `transformers` version exposes `Cache` internals differently.)")

    print("\n" + "=" * 88)
    print("All BD3LM verification checks passed.")
    print("=" * 88)
