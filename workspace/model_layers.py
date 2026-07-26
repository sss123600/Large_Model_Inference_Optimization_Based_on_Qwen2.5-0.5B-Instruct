"""
Core building blocks for Qwen2.5-0.5B-Instruct.
RMSNorm / RoPE / SwiGLU-MLP / GQA-Attention / TransformerBlock / Qwen2Model / Qwen2ForCausalLM
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

if TYPE_CHECKING:
    from .kv_cache import KVCache, PagedKVCache, SeqKVPool


@dataclass
class ModelConfig:
    dim: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    num_layers: int
    intermediate_size: int
    vocab_size: int
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    max_position_embeddings: int = 32768


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xf = x.float()
        norm = torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        return (xf * norm).to(x.dtype) * self.weight


# ---------------------------------------------------------------------------
# Rotary Positional Embedding (RoPE)
# ---------------------------------------------------------------------------
class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int = 32768, theta: float = 1_000_000.0):
        super().__init__()
        self.head_dim = head_dim
        self.theta = theta
        self._build_cache(max_seq_len)

    def _build_cache(self, max_len: int):
        inv_freq = 1.0 / (
            self.theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        )
        pos = torch.arange(max_len, dtype=torch.float32)
        freqs = torch.outer(pos, inv_freq)  # [max_len, head_dim//2]
        emb = torch.cat([freqs, freqs], dim=-1)  # [max_len, head_dim]
        self.register_buffer("cos_cached", torch.cos(emb), persistent=False)
        self.register_buffer("sin_cached", torch.sin(emb), persistent=False)
        self.max_len = max_len

    def ensure_length(self, needed: int, device: torch.device):
        if needed <= self.max_len:
            return
        self._build_cache(needed + 1024)
        self.cos_cached = self.cos_cached.to(device=device)
        self.sin_cached = self.sin_cached.to(device=device)

    def apply_rotary(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cached[position_ids].unsqueeze(1).to(q.dtype)  # [B, 1, S, D]
        sin = self.sin_cached[position_ids].unsqueeze(1).to(q.dtype)
        q = (q * cos) + (self._rotate_half(q) * sin)
        k = (k * cos) + (self._rotate_half(k) * sin)
        return q, k

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)


# ---------------------------------------------------------------------------
# SwiGLU MLP
# ---------------------------------------------------------------------------
class SwiGLUMLP(nn.Module):
    def __init__(self, dim: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.up_proj = nn.Linear(dim, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# Grouped Query Attention (GQA)
# ---------------------------------------------------------------------------
class GQAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, num_kv_heads: int, head_dim: int):
        super().__init__()
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.kv_groups = num_heads // num_kv_heads

        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=True)
        self.k_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=True)
        self.v_proj = nn.Linear(dim, num_kv_heads * head_dim, bias=True)
        self.o_proj = nn.Linear(num_heads * head_dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        paged_cache: Optional[PagedKVCache] = None,
        block_tables: Optional[list[list[int]]] = None,
        context_lengths: Optional[torch.Tensor] = None,
        seq_pool: Optional[SeqKVPool] = None,
        slot_ids: Optional[list[int]] = None,
        ctx_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        B, S, _ = x.shape

        q = self.q_proj(x).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = rope.apply_rotary(q, k, position_ids)

        # --- SeqKVPool path (continuous batching) ---
        if seq_pool is not None and slot_ids is not None:
            if is_prefill:
                seq_pool.update_prefill(layer_idx, slot_ids, k, v, ctx_lens)
                k_for_attn = k
                v_for_attn = v
            else:
                seq_pool.update_decode(layer_idx, slot_ids, k, v, ctx_lens)
                # Gather full KV including the token just written
                new_ctx = [c + 1 for c in ctx_lens]
                k_for_attn, v_for_attn = seq_pool.gather(layer_idx, slot_ids, new_ctx)

            if self.kv_groups > 1:
                B_kv, H_kv, S_kv, D_kv = k_for_attn.shape
                k_exp = k_for_attn[:, :, None, :, :].expand(B_kv, H_kv, self.kv_groups, S_kv, D_kv).reshape(B_kv, -1, S_kv, D_kv)
                v_exp = v_for_attn[:, :, None, :, :].expand(B_kv, H_kv, self.kv_groups, S_kv, D_kv).reshape(B_kv, -1, S_kv, D_kv)
            else:
                k_exp, v_exp = k_for_attn, v_for_attn

            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=attn_mask, is_causal=(is_prefill and attn_mask is None),
            )

        # --- Paged KV Cache path ---
        elif block_tables is not None and paged_cache is not None:
            if is_prefill:
                paged_cache.update_batched(layer_idx, k, v, block_tables, context_lengths)
                k_for_attn = k
                v_for_attn = v
            else:
                paged_cache.update(layer_idx, k, v, block_tables, context_lengths)
                new_ctx = context_lengths + S
                k_for_attn, v_for_attn = paged_cache.gather_kv(layer_idx, block_tables, new_ctx)

            if self.kv_groups > 1:
                B_kv, H_kv, S_kv, D_kv = k_for_attn.shape
                k_exp = k_for_attn[:, :, None, :, :].expand(B_kv, H_kv, self.kv_groups, S_kv, D_kv).reshape(B_kv, -1, S_kv, D_kv)
                v_exp = v_for_attn[:, :, None, :, :].expand(B_kv, H_kv, self.kv_groups, S_kv, D_kv).reshape(B_kv, -1, S_kv, D_kv)
            else:
                k_exp, v_exp = k_for_attn, v_for_attn

            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=attn_mask, is_causal=(is_prefill and attn_mask is None),
            )

        # --- Legacy contiguous KV Cache path (fallback) ---
        else:
            if kv_cache is not None:
                k, v = kv_cache.update(layer_idx, k, v)

            if self.kv_groups > 1:
                B_kv, H_kv, S_kv, D_kv = k.shape
                k_exp = k[:, :, None, :, :].expand(B_kv, H_kv, self.kv_groups, S_kv, D_kv).reshape(B_kv, -1, S_kv, D_kv)
                v_exp = v[:, :, None, :, :].expand(B_kv, H_kv, self.kv_groups, S_kv, D_kv).reshape(B_kv, -1, S_kv, D_kv)
            else:
                k_exp, v_exp = k, v

            attn_out = F.scaled_dot_product_attention(
                q, k_exp, v_exp, attn_mask=attn_mask, is_causal=(is_prefill and attn_mask is None),
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, -1)
        return self.o_proj(attn_out)


# ---------------------------------------------------------------------------
# Transformer Block
# ---------------------------------------------------------------------------
class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.input_layernorm = RMSNorm(config.dim, config.rms_norm_eps)
        self.self_attn = GQAttention(
            config.dim, config.num_heads, config.num_kv_heads, config.head_dim,
        )
        self.post_attention_layernorm = RMSNorm(config.dim, config.rms_norm_eps)
        self.mlp = SwiGLUMLP(config.dim, config.intermediate_size)

    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor,
        rope: RotaryEmbedding,
        kv_cache: Optional[KVCache] = None,
        layer_idx: int = 0,
        attn_mask: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        paged_cache: Optional[PagedKVCache] = None,
        block_tables: Optional[list[list[int]]] = None,
        context_lengths: Optional[torch.Tensor] = None,
        seq_pool: Optional[SeqKVPool] = None,
        slot_ids: Optional[list[int]] = None,
        ctx_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        normed = self.input_layernorm(x)
        attn_out = self.self_attn(
            normed, position_ids, rope, kv_cache, layer_idx, attn_mask, is_prefill,
            paged_cache=paged_cache, block_tables=block_tables, context_lengths=context_lengths,
            seq_pool=seq_pool, slot_ids=slot_ids, ctx_lens=ctx_lens,
        )
        x = x + attn_out
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# ---------------------------------------------------------------------------
# Qwen2 Model (embedding + layers + final norm)
# ---------------------------------------------------------------------------
class Qwen2Model(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(
            [TransformerBlock(config) for _ in range(config.num_layers)]
        )
        self.norm = RMSNorm(config.dim, config.rms_norm_eps)
        self.rope = RotaryEmbedding(config.head_dim, config.max_position_embeddings, config.rope_theta)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        paged_cache: Optional[PagedKVCache] = None,
        block_tables: Optional[list[list[int]]] = None,
        context_lengths: Optional[torch.Tensor] = None,
        seq_pool: Optional[SeqKVPool] = None,
        slot_ids: Optional[list[int]] = None,
        ctx_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        x = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            x = layer(
                x, position_ids, self.rope, kv_cache, i, attn_mask, is_prefill,
                paged_cache=paged_cache, block_tables=block_tables, context_lengths=context_lengths,
                seq_pool=seq_pool, slot_ids=slot_ids, ctx_lens=ctx_lens,
            )
        if kv_cache is not None:
            kv_cache.advance(input_ids.shape[1])
        return x


# ---------------------------------------------------------------------------
# Qwen2 For Causal LM (model + lm_head)
# ---------------------------------------------------------------------------
class Qwen2ForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.model = Qwen2Model(config)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
        attn_mask: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        paged_cache: Optional[PagedKVCache] = None,
        block_tables: Optional[list[list[int]]] = None,
        context_lengths: Optional[torch.Tensor] = None,
        seq_pool: Optional[SeqKVPool] = None,
        slot_ids: Optional[list[int]] = None,
        ctx_lens: Optional[list[int]] = None,
    ) -> torch.Tensor:
        """Returns logits for the LAST token: [batch, vocab_size]."""
        x = self.model(
            input_ids, position_ids, kv_cache, attn_mask, is_prefill,
            paged_cache=paged_cache, block_tables=block_tables, context_lengths=context_lengths,
            seq_pool=seq_pool, slot_ids=slot_ids, ctx_lens=ctx_lens,
        )
        last = x[:, -1, :].float()
        norm_w = self.model.norm.weight.float()
        h = last * torch.rsqrt(last.pow(2).mean(-1, keepdim=True) + self.model.norm.eps)
        h = (h * norm_w).to(x.dtype)
        return F.linear(h, self.lm_head.weight)

    def tie_weights(self):
        self.lm_head.weight = self.model.embed_tokens.weight
