"""
KV Cache implementations:
1. KVCache — Pre-allocated contiguous cache for fixed-batch generation.
2. PagedKVCache — Block-level paged cache with memory pooling.
3. SeqKVPool — Lightweight per-sequence contiguous buffers for continuous batching
               (avoids gather overhead while enabling dynamic scheduling).
"""

from __future__ import annotations

from typing import Optional

import torch


# ===========================================================================
# SeqKVPool: per-sequence contiguous cache for continuous batching
# ===========================================================================

class SeqKVPool:
    """Per-sequence KV buffers backed by a single pre-allocated pool.

    Each sequence gets a contiguous slice. Write is a simple memcpy; read is
    a direct slice — no scatter/gather. This trades memory flexibility for
    speed, which is the right trade-off for a small model on a single GPU.
    """

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        max_seqs: int,
        max_seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.max_seqs = max_seqs
        self.max_seq_len = max_seq_len
        self.device = device
        self.dtype = dtype

        # Pool shape: [num_layers, max_seqs, num_kv_heads, max_seq_len, head_dim]
        self.k_pool = torch.zeros(
            (num_layers, max_seqs, num_kv_heads, max_seq_len, head_dim),
            device=device, dtype=dtype,
        )
        self.v_pool = torch.zeros(
            (num_layers, max_seqs, num_kv_heads, max_seq_len, head_dim),
            device=device, dtype=dtype,
        )

        # Slot management
        self._free_slots: list[int] = list(range(max_seqs - 1, -1, -1))

    @property
    def num_free_slots(self) -> int:
        return len(self._free_slots)

    def allocate(self) -> int:
        """Allocate one slot. Returns slot index."""
        if not self._free_slots:
            raise RuntimeError("SeqKVPool: no free slots")
        return self._free_slots.pop()

    def free(self, slot: int) -> None:
        """Release a slot back to the pool."""
        self._free_slots.append(slot)

    def update_prefill(
        self,
        layer_idx: int,
        slot_ids: list[int],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        context_lengths: list[int],
    ) -> None:
        """Write prefill K/V for a batch of sequences.

        k_new/v_new: [batch, num_kv_heads, seq_len, head_dim]
        Each sequence in the batch may have different start offsets (context_lengths).
        For fresh prefill, context_lengths should be [0, 0, ...].
        """
        for i, slot in enumerate(slot_ids):
            ctx = context_lengths[i]
            seq_len = k_new.shape[2]
            end = ctx + seq_len
            self.k_pool[layer_idx, slot, :, ctx:end, :] = k_new[i]
            self.v_pool[layer_idx, slot, :, ctx:end, :] = v_new[i]

    def update_decode(
        self,
        layer_idx: int,
        slot_ids: list[int],
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        positions: list[int],
    ) -> None:
        """Write one decode token per sequence.

        k_new/v_new: [batch, num_kv_heads, 1, head_dim]
        positions[i] = where to write in that slot's sequence dimension.
        """
        if len(slot_ids) == 1:
            # Fast path: single sequence, direct write
            self.k_pool[layer_idx, slot_ids[0], :, positions[0], :] = k_new[0, :, 0, :]
            self.v_pool[layer_idx, slot_ids[0], :, positions[0], :] = v_new[0, :, 0, :]
        else:
            # Vectorized scatter using advanced indexing
            slot_t = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
            pos_t = torch.tensor(positions, device=self.device, dtype=torch.long)
            # k_pool[layer_idx] shape: [max_seqs, num_kv_heads, max_seq_len, head_dim]
            self.k_pool[layer_idx, slot_t, :, pos_t, :] = k_new[:, :, 0, :]
            self.v_pool[layer_idx, slot_t, :, pos_t, :] = v_new[:, :, 0, :]

    def gather(
        self,
        layer_idx: int,
        slot_ids: list[int],
        context_lengths: list[int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Gather KV for attention. Returns [batch, heads, max_ctx, dim].

        Fast path: single sequence → direct slice (zero-copy view).
        Batch path: index_select + truncate to max_ctx.
        """
        max_ctx = max(context_lengths)

        if len(slot_ids) == 1:
            # Zero-copy: direct view into pool, no allocation
            slot = slot_ids[0]
            k = self.k_pool[layer_idx, slot:slot+1, :, :max_ctx, :]
            v = self.v_pool[layer_idx, slot:slot+1, :, :max_ctx, :]
            return k, v

        # Batch: use persistent slot tensor to avoid repeated allocation
        slot_tensor = torch.tensor(slot_ids, device=self.device, dtype=torch.long)
        k = self.k_pool[layer_idx].index_select(0, slot_tensor)[:, :, :max_ctx, :]
        v = self.v_pool[layer_idx].index_select(0, slot_tensor)[:, :, :max_ctx, :]
        return k.contiguous(), v.contiguous()

    def reset(self) -> None:
        """Wipe all slots."""
        self.k_pool.zero_()
        self.v_pool.zero_()
        self._free_slots = list(range(self.max_seqs - 1, -1, -1))


# ===========================================================================
# PagedKVCache: Block-level paged cache with memory pooling
# ===========================================================================

class PagedKVCache:
    """Block-level paged KV cache manager (simplified PagedAttention)."""

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.device = device
        self.dtype = dtype

        self.k_pool = torch.zeros(
            (num_layers, num_blocks, num_kv_heads, block_size, head_dim),
            device=device, dtype=dtype,
        )
        self.v_pool = torch.zeros(
            (num_layers, num_blocks, num_kv_heads, block_size, head_dim),
            device=device, dtype=dtype,
        )

        self._free_blocks: list[int] = list(range(num_blocks - 1, -1, -1))

    @property
    def num_free_blocks(self) -> int:
        return len(self._free_blocks)

    def _alloc_block(self) -> int:
        if not self._free_blocks:
            raise RuntimeError("PagedKVCache: out of free blocks")
        return self._free_blocks.pop()

    def allocate_seq(self, num_tokens: int = 0) -> list[int]:
        needed = max(1, (num_tokens + self.block_size - 1) // self.block_size)
        if needed > self.num_free_blocks:
            raise RuntimeError(
                f"PagedKVCache: need {needed} blocks but only {self.num_free_blocks} free"
            )
        return [self._alloc_block() for _ in range(needed)]

    def free_seq(self, block_table: list[int]) -> None:
        for blk_idx in block_table:
            self.k_pool[:, blk_idx].zero_()
            self.v_pool[:, blk_idx].zero_()
        self._free_blocks.extend(block_table)

    def ensure_capacity(self, block_table: list[int], context_len: int, new_tokens: int) -> list[int]:
        total_needed = context_len + new_tokens
        blocks_needed = (total_needed + self.block_size - 1) // self.block_size
        while len(block_table) < blocks_needed:
            block_table.append(self._alloc_block())
        return block_table

    def update(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        block_tables: list[list[int]],
        context_lengths: torch.Tensor,
    ) -> None:
        batch_size = k_new.shape[0]
        new_tokens = k_new.shape[2]

        for b in range(batch_size):
            ctx_len = int(context_lengths[b].item()) if isinstance(context_lengths, torch.Tensor) else int(context_lengths[b])
            bt = block_tables[b]

            for t in range(new_tokens):
                global_pos = ctx_len + t
                block_idx_in_table = global_pos // self.block_size
                offset_in_block = global_pos % self.block_size
                phys_block = bt[block_idx_in_table]

                self.k_pool[layer_idx, phys_block, :, offset_in_block, :] = k_new[b, :, t, :]
                self.v_pool[layer_idx, phys_block, :, offset_in_block, :] = v_new[b, :, t, :]

    def update_batched(
        self,
        layer_idx: int,
        k_new: torch.Tensor,
        v_new: torch.Tensor,
        block_tables: list[list[int]],
        context_lengths: torch.Tensor,
    ) -> None:
        batch_size = k_new.shape[0]
        new_tokens = k_new.shape[2]

        for b in range(batch_size):
            ctx_len = int(context_lengths[b].item()) if isinstance(context_lengths, torch.Tensor) else int(context_lengths[b])
            bt = block_tables[b]

            written = 0
            while written < new_tokens:
                global_pos = ctx_len + written
                block_idx_in_table = global_pos // self.block_size
                offset_in_block = global_pos % self.block_size
                phys_block = bt[block_idx_in_table]

                space_in_block = self.block_size - offset_in_block
                chunk = min(space_in_block, new_tokens - written)

                end_offset = offset_in_block + chunk
                self.k_pool[layer_idx, phys_block, :, offset_in_block:end_offset, :] = k_new[b, :, written:written + chunk, :]
                self.v_pool[layer_idx, phys_block, :, offset_in_block:end_offset, :] = v_new[b, :, written:written + chunk, :]

                written += chunk

    def gather_kv(
        self,
        layer_idx: int,
        block_tables: list[list[int]],
        context_lengths: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = len(block_tables)
        if isinstance(context_lengths, torch.Tensor):
            max_ctx = int(context_lengths.max().item())
        else:
            max_ctx = max(context_lengths)

        k_out = torch.zeros(
            (batch_size, self.num_kv_heads, max_ctx, self.head_dim),
            device=self.device, dtype=self.dtype,
        )
        v_out = torch.zeros(
            (batch_size, self.num_kv_heads, max_ctx, self.head_dim),
            device=self.device, dtype=self.dtype,
        )

        for b in range(batch_size):
            ctx_len = int(context_lengths[b].item()) if isinstance(context_lengths, torch.Tensor) else int(context_lengths[b])
            bt = block_tables[b]

            copied = 0
            block_i = 0
            while copied < ctx_len:
                phys_block = bt[block_i]
                chunk = min(self.block_size, ctx_len - copied)
                k_out[b, :, copied:copied + chunk, :] = self.k_pool[layer_idx, phys_block, :, :chunk, :]
                v_out[b, :, copied:copied + chunk, :] = self.v_pool[layer_idx, phys_block, :, :chunk, :]
                copied += chunk
                block_i += 1

        return k_out, v_out

    def reset(self) -> None:
        self.k_pool.zero_()
        self.v_pool.zero_()
        self._free_blocks = list(range(self.num_blocks - 1, -1, -1))


# ===========================================================================
# Legacy KVCache (kept for backward compatibility with current engine)
# ===========================================================================

class KVCache:
    """Pre-allocated contiguous KV cache (original implementation)."""

    __slots__ = ("k", "v", "seq_len")

    def __init__(
        self,
        num_layers: int,
        batch_size: int,
        max_total_len: int,
        num_kv_heads: int,
        head_dim: int,
        device: torch.device,
        dtype: torch.dtype,
    ):
        shape = (num_layers, batch_size, num_kv_heads, max_total_len, head_dim)
        self.k = torch.zeros(shape, device=device, dtype=dtype)
        self.v = torch.zeros(shape, device=device, dtype=dtype)
        self.seq_len = 0

    def update(
        self, layer_idx: int, k_new: torch.Tensor, v_new: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Write new K/V into the buffer and return the filled slice."""
        new_len = k_new.shape[2]
        start = self.seq_len
        end = start + new_len
        self.k[layer_idx, :, :, start:end, :] = k_new
        self.v[layer_idx, :, :, start:end, :] = v_new
        return (
            self.k[layer_idx, :, :, :end, :],
            self.v[layer_idx, :, :, :end, :],
        )

    def advance(self, n: int):
        """Call once per step after all layers have written."""
        self.seq_len += n
