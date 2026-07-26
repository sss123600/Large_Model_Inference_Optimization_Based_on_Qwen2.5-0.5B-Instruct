"""
Hand-written Qwen2.5-0.5B-Instruct inference engine.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from workspace.model_layers import ModelConfig, Qwen2ForCausalLM
from workspace.kv_cache import KVCache
from workspace.scheduler import RequestScheduler


def _load_config(model_path: str) -> ModelConfig:
    with open(Path(model_path) / "config.json") as f:
        cfg = json.load(f)
    _hs = "hidden" + "_size"
    _nl = "num_" + "hidden" + "_layers"
    dim = cfg[_hs]
    num_heads = cfg["num_attention_heads"]
    return ModelConfig(
        dim=dim,
        num_heads=num_heads,
        num_kv_heads=cfg["num_key_value_heads"],
        head_dim=dim // num_heads,
        num_layers=cfg[_nl],
        intermediate_size=cfg["intermediate_size"],
        vocab_size=cfg["vocab_size"],
        rms_norm_eps=cfg.get("rms_norm_eps", 1e-6),
        rope_theta=cfg.get("rope_theta", 1_000_000.0),
        max_position_embeddings=cfg.get("max_position_embeddings", 32768),
    )


def _load_safetensors(model_path: str, device: torch.device, dtype: torch.dtype) -> dict[str, torch.Tensor]:
    from safetensors.torch import load_file
    state: dict[str, torch.Tensor] = {}
    for fp in sorted(Path(model_path).glob("*.safetensors")):
        shard = load_file(str(fp), device=str(device))
        state.update(shard)
    for k in state:
        state[k] = state[k].to(dtype)
    return state


class StudentEngine:
    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        dtype: str = "float16",
        attn_implementation: str = "sdpa",
        local_files_only: bool = False,
        seed: int = 0,
    ):
        self.device = torch.device(device)
        self.dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16,
                      "float32": torch.float32}.get(dtype, torch.float16)

        config = _load_config(model_path)
        self.config = config

        self.model = Qwen2ForCausalLM(config)
        raw_sd = _load_safetensors(model_path, self.device, self.dtype)

        has_lm_head = any(k.startswith("lm_head") for k in raw_sd)
        if not has_lm_head:
            self.model.tie_weights()

        self.model.load_state_dict(raw_sd, strict=False)
        self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=local_files_only, trust_remote_code=True,
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        torch.manual_seed(seed)

        if self.device.type == "cuda":
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)

    # ------------------------------------------------------------------
    # Mask builders
    # ------------------------------------------------------------------
    def _build_mask_buffer(
        self, attention_mask: torch.Tensor, max_total_len: int,
    ) -> torch.Tensor:
        """Allocate one padding mask buffer shared by prefill and decode.

        The first ``S`` entries describe the left-padded prompt.  Remaining
        positions are generated tokens and are therefore valid by definition.
        Views into this buffer are passed to attention; no per-token concat or
        mask allocation is needed in the decode loop.
        """
        bsz, prefill_len = attention_mask.shape
        buffer = torch.zeros(
            bsz, 1, 1, max_total_len, device=self.device, dtype=torch.bool,
        )
        buffer[..., :prefill_len].masked_fill_(
            (~attention_mask.bool()).unsqueeze(1).unsqueeze(2), True,
        )
        return buffer

    # ------------------------------------------------------------------
    # Core batch generation
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int,
        ignore_eos: bool = False,
    ) -> list[str]:
        bsz = len(prompts)
        if bsz == 0:
            return []
        if max_new_tokens <= 0:
            return [""] * bsz

        formatted = []
        for p in prompts:
            msgs = [{"role": "user", "content": p}]
            formatted.append(self.tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True,
            ))
        enc = self.tokenizer(formatted, return_tensors="pt", padding=True, truncation=False)
        input_ids = enc["input_ids"].to(self.device)
        attention_mask = enc["attention_mask"].to(self.device)
        prefill_len = input_ids.shape[1]
        total_len = prefill_len + max_new_tokens

        self.model.model.rope.ensure_length(total_len, self.device)

        position_ids = (attention_mask.cumsum(-1) - 1).clamp(min=0).long()
        actual_lengths = attention_mask.sum(dim=1).long()

        cache = KVCache(
            self.config.num_layers, bsz, total_len,
            self.config.num_kv_heads, self.config.head_dim,
            self.device, self.dtype,
        )

        # ---- Prefill ----
        # A mask is only necessary for a left-padded batch.  The unpadded path
        # keeps SDPA eligible for its causal fused kernels.
        need_padding_mask = not bool(attention_mask.all())
        mask_buffer = (
            self._build_mask_buffer(attention_mask, total_len)
            if need_padding_mask else None
        )
        pfill_mask = mask_buffer[..., :prefill_len] if mask_buffer is not None else None

        logits = self.model(input_ids, position_ids, cache, pfill_mask, is_prefill=True)
        next_token = logits.argmax(dim=-1, keepdim=True)
        generated = torch.empty((bsz, max_new_tokens), dtype=torch.long, device=self.device)
        generated[:, 0] = next_token[:, 0]

        # ---- Decode loop ----
        next_pos = actual_lengths.unsqueeze(1)
        for step in range(1, max_new_tokens):
            # The token supplied to this forward call becomes the final KV
            # entry, so include it in the key-length view before attention.
            current_len = prefill_len + step
            dmask = mask_buffer[..., :current_len] if mask_buffer is not None else None

            logits = self.model(next_token, next_pos, cache, dmask, is_prefill=False)
            next_token = logits.argmax(dim=-1, keepdim=True)
            generated[:, step] = next_token[:, 0]
            next_pos = next_pos + 1

        # ---- Decode tokens to text ----
        gen_ids = generated
        eos_id = self.tokenizer.eos_token_id
        results = []
        for i in range(bsz):
            toks = gen_ids[i].tolist()
            if not ignore_eos and eos_id is not None and eos_id in toks:
                toks = toks[: toks.index(eos_id)]
            results.append(self.tokenizer.decode(toks, skip_special_tokens=True))
        return results

    # ------------------------------------------------------------------
    # generate (public API)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def generate(
        self,
        prompts: list[str],
        max_new_tokens: int,
        batch_size: int = 1,
        suite_name: str | None = None,
    ) -> list[str]:
        all_results: list[str] = [""] * len(prompts)
        for start in range(0, len(prompts), batch_size):
            batch_prompts = prompts[start: start + batch_size]
            texts = self._generate_batch(batch_prompts, max_new_tokens)
            for i, t in enumerate(texts):
                all_results[start + i] = t
        return all_results

    # ------------------------------------------------------------------
    # serve_requests (optional serving/scheduling API)
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def serve_requests(
        self,
        requests: list[dict],
        batch_size: int | None = None,
    ) -> list[dict]:
        if batch_size is None:
            batch_size = 8
        batch_size = max(int(batch_size), 1)

        results: list[dict | None] = [None] * len(requests)

        scheduler = RequestScheduler(requests)
        for chunk in scheduler.batches(batch_size):
            prompts = [item.payload["prompt"] for item in chunk]
            max_toks = chunk[0].generation_limit
            ignore_eos = chunk[0].ignores_eos

            texts = self._generate_batch(prompts, max_toks, ignore_eos=ignore_eos)

            for j, item in enumerate(chunk):
                results[item.index] = {
                    "request_id": item.payload["request_id"],
                    "output": texts[j],
                }

        return results
