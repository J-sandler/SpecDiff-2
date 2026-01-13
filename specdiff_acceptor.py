#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import math
import os
import sys
import time
import types
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, StaticCache

try:
    from diffucoder_semi_ar import _fuse_dream_mlps as _phase1_fuse_mlps
    from diffucoder_semi_ar import _fuse_qkv_projections as _phase1_fuse_qkv
except Exception:
    _phase1_fuse_mlps = None
    _phase1_fuse_qkv = None


def _get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _supports_num_logits_to_keep(model) -> bool:
    try:
        return "num_logits_to_keep" in model.forward.__code__.co_varnames
    except Exception:
        return False


def _build_sdp_context(backend: str, *, verbose: bool) -> contextlib.AbstractContextManager:
    backend = (backend or "auto").lower()
    if not torch.cuda.is_available():
        return contextlib.nullcontext()

    enable_flash = True
    enable_mem = True
    enable_math = True
    if backend != "auto":
        enable_flash = backend == "flash"
        enable_mem = backend in ("mem", "mem_efficient", "memory", "mem-efficient")
        enable_math = backend == "math"

    ctx = contextlib.nullcontext()
    used_ctx = False
    sdp_kernel = None
    if hasattr(torch.nn, "attention") and hasattr(torch.nn.attention, "sdpa_kernel"):
        sdp_kernel = torch.nn.attention.sdpa_kernel
    elif hasattr(torch.backends.cuda, "sdp_kernel"):
        sdp_kernel = torch.backends.cuda.sdp_kernel
    if sdp_kernel is not None:
        try:
            ctx = sdp_kernel(
                enable_flash=enable_flash,
                enable_mem_efficient=enable_mem,
                enable_math=enable_math,
            )
            used_ctx = True
        except Exception:
            ctx = contextlib.nullcontext()

    if not used_ctx:
        if hasattr(torch.backends.cuda, "enable_flash_sdp"):
            try:
                torch.backends.cuda.enable_flash_sdp(enable_flash)
            except Exception:
                pass
        if hasattr(torch.backends.cuda, "enable_mem_efficient_sdp"):
            try:
                torch.backends.cuda.enable_mem_efficient_sdp(enable_mem)
            except Exception:
                pass
        if hasattr(torch.backends.cuda, "enable_math_sdp"):
            try:
                torch.backends.cuda.enable_math_sdp(enable_math)
            except Exception:
                pass

    if verbose:
        parts = [f"backend={backend}"]
        for name in ("flash", "mem_efficient", "math"):
            fn = getattr(torch.backends.cuda, f"{name}_sdp_enabled", None)
            if callable(fn):
                try:
                    parts.append(f"{name}={fn()}")
                except Exception:
                    continue
        avail_fn = getattr(torch.backends.cuda, "is_flash_sdp_available", None)
        if callable(avail_fn):
            try:
                parts.append(f"flash_available={avail_fn()}")
            except Exception:
                pass
        print("[SDPA] " + " ".join(parts))

    return ctx


def _tokenizers_compatible(diff_tokenizer, verify_tokenizer) -> Tuple[bool, str]:
    vocab_a = diff_tokenizer.get_vocab()
    vocab_b = verify_tokenizer.get_vocab()
    tokens_a = set(vocab_a.keys())
    tokens_b = set(vocab_b.keys())

    special_a = set(getattr(diff_tokenizer, "all_special_tokens", []) or [])
    special_b = set(getattr(verify_tokenizer, "all_special_tokens", []) or [])
    added_a = set(getattr(diff_tokenizer, "get_added_vocab", lambda: {})().keys())
    added_b = set(getattr(verify_tokenizer, "get_added_vocab", lambda: {})().keys())
    special = special_a | special_b | added_a | added_b

    shared = tokens_a & tokens_b
    mismatched = [tok for tok in shared if vocab_a[tok] != vocab_b[tok]]
    bad_mismatch = [tok for tok in mismatched if tok not in special]

    only_a = [tok for tok in tokens_a - tokens_b if tok not in special]
    only_b = [tok for tok in tokens_b - tokens_a if tok not in special]

    compatible = not bad_mismatch and not only_a and not only_b
    overlap = len(shared) / max(1, min(len(tokens_a), len(tokens_b)))
    msg = (
        f"shared={len(shared)} overlap={overlap:.5f} "
        f"mismatch={len(mismatched)} bad_mismatch={len(bad_mismatch)} "
        f"only_diff={len(only_a)} only_verify={len(only_b)}"
    )
    return compatible, msg


def _special_token_mismatch(diff_tokenizer, verify_tokenizer) -> List[str]:
    mismatches: List[str] = []
    for name in ("bos_token", "eos_token", "pad_token", "cls_token", "sep_token"):
        if getattr(diff_tokenizer, name, None) != getattr(verify_tokenizer, name, None):
            mismatches.append(name)
    return mismatches


def _build_token_id_maps(
    diff_tokenizer,
    verify_tokenizer,
    diff_size: Optional[int] = None,
    verify_size: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
    diff_size = int(diff_size) if diff_size is not None else len(diff_tokenizer)
    verify_size = int(verify_size) if verify_size is not None else len(verify_tokenizer)
    diff_to_verify = torch.full((diff_size,), -1, dtype=torch.long)
    verify_to_diff = torch.full((verify_size,), -1, dtype=torch.long)
    verify_token_to_ids: Dict[str, List[int]] = {}
    for vid in range(verify_size):
        tok = verify_tokenizer.convert_ids_to_tokens(vid)
        verify_token_to_ids.setdefault(tok, []).append(vid)
    diff_token_to_ids: Dict[str, List[int]] = {}
    for did in range(diff_size):
        tok = diff_tokenizer.convert_ids_to_tokens(did)
        diff_token_to_ids.setdefault(tok, []).append(did)
    for did in range(diff_size):
        tok = diff_tokenizer.convert_ids_to_tokens(did)
        ids = verify_token_to_ids.get(tok)
        if ids:
            diff_to_verify[did] = int(ids[0])
    for vid in range(verify_size):
        tok = verify_tokenizer.convert_ids_to_tokens(vid)
        ids = diff_token_to_ids.get(tok)
        if ids:
            verify_to_diff[vid] = int(ids[0])
    missing_diff = int((diff_to_verify < 0).sum().item())
    missing_verify = int((verify_to_diff < 0).sum().item())
    return diff_to_verify, verify_to_diff, missing_diff, missing_verify


def _maybe_add_mask_token(model, tokenizer, mask_token: str = "[MASK]") -> int:
    if getattr(tokenizer, "mask_token_id", None) is not None:
        return int(tokenizer.mask_token_id)
    if mask_token in tokenizer.get_vocab():
        return int(tokenizer.convert_tokens_to_ids(mask_token))
    added = tokenizer.add_special_tokens({"additional_special_tokens": [mask_token]})
    if added <= 0:
        raise RuntimeError("Failed to add mask token to tokenizer.")
    model.resize_token_embeddings(len(tokenizer))
    mask_id = int(tokenizer.convert_tokens_to_ids(mask_token))
    with torch.no_grad():
        emb = model.get_input_embeddings().weight
        mean_vec = emb[:-1].mean(dim=0)
        emb[mask_id].copy_(mean_vec)
        out_emb = model.get_output_embeddings()
        if out_emb is not None and out_emb.weight.shape[0] == emb.shape[0]:
            out_emb.weight[mask_id].copy_(mean_vec.to(dtype=out_emb.weight.dtype))
    return mask_id


def _filter_logits(logits: torch.Tensor, top_k: int, top_p: float) -> torch.Tensor:
    if top_k <= 0 and top_p >= 1.0:
        return logits
    logits = logits.clone()
    if top_k > 0:
        k = min(int(top_k), logits.shape[-1])
        values, _ = torch.topk(logits, k=k, dim=-1)
        cutoff = values[..., -1, None]
        logits = logits.masked_fill(logits < cutoff, float("-inf"))
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = cumprobs > float(top_p)
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.zeros_like(logits).scatter(-1, sorted_idx, sorted_logits)
    return logits


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1)
    logits = logits / float(temperature)
    logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _sample_from_probs(
    probs: torch.Tensor,
    *,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    total = probs.sum()
    if total <= 0:
        probs = torch.full_like(probs, 1.0 / probs.numel())
    else:
        probs = probs / total
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)


def _sample_k_sequences(
    logits: torch.Tensor,
    k: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    # logits: (1, gamma, vocab)
    logits = logits.squeeze(0)
    if temperature <= 0:
        greedy = torch.argmax(logits, dim=-1)
        return greedy.unsqueeze(0).repeat(int(k), 1)
    logits = logits / float(temperature)
    logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
    # sample K per position -> (gamma, K)
    picks = torch.multinomial(probs, num_samples=int(k), replacement=True, generator=generator)
    return picks.transpose(0, 1).contiguous()


def _draft_probs_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    if logits.dim() == 3:
        logits = logits.squeeze(0)
    if temperature <= 0:
        argmax = torch.argmax(logits, dim=-1, keepdim=True)
        probs = torch.zeros_like(logits, dtype=torch.float32)
        probs.scatter_(-1, argmax, 1.0)
        return probs
    logits = logits / float(temperature)
    logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
    return torch.softmax(logits.to(dtype=torch.float32), dim=-1)


class FrozenCache(Cache):
    def __init__(self, base_cache, prefix_len: Optional[int] = None):
        super().__init__()
        self.base_cache = base_cache
        self.prefix_len = prefix_len

    def __len__(self):
        try:
            return len(self.base_cache)
        except Exception:
            return 0

    def __getitem__(self, layer_idx: int):
        return self.base_cache[layer_idx]

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield self.base_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if self.prefix_len is not None:
            return int(self.prefix_len)
        if self.base_cache is None:
            return 0
        if hasattr(self.base_cache, "get_seq_length"):
            return self.base_cache.get_seq_length(layer_idx)
        return 0

    def get_max_cache_shape(self) -> Optional[int]:
        if self.base_cache is None:
            return None
        if hasattr(self.base_cache, "get_max_cache_shape"):
            return self.base_cache.get_max_cache_shape()
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        base = self.base_cache
        if base is None:
            return key_states, value_states
        if hasattr(base, "key_cache") and hasattr(base, "value_cache"):
            if len(base.key_cache) <= layer_idx:
                return key_states, value_states
            k = base.key_cache[layer_idx]
            v = base.value_cache[layer_idx]
            if isinstance(k, list) or isinstance(v, list) or len(k) == 0:
                return key_states, value_states
            prefix_len = self.prefix_len
            if prefix_len is None and hasattr(base, "get_seq_length"):
                prefix_len = base.get_seq_length(layer_idx)
            if prefix_len is not None and k.shape[-2] > prefix_len:
                k = k[..., :prefix_len, :]
                v = v[..., :prefix_len, :]
            return torch.cat([k, key_states], dim=-2), torch.cat([v, value_states], dim=-2)
        return key_states, value_states


class PreallocCache(Cache):
    def __init__(
        self,
        config,
        batch_size: int,
        max_cache_len: int,
        device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.max_cache_len = max_cache_len
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        for _ in range(config.num_hidden_layers):
            k = torch.zeros(
                (batch_size, self.num_key_value_heads, max_cache_len, self.head_dim),
                device=device,
                dtype=dtype,
            )
            v = torch.zeros(
                (batch_size, self.num_key_value_heads, max_cache_len, self.head_dim),
                device=device,
                dtype=dtype,
            )
            self.key_cache.append(k)
            self.value_cache.append(v)
        self.current_length = 0

    def __len__(self):
        return len(self.key_cache)

    def __getitem__(self, layer_idx: int):
        return (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.key_cache[layer_idx], self.value_cache[layer_idx])

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        return int(self.current_length)

    def get_max_cache_shape(self) -> Optional[int]:
        return None

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        cache_position = None
        if cache_kwargs is not None:
            cache_position = cache_kwargs.get("cache_position")
        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        if cache_position is None:
            pos = self.current_length
            k_out[:, :, pos : pos + key_states.shape[-2], :] = key_states
            v_out[:, :, pos : pos + value_states.shape[-2], :] = value_states
            self.current_length += key_states.shape[-2]
        else:
            k_out.index_copy_(2, cache_position, key_states)
            v_out.index_copy_(2, cache_position, value_states)
            self.current_length = max(self.current_length, int(cache_position.max().item()) + 1)
        return k_out[:, :, : self.current_length, :], v_out[:, :, : self.current_length, :]


def _cache_key_value(cache, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    if cache is None:
        return None
    if isinstance(cache, FrozenCache):
        cache = cache.base_cache
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        try:
            return cache.key_cache[layer_idx], cache.value_cache[layer_idx]
        except Exception:
            return None
    return None


def _truncate_cache(cache, new_len: int) -> None:
    if cache is None:
        return
    if isinstance(cache, FrozenCache):
        cache = cache.base_cache
    if hasattr(cache, "current_length"):
        try:
            cache.current_length = int(new_len)
        except Exception:
            return
    elif hasattr(cache, "cache_length"):
        try:
            cache.cache_length = int(new_len)
        except Exception:
            return


def _reuse_tree_cache_for_accepted(
    *,
    verify_cache,
    tree_cache,
    prefix_len: int,
    accepted_nodes: Sequence[int],
) -> Optional[Cache]:
    if verify_cache is None or tree_cache is None:
        return verify_cache
    if not accepted_nodes:
        return verify_cache
    try:
        num_layers = len(verify_cache)
    except Exception:
        return verify_cache
    # Handle in-place reuse (tree_cache == verify_cache) safely.
    same_cache = verify_cache is tree_cache
    gather_idx = None
    if same_cache:
        cache0 = _cache_key_value(tree_cache, 0)
        if cache0 is not None:
            gather_idx = torch.tensor(
                [int(prefix_len + node) for node in accepted_nodes],
                device=cache0[0].device,
                dtype=torch.long,
            )
        else:
            same_cache = False
    for layer_idx in range(num_layers):
        dst = _cache_key_value(verify_cache, layer_idx)
        src = _cache_key_value(tree_cache, layer_idx)
        if dst is None or src is None:
            continue
        k_dst, v_dst = dst
        k_src, v_src = src
        if same_cache and gather_idx is not None:
            k_tmp = k_src.index_select(2, gather_idx).clone()
            v_tmp = v_src.index_select(2, gather_idx).clone()
            for out_idx in range(len(accepted_nodes)):
                dst_pos = int(prefix_len + out_idx)
                k_dst[:, :, dst_pos : dst_pos + 1, :] = k_tmp[:, :, out_idx : out_idx + 1, :]
                v_dst[:, :, dst_pos : dst_pos + 1, :] = v_tmp[:, :, out_idx : out_idx + 1, :]
        else:
            for out_idx, node_idx in enumerate(accepted_nodes):
                src_pos = int(prefix_len + node_idx)
                dst_pos = int(prefix_len + out_idx)
                k_dst[:, :, dst_pos : dst_pos + 1, :] = k_src[:, :, src_pos : src_pos + 1, :]
                v_dst[:, :, dst_pos : dst_pos + 1, :] = v_src[:, :, src_pos : src_pos + 1, :]
    if hasattr(verify_cache, "current_length"):
        try:
            verify_cache.current_length = int(prefix_len + len(accepted_nodes))
        except Exception:
            pass
    return verify_cache


def _copy_cache_range(
    *,
    dst_cache,
    src_cache,
    start: int,
    length: int,
) -> bool:
    if length <= 0 or dst_cache is None or src_cache is None:
        return False
    if isinstance(dst_cache, FrozenCache):
        dst_cache = dst_cache.base_cache
    if isinstance(src_cache, FrozenCache):
        src_cache = src_cache.base_cache
    if not (hasattr(dst_cache, "key_cache") and hasattr(dst_cache, "value_cache")):
        return False
    if not (hasattr(src_cache, "key_cache") and hasattr(src_cache, "value_cache")):
        return False
    end = int(start + length)
    num_layers = min(len(dst_cache.key_cache), len(src_cache.key_cache))
    for layer_idx in range(num_layers):
        k_src = src_cache.key_cache[layer_idx]
        v_src = src_cache.value_cache[layer_idx]
        k_dst = dst_cache.key_cache[layer_idx]
        v_dst = dst_cache.value_cache[layer_idx]
        k_dst[:, :, start:end, :] = k_src[:, :, start:end, :]
        v_dst[:, :, start:end, :] = v_src[:, :, start:end, :]
    if hasattr(dst_cache, "current_length"):
        try:
            dst_cache.current_length = max(int(dst_cache.current_length), int(end))
        except Exception:
            pass
    return True


def _build_tree_attention_mask(
    *,
    prefix_len: int,
    parents: Sequence[int],
    depths: Sequence[int],
    device,
    dtype: torch.dtype,
    mask_dtype: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    tree_len = len(parents)
    total_len = prefix_len + tree_len
    use_bool = str(mask_dtype).lower() == "bool"
    if use_bool:
        mask = torch.zeros((total_len, total_len), dtype=torch.bool, device=device)
        for i in range(prefix_len):
            mask[i, : i + 1] = True
        for idx in range(tree_len):
            row = prefix_len + idx
            mask[row, :prefix_len] = True
            anc = parents[idx]
            while anc >= 0:
                mask[row, prefix_len + anc] = True
                anc = parents[anc]
            mask[row, row] = True
        attn_mask = mask.unsqueeze(0).unsqueeze(0)
    else:
        neg = -1e9
        mask = torch.full((total_len, total_len), neg, dtype=torch.float32, device=device)
        for i in range(prefix_len):
            mask[i, : i + 1] = 0.0
        for idx in range(tree_len):
            row = prefix_len + idx
            mask[row, :prefix_len] = 0.0
            anc = parents[idx]
            while anc >= 0:
                mask[row, prefix_len + anc] = 0.0
                anc = parents[anc]
            mask[row, row] = 0.0
        attn_mask = mask.unsqueeze(0).unsqueeze(0).to(dtype=dtype)

    pos_ids = torch.arange(total_len, device=device, dtype=torch.long).unsqueeze(0)
    for idx, depth in enumerate(depths):
        pos_ids[0, prefix_len + idx] = prefix_len + depth - 1

    return attn_mask, pos_ids


def _build_tree_mask(*, parents: Sequence[int], device) -> torch.Tensor:
    tree_len = len(parents)
    mask = torch.zeros((tree_len, tree_len), dtype=torch.float32, device=device)
    for idx in range(tree_len):
        mask[idx, idx] = 1.0
        anc = parents[idx]
        while anc >= 0:
            mask[idx, anc] = 1.0
            anc = parents[anc]
    return mask.unsqueeze(0).unsqueeze(0)


def _patch_tree_mask_support(model) -> None:
    base = getattr(model, "model", model)
    if getattr(base, "_tree_mask_patched", False):
        return
    orig_prepare = base._prepare_4d_causal_attention_mask_with_cache_position

    def _prepare_with_tree(
        self,
        attention_mask: torch.Tensor,
        sequence_length: int,
        target_length: int,
        dtype: torch.dtype,
        device: torch.device,
        cache_position: torch.Tensor,
        batch_size: int,
        config,
        past_key_values,
    ):
        causal_mask = orig_prepare(
            attention_mask,
            sequence_length=sequence_length,
            target_length=target_length,
            dtype=dtype,
            device=device,
            cache_position=cache_position,
            batch_size=batch_size,
            config=config,
            past_key_values=past_key_values,
        )
        tree_mask = getattr(self, "tree_mask", None)
        if tree_mask is not None:
            tree_len = tree_mask.size(-1)
            min_val = torch.finfo(causal_mask.dtype).min
            causal_mask[:, :, -tree_len:, -tree_len:][tree_mask == 0] = min_val
        return causal_mask

    base._prepare_4d_causal_attention_mask_with_cache_position = types.MethodType(
        _prepare_with_tree, base
    )
    base._tree_mask_patched = True


def _patch_qwen2_tree_kernel(model, *, tree_mask: torch.Tensor, kernel: str) -> None:
    if kernel == "none":
        return
    try:
        from flash_attn.flash_attn_interface import _flash_attn_forward
    except Exception as exc:
        print(f"[Warn] flash_attn not available for tree kernel: {exc}")
        return

    from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv

    base = getattr(model, "model", model)
    if not hasattr(base, "_tree_kernel_enabled"):
        base._tree_kernel_enabled = False
    if not hasattr(base, "_tree_kernel_patched"):
        base._tree_kernel_patched = False

    if not hasattr(base, "_orig_update_causal_mask"):
        base._orig_update_causal_mask = base._update_causal_mask

        def _update_causal_mask(
            self,
            attention_mask,
            input_tensor,
            cache_position,
            past_key_values,
            output_attentions,
        ):
            if getattr(self, "_tree_kernel_enabled", False):
                return None
            return self._orig_update_causal_mask(
                attention_mask, input_tensor, cache_position, past_key_values, output_attentions
            )

        base._update_causal_mask = types.MethodType(_update_causal_mask, base)

    if not base._tree_kernel_patched:
        for layer in base.layers:
            attn = layer.self_attn
            if not hasattr(attn, "_orig_forward"):
                attn._orig_forward = attn.forward

            def _forward_with_tree(
                self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.LongTensor] = None,
                past_key_value: Optional[Cache] = None,
                output_attentions: bool = False,
                use_cache: bool = False,
                cache_position: Optional[torch.LongTensor] = None,
                position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
            ):
                if output_attentions:
                    return self._orig_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                if getattr(self, "_tree_kernel", "none") == "none":
                    return self._orig_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                if not getattr(base, "_tree_kernel_enabled", False):
                    return self._orig_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                if past_key_value is None:
                    return self._orig_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                tree_mask_local = getattr(self, "_tree_mask", None)
                if tree_mask_local is None:
                    return self._orig_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                bsz, q_len, _ = hidden_states.size()
                if tree_mask_local.size(-1) != q_len:
                    return self._orig_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )

                query_states = self.q_proj(hidden_states)
                key_states = self.k_proj(hidden_states)
                value_states = self.v_proj(hidden_states)

                query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
                key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
                value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

                if position_embeddings is None:
                    cos, sin = self.rotary_emb(value_states, position_ids)
                else:
                    cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                key_states_kv = key_states
                value_states_kv = value_states

                base_cache = past_key_value
                if isinstance(past_key_value, FrozenCache):
                    base_cache = past_key_value.base_cache
                if base_cache is None or not hasattr(base_cache, "key_cache"):
                    return self._orig_forward(
                        hidden_states=hidden_states,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        past_key_value=past_key_value,
                        output_attentions=output_attentions,
                        use_cache=use_cache,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                prefix_len = 0
                if hasattr(past_key_value, "get_seq_length"):
                    prefix_len = past_key_value.get_seq_length(self.layer_idx)

                k_prefix = None
                v_prefix = None
                if prefix_len > 0 and base_cache is not None and hasattr(base_cache, "key_cache"):
                    k_prefix = base_cache.key_cache[self.layer_idx][..., :prefix_len, :]
                    v_prefix = base_cache.value_cache[self.layer_idx][..., :prefix_len, :]

                key_states = repeat_kv(key_states, self.num_key_value_groups)
                value_states = repeat_kv(value_states, self.num_key_value_groups)

                if k_prefix is not None:
                    k_prefix = repeat_kv(k_prefix, self.num_key_value_groups)
                    v_prefix = repeat_kv(v_prefix, self.num_key_value_groups)

                if bool(use_cache) and base_cache is not None and not isinstance(past_key_value, FrozenCache):
                    wrote_cache = False
                    if hasattr(base_cache, "key_cache") and hasattr(base_cache, "value_cache"):
                        try:
                            k_store = base_cache.key_cache[self.layer_idx]
                            v_store = base_cache.value_cache[self.layer_idx]
                            end = int(prefix_len + q_len)
                            if k_store.shape[-2] >= end:
                                k_store[:, :, int(prefix_len) : end, :] = key_states_kv
                                v_store[:, :, int(prefix_len) : end, :] = value_states_kv
                                if hasattr(base_cache, "current_length"):
                                    base_cache.current_length = max(int(end), int(base_cache.current_length))
                                wrote_cache = True
                        except Exception:
                            wrote_cache = False
                    if not wrote_cache and hasattr(base_cache, "update"):
                        cache_position = torch.arange(
                            q_len, device=hidden_states.device, dtype=torch.long
                        ) + int(prefix_len)
                        try:
                            base_cache.update(
                                key_states_kv,
                                value_states_kv,
                                self.layer_idx,
                                cache_kwargs={"cache_position": cache_position},
                            )
                        except Exception:
                            pass

                scale = 1.0 / math.sqrt(self.head_dim)
                q_flash = query_states.transpose(1, 2).contiguous()

                if k_prefix is None:
                    out_prefix = torch.zeros_like(query_states)
                    lse_prefix = torch.full(
                        (bsz, self.num_heads, q_len),
                        float("-inf"),
                        device=query_states.device,
                        dtype=torch.float32,
                    )
                else:
                    k_flash = k_prefix.transpose(1, 2).contiguous()
                    v_flash = v_prefix.transpose(1, 2).contiguous()
                    out_prefix, _, _, _, _, lse_prefix, _, _ = _flash_attn_forward(
                        q_flash,
                        k_flash,
                        v_flash,
                        0.0,
                        scale,
                        False,
                        (-1, -1),
                        None,
                        False,
                    )
                    out_prefix = out_prefix.transpose(1, 2)
                    lse_prefix = lse_prefix.to(dtype=torch.float32)

                scores_tree = torch.matmul(query_states.float(), key_states.transpose(2, 3).float()) * scale
                scores_tree = scores_tree.masked_fill(tree_mask_local == 0, float("-inf"))
                lse_tree = torch.logsumexp(scores_tree, dim=-1)
                attn_tree = torch.softmax(scores_tree, dim=-1, dtype=torch.float32)
                out_tree = torch.matmul(attn_tree, value_states.float()).to(dtype=query_states.dtype)

                total_lse = torch.logaddexp(lse_prefix, lse_tree)
                w_prefix = torch.exp(lse_prefix - total_lse).unsqueeze(-1)
                w_tree = torch.exp(lse_tree - total_lse).unsqueeze(-1)
                attn_output = (out_prefix * w_prefix + out_tree * w_tree).to(dtype=query_states.dtype)

                attn_output = attn_output.transpose(1, 2).contiguous()
                attn_output = attn_output.view(bsz, q_len, self.hidden_size)
                attn_output = self.o_proj(attn_output)

                return attn_output, None, past_key_value

            attn.forward = types.MethodType(_forward_with_tree, attn)

        base._tree_kernel_patched = True

    for layer in base.layers:
        attn = layer.self_attn
        attn._tree_kernel = kernel
        attn._tree_mask = tree_mask


def _warmup_tree_kernel(
    *,
    model,
    tree_len: int,
    prefix_len: int,
    dtype: torch.dtype,
    device: torch.device,
) -> None:
    if tree_len <= 0 or prefix_len <= 0:
        return
    try:
        from flash_attn.flash_attn_interface import _flash_attn_forward
    except Exception:
        return
    num_heads = int(getattr(model.config, "num_attention_heads", 0))
    head_dim = int(getattr(model.config, "head_dim", model.config.hidden_size // num_heads))
    q = torch.zeros((1, tree_len, num_heads, head_dim), device=device, dtype=dtype)
    k = torch.zeros((1, prefix_len, num_heads, head_dim), device=device, dtype=dtype)
    v = torch.zeros((1, prefix_len, num_heads, head_dim), device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(head_dim)
    _flash_attn_forward(q, k, v, 0.0, scale, False, (-1, -1), None, False)
    qh = q.transpose(1, 2)
    kh = torch.zeros((1, num_heads, tree_len, head_dim), device=device, dtype=dtype)
    _ = torch.matmul(qh.float(), kh.transpose(2, 3).float())


def _build_tree_from_sequences(seqs: torch.Tensor) -> Tuple[torch.Tensor, List[int], List[int], List[List[int]]]:
    # seqs: (K, gamma)
    seqs = seqs.tolist()
    node_tokens: List[int] = []
    parents: List[int] = []
    depths: List[int] = []
    child_maps: Dict[int, Dict[int, int]] = {-1: {}}
    seq_nodes: List[List[int]] = []

    for seq in seqs:
        parent = -1
        nodes = []
        for depth, tok in enumerate(seq, start=1):
            tok = int(tok)
            child_map = child_maps.get(parent)
            if child_map is None:
                child_map = {}
                child_maps[parent] = child_map
            if tok in child_map:
                idx = child_map[tok]
            else:
                idx = len(node_tokens)
                node_tokens.append(tok)
                parents.append(parent)
                depths.append(depth)
                child_maps[parent][tok] = idx
                child_maps[idx] = {}
            nodes.append(idx)
            parent = idx
        seq_nodes.append(nodes)

    if node_tokens:
        children: List[List[int]] = [[] for _ in range(len(node_tokens))]
        for idx, parent in enumerate(parents):
            if parent >= 0:
                children[parent].append(idx)
        roots = [idx for idx, parent in enumerate(parents) if parent < 0]
        order: List[int] = []
        queue = list(roots)
        q_idx = 0
        while q_idx < len(queue):
            node = queue[q_idx]
            q_idx += 1
            order.append(node)
            queue.extend(children[node])
        if len(order) == len(node_tokens) and order != list(range(len(node_tokens))):
            remap = {old: new for new, old in enumerate(order)}
            node_tokens = [node_tokens[i] for i in order]
            parents = [
                -1 if parents[i] < 0 else remap[int(parents[i])] for i in order
            ]
            depths = [0 for _ in range(len(node_tokens))]
            for idx in range(len(node_tokens)):
                depth = 1
                parent = parents[idx]
                while parent >= 0:
                    depth += 1
                    parent = parents[parent]
                depths[idx] = depth
            seq_nodes = [[remap[int(idx)] for idx in nodes] for nodes in seq_nodes]

    return (
        torch.tensor(node_tokens, dtype=torch.long),
        parents,
        depths,
        seq_nodes,
    )


def _log_topk(logits: torch.Tensor, tokenizer, label: str, topk: int, max_rows: int) -> None:
    if topk <= 0 or max_rows <= 0:
        return
    rows = min(int(max_rows), logits.shape[0])
    topk_vals, topk_idx = torch.topk(logits[:rows], k=topk, dim=-1)
    probs = torch.softmax(topk_vals.to(dtype=torch.float32), dim=-1)
    for i in range(rows):
        tokens = [tokenizer.decode([int(t)]) for t in topk_idx[i].tolist()]
        probs_i = [float(p) for p in probs[i].tolist()]
        print(f"[Qual][{label}] node={i} topk={list(zip(tokens, probs_i))}")


def _root_indices(parents: Sequence[int]) -> List[int]:
    return [idx for idx, parent in enumerate(parents) if parent < 0]


def _log_root_candidates(
    root_tokens: torch.Tensor,
    argmax_prefix: int,
    tokenizer,
    topk: int,
    token_group: Optional[set] = None,
) -> None:
    if root_tokens.numel() == 0:
        return
    uniq, counts = torch.unique(root_tokens, return_counts=True)
    order = torch.argsort(counts, descending=True)
    top = order[: max(1, int(topk))]
    items = []
    for idx in top.tolist():
        tok_id = int(uniq[idx].item())
        items.append((tok_id, tokenizer.convert_ids_to_tokens(tok_id), int(counts[idx].item())))
    if token_group is not None:
        mask = torch.zeros_like(root_tokens, dtype=torch.bool)
        for tid in token_group:
            mask |= root_tokens == int(tid)
        match = int(mask.sum().item())
    else:
        match = int((root_tokens == int(argmax_prefix)).sum().item())
    argmax_tok = tokenizer.convert_ids_to_tokens(int(argmax_prefix))
    print(
        f"[Debug] root_match={match}/{root_tokens.numel()} "
        f"argmax_prefix_id={int(argmax_prefix)} argmax_prefix={repr(argmax_tok)}"
    )
    group_size = len(token_group) if token_group is not None else 0
    print(f"[Debug] root_group_size={group_size} root_top_counts={items}")


def _debug_prefix_alignment(
    *,
    verify_tokenizer,
    diff_tokenizer,
    verify_prefix_ids: torch.Tensor,
    diff_prefix_ids: torch.Tensor,
    verify_to_diff: Optional[torch.Tensor],
    max_tail: int,
) -> None:
    if verify_prefix_ids.numel() == 0 or diff_prefix_ids.numel() == 0:
        return
    verify_ids = verify_prefix_ids[0].tolist()
    diff_ids = diff_prefix_ids[0].tolist()
    verify_text = verify_tokenizer.decode(verify_ids, skip_special_tokens=False)
    diff_text = diff_tokenizer.decode(diff_ids, skip_special_tokens=False)
    text_match = verify_text == diff_text
    print(
        f"[Debug] prefix_text_match={text_match} verify_len={len(verify_ids)} diff_len={len(diff_ids)}"
    )
    if not text_match:
        verify_tail = verify_tokenizer.decode(verify_ids[-max_tail:], skip_special_tokens=False)
        diff_tail = diff_tokenizer.decode(diff_ids[-max_tail:], skip_special_tokens=False)
        print(f"[Debug] prefix_text_tail_verify={repr(verify_tail)}")
        print(f"[Debug] prefix_text_tail_diff={repr(diff_tail)}")

    if verify_to_diff is not None:
        mapped = verify_to_diff[verify_prefix_ids]
        if mapped.shape == diff_prefix_ids.shape:
            mismatch = (mapped != diff_prefix_ids) & (mapped >= 0)
            mismatch_count = int(mismatch.sum().item())
            if mismatch_count:
                first = int(torch.nonzero(mismatch, as_tuple=False)[0].item())
                v_id = int(verify_prefix_ids[0, first].item())
                d_id = int(diff_prefix_ids[0, first].item())
                m_id = int(mapped[0, first].item())
                v_tok = verify_tokenizer.convert_ids_to_tokens(v_id)
                d_tok = diff_tokenizer.convert_ids_to_tokens(d_id)
                m_tok = diff_tokenizer.convert_ids_to_tokens(m_id) if m_id >= 0 else "<unk>"
                print(
                    "[Debug] prefix_id_map_mismatch="
                    f"{mismatch_count}/{mapped.numel()} first_idx={first} "
                    f"verify={v_id}/{repr(v_tok)} diff={d_id}/{repr(d_tok)} mapped={m_id}/{repr(m_tok)}"
                )
            else:
                print(f"[Debug] prefix_id_map_mismatch=0/{mapped.numel()}")
        else:
            print("[Debug] prefix_id_map_mismatch=shape_mismatch")

    diff_round = diff_tokenizer(verify_text, return_tensors="pt", add_special_tokens=False).input_ids
    if diff_round.shape == diff_prefix_ids.shape:
        if not torch.equal(diff_round.to(diff_prefix_ids.device), diff_prefix_ids):
            print("[Debug] diff_roundtrip_mismatch=1")
        else:
            print("[Debug] diff_roundtrip_mismatch=0")
    else:
        print("[Debug] diff_roundtrip_mismatch=shape_mismatch")


def _accept_greedy(
    seqs: torch.Tensor,
    seq_nodes: List[List[int]],
    tree_logits: torch.Tensor,
    prefix_next_logits: torch.Tensor,
    prefix_token_group: Optional[set],
    *,
    verify_temperature: float,
    verify_top_k: int,
    verify_top_p: float,
    generator: Optional[torch.Generator],
) -> Tuple[List[int], int, int, Optional[int]]:
    argmax_prefix = torch.argmax(prefix_next_logits, dim=-1)
    argmax_tree = torch.argmax(tree_logits, dim=-1)
    best_len = -1
    best_idx = 0
    for idx, seq in enumerate(seqs.tolist()):
        accept_len = 0
        for depth, tok in enumerate(seq):
            if depth == 0:
                if prefix_token_group is not None:
                    match = int(tok) in prefix_token_group
                else:
                    match = int(argmax_prefix.item()) == int(tok)
            else:
                parent_idx = seq_nodes[idx][depth - 1]
                match = int(argmax_tree[parent_idx].item()) == int(tok)
            if match:
                accept_len += 1
            else:
                break
        if accept_len > best_len:
            best_len = accept_len
            best_idx = idx

    if best_len <= 0:
        for idx, seq in enumerate(seqs.tolist()):
            tok = int(seq[0])
            if prefix_token_group is not None:
                if tok in prefix_token_group:
                    best_len = 1
                    best_idx = idx
                    break
            else:
                if tok == int(argmax_prefix.item()):
                    best_len = 1
                    best_idx = idx
                    break

    accepted = []
    if best_len > 0:
        accepted = [int(t) for t in seqs[best_idx, :best_len].tolist()]
    if best_len == 0:
        rep_logits = prefix_next_logits
    else:
        last_node = seq_nodes[best_idx][best_len - 1]
        rep_logits = tree_logits[last_node]
    replacement = _sample_from_logits(
        rep_logits,
        temperature=float(verify_temperature),
        top_k=int(verify_top_k),
        top_p=float(verify_top_p),
        generator=generator,
    )
    return accepted, int(best_len), int(best_idx), int(replacement.item())


def _rand_scalar(generator: Optional[torch.Generator], device: torch.device) -> torch.Tensor:
    if generator is None:
        return torch.rand((), device=device)
    return torch.rand((), generator=generator, device=device)


def _accept_leviathan(
    seqs_verify: torch.Tensor,
    seq_nodes: List[List[int]],
    tree_logits: torch.Tensor,
    prefix_next_logits: torch.Tensor,
    diff_logits: torch.Tensor,
    diff_to_verify: Optional[torch.Tensor],
    verify_vocab_size: int,
    *,
    diffusion_temperature: float,
    diffusion_top_k: int,
    diffusion_top_p: float,
    verify_temperature: float,
    verify_top_k: int,
    verify_top_p: float,
    generator: Optional[torch.Generator],
    eps: float = 1e-8,
) -> Tuple[List[int], int, int, Optional[int]]:
    if seqs_verify.numel() == 0:
        replacement = _sample_from_logits(
            prefix_next_logits,
            temperature=verify_temperature,
            top_k=verify_top_k,
            top_p=verify_top_p,
            generator=generator,
        )
        return [], 0, 0, int(replacement.item())

    seq_count, seq_len = seqs_verify.shape
    device = seqs_verify.device

    diff_probs = _draft_probs_from_logits(
        diff_logits,
        temperature=diffusion_temperature,
        top_k=diffusion_top_k,
        top_p=diffusion_top_p,
    )
    q_verify = torch.zeros(
        (seq_len, verify_vocab_size), device=device, dtype=diff_probs.dtype
    )
    if diff_to_verify is not None:
        idx = diff_to_verify.to(device=device)
        valid = idx >= 0
        if valid.any():
            valid_idx = idx[valid]
            for depth in range(seq_len):
                q_verify[depth].scatter_add_(0, valid_idx, diff_probs[depth, valid])

    p_prefix = _draft_probs_from_logits(
        prefix_next_logits,
        temperature=verify_temperature,
        top_k=verify_top_k,
        top_p=verify_top_p,
    )
    if tree_logits.numel() > 0:
        p_tree = _draft_probs_from_logits(
            tree_logits,
            temperature=verify_temperature,
            top_k=verify_top_k,
            top_p=verify_top_p,
        )
    else:
        p_tree = torch.empty((0, verify_vocab_size), device=device, dtype=p_prefix.dtype)

    accepted: List[int] = []
    candidate_mask = torch.ones(seq_count, dtype=torch.bool, device=device)
    chosen_idx = 0
    current_node = None
    seq_tokens = seqs_verify
    deterministic = verify_temperature <= 0.0
    if deterministic:
        prefix_argmax = torch.argmax(p_prefix).to(device)
        tree_argmax = (
            torch.argmax(tree_logits, dim=1).to(device)
            if tree_logits.numel() > 0
            else torch.empty((0,), device=device, dtype=torch.long)
        )

    def _target_token(depth: int) -> torch.Tensor:
        if depth == 0:
            return prefix_argmax
        if current_node is None or current_node >= len(tree_argmax):
            raise RuntimeError("Missing node for deterministic acceptance.")
        return tree_argmax[current_node]

    for depth in range(seq_len):
        candidate_indices = torch.nonzero(candidate_mask, as_tuple=True)[0]
        if candidate_indices.numel() == 0:
            break
        if depth == 0:
            p_base = p_prefix
        else:
            if current_node is None:
                break
            p_base = p_tree[current_node]
        q_dist = q_verify[depth]
        p_work = p_base.clone()
        tokens = seq_tokens[:, depth]

        if deterministic:
            target_token = _target_token(depth)
            matches = tokens == target_token
            valid_matches = torch.nonzero(candidate_mask & matches, as_tuple=True)[0]
            if valid_matches.numel() == 0:
                replacement = int(target_token.item())
                return accepted, len(accepted), chosen_idx, replacement
            accepted_tok = int(target_token.item())
            chosen_idx = int(valid_matches[0].item())
            accepted.append(accepted_tok)
            candidate_mask &= matches
            if candidate_mask.any():
                first_idx = int(torch.nonzero(candidate_mask, as_tuple=True)[0][0].item())
                current_node = seq_nodes[first_idx][depth]
            else:
                current_node = None
            continue

        accepted_tok: Optional[int] = None
        for ci in candidate_indices:
            tok = tokens[ci]
            q_tok = q_dist[tok]
            p_tok = p_work[tok]
            denom = torch.maximum(q_tok, torch.tensor(eps, device=device, dtype=q_tok.dtype))
            accept_prob = torch.minimum(
                torch.tensor(1.0, device=device, dtype=p_tok.dtype), p_tok / denom
            )
            u = _rand_scalar(generator, device=device)
            if u <= accept_prob:
                accepted_tok = int(tok.item())
                chosen_idx = int(ci.item())
                break
            p_work = (p_work - q_dist).clamp(min=0.0)
            total = p_work.sum()
            if total > 0:
                p_work = p_work / total
        if accepted_tok is None:
            replacement = _sample_from_probs(p_work, generator=generator)
            return accepted, len(accepted), chosen_idx, int(replacement.item())

        accepted.append(accepted_tok)
        match_mask = tokens == accepted_tok
        candidate_mask &= match_mask
        if candidate_mask.any():
            first_idx = int(torch.nonzero(candidate_mask, as_tuple=True)[0][0].item())
            current_node = seq_nodes[first_idx][depth]
        else:
            current_node = None

    if current_node is None:
        replacement = _sample_from_probs(p_prefix, generator=generator)
    else:
        replacement = _sample_from_probs(p_tree[current_node], generator=generator)
    return accepted, len(accepted), int(chosen_idx), int(replacement.item())


def _measure_stage(start_event, end_event, sync: bool, fn):
    if start_event is not None:
        start_event.record()
    t0 = time.perf_counter()
    out = fn()
    if end_event is not None:
        end_event.record()
        if sync:
            torch.cuda.synchronize()
            cuda_ms = float(start_event.elapsed_time(end_event)) if start_event is not None else 0.0
        else:
            cuda_ms = 0.0
    else:
        cuda_ms = 0.0
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return out, cuda_ms, wall_ms


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 3: diffusion draft -> tree verify -> accept.")
    p.add_argument("--diffusion_model_name", type=str, default="apple/DiffuCoder-7B-cpGRPO")
    p.add_argument("--verify_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--gamma", type=int, default=8)
    p.add_argument("--k", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--steps", type=int, default=0)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--verify_chat_template", action="store_true", default=False)
    p.add_argument("--verify_system_prompt", type=str, default="")
    p.add_argument("--diffusion_temperature", type=float, default=1.0)
    p.add_argument("--diffusion_top_k", type=int, default=0)
    p.add_argument("--diffusion_top_p", type=float, default=1.0)
    p.add_argument("--diffusion_next_token", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--diffusion_sdp_backend",
        type=str,
        default="auto",
        choices=("auto", "flash", "mem_efficient", "math"),
    )
    p.add_argument("--diffusion_log_attention_backend", action="store_true", default=False)
    p.add_argument("--diffusion_omit_attention_mask", action="store_true", default=False)
    p.add_argument("--diffusion_fuse_mlp", action="store_true", default=False)
    p.add_argument("--diffusion_fuse_qkv", action="store_true", default=False)
    p.add_argument("--verify_temperature", type=float, default=0.0)
    p.add_argument("--verify_top_k", type=int, default=0)
    p.add_argument("--verify_top_p", type=float, default=1.0)
    p.add_argument(
        "--verify_update_mode",
        type=str,
        default="reuse_tree",
        choices=("forward", "reuse_tree"),
    )
    p.add_argument(
        "--verify_sdp_backend",
        type=str,
        default="auto",
        choices=("auto", "flash", "mem_efficient", "math"),
    )
    p.add_argument("--verify_log_attention_backend", action="store_true", default=False)
    p.add_argument("--accept_mode", type=str, default="leviathan", choices=("greedy", "leviathan"))
    p.add_argument("--approx_kv_cache", action="store_true", default=False)
    p.add_argument("--verify_cache_type", type=str, default="prealloc", choices=("dynamic", "static", "prealloc"))
    p.add_argument("--verify_max_cache_len", type=int, default=0)
    p.add_argument("--verify_freeze_cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tree_mask_dtype", type=str, default="float", choices=("bool", "float"))
    p.add_argument("--tree_mask_mode", type=str, default="model", choices=("model", "explicit"))
    p.add_argument("--tree_kernel", type=str, default="flash_prefix", choices=("none", "flash_prefix"))
    p.add_argument("--tree_kernel_warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tree_kernel_warmup_prefix", type=int, default=0)
    p.add_argument("--tree_kernel_fallback", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tree_kernel_fallback_tol", type=float, default=1e-3)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--warmup_steps", type=int, default=1)
    p.add_argument("--sync_timing", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--debug_topk", type=int, default=5)
    p.add_argument("--debug_nodes", type=int, default=0)
    p.add_argument("--debug_tree_compare", action="store_true", default=False)
    p.add_argument("--debug_tree_compare_depths", type=int, default=2)
    p.add_argument("--debug_prefix_every", type=int, default=0)
    p.add_argument("--debug_prefix_tail", type=int, default=16)
    p.add_argument("--allow_tokenizer_mismatch", action="store_true", default=False)
    p.add_argument("--add_special_tokens", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--sync_prefix_text", action="store_true", default=False)
    p.add_argument("--sync_prefix_full", action=argparse.BooleanOptionalAction, default=False)
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    device = torch.device(args.device)
    dtype = _get_dtype(args.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))
    k_val = int(args.k)
    if str(args.accept_mode) == "leviathan" and k_val > 1:
        print(
            "[Info] accept_mode=leviathan uses multi-round tree acceptance; k>1 can improve "
            "accept_len at extra verify cost."
        )

    diff_sdp_backend = str(args.diffusion_sdp_backend)
    diff_sdp_verbose = bool(args.diffusion_log_attention_backend)

    def _diff_sdp_ctx():
        nonlocal diff_sdp_verbose
        ctx = _build_sdp_context(diff_sdp_backend, verbose=diff_sdp_verbose)
        diff_sdp_verbose = False
        return ctx

    verify_sdp_backend = str(args.verify_sdp_backend)
    verify_sdp_verbose = bool(args.verify_log_attention_backend)

    def _verify_sdp_ctx():
        nonlocal verify_sdp_verbose
        ctx = _build_sdp_context(verify_sdp_backend, verbose=verify_sdp_verbose)
        verify_sdp_verbose = False
        return ctx

    diff_tokenizer = AutoTokenizer.from_pretrained(
        args.diffusion_model_name, trust_remote_code=bool(args.trust_remote_code)
    )
    verify_tokenizer = AutoTokenizer.from_pretrained(args.verify_model_name)

    if diff_tokenizer.pad_token is None and diff_tokenizer.eos_token is not None:
        diff_tokenizer.pad_token = diff_tokenizer.eos_token
    if verify_tokenizer.pad_token is None and verify_tokenizer.eos_token is not None:
        verify_tokenizer.pad_token = verify_tokenizer.eos_token

    compatible, compat_msg = _tokenizers_compatible(diff_tokenizer, verify_tokenizer)
    if not compatible and not bool(args.allow_tokenizer_mismatch):
        raise ValueError(
            "Tokenizers do not match (non-special tokens differ). "
            f"{compat_msg}. Use --allow_tokenizer_mismatch to override."
        )
    if not compatible:
        print(f"[Warn] Tokenizers differ beyond special tokens; proceeding due to override. {compat_msg}")
    else:
        if diff_tokenizer.get_vocab() != verify_tokenizer.get_vocab():
            print(f"[Info] Tokenizers differ only in special/added tokens; proceeding. {compat_msg}")
    special_mismatch = _special_token_mismatch(diff_tokenizer, verify_tokenizer)
    if special_mismatch:
        print(f"[Info] special_token_mismatch={','.join(special_mismatch)}")
        if bool(args.add_special_tokens):
            print(
                "[Warn] add_special_tokens=True with mismatched special tokens can desync prefixes. "
                "Consider --no-add_special_tokens."
            )

    diff_model = None
    load_errors = []
    for factory in (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel):
        try:
            diff_model = factory.from_pretrained(
                args.diffusion_model_name,
                torch_dtype=dtype,
                trust_remote_code=bool(args.trust_remote_code),
            )
            break
        except Exception as exc:
            load_errors.append(str(exc))
    if diff_model is None:
        raise RuntimeError("Failed to load diffusion model. Errors:\n" + "\n".join(load_errors))

    verify_model = AutoModelForCausalLM.from_pretrained(
        args.verify_model_name,
        torch_dtype=dtype,
    )

    diff_model.to(device)
    verify_model.to(device)
    diff_model.eval()
    verify_model.eval()
    if bool(args.diffusion_fuse_mlp):
        if _phase1_fuse_mlps is None:
            print("[Warn] diffusion_fuse_mlp requested but phase1 fuser is unavailable.")
        else:
            _phase1_fuse_mlps(diff_model, verbose=True)
    if bool(args.diffusion_fuse_qkv):
        if _phase1_fuse_qkv is None:
            print("[Warn] diffusion_fuse_qkv requested but phase1 fuser is unavailable.")
        else:
            _phase1_fuse_qkv(diff_model, verbose=True)
    use_num_logits_to_keep = _supports_num_logits_to_keep(diff_model)
    print(f"[Info] verifier={args.verify_model_name}")

    mask_id = _maybe_add_mask_token(diff_model, diff_tokenizer)

    diff_vocab_size = None
    verify_vocab_size = None
    diff_out = diff_model.get_output_embeddings()
    if diff_out is not None:
        diff_vocab_size = diff_out.weight.shape[0]
    if diff_vocab_size is None and hasattr(diff_model.config, "vocab_size"):
        diff_vocab_size = int(diff_model.config.vocab_size)
    verify_out = verify_model.get_output_embeddings()
    if verify_out is not None:
        verify_vocab_size = verify_out.weight.shape[0]
    if verify_vocab_size is None and hasattr(verify_model.config, "vocab_size"):
        verify_vocab_size = int(verify_model.config.vocab_size)

    diff_to_verify, verify_to_diff, missing_diff, missing_verify = _build_token_id_maps(
        diff_tokenizer,
        verify_tokenizer,
        diff_size=diff_vocab_size,
        verify_size=verify_vocab_size,
    )
    diff_to_verify = diff_to_verify.to(device)
    verify_to_diff = verify_to_diff.to(device)
    if missing_diff or missing_verify:
        print(
            f"[Info] token id map missing: diff_missing={missing_diff} verify_missing={missing_verify}."
        )
    diff_valid_mask = (diff_to_verify >= 0).to(device=device)

    verify_size = verify_vocab_size if verify_vocab_size is not None else len(verify_tokenizer)
    verify_token_groups: Dict[str, set] = {}
    for vid in range(int(verify_size)):
        tok = verify_tokenizer.convert_ids_to_tokens(vid)
        verify_token_groups.setdefault(tok, set()).add(int(vid))

    add_special = bool(args.add_special_tokens)
    prompt_text = str(args.prompt)
    if bool(args.verify_chat_template):
        messages = []
        if str(args.verify_system_prompt):
            messages.append({"role": "system", "content": str(args.verify_system_prompt)})
        messages.append({"role": "user", "content": prompt_text})
        prompt_text = verify_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        if not bool(args.sync_prefix_full):
            print("[Info] verify_chat_template enabled; forcing sync_prefix_full for diffusion prefix.")
            args.sync_prefix_full = True

    verify_enc = verify_tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special)
    verify_prefix_ids = verify_enc.input_ids.to(device=device, dtype=torch.long)
    if bool(args.sync_prefix_text) or bool(args.sync_prefix_full):
        diff_enc = diff_tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special)
        diff_prefix_ids = diff_enc.input_ids.to(device=device, dtype=torch.long)
        if bool(args.sync_prefix_full):
            print("[Info] sync_prefix_full enabled; using diffusion tokenizer for prefix.")
        else:
            print("[Info] sync_prefix_text enabled; using diffusion tokenizer for prefix.")
    else:
        diff_prefix_ids = verify_to_diff[verify_prefix_ids]
        if (diff_prefix_ids < 0).any():
            diff_enc = diff_tokenizer(prompt_text, return_tensors="pt", add_special_tokens=add_special)
            diff_prefix_ids = diff_enc.input_ids.to(device=device, dtype=torch.long)
            print("[Warn] prompt token map incomplete; using diffusion tokenizer for prefix.")
    print(
        "[Info] prefix_mode "
        f"add_special_tokens={add_special} "
        f"sync_prefix_text={bool(args.sync_prefix_text)} "
        f"sync_prefix_full={bool(args.sync_prefix_full)}"
    )

    if int(args.debug_prefix_every) > 0:
        _debug_prefix_alignment(
            verify_tokenizer=verify_tokenizer,
            diff_tokenizer=diff_tokenizer,
            verify_prefix_ids=verify_prefix_ids,
            diff_prefix_ids=diff_prefix_ids,
            verify_to_diff=verify_to_diff,
            max_tail=int(args.debug_prefix_tail),
        )

    use_cuda = device.type == "cuda"
    start_event = torch.cuda.Event(enable_timing=True) if use_cuda else None
    end_event = torch.cuda.Event(enable_timing=True) if use_cuda else None

    # Prefill diffusion cache (approx mode only).
    diff_past = None
    diff_prefix_next_logits = None
    diff_shift = -1 if bool(args.diffusion_next_token) else 0
    diff_shift_locked = False
    warned_approx_shift = False
    if bool(args.approx_kv_cache):
        diff_past = DynamicCache()
        prefill_kwargs = {}
        if use_num_logits_to_keep:
            prefill_kwargs["num_logits_to_keep"] = 0
        with torch.inference_mode(), _diff_sdp_ctx():
            out = diff_model(
                input_ids=diff_prefix_ids,
                attention_mask=None,
                use_cache=True,
                past_key_values=diff_past,
                return_dict=True,
                **prefill_kwargs,
            )
        if bool(args.diffusion_next_token):
            diff_prefix_next_logits = out.logits[:, -1, :].squeeze(0)
        print("[ApproxCache] diffusion approx_kv_cache enabled.")
    elif bool(args.diffusion_next_token):
        prefill_kwargs = {}
        if use_num_logits_to_keep:
            prefill_kwargs["num_logits_to_keep"] = 1
        with torch.inference_mode(), _diff_sdp_ctx():
            out = diff_model(
                input_ids=diff_prefix_ids,
                attention_mask=None,
                use_cache=False,
                return_dict=True,
                **prefill_kwargs,
            )
        diff_prefix_next_logits = out.logits[:, -1, :].squeeze(0)

    # Prefill verify cache.
    verify_past = None
    prefix_next_logits = None
    if args.verify_cache_type == "static":
        if int(args.verify_max_cache_len) <= 0:
            raise ValueError("--verify_max_cache_len must be > 0 for static cache")
        verify_past = StaticCache(
            config=verify_model.config,
            batch_size=1,
            max_cache_len=int(args.verify_max_cache_len),
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode(), _verify_sdp_ctx():
            out = verify_model(
                input_ids=verify_prefix_ids, past_key_values=verify_past, use_cache=True, return_dict=True
            )
        prefix_next_logits = out.logits[:, -1, :].squeeze(0)
    elif args.verify_cache_type == "prealloc":
        if int(args.verify_max_cache_len) <= 0:
            raise ValueError("--verify_max_cache_len must be > 0 for prealloc cache")
        verify_past = PreallocCache(
            config=verify_model.config,
            batch_size=1,
            max_cache_len=int(args.verify_max_cache_len),
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode(), _verify_sdp_ctx():
            out = verify_model(
                input_ids=verify_prefix_ids, past_key_values=verify_past, use_cache=True, return_dict=True
            )
        prefix_next_logits = out.logits[:, -1, :].squeeze(0)
    else:
        with torch.inference_mode(), _verify_sdp_ctx():
            out = verify_model(input_ids=verify_prefix_ids, use_cache=True, return_dict=True)
        verify_past = out.past_key_values
        prefix_next_logits = out.logits[:, -1, :].squeeze(0)

    tree_cache = None
    tree_cache_prefix_len = int(verify_prefix_ids.shape[1])
    if str(args.verify_update_mode) == "reuse_tree":
        if int(args.verify_max_cache_len) > 0:
            tree_cache = PreallocCache(
                config=verify_model.config,
                batch_size=1,
                max_cache_len=int(args.verify_max_cache_len),
                device=device,
                dtype=dtype,
            )
            if verify_past is not None:
                _copy_cache_range(
                    dst_cache=tree_cache,
                    src_cache=verify_past,
                    start=0,
                    length=tree_cache_prefix_len,
                )
        else:
            print("[Warn] reuse_tree requested without verify_max_cache_len; disabling tree cache.")

    # Tree kernel patch.
    tree_kernel = str(args.tree_kernel)
    tree_mask_mode = str(args.tree_mask_mode)
    if tree_kernel != "none" and bool(args.tree_kernel_warmup):
        warm_prefix = int(args.tree_kernel_warmup_prefix)
        if warm_prefix <= 0:
            warm_prefix = int(verify_prefix_ids.shape[1])
        warm_tree_len = int(k_val) * int(args.gamma)
        _warmup_tree_kernel(
            model=verify_model,
            tree_len=warm_tree_len,
            prefix_len=warm_prefix,
            dtype=dtype,
            device=device,
        )

    total_generated = 0
    max_new_tokens = int(args.max_new_tokens)
    max_steps = int(args.steps)
    if max_steps <= 0:
        max_steps = (max_new_tokens + int(args.gamma) - 1) // int(args.gamma)

    t_start = time.perf_counter()
    total_steps = 0
    total_accept_len = 0
    total_added = 0
    total_nodes = 0

    for step in range(max_steps):
        if total_generated >= max_new_tokens:
            break

        step_prefix_len = int(verify_prefix_ids.shape[1])
        reuse_tree_enabled = str(args.verify_update_mode) == "reuse_tree" and tree_cache is not None
        if verify_past is not None and hasattr(verify_past, "get_seq_length"):
            try:
                cache_len = int(verify_past.get_seq_length(0))
            except Exception:
                cache_len = step_prefix_len
            if cache_len != step_prefix_len:
                print(
                    f"[Warn] verify_cache_len drift: cache_len={cache_len} prefix_len={step_prefix_len}. "
                    "Truncating cache."
                )
                _truncate_cache(verify_past, step_prefix_len)

        if tree_cache is not None:
            if tree_cache_prefix_len < step_prefix_len:
                copied = _copy_cache_range(
                    dst_cache=tree_cache,
                    src_cache=verify_past,
                    start=int(tree_cache_prefix_len),
                    length=int(step_prefix_len - tree_cache_prefix_len),
                )
                if not copied:
                    print("[Warn] tree_cache prefix sync failed; disabling tree cache.")
                    tree_cache = None
                else:
                    tree_cache_prefix_len = int(step_prefix_len)
            elif tree_cache_prefix_len > step_prefix_len:
                _truncate_cache(tree_cache, int(step_prefix_len))
                tree_cache_prefix_len = int(step_prefix_len)
            else:
                _truncate_cache(tree_cache, int(step_prefix_len))

        # Diffusion forward.
        def _diff_forward():
            nonlocal warned_approx_shift
            step_gamma = int(args.gamma)
            mask_tokens = torch.full((1, step_gamma), mask_id, device=device, dtype=torch.long)
            model_kwargs = {}
            if use_num_logits_to_keep:
                keep = int(step_gamma)
                if bool(args.diffusion_next_token) and not bool(args.approx_kv_cache):
                    keep += 1
                model_kwargs["num_logits_to_keep"] = int(keep)
            if bool(args.approx_kv_cache):
                with torch.inference_mode(), _diff_sdp_ctx():
                    outputs = diff_model(
                        input_ids=mask_tokens,
                        attention_mask=None,
                        past_key_values=diff_past,
                        use_cache=True,
                        return_dict=True,
                        **model_kwargs,
                    )
            else:
                input_ids = torch.cat([diff_prefix_ids, mask_tokens], dim=1)
                attn_mask = None
                if not bool(args.diffusion_omit_attention_mask):
                    attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
                with torch.inference_mode(), _diff_sdp_ctx():
                    outputs = diff_model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        use_cache=False,
                        return_dict=True,
                        **model_kwargs,
                    )
            logits = outputs.logits
            if bool(args.diffusion_next_token):
                if bool(args.approx_kv_cache):
                    if diff_prefix_next_logits is not None and logits.shape[1] >= 1:
                        prefix_log = diff_prefix_next_logits.unsqueeze(0).unsqueeze(1)
                        logits = torch.cat([prefix_log, logits[:, :-1, :]], dim=1)
                    elif not warned_approx_shift:
                        print(
                            "[Warn] diffusion_next_token with approx_kv_cache could not align; "
                            "prefill prefix_next_logits missing."
                        )
                        warned_approx_shift = True
                else:
                    nonlocal diff_shift
                    nonlocal diff_shift_locked
                    if (
                        not diff_shift_locked
                        and diff_prefix_next_logits is not None
                        and logits.shape[1] > 1
                    ):
                        prefix_len = int(diff_prefix_ids.shape[1])
                        cand_left = max(0, min(prefix_len - 1, logits.shape[1] - 1))
                        cand_right = max(0, min(prefix_len, logits.shape[1] - 1))
                        left_logits = logits[:, cand_left, :].squeeze(0)
                        right_logits = logits[:, cand_right, :].squeeze(0)
                        diff_left = float((left_logits - diff_prefix_next_logits).abs().max().item())
                        diff_right = float((right_logits - diff_prefix_next_logits).abs().max().item())
                        diff_shift = -1 if diff_left <= diff_right else 0
                        diff_shift_locked = True
                        print(
                            "[Info] diffusion_shift_auto "
                            f"shift={diff_shift} diff_left={diff_left:.4e} diff_right={diff_right:.4e}"
                        )
                    if logits.shape[1] == step_gamma + 1:
                        logits = logits[:, :-1, :]
                    elif logits.shape[1] > step_gamma:
                        prefix_len = int(diff_prefix_ids.shape[1])
                        start = max(0, prefix_len + diff_shift)
                        logits = logits[:, start : start + step_gamma, :]
            if logits.shape[1] != step_gamma:
                logits = logits[:, -step_gamma:, :]
            return logits, outputs

        (diff_logits, diff_outputs), diff_cuda_ms, diff_wall_ms = _measure_stage(
            start_event, end_event, bool(args.sync_timing), _diff_forward
        )
        if diff_valid_mask is not None:
            mask = diff_valid_mask
            if mask.numel() != diff_logits.shape[-1]:
                tmp = torch.zeros(diff_logits.shape[-1], device=device, dtype=torch.bool)
                size = min(tmp.numel(), mask.numel())
                tmp[:size] = mask[:size]
                mask = tmp
            diff_logits = diff_logits.masked_fill(~mask, float("-inf"))
        if bool(args.approx_kv_cache):
            diff_past = diff_outputs.past_key_values

        # Sample K sequences.
        def _sample_stage():
            return _sample_k_sequences(
                diff_logits,
                int(k_val),
                temperature=float(args.diffusion_temperature),
                top_k=int(args.diffusion_top_k),
                top_p=float(args.diffusion_top_p),
                generator=generator,
            )

        samples, sample_cuda_ms, sample_wall_ms = _measure_stage(
            None, None, False, _sample_stage
        )
        samples_verify = diff_to_verify[samples]
        if (samples_verify < 0).any():
            verify_unk = verify_tokenizer.unk_token_id
            if verify_unk is None:
                verify_unk = 0
            samples_verify = torch.where(
                samples_verify < 0,
                torch.tensor(int(verify_unk), device=device),
                samples_verify,
            )
        if int(args.debug_nodes) > 0 and prefix_next_logits is not None:
            diff_probs_all = _draft_probs_from_logits(
                diff_logits,
                temperature=float(args.diffusion_temperature),
                top_k=int(args.diffusion_top_k),
                top_p=float(args.diffusion_top_p),
            )
            diff_probs = diff_probs_all[0]
            argmax_ver = int(torch.argmax(prefix_next_logits).item())
            diff_id = int(verify_to_diff[argmax_ver].item()) if verify_to_diff is not None else -1
            p_match = 0.0
            if diff_id >= 0 and diff_id < diff_probs.numel():
                p_match = float(diff_probs[diff_id].item())
            exp_root_hit = 1.0 - (1.0 - p_match) ** float(max(1, int(k_val)))
            diff_argmax = int(torch.argmax(diff_probs).item())
            diff_argmax_tok = diff_tokenizer.convert_ids_to_tokens(diff_argmax)
            ver_tok = verify_tokenizer.convert_ids_to_tokens(argmax_ver)
            print(
                f"[Debug] align p_match={p_match:.4f} exp_root_hit={exp_root_hit:.3f} "
                f"ver_top={repr(ver_tok)} diff_top={repr(diff_argmax_tok)}"
            )
            if p_match < 0.01 and exp_root_hit < 0.1:
                print(
                    "[Warn] align p_match is low; expected acceptance is near-zero. "
                    "Check prefix alignment, special tokens, or draft temperature."
                )
            topk = max(1, int(args.debug_topk))
            ver_topk_ids = torch.topk(prefix_next_logits, k=min(topk, prefix_next_logits.numel())).indices
            diff_topk_ids = torch.topk(diff_probs, k=min(topk, diff_probs.numel())).indices
            overlap_tokens = []
            if diff_to_verify is not None:
                mapped = diff_to_verify[diff_topk_ids]
                valid = mapped >= 0
                overlap = set(ver_topk_ids.tolist()) & set(mapped[valid].tolist())
                for tid in list(overlap)[:topk]:
                    overlap_tokens.append(verify_tokenizer.convert_ids_to_tokens(int(tid)))
                print(
                    f"[Debug] align_topk overlap={len(overlap)}/{topk} tokens={overlap_tokens}"
                )

        # Build tree.
        def _tree_stage():
            return _build_tree_from_sequences(samples_verify)

        (flat_tokens, parents, depths, seq_nodes), tree_cuda_ms, tree_wall_ms = _measure_stage(
            None, None, False, _tree_stage
        )
        max_nodes = int(k_val) * int(args.gamma)
        if len(parents) > max_nodes:
            print(
                f"[Warn] tree_nodes={len(parents)} exceeds k*gamma={max_nodes}; "
                "check draft logits slicing."
            )

        # Verify tree.
        tree_past = None
        tree_prefix_len = None

        def _verify_stage():
            nonlocal tree_past, tree_prefix_len
            if len(parents) == 0:
                return torch.empty((0, diff_logits.shape[-1]), device=device, dtype=diff_logits.dtype)
            flat_tokens_device = flat_tokens.to(device=device)
            want_tree_cache = False
            if tree_kernel != "none":
                tree_mask = _build_tree_mask(parents=parents, device=device)
                _patch_qwen2_tree_kernel(verify_model, tree_mask=tree_mask, kernel=tree_kernel)
                base = getattr(verify_model, "model", verify_model)
                base._tree_kernel_enabled = True
                prefix_len = int(verify_prefix_ids.shape[1])
                tree_prefix_len = int(prefix_len)
                pos_ids = (torch.tensor(depths, device=device, dtype=torch.long) + prefix_len - 1).unsqueeze(0)
                use_past = verify_past
                if bool(args.verify_freeze_cache) and verify_past is not None:
                    use_past = FrozenCache(verify_past, prefix_len=prefix_len)
                with torch.inference_mode(), _verify_sdp_ctx():
                    out_tree = verify_model(
                        input_ids=flat_tokens_device.unsqueeze(0),
                        past_key_values=use_past,
                        position_ids=pos_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                base._tree_kernel_enabled = False
                tree_past = None
                tree_logits = out_tree.logits[:, -len(parents) :, :].squeeze(0)
                if bool(args.tree_kernel_fallback) and bool(args.debug_tree_compare):
                    attn_mask, pos_ids = _build_tree_attention_mask(
                        prefix_len=int(verify_prefix_ids.shape[1]),
                        parents=parents,
                        depths=depths,
                        device=device,
                        dtype=dtype,
                        mask_dtype=str(args.tree_mask_dtype),
                    )
                    full_ids = torch.cat([verify_prefix_ids, flat_tokens_device.unsqueeze(0)], dim=1)
                    with torch.inference_mode(), _verify_sdp_ctx():
                        out_explicit = verify_model(
                            input_ids=full_ids,
                            attention_mask=attn_mask,
                            position_ids=pos_ids,
                            use_cache=False,
                            return_dict=True,
                        )
                    explicit_logits = out_explicit.logits[:, -len(parents) :, :].squeeze(0)
                    max_diff = float((tree_logits - explicit_logits).abs().max().item())
                    if max_diff > float(args.tree_kernel_fallback_tol):
                        print(
                            f"[Warn] tree_kernel mismatch max_abs_diff={max_diff:.4e}; "
                            "falling back to explicit tree attention."
                        )
                        tree_logits = explicit_logits
                return tree_logits

            if tree_mask_mode == "model":
                tree_mask = _build_tree_mask(parents=parents, device=device)
                _patch_tree_mask_support(verify_model)
                prefix_len = int(verify_prefix_ids.shape[1])
                pos_ids = (torch.tensor(depths, device=device, dtype=torch.long) + prefix_len - 1).unsqueeze(0)
                use_past = verify_past
                if bool(args.verify_freeze_cache) and verify_past is not None:
                    use_past = FrozenCache(verify_past, prefix_len=prefix_len)
                verify_model.model.tree_mask = tree_mask
                with torch.inference_mode(), _verify_sdp_ctx():
                    out_tree = verify_model(
                        input_ids=flat_tokens_device.unsqueeze(0),
                        past_key_values=use_past,
                        position_ids=pos_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                verify_model.model.tree_mask = None
                tree_past = None
                return out_tree.logits[:, -len(parents) :, :].squeeze(0)

            attn_mask, pos_ids = _build_tree_attention_mask(
                prefix_len=int(verify_prefix_ids.shape[1]),
                parents=parents,
                depths=depths,
                device=device,
                dtype=dtype,
                mask_dtype=str(args.tree_mask_dtype),
            )
            full_ids = torch.cat([verify_prefix_ids, flat_tokens_device.unsqueeze(0)], dim=1)
            with torch.inference_mode(), _verify_sdp_ctx():
                out_tree = verify_model(
                    input_ids=full_ids,
                    attention_mask=attn_mask,
                    position_ids=pos_ids,
                    use_cache=False,
                    return_dict=True,
                )
            return out_tree.logits[:, -len(parents) :, :].squeeze(0)

        tree_logits, verify_cuda_ms, verify_wall_ms = _measure_stage(
            start_event, end_event, bool(args.sync_timing), _verify_stage
        )
        prefix_cuda_ms = 0.0
        prefix_wall_ms = 0.0
        if prefix_next_logits is None:
            with torch.inference_mode(), _verify_sdp_ctx():
                out_next = verify_model(input_ids=verify_prefix_ids, use_cache=False, return_dict=True)
            prefix_next_logits = out_next.logits[:, -1, :].squeeze(0)

        prefix_next_logits_used = prefix_next_logits
        accept_argmax_prefix = None
        accept_root_tokens = None
        accept_prefix_group = None
        if samples_verify.numel() > 0 and prefix_next_logits_used.numel() > 0:
            accept_argmax_prefix = int(torch.argmax(prefix_next_logits_used).item())
            accept_root_tokens = samples_verify[:, 0]
            argmax_tok = verify_tokenizer.convert_ids_to_tokens(int(accept_argmax_prefix))
            accept_prefix_group = verify_token_groups.get(argmax_tok)

        # Accept tokens.
        def _accept_stage():
            if str(args.accept_mode) == "leviathan":
                return _accept_leviathan(
                    samples_verify,
                    seq_nodes,
                    tree_logits,
                    prefix_next_logits_used,
                    diff_logits,
                    diff_to_verify,
                    int(verify_size),
                    diffusion_temperature=float(args.diffusion_temperature),
                    diffusion_top_k=int(args.diffusion_top_k),
                    diffusion_top_p=float(args.diffusion_top_p),
                    verify_temperature=float(args.verify_temperature),
                    verify_top_k=int(args.verify_top_k),
                    verify_top_p=float(args.verify_top_p),
                    generator=generator,
                )
            return _accept_greedy(
                samples_verify,
                seq_nodes,
                tree_logits,
                prefix_next_logits_used,
                accept_prefix_group,
                verify_temperature=float(args.verify_temperature),
                verify_top_k=int(args.verify_top_k),
                verify_top_p=float(args.verify_top_p),
                generator=generator,
            )

        (accepted, accept_len, best_idx, replacement), accept_cuda_ms, accept_wall_ms = _measure_stage(
            None, None, False, _accept_stage
        )

        if bool(args.debug_tree_compare) and samples_verify.numel() > 0:
            cmp_idx = min(int(best_idx), int(samples_verify.shape[0] - 1))
            path = samples_verify[cmp_idx].unsqueeze(0)
            full_ids = torch.cat([verify_prefix_ids, path], dim=1)
            with torch.inference_mode(), _verify_sdp_ctx():
                out_ar = verify_model(input_ids=full_ids, use_cache=False, return_dict=True)
            ar_logits = out_ar.logits.squeeze(0)
            prefix_len = int(verify_prefix_ids.shape[1])
            if prefix_next_logits_used is not None and ar_logits.shape[0] > 0:
                ar_pos = max(0, prefix_len - 1)
                if ar_pos < ar_logits.shape[0]:
                    diff = (prefix_next_logits_used - ar_logits[ar_pos]).abs().max().item()
                    print(f"[Debug] tree_compare prefix max_abs_diff={diff:.4e}")
            max_depth = min(int(args.debug_tree_compare_depths), int(path.shape[1]) - 1)
            for d in range(max_depth):
                node_idx = seq_nodes[cmp_idx][d]
                ar_pos = prefix_len + d
                if ar_pos >= ar_logits.shape[0]:
                    break
                diff = (tree_logits[node_idx] - ar_logits[ar_pos]).abs().max().item()
                print(f"[Debug] tree_compare depth={d+1} max_abs_diff={diff:.4e}")

        tokens_to_add = accepted + [replacement]
        remaining = max_new_tokens - total_generated
        if remaining < len(tokens_to_add):
            tokens_to_add = tokens_to_add[:remaining]
        add_tensor = torch.tensor(tokens_to_add, device=device, dtype=torch.long).unsqueeze(0)
        tokens_to_add_diff = tokens_to_add
        if bool(args.sync_prefix_full):
            tokens_to_add_diff = []
        elif bool(args.sync_prefix_text):
            new_text = verify_tokenizer.decode(tokens_to_add, skip_special_tokens=False)
            diff_ids = diff_tokenizer(new_text, return_tensors="pt", add_special_tokens=False).input_ids
            tokens_to_add_diff = diff_ids.squeeze(0).to(device=device, dtype=torch.long).tolist()
        else:
            mapped = verify_to_diff[add_tensor.squeeze(0)]
            if (mapped < 0).any():
                new_text = verify_tokenizer.decode(tokens_to_add, skip_special_tokens=False)
                diff_ids = diff_tokenizer(new_text, return_tensors="pt", add_special_tokens=False).input_ids
                tokens_to_add_diff = diff_ids.squeeze(0).to(device=device, dtype=torch.long).tolist()
            else:
                tokens_to_add_diff = mapped.tolist()

        def _update_stage():
            nonlocal verify_past
            nonlocal tree_cache_prefix_len
            if not tokens_to_add:
                return
            if reuse_tree_enabled and accept_len > 0 and tree_past is not None:
                accepted_nodes = seq_nodes[best_idx][:accept_len]
                verify_past = _reuse_tree_cache_for_accepted(
                    verify_cache=verify_past,
                    tree_cache=tree_past,
                    prefix_len=int(verify_prefix_ids.shape[1]),
                    accepted_nodes=accepted_nodes,
                )
                has_replacement = len(tokens_to_add) > accept_len
                update_tokens = [int(replacement)] if has_replacement else []
            else:
                if reuse_tree_enabled and tree_prefix_len is not None:
                    _truncate_cache(verify_past, int(tree_prefix_len))
                update_tokens = tokens_to_add
            if not update_tokens:
                if tree_cache is not None and len(tokens_to_add) > 0:
                    start = int(step_prefix_len)
                    _truncate_cache(tree_cache, start)
                    copied = _copy_cache_range(
                        dst_cache=tree_cache,
                        src_cache=verify_past,
                        start=start,
                        length=len(tokens_to_add),
                    )
                    if copied:
                        tree_cache_prefix_len = int(start + len(tokens_to_add))
                return
            update_tensor = torch.tensor(update_tokens, device=device, dtype=torch.long).unsqueeze(0)
            with torch.inference_mode(), _verify_sdp_ctx():
                out = verify_model(
                    input_ids=update_tensor,
                    past_key_values=verify_past,
                    use_cache=True,
                    return_dict=True,
                )
            verify_past = out.past_key_values
            nonlocal prefix_next_logits
            prefix_next_logits = out.logits[:, -1, :].squeeze(0)
            if tree_cache is not None:
                start = int(step_prefix_len)
                _truncate_cache(tree_cache, start)
                copied = _copy_cache_range(
                    dst_cache=tree_cache,
                    src_cache=verify_past,
                    start=start,
                    length=len(tokens_to_add),
                )
                if copied:
                    tree_cache_prefix_len = int(start + len(tokens_to_add))

        _, update_cuda_ms, update_wall_ms = _measure_stage(
            start_event, end_event, bool(args.sync_timing), _update_stage
        )

        verify_prefix_ids = torch.cat([verify_prefix_ids, add_tensor], dim=1)
        if bool(args.sync_prefix_full):
            full_text = verify_tokenizer.decode(
                verify_prefix_ids[0].tolist(), skip_special_tokens=False
            )
            diff_prefix_ids = diff_tokenizer(
                full_text, return_tensors="pt", add_special_tokens=False
            ).input_ids.to(device=device, dtype=torch.long)
        elif tokens_to_add_diff:
            diff_add_tensor = torch.tensor(tokens_to_add_diff, device=device, dtype=torch.long).unsqueeze(0)
            diff_prefix_ids = torch.cat([diff_prefix_ids, diff_add_tensor], dim=1)
        total_generated += len(tokens_to_add)
        total_steps += 1
        total_accept_len += int(accept_len)
        total_added += len(tokens_to_add)
        total_nodes += len(parents)

        if int(args.debug_prefix_every) > 0 and (step % max(1, int(args.debug_prefix_every)) == 0):
            _debug_prefix_alignment(
                verify_tokenizer=verify_tokenizer,
                diff_tokenizer=diff_tokenizer,
                verify_prefix_ids=verify_prefix_ids,
                diff_prefix_ids=diff_prefix_ids,
                verify_to_diff=verify_to_diff,
                max_tail=int(args.debug_prefix_tail),
            )

        if step >= int(args.warmup_steps) and (step % max(1, int(args.log_every)) == 0):
            print(f"[Step {step:4d}] prefix_len={step_prefix_len} nodes={len(parents)}")
            print(
                f"  [Diff] forward_ms={diff_cuda_ms:.2f} wall_ms={diff_wall_ms:.2f}"
                f"  [Sample] wall_ms={sample_wall_ms:.2f}"
                f"  [Tree] wall_ms={tree_wall_ms:.2f}"
            )
            print(
                f"  [Verify] ms={verify_cuda_ms:.2f} wall_ms={verify_wall_ms:.2f}"
                f"  [Prefix] ms={prefix_cuda_ms:.2f} wall_ms={prefix_wall_ms:.2f}"
                f"  [Accept] wall_ms={accept_wall_ms:.2f}"
                f"  [Update] ms={update_cuda_ms:.2f} wall_ms={update_wall_ms:.2f}"
            )
            print(
                f"  [Accept] mode={args.accept_mode} best_idx={best_idx} "
                f"accept_len={accept_len} add={len(tokens_to_add)}"
            )
            if len(tree_logits) > 0:
                _log_topk(tree_logits, verify_tokenizer, "tree", int(args.debug_topk), int(args.debug_nodes))
                root_idx = _root_indices(parents)
                if root_idx:
                    root_logits = tree_logits[root_idx]
                    _log_topk(
                        root_logits,
                        verify_tokenizer,
                        "tree_root",
                        int(args.debug_topk),
                        min(int(args.debug_nodes), len(root_idx)),
                    )
            if int(args.debug_nodes) > 0 and prefix_next_logits_used.numel() > 0:
                _log_topk(
                    prefix_next_logits_used.unsqueeze(0),
                    verify_tokenizer,
                    "verifier_next",
                    int(args.debug_topk),
                    1,
                )
            if int(args.debug_nodes) > 0 and accept_root_tokens is not None:
                _log_root_candidates(
                    accept_root_tokens,
                    int(accept_argmax_prefix),
                    verify_tokenizer,
                    int(args.debug_topk),
                    accept_prefix_group,
                )
                if prefix_next_logits_used.numel() > 0:
                    topk = min(int(args.debug_topk), int(prefix_next_logits_used.numel()))
                    topk_ids = torch.topk(prefix_next_logits_used, k=topk).indices
                    topk_set = set(topk_ids.tolist())
                    root_topk_match = sum(
                        1 for t in accept_root_tokens.tolist() if int(t) in topk_set
                    )
                    print(
                        f"[Debug] root_topk_match={root_topk_match}/{accept_root_tokens.numel()} topk={topk}"
                    )

        if int(args.debug_nodes) > 0 and step >= int(args.warmup_steps):
            _log_topk(diff_logits.squeeze(0), diff_tokenizer, "diff", int(args.debug_topk), 1)

        if total_generated >= max_new_tokens:
            break

    elapsed = time.perf_counter() - t_start
    toks_per_sec = total_generated / max(elapsed, 1e-6)
    print(f"[Info] generated={total_generated} tokens in {elapsed:.2f}s ({toks_per_sec:.2f} tok/s)")
    if total_steps > 0:
        avg_accept = total_accept_len / float(total_steps)
        avg_add = total_added / float(total_steps)
        avg_nodes = total_nodes / float(total_steps)
        accept_rate = total_accept_len / float(max(1, total_steps * int(args.gamma)))
        print(
            "[Summary] "
            f"steps={total_steps} avg_accept_len={avg_accept:.2f} "
            f"avg_add={avg_add:.2f} accept_rate={accept_rate:.3f} "
            f"avg_nodes={avg_nodes:.1f} expected_streak={avg_accept:.2f}"
        )
    final_text = verify_tokenizer.decode(verify_prefix_ids[0].tolist(), skip_special_tokens=False)
    print("[Final] text=")
    print(final_text)


if __name__ == "__main__":
    main()
