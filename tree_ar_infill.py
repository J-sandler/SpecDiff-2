#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import math
import os
import time
import types
import sys
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, StaticCache


def _get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _ensure_dynamic_cache(past_key_values):
    if past_key_values is None:
        return None
    if isinstance(past_key_values, Cache):
        return past_key_values
    if isinstance(past_key_values, DynamicCache):
        return past_key_values
    if hasattr(DynamicCache, "from_legacy_cache"):
        try:
            return DynamicCache.from_legacy_cache(past_key_values)
        except Exception:
            return past_key_values
    return past_key_values


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
        try:
            return self.base_cache.key_cache[layer_idx].shape[-2]
        except Exception:
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
                try:
                    prefix_len = base.get_seq_length(layer_idx)
                except Exception:
                    prefix_len = None
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

    def batch_repeat_interleave(self, repeats: int):
        out = PreallocCache.__new__(PreallocCache)
        Cache.__init__(out)
        out.batch_size = self.batch_size * repeats
        out.max_cache_len = self.max_cache_len
        out.num_key_value_heads = self.num_key_value_heads
        out.head_dim = self.head_dim
        out.key_cache = [t.repeat_interleave(repeats, dim=0) for t in self.key_cache]
        out.value_cache = [t.repeat_interleave(repeats, dim=0) for t in self.value_cache]
        out.current_length = self.current_length
        return out


def _expand_past(past_key_values, repeats: int):
    past_key_values = _ensure_dynamic_cache(past_key_values)
    if past_key_values is None:
        return None
    if hasattr(past_key_values, "batch_repeat_interleave"):
        if isinstance(past_key_values, PreallocCache):
            return past_key_values.batch_repeat_interleave(repeats)
        expanded = DynamicCache()
        expanded._seen_tokens = getattr(past_key_values, "_seen_tokens", 0)
        expanded.key_cache = []
        expanded.value_cache = []
        for k, v in zip(past_key_values.key_cache, past_key_values.value_cache):
            if isinstance(k, list) or isinstance(v, list):
                expanded.key_cache.append([])
                expanded.value_cache.append([])
            else:
                expanded.key_cache.append(k.repeat_interleave(repeats, dim=0))
                expanded.value_cache.append(v.repeat_interleave(repeats, dim=0))
        return expanded
    out = []
    for layer in past_key_values:
        k, v = layer
        out.append((k.repeat_interleave(repeats, dim=0), v.repeat_interleave(repeats, dim=0)))
    return tuple(out)


def _select_branch_tokens(
    logits: torch.Tensor,
    branching: int,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    method: str,
) -> torch.Tensor:
    if branching <= 0:
        raise ValueError("branching must be >= 1")
    if method == "topk":
        k = max(branching, 1)
        return torch.topk(logits, k=k, dim=-1).indices[:, :branching]
    if temperature <= 0:
        return torch.topk(logits, k=branching, dim=-1).indices
    scaled = logits / float(temperature)
    scaled = scaled.to(dtype=torch.float32)
    if top_k > 0:
        k = min(int(top_k), scaled.shape[-1])
        vals, idx = torch.topk(scaled, k=k, dim=-1)
        if top_p < 1.0:
            probs = torch.softmax(vals, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumprobs > float(top_p)
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            picks = torch.multinomial(sorted_probs, num_samples=branching, replacement=False)
            chosen = sorted_idx.gather(dim=-1, index=picks)
            return idx.gather(dim=-1, index=chosen)
        probs = torch.softmax(vals, dim=-1)
        picks = torch.multinomial(probs, num_samples=branching, replacement=False)
        return idx.gather(dim=-1, index=picks)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(scaled, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = cumprobs > float(top_p)
        cutoff[..., 0] = False
        probs = probs.masked_fill(cutoff, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        picks = torch.multinomial(probs, num_samples=branching, replacement=False)
        return sorted_idx.gather(dim=-1, index=picks)
    probs = torch.softmax(scaled, dim=-1)
    return torch.multinomial(probs, num_samples=branching, replacement=False)


def _top1_stats(logits: torch.Tensor) -> Tuple[float, float]:
    top2 = torch.topk(logits, k=2, dim=-1).values
    lse = torch.logsumexp(logits, dim=-1)
    top1_prob = torch.exp(top2[..., 0] - lse)
    gap = top2[..., 0] - top2[..., 1]
    return float(top1_prob.mean().item()), float(gap.mean().item())


def _build_tree_from_prefix(
    *,
    model,
    prefix_ids: torch.Tensor,
    prefix_past,
    prefix_logits: torch.Tensor,
    gamma: int,
    branching: int,
    temperature: float,
    top_k: int,
    top_p: float,
    select_method: str,
) -> Tuple[List[torch.Tensor], List[Tuple[float, float]], List[torch.Tensor], List[torch.Tensor]]:
    tree_tokens_by_depth: List[torch.Tensor] = []
    stats_by_depth: List[Tuple[float, float]] = []
    logits_by_depth: List[torch.Tensor] = []
    logp_by_depth: List[torch.Tensor] = []

    frontier_logits = prefix_logits
    frontier_past = prefix_past

    for _depth in range(gamma):
        tokens = _select_branch_tokens(
            frontier_logits,
            branching,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            method=select_method,
        )
        tree_tokens_by_depth.append(tokens)
        logits_by_depth.append(frontier_logits)
        stats_by_depth.append(_top1_stats(frontier_logits))

        log_probs = torch.log_softmax(frontier_logits.to(dtype=torch.float32), dim=-1)
        sel_logp = log_probs.gather(dim=-1, index=tokens)
        logp_by_depth.append(sel_logp.detach())

        flat_tokens = tokens.reshape(-1)
        expanded_past = _expand_past(frontier_past, branching)

        outputs = model(
            input_ids=flat_tokens.unsqueeze(-1),
            past_key_values=expanded_past,
            use_cache=True,
            return_dict=True,
        )
        frontier_past = _ensure_dynamic_cache(outputs.past_key_values)
        frontier_logits = outputs.logits[:, -1, :]

    return tree_tokens_by_depth, stats_by_depth, logits_by_depth, logp_by_depth


def _flatten_tree(tokens_by_depth: Sequence[torch.Tensor]) -> torch.Tensor:
    flat_tokens: List[int] = []
    for tokens in tokens_by_depth:
        tokens = tokens.detach().cpu()
        flat_tokens.extend(tokens.reshape(-1).tolist())
    return torch.tensor(flat_tokens, dtype=torch.long)


def _build_tree_structure(branching: int, gamma: int) -> Tuple[List[int], List[int]]:
    if branching <= 0 or gamma <= 0:
        return [], []
    parents: List[int] = []
    depths: List[int] = []
    prev_start = 0
    prev_count = 1  # root (not stored in parents list)
    total_nodes = 0
    for depth in range(1, gamma + 1):
        num_parents = prev_count
        for p in range(num_parents):
            for _ in range(branching):
                if depth == 1:
                    parents.append(-1)
                else:
                    parents.append(prev_start + p)
                depths.append(depth)
        total_nodes += num_parents * branching
        prev_start = total_nodes - (num_parents * branching)
        prev_count = num_parents * branching
    return parents, depths


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


def _build_tree_mask(
    *,
    parents: Sequence[int],
    device,
) -> torch.Tensor:
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
    if not hasattr(base, "_tree_kernel_enabled"):
        base._tree_kernel_enabled = False
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


def _log_topk(
    *,
    logits: torch.Tensor,
    tokenizer,
    label: str,
    topk: int,
    max_rows: int,
) -> None:
    if topk <= 0 or max_rows <= 0:
        return
    rows = min(int(max_rows), logits.shape[0])
    topk_vals, topk_idx = torch.topk(logits[:rows], k=topk, dim=-1)
    probs = torch.softmax(topk_vals.to(dtype=torch.float32), dim=-1)
    for i in range(rows):
        tokens = [tokenizer.decode([int(t)]) for t in topk_idx[i].tolist()]
        probs_i = [float(p) for p in probs[i].tolist()]
        print(f"[Qual][{label}] node={i} topk={list(zip(tokens, probs_i))}")


def _step_timing(start_event, end_event, sync: bool) -> float:
    if start_event is None or end_event is None:
        return 0.0
    end_event.record()
    if sync:
        torch.cuda.synchronize()
    return float(start_event.elapsed_time(end_event))


def _load_eagle_qwen2(
    *,
    model_name: str,
    repo_path: str,
    dtype: torch.dtype,
    device: torch.device,
    revision: Optional[str],
    local_files_only: bool,
    attn_implementation: str,
):
    repo_path = os.path.expanduser(repo_path)
    if not os.path.isabs(repo_path):
        repo_path = os.path.join(os.path.dirname(__file__), repo_path)
    if repo_path not in sys.path:
        sys.path.insert(0, repo_path)
    from eagle.model.modeling_qwen2_kv import Qwen2ForCausalLM as EagleQwen2ForCausalLM

    eagle_kwargs: Dict[str, object] = {
        "torch_dtype": dtype,
        "revision": revision,
        "local_files_only": bool(local_files_only),
    }
    if attn_implementation != "auto":
        eagle_kwargs["attn_implementation"] = attn_implementation

    model = EagleQwen2ForCausalLM.from_pretrained(model_name, **eagle_kwargs)
    model.to(device)
    model.eval()
    return model


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
            if past_key_value is None or not hasattr(past_key_value, "key_cache"):
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

            base_cache = past_key_value
            if isinstance(past_key_value, FrozenCache):
                base_cache = past_key_value.base_cache
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

            k_tree = key_states
            v_tree = value_states
            scores_tree = torch.matmul(query_states.float(), k_tree.transpose(2, 3).float()) * scale
            scores_tree = scores_tree.masked_fill(tree_mask_local == 0, float("-inf"))
            lse_tree = torch.logsumexp(scores_tree, dim=-1)
            attn_tree = torch.softmax(scores_tree, dim=-1, dtype=torch.float32)
            out_tree = torch.matmul(attn_tree, v_tree.float()).to(dtype=query_states.dtype)

            total_lse = torch.logaddexp(lse_prefix, lse_tree)
            w_prefix = torch.exp(lse_prefix - total_lse).unsqueeze(-1)
            w_tree = torch.exp(lse_tree - total_lse).unsqueeze(-1)
            attn_output = out_prefix * w_prefix + out_tree * w_tree

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, q_len, self.hidden_size)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        attn.forward = types.MethodType(_forward_with_tree, attn)
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


def main() -> None:
    p = argparse.ArgumentParser(description="Tree-attention infilling for AR models (Qwen2.5).")
    p.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--draft_model_name", type=str, default="")
    p.add_argument("--draft_max_prefix_len", type=int, default=0)
    p.add_argument("--draft_cache_type", type=str, default="dynamic", choices=("dynamic", "static", "prealloc"))
    p.add_argument("--draft_max_cache_len", type=int, default=0)
    p.add_argument("--revision", type=str, default=None)
    p.add_argument("--local_files_only", action="store_true", default=False)
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--gamma", type=int, default=6)
    p.add_argument("--branching", type=int, default=2)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--method", type=str, default="both", choices=("kv_cache", "tree_attention", "both"))
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--select_method", type=str, default="topk", choices=("topk", "sample"))
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--log_kv", action="store_true", default=False)
    p.add_argument("--warmup_steps", type=int, default=1)
    p.add_argument("--sync_timing", action="store_true", default=True)
    p.add_argument("--debug_topk", type=int, default=5)
    p.add_argument("--debug_nodes", type=int, default=2)
    p.add_argument("--attn_implementation", type=str, default="auto")
    p.add_argument("--draft_attn_implementation", type=str, default="auto")
    p.add_argument("--verify_attn_implementation", type=str, default="sdpa")
    p.add_argument("--tree_mask_dtype", type=str, default="bool", choices=("bool", "float"))
    p.add_argument("--verify_max_prefix_len", type=int, default=0)
    p.add_argument("--verify_cache_type", type=str, default="dynamic", choices=("dynamic", "static", "prealloc"))
    p.add_argument("--verify_max_cache_len", type=int, default=0)
    p.add_argument("--verify_freeze_cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--benchmark_eagle_steps", type=int, default=0)
    p.add_argument("--benchmark_eagle_repo", type=str, default="EAGLE")
    p.add_argument("--tree_kernel", type=str, default="flash_prefix", choices=("none", "flash_prefix"))
    p.add_argument("--tree_kernel_warmup", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--tree_kernel_warmup_prefix", type=int, default=0)
    p.add_argument(
        "--tree_mask_mode",
        type=str,
        default="model",
        choices=("model", "explicit"),
        help="model=inject tree mask into model; explicit=build full attention mask",
    )

    args = p.parse_args()

    torch.manual_seed(int(args.seed))
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    dtype = _get_dtype(args.dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        revision=args.revision,
        local_files_only=bool(args.local_files_only),
    )
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: Dict[str, object] = {
        "torch_dtype": dtype,
        "revision": args.revision,
        "local_files_only": bool(args.local_files_only),
    }
    if args.attn_implementation != "auto":
        model_kwargs["attn_implementation"] = str(args.attn_implementation)

    model = AutoModelForCausalLM.from_pretrained(args.model_name, **model_kwargs)
    model.to(device)
    model.eval()

    draft_model = model
    draft_name = str(args.draft_model_name).strip()
    draft_impl = str(args.draft_attn_implementation)
    if not draft_name:
        draft_name = str(args.model_name)
    if draft_impl == "auto":
        draft_impl = str(args.attn_implementation)
    if draft_name != str(args.model_name) or draft_impl != str(args.attn_implementation):
        draft_kwargs = dict(model_kwargs)
        if draft_impl != "auto":
            draft_kwargs["attn_implementation"] = draft_impl
        draft_model = AutoModelForCausalLM.from_pretrained(draft_name, **draft_kwargs)
        draft_model.to(device)
        draft_model.eval()

    model_verify = model
    verify_impl = str(args.verify_attn_implementation)
    if args.method in ("tree_attention", "both"):
        if verify_impl in ("auto", "flash_attention_2"):
            print("[Info] Tree-attention mask is incompatible with flash_attention_2; using sdpa for verification.")
            verify_impl = "sdpa"
        if verify_impl != str(args.attn_implementation):
            verify_kwargs = dict(model_kwargs)
            verify_kwargs["attn_implementation"] = verify_impl
            model_verify = AutoModelForCausalLM.from_pretrained(args.model_name, **verify_kwargs)
            model_verify.to(device)
            model_verify.eval()

    tree_mask_dtype = str(args.tree_mask_dtype)
    model_type = getattr(model_verify.config, "model_type", "")
    if tree_mask_dtype == "bool" and str(model_type).startswith("qwen2"):
        tree_mask_dtype = "float"
        print("[Info] Qwen2 attention mask requires float; forcing tree_mask_dtype=float.")

    enc = tokenizer(args.prompt, return_tensors="pt")
    prefix_ids_full = enc.input_ids.to(device=device, dtype=torch.long)
    prefix_ids_draft = prefix_ids_full
    prefix_ids_verify = prefix_ids_full

    eagle_model = None
    eagle_prefix_past = None
    prefix_ids_eagle = prefix_ids_full
    if int(args.benchmark_eagle_steps) > 0:
        try:
            eagle_model = _load_eagle_qwen2(
                model_name=str(args.model_name),
                repo_path=str(args.benchmark_eagle_repo),
                dtype=dtype,
                device=device,
                revision=args.revision,
                local_files_only=bool(args.local_files_only),
                attn_implementation=str(args.verify_attn_implementation),
            )
        except Exception as exc:
            print(f"[Warn] Failed to load EAGLE model: {exc}")
            eagle_model = None

    start_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
    end_event = torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None

    # Prime cache on prompt for draft.
    prefix_past = None
    if str(args.draft_cache_type) == "static":
        if int(args.draft_max_cache_len) <= 0:
            raise ValueError("--draft_max_cache_len must be > 0 when using static draft cache")
        prefix_past = StaticCache(
            config=draft_model.config,
            batch_size=1,
            max_cache_len=int(args.draft_max_cache_len),
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            out = draft_model(
                input_ids=prefix_ids_full,
                past_key_values=prefix_past,
                use_cache=True,
                return_dict=True,
            )
        prefix_past = out.past_key_values
    elif str(args.draft_cache_type) == "prealloc":
        if int(args.draft_max_cache_len) <= 0:
            raise ValueError("--draft_max_cache_len must be > 0 when using prealloc draft cache")
        prefix_past = PreallocCache(
            config=draft_model.config,
            batch_size=1,
            max_cache_len=int(args.draft_max_cache_len),
            device=device,
            dtype=dtype,
        )
        with torch.inference_mode():
            out = draft_model(
                input_ids=prefix_ids_full,
                past_key_values=prefix_past,
                use_cache=True,
                return_dict=True,
            )
        prefix_past = out.past_key_values
    else:
        with torch.inference_mode():
            out = draft_model(input_ids=prefix_ids_full, use_cache=True, return_dict=True)
        prefix_past = _ensure_dynamic_cache(out.past_key_values)
    prefix_logits = out.logits[:, -1, :]
    if int(args.draft_max_prefix_len) > 0 and prefix_ids_full.shape[1] > int(args.draft_max_prefix_len):
        if hasattr(prefix_past, "crop"):
            prefix_past.crop(int(args.draft_max_prefix_len))
            prefix_ids_draft = prefix_ids_full[:, -int(args.draft_max_prefix_len) :]
            print(f"[Draft] cropped prefix to {int(args.draft_max_prefix_len)} tokens")
        else:
            print("[Draft] crop requested but cache does not support crop; skipping.")

    prefix_past_verify = None
    if args.method in ("tree_attention", "both"):
        if str(args.verify_cache_type) == "static":
            if int(args.verify_max_cache_len) <= 0:
                raise ValueError("--verify_max_cache_len must be > 0 when using static verify cache")
            prefix_past_verify = StaticCache(
                config=model_verify.config,
                batch_size=1,
                max_cache_len=int(args.verify_max_cache_len),
                device=device,
                dtype=dtype,
            )
            with torch.inference_mode():
                out_verify = model_verify(
                    input_ids=prefix_ids_full,
                    past_key_values=prefix_past_verify,
                    use_cache=True,
                    return_dict=True,
                )
            prefix_past_verify = out_verify.past_key_values
        elif str(args.verify_cache_type) == "prealloc":
            if int(args.verify_max_cache_len) <= 0:
                raise ValueError("--verify_max_cache_len must be > 0 when using prealloc verify cache")
            prefix_past_verify = PreallocCache(
                config=model_verify.config,
                batch_size=1,
                max_cache_len=int(args.verify_max_cache_len),
                device=device,
                dtype=dtype,
            )
            with torch.inference_mode():
                out_verify = model_verify(
                    input_ids=prefix_ids_full,
                    past_key_values=prefix_past_verify,
                    use_cache=True,
                    return_dict=True,
                )
            prefix_past_verify = out_verify.past_key_values
        else:
            with torch.inference_mode():
                out_verify = model_verify(input_ids=prefix_ids_full, use_cache=True, return_dict=True)
            prefix_past_verify = _ensure_dynamic_cache(out_verify.past_key_values)
        if int(args.verify_max_prefix_len) > 0 and prefix_ids_full.shape[1] > int(args.verify_max_prefix_len):
            if hasattr(prefix_past_verify, "crop"):
                prefix_past_verify.crop(int(args.verify_max_prefix_len))
                prefix_ids_verify = prefix_ids_full[:, -int(args.verify_max_prefix_len) :]
                print(f"[Verify] cropped prefix to {int(args.verify_max_prefix_len)} tokens (approx)")
            else:
                print("[Verify] crop requested but cache does not support crop; skipping.")

    if eagle_model is not None:
        with torch.inference_mode():
            out_eagle = eagle_model(input_ids=prefix_ids_full, use_cache=True, return_dict=True)
        eagle_prefix_past = _ensure_dynamic_cache(out_eagle.past_key_values)

    parents: List[int] = []
    depths: List[int] = []
    depth_offsets = None
    tree_mask = None
    if args.method in ("tree_attention", "both"):
        parents, depths = _build_tree_structure(int(args.branching), int(args.gamma))
        if str(args.tree_mask_mode) == "model":
            tree_mask = _build_tree_mask(parents=parents, device=device)
            _patch_tree_mask_support(model_verify)
            depth_offsets = torch.tensor(depths, device=device, dtype=torch.long) - 1
        if str(args.tree_kernel) != "none":
            if str(args.tree_mask_mode) != "model":
                print("[Info] tree_kernel requires tree_mask_mode=model; forcing.")
            tree_mask = _build_tree_mask(parents=parents, device=device)
            depth_offsets = torch.tensor(depths, device=device, dtype=torch.long) - 1
            _patch_qwen2_tree_kernel(model_verify, tree_mask=tree_mask, kernel=str(args.tree_kernel))
            if bool(args.tree_kernel_warmup):
                warm_prefix = int(args.tree_kernel_warmup_prefix)
                if warm_prefix <= 0:
                    warm_prefix = int(prefix_ids_full.shape[1]) + int(args.steps) * int(args.gamma)
                if int(args.verify_max_cache_len) > 0:
                    warm_prefix = min(warm_prefix, int(args.verify_max_cache_len))
                _warmup_tree_kernel(
                    model=model_verify,
                    tree_len=len(parents),
                    prefix_len=warm_prefix,
                    dtype=dtype,
                    device=device,
                )

    for step in range(int(args.steps)):
        if step % max(1, int(args.log_every)) == 0 and step >= int(args.warmup_steps):
            print(f"[Step {step:4d}] prefix_len={prefix_ids_full.shape[1]}")

        # Build tree with KV cache.
        kv_ms = 0.0
        kv_wall_ms = 0.0
        if bool(args.log_kv):
            if start_event is not None:
                start_event.record()
            t0 = time.perf_counter()
        tree_tokens_by_depth, stats_by_depth, logits_by_depth, logp_by_depth = _build_tree_from_prefix(
            model=draft_model,
            prefix_ids=prefix_ids_draft,
            prefix_past=prefix_past,
            prefix_logits=prefix_logits,
            gamma=int(args.gamma),
            branching=int(args.branching),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            select_method=str(args.select_method),
        )
        if bool(args.log_kv):
            kv_ms = _step_timing(start_event, end_event, bool(args.sync_timing))
            kv_wall_ms = (time.perf_counter() - t0) * 1000.0

            if step % max(1, int(args.log_every)) == 0 and step >= int(args.warmup_steps):
                p_mean, gap_mean = stats_by_depth[-1]
                print(
                    f"  [KV] tree_ms={kv_ms:.2f} wall_ms={kv_wall_ms:.2f} "
                    f"depth{args.gamma}_top1_prob={p_mean:.3f} gap={gap_mean:.3f}"
                )

        if args.method in ("tree_attention", "both"):
            flat_tokens = _flatten_tree(tree_tokens_by_depth)
            flat_tokens = flat_tokens.to(device=device)

            if start_event is not None:
                start_event.record()
            t1 = time.perf_counter()
            with torch.inference_mode():
                verify_past = prefix_past_verify
                if bool(args.verify_freeze_cache) and prefix_past_verify is not None:
                    verify_past = FrozenCache(prefix_past_verify, prefix_len=int(prefix_ids_verify.shape[1]))
                if str(args.tree_kernel) != "none":
                    prefix_len_verify = int(prefix_ids_verify.shape[1])
                    pos_ids = (depth_offsets + prefix_len_verify).unsqueeze(0)
                    model_verify.model._tree_kernel_enabled = True
                    out_tree = model_verify(
                        input_ids=flat_tokens.unsqueeze(0),
                        past_key_values=verify_past,
                        position_ids=pos_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    model_verify.model._tree_kernel_enabled = False
                elif str(args.tree_mask_mode) == "model":
                    prefix_len_verify = int(prefix_ids_verify.shape[1])
                    pos_ids = (depth_offsets + prefix_len_verify).unsqueeze(0)
                    model_verify.model.tree_mask = tree_mask
                    out_tree = model_verify(
                        input_ids=flat_tokens.unsqueeze(0),
                        past_key_values=verify_past,
                        position_ids=pos_ids,
                        use_cache=False,
                        return_dict=True,
                    )
                    model_verify.model.tree_mask = None
                else:
                    attn_mask, pos_ids = _build_tree_attention_mask(
                        prefix_len=int(prefix_ids_verify.shape[1]),
                        parents=parents,
                        depths=depths,
                        device=device,
                        dtype=dtype,
                        mask_dtype=tree_mask_dtype,
                    )
                    full_ids = torch.cat([prefix_ids_verify, flat_tokens.unsqueeze(0)], dim=1)
                    out_tree = model_verify(
                        input_ids=full_ids,
                        attention_mask=attn_mask,
                        position_ids=pos_ids,
                        use_cache=False,
                        return_dict=True,
                    )
            ta_ms = _step_timing(start_event, end_event, bool(args.sync_timing))
            ta_wall_ms = (time.perf_counter() - t1) * 1000.0

            tree_logits = out_tree.logits[:, -len(parents) :, :].squeeze(0)
            top1_prob, gap = _top1_stats(tree_logits)
            if step % max(1, int(args.log_every)) == 0 and step >= int(args.warmup_steps):
                print(
                    f"  [TreeAttn] verify_ms={ta_ms:.2f} wall_ms={ta_wall_ms:.2f} "
                    f"top1_prob={top1_prob:.3f} gap={gap:.3f} nodes={len(parents)}"
                )
            _log_topk(
                logits=tree_logits,
                tokenizer=tokenizer,
                label="tree",
                topk=int(args.debug_topk),
                max_rows=int(args.debug_nodes),
            )

            if eagle_model is not None and step < int(args.benchmark_eagle_steps):
                if start_event is not None:
                    start_event.record()
                t1e = time.perf_counter()
                with torch.inference_mode():
                    prefix_len_eagle = int(prefix_ids_eagle.shape[1])
                    pos_ids_eagle = (depth_offsets + prefix_len_eagle).unsqueeze(0)
                    eagle_past = eagle_prefix_past
                    if eagle_past is not None:
                        eagle_past = FrozenCache(eagle_past, prefix_len=prefix_len_eagle)
                    eagle_model.model.tree_mask = tree_mask
                    out_eagle = eagle_model(
                        input_ids=flat_tokens.unsqueeze(0),
                        past_key_values=eagle_past,
                        position_ids=pos_ids_eagle,
                        use_cache=False,
                        return_dict=True,
                    )
                    eagle_model.model.tree_mask = None
                eagle_ms = _step_timing(start_event, end_event, bool(args.sync_timing))
                eagle_wall_ms = (time.perf_counter() - t1e) * 1000.0
                if step % max(1, int(args.log_every)) == 0 and step >= int(args.warmup_steps):
                    print(f"  [EAGLE] verify_ms={eagle_ms:.2f} wall_ms={eagle_wall_ms:.2f}")

        # Qualitative logging for KV draft.
        if bool(args.log_kv) and step % max(1, int(args.log_every)) == 0 and step >= int(args.warmup_steps):
            depth_logits = logits_by_depth[0]
            _log_topk(
                logits=depth_logits,
                tokenizer=tokenizer,
                label="kv_depth1",
                topk=int(args.debug_topk),
                max_rows=int(args.debug_nodes),
            )

        # Extend prefix with the leftmost path (branch 0) to grow context.
        path_tokens = []
        for depth_tokens in tree_tokens_by_depth:
            path_tokens.append(int(depth_tokens[0, 0].item()))
        path_ids = torch.tensor(path_tokens, device=device, dtype=torch.long).unsqueeze(0)

        with torch.inference_mode():
            out = draft_model(input_ids=path_ids, past_key_values=prefix_past, use_cache=True, return_dict=True)
        prefix_past = _ensure_dynamic_cache(out.past_key_values)
        prefix_logits = out.logits[:, -1, :]
        prefix_ids_full = torch.cat([prefix_ids_full, path_ids], dim=1)
        prefix_ids_draft = torch.cat([prefix_ids_draft, path_ids], dim=1)
        if int(args.draft_max_prefix_len) > 0 and prefix_ids_draft.shape[1] > int(args.draft_max_prefix_len):
            prefix_past.crop(int(args.draft_max_prefix_len))
            prefix_ids_draft = prefix_ids_draft[:, -int(args.draft_max_prefix_len) :]

        if args.method in ("tree_attention", "both"):
            prefix_ids_verify = torch.cat([prefix_ids_verify, path_ids], dim=1)
            with torch.inference_mode():
                out_verify = model_verify(
                    input_ids=path_ids,
                    past_key_values=prefix_past_verify,
                    use_cache=True,
                    return_dict=True,
                )
            prefix_past_verify = _ensure_dynamic_cache(out_verify.past_key_values)
            if int(args.verify_max_prefix_len) > 0 and prefix_ids_verify.shape[1] > int(args.verify_max_prefix_len):
                if hasattr(prefix_past_verify, "crop"):
                    prefix_past_verify.crop(int(args.verify_max_prefix_len))
                    prefix_ids_verify = prefix_ids_verify[:, -int(args.verify_max_prefix_len) :]
                else:
                    prefix_ids_verify = prefix_ids_verify[:, -int(args.verify_max_prefix_len) :]

        if eagle_model is not None:
            prefix_ids_eagle = torch.cat([prefix_ids_eagle, path_ids], dim=1)
            with torch.inference_mode():
                out_eagle = eagle_model(
                    input_ids=path_ids,
                    past_key_values=eagle_prefix_past,
                    use_cache=True,
                    return_dict=True,
                )
            eagle_prefix_past = _ensure_dynamic_cache(out_eagle.past_key_values)


if __name__ == "__main__":
    main()
