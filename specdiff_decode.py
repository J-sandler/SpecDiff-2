#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import json
import math
import os
import re
import shlex
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoModelForMaskedLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache, StaticCache

from diffu_utils import ensure_kv_cache_model

try:
    from diffucoder_semi_ar import _fuse_dream_mlps as _phase1_fuse_mlps
    from diffucoder_semi_ar import _fuse_qkv_projections as _phase1_fuse_qkv
except Exception:
    _phase1_fuse_mlps = None
    _phase1_fuse_qkv = None


def _load_generate_tree_buffers():
    import utils

    utils.Timer = lambda *_args, **_kwargs: contextlib.nullcontext()
    return utils.generate_tree_buffers

def _load_qwen2_verifier():
    from qwen2_kv import Qwen2ForCausalLM
    return Qwen2ForCausalLM

def _load_kv_cache():
    from kv_cache import KVCache, initialize_past_key_values
    return KVCache, initialize_past_key_values

def _make_past_view_single_device(model, past_key_values_data_list, current_length_data, KVCacheCls):
    if not past_key_values_data_list:
        raise ValueError("past_key_values_data_list is empty")
    data = past_key_values_data_list[0]
    num_layers = int(model.config.num_hidden_layers)
    past = []
    for layer_idx in range(num_layers):
        past.append(
            [
                KVCacheCls(data[2 * layer_idx + 0], current_length_data[2 * layer_idx + 0]),
                KVCacheCls(data[2 * layer_idx + 1], current_length_data[2 * layer_idx + 1]),
            ]
        )
    return past


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



_ARG_ALIASES = {
    "--verifier": "--verify_model_name",
    "--draft_temperature": "--diffusion_temperature",
    "--verify_temp": "--verify_temperature",
}


def _apply_aliases(argv: Sequence[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        matched = False
        for legacy, canonical in _ARG_ALIASES.items():
            if token == legacy:
                out.append(canonical)
                matched = True
                i += 1
                if i < len(argv) and not argv[i].startswith("--"):
                    out.append(argv[i])
                    i += 1
                break
            if token.startswith(legacy + "="):
                suffix = token[len(legacy) :]
                out.append(canonical + suffix)
                matched = True
                i += 1
                break
        if matched:
            continue
        out.append(token)
        i += 1
    return out


_DIFF_LINE_RE = re.compile(r"\[Diff\]\s+wall_ms=([0-9.]+)")
_INFO_LINE_RE = re.compile(r"\[Info\]\s+generated=(\d+)\s+tokens in\s+([0-9.]+)s")
_STAGE_TIME_RE = re.compile(r"\[(?P<stage>[A-Za-z]+)\]\s+wall_ms=(?P<ms>[0-9.]+)")


_BATCH_SKIP_OPTS = {
    "--prompt",
    "--prompts_file",
    "--prompt_field",
    "--max_prompts",
    "--batch_stats_output",
}


@dataclass(frozen=True)
class _PromptRunResult:
    prompt: str
    diff_times: List[float]
    generated_tokens: int
    generated_seconds: float
    stage_totals: Dict[str, float] = field(default_factory=dict)
    stage_counts: Dict[str, int] = field(default_factory=dict)


def _read_prompts(path: str, prompt_field: Optional[str], limit: Optional[int]) -> List[str]:
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as reader:
        if prompt_field:
            for line in reader:
                raw = line.strip()
                if not raw:
                    continue
                try:
                    payload = json.loads(raw)
                except json.JSONDecodeError:
                    continue
                value = payload.get(prompt_field)
                if value is None:
                    continue
                if isinstance(value, list):
                    prompt = "\n".join(str(entry) for entry in value if entry is not None)
                else:
                    prompt = str(value)
                prompts.append(prompt)
                if limit and len(prompts) >= limit:
                    break
        else:
            buffer: List[str] = []
            for line in reader:
                raw = line.rstrip("\n")
                stripped = raw.strip()
                if stripped.startswith("Question:") and buffer:
                    prompts.append("\n".join(buffer))
                    buffer = []
                    if limit and len(prompts) >= limit:
                        break
                if not stripped:
                    if buffer and stripped == "":
                        prompts.append("\n".join(buffer))
                        buffer = []
                        if limit and len(prompts) >= limit:
                            break
                    continue
                buffer.append(raw)
            if buffer and (limit is None or len(prompts) < limit):
                prompts.append("\n".join(buffer))
    return prompts[:limit] if limit and len(prompts) > limit else prompts


def _filter_batch_args(argv: Sequence[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        if token in _BATCH_SKIP_OPTS:
            i += 1
            if token == "--prompt":
                if i < len(argv):
                    i += 1
            elif i < len(argv):
                i += 1
            continue
        if any(token.startswith(opt + "=") for opt in _BATCH_SKIP_OPTS):
            i += 1
            continue
        out.append(token)
        i += 1
    return out


def _extract_diff_ms(line: str) -> Optional[float]:
    match = _DIFF_LINE_RE.search(line)
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass
    return None


def _extract_generated_info(line: str) -> Optional[Tuple[int, float]]:
    match = _INFO_LINE_RE.search(line)
    if match:
        try:
            tokens = int(match.group(1))
            elapsed = float(match.group(2))
            return tokens, elapsed
        except ValueError:
            pass
    return None


def _run_prompt_subprocess(base_cmd: Sequence[str], prompt: str) -> _PromptRunResult:
    cmd = list(base_cmd) + ["--prompt", prompt]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    assert proc.stdout is not None
    diff_times: List[float] = []
    generated_tokens = 0
    generated_seconds = 0.0
    stage_totals = defaultdict(float)
    stage_counts = defaultdict(int)
    for raw_line in proc.stdout:
        print(raw_line, end="")
        diff = _extract_diff_ms(raw_line)
        if diff is not None:
            diff_times.append(diff)
        generated = _extract_generated_info(raw_line)
        if generated is not None:
            generated_tokens, generated_seconds = generated
        for match in _STAGE_TIME_RE.finditer(raw_line):
            try:
                stage = match.group("stage")
                ms = float(match.group("ms"))
            except ValueError:
                continue
            stage_totals[stage] += ms
            stage_counts[stage] += 1
    rc = proc.wait()
    if rc != 0:
        raise SystemExit(rc)
    return _PromptRunResult(
        prompt=prompt,
        diff_times=diff_times,
        generated_tokens=generated_tokens,
        generated_seconds=generated_seconds,
        stage_totals=dict(stage_totals),
        stage_counts=dict(stage_counts),
    )


def _report_batch_summary(
    results: Sequence[_PromptRunResult],
    output_path: Optional[str],
    base_cmd: Sequence[str],
    args: argparse.Namespace,
) -> None:
    prompt_count = len(results)
    diff_values = [value for result in results for value in result.diff_times]
    diff_steps = len(diff_values)
    total_diff_ms = sum(diff_values)
    total_tokens = sum(result.generated_tokens for result in results)
    total_seconds = sum(result.generated_seconds for result in results)
    throughput = float("nan")
    if total_seconds > 0:
        throughput = total_tokens / total_seconds
    stage_totals = defaultdict(float)
    stage_counts = defaultdict(int)
    for result in results:
        for stage, total in result.stage_totals.items():
            stage_totals[stage] += total
        for stage, count in result.stage_counts.items():
            stage_counts[stage] += count
    avg_stage_ms: Dict[str, float] = {}
    for stage, total in stage_totals.items():
        count = stage_counts.get(stage, 0)
        if count > 0:
            avg_stage_ms[stage] = total / float(count)
    if diff_steps > 0:
        mean_diff = total_diff_ms / diff_steps
        min_diff = min(diff_values)
        max_diff = max(diff_values)
    else:
        mean_diff = float("nan")
        min_diff = float("nan")
        max_diff = float("nan")
    print(
        "[BatchSummary] prompts={} diff_steps={} total_diff_ms={:.2f} mean_diff_ms={:.2f} "
        "min_diff_ms={:.2f} max_diff_ms={:.2f} throughput_tokens_s={:.2f}".format(
            prompt_count, diff_steps, total_diff_ms, mean_diff, min_diff, max_diff, throughput
        )
    )
    print(f"[BatchSummary] total_generated_tokens={total_tokens} total_generated_seconds={total_seconds:.2f}")
    if avg_stage_ms:
        stage_info = " ".join(f"{stage}={avg:.2f}ms" for stage, avg in sorted(avg_stage_ms.items()))
        print(f"[BatchSummary] avg_stage_ms={stage_info}")
    if output_path:
        payload = {
            "prompts": prompt_count,
            "diff_steps": diff_steps,
            "total_diff_ms": total_diff_ms,
            "mean_diff_ms": mean_diff,
            "min_diff_ms": min_diff,
            "max_diff_ms": max_diff,
            "throughput_tokens_per_s": throughput,
            "total_generated_tokens": total_tokens,
            "total_generated_seconds": total_seconds,
            "command": " ".join(shlex.quote(token) for token in base_cmd),
            "run_config": {
                "diffusion_model_name": str(getattr(args, "diffusion_model_name", "")),
                "verify_model_name": str(getattr(args, "verify_model_name", "")),
                "prompts_file": str(getattr(args, "prompts_file", "")) if getattr(args, "prompts_file", None) else None,
                "max_prompts": int(getattr(args, "max_prompts", 0)) if getattr(args, "max_prompts", None) else None,
                "gamma": int(getattr(args, "gamma", 0)),
                "k": int(getattr(args, "k", 0)),
                "max_new_tokens": int(getattr(args, "max_new_tokens", 0)),
                "diffusion_temperature": float(getattr(args, "diffusion_temperature", 0)),
                "diffusion_top_k": int(getattr(args, "diffusion_top_k", 0)),
                "diffusion_top_p": float(getattr(args, "diffusion_top_p", 0)),
                "verify_temperature": float(getattr(args, "verify_temperature", 0)),
                "verify_top_k": int(getattr(args, "verify_top_k", 0)),
                "verify_top_p": float(getattr(args, "verify_top_p", 0)),
                "approx_kv_cache": bool(getattr(args, "approx_kv_cache", False)),
                "diffusion_next_token": bool(getattr(args, "diffusion_next_token", False)),
            },
            "avg_stage_ms": avg_stage_ms,
        }
        exists = os.path.exists(output_path)
        with open(output_path, "a", encoding="utf-8") as writer:
            if exists and os.path.getsize(output_path) > 0:
                writer.write("\n")
            json.dump(payload, writer, indent=2)


def _run_batch_mode(args: argparse.Namespace, normalized_args: Sequence[str]) -> None:
    assert args.prompts_file
    prompts = _read_prompts(args.prompts_file, args.prompt_field, args.max_prompts)
    if not prompts:
        raise SystemExit(f"No prompts found in {args.prompts_file}")
    filtered_args = _filter_batch_args(normalized_args)
    script_path = os.path.abspath(sys.argv[0])
    base_cmd = [sys.executable, script_path] + filtered_args
    results: List[_PromptRunResult] = []
    for idx, prompt in enumerate(prompts, start=1):
        print(f"[Batch] running prompt {idx}/{len(prompts)}")
        result = _run_prompt_subprocess(base_cmd, prompt)
        results.append(result)
    _report_batch_summary(results, args.batch_stats_output, base_cmd, args)


def _has_full_model_files(path: str) -> bool:
    if not path or not os.path.isdir(path):
        return False
    config_path = os.path.join(path, "config.json")
    if not os.path.isfile(config_path):
        return False
    weight_candidates = (
        "pytorch_model.bin",
        "model.safetensors",
    )
    for name in weight_candidates:
        if os.path.isfile(os.path.join(path, name)):
            return True
    for fname in os.listdir(path):
        if fname.startswith("model-") and fname.endswith(".safetensors"):
            return True
        if fname.startswith("pytorch_model-") and fname.endswith(".bin"):
            return True
    return False


def _find_adapter_dir(path: str) -> Optional[str]:
    if not path:
        return None
    if os.path.isdir(path):
        if _has_full_model_files(path):
            return None
        direct = os.path.join(path, "adapter_config.json")
        if os.path.isfile(direct):
            return path
        nested = os.path.join(path, "adapter", "adapter_config.json")
        if os.path.isfile(nested):
            return os.path.join(path, "adapter")
    return None


def _resolve_base_model(adapter_dir: str, fallback: str) -> str:
    cfg_path = os.path.join(adapter_dir, "adapter_config.json")
    with open(cfg_path, "r", encoding="utf-8") as reader:
        cfg = json.load(reader)
    base_name = cfg.get("base_model_name_or_path") or fallback
    if not base_name:
        raise ValueError(f"Missing base_model_name_or_path in {cfg_path}")
    return base_name


def _load_adapter_state(adapter_dir: str) -> Dict[str, torch.Tensor]:
    safetensors_path = os.path.join(adapter_dir, "adapter_model.safetensors")
    bin_path = os.path.join(adapter_dir, "adapter_model.bin")
    if os.path.isfile(safetensors_path):
        try:
            from safetensors.torch import load_file
        except Exception as exc:
            raise RuntimeError(f"safetensors is required to load {safetensors_path}: {exc}") from exc
        return load_file(safetensors_path, device="cpu")
    if os.path.isfile(bin_path):
        return torch.load(bin_path, map_location="cpu")
    raise FileNotFoundError(f"No adapter_model.safetensors or adapter_model.bin found in {adapter_dir}")


def _load_and_merge_adapter(diff_model: torch.nn.Module, adapter_dir: str) -> torch.nn.Module:
    try:
        from peft import PeftConfig, get_peft_model
        from peft.utils.save_and_load import set_peft_model_state_dict
    except Exception as exc:
        raise RuntimeError(f"peft is required to load adapters from {adapter_dir}: {exc}") from exc

    peft_cfg = PeftConfig.from_pretrained(adapter_dir)
    peft_model = get_peft_model(diff_model, peft_cfg)
    adapter_state = _load_adapter_state(adapter_dir)
    set_peft_model_state_dict(peft_model, adapter_state, ignore_mismatched_sizes=True)
    if hasattr(peft_model, "merge_and_unload"):
        merged = peft_model.merge_and_unload()
        print(f"[Info] merged adapter into diffusion model from {adapter_dir}")
        return merged
    print(f"[Warn] adapter loaded from {adapter_dir}, but merge_and_unload is unavailable")
    return peft_model


def _load_diffusion_model_and_tokenizer(args: argparse.Namespace):
    adapter_dir = _find_adapter_dir(str(args.diffusion_model_name))
    base_name = str(args.diffusion_model_name)
    if adapter_dir:
        base_name = _resolve_base_model(adapter_dir, base_name)

    tok_source = str(args.diffusion_model_name)
    try:
        diff_tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=bool(args.trust_remote_code))
    except Exception:
        diff_tokenizer = AutoTokenizer.from_pretrained(base_name, trust_remote_code=bool(args.trust_remote_code))

    diff_model = None
    load_errors = []
    for factory in (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel):
        try:
            diff_model = factory.from_pretrained(
                base_name,
                torch_dtype=_get_dtype(args.dtype),
                trust_remote_code=bool(args.trust_remote_code),
            )
            break
        except Exception as exc:
            load_errors.append(str(exc))
    if diff_model is None:
        raise RuntimeError("Failed to load diffusion model. Errors:\n" + "\n".join(load_errors))

    if adapter_dir:
        diff_model = _load_and_merge_adapter(diff_model, adapter_dir)

    if hasattr(diff_model, "config") and hasattr(diff_model.config, "use_cache"):
        diff_model.config.use_cache = True
    if bool(args.approx_kv_cache):
        diff_model = ensure_kv_cache_model(diff_model)

    return diff_model, diff_tokenizer


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


def _filter_logits(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    if top_k and top_k > 0:
        top_k = min(int(top_k), logits.shape[-1])
        kth_vals = torch.topk(logits, k=top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth_vals, float("-inf"))
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        probs = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs, dim=-1)
        cutoff = cum > float(top_p)
        cutoff[..., 0] = False
        sorted_logits = sorted_logits.masked_fill(cutoff, float("-inf"))
        logits = torch.full_like(logits, float("-inf"))
        logits.scatter_(-1, sorted_idx, sorted_logits)
    return logits


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
    logits = logits.squeeze(0)
    if temperature <= 0:
        greedy = torch.argmax(logits, dim=-1)
        return greedy.unsqueeze(0).repeat(int(k), 1)
    logits = logits / float(temperature)

    # Fast-path: top-k only (no nucleus). Avoid full-vocab softmax.
    if top_k and top_k > 0 and (not top_p or float(top_p) >= 1.0):
        top_k = min(int(top_k), int(logits.shape[-1]))
        topv, topi = torch.topk(logits, k=top_k, dim=-1)
        probs = torch.softmax(topv.to(dtype=torch.float32), dim=-1)
        local = torch.multinomial(probs, num_samples=int(k), replacement=True, generator=generator)
        picks = topi.gather(-1, local)
        return picks.transpose(0, 1).contiguous()

    logits = _filter_logits(logits, top_k=top_k, top_p=top_p)
    probs = torch.softmax(logits.to(dtype=torch.float32), dim=-1)
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

def _build_tree_from_sequences(seqs: torch.Tensor) -> Tuple[torch.Tensor, List[int], List[int], List[List[int]]]:
    # NOTE: `Tensor.tolist()` is very expensive (nested Python lists). For the small
    # (K, gamma) tree we only need token ids, so we transfer to CPU once and read
    # scalars in the Python loop.
    seqs_cpu = seqs
    if seqs_cpu.is_cuda:
        seqs_cpu = seqs_cpu.to(device="cpu", non_blocking=True)
    seqs_cpu = seqs_cpu.contiguous()
    k = int(seqs_cpu.shape[0])
    gamma = int(seqs_cpu.shape[1]) if k > 0 else 0
    parents: List[int] = []
    tokens: List[int] = []
    depths: List[int] = []
    node_for_prefix: Dict[Tuple[int, ...], int] = {}
    seq_nodes: List[List[int]] = [[-1] * gamma for _ in range(k)]

    for i in range(k):
        prefix: List[int] = []
        prev_node = -1
        for d in range(gamma):
            tok = int(seqs_cpu[i, d].item())
            prefix.append(tok)
            key = tuple(prefix)
            node = node_for_prefix.get(key)
            if node is None:
                node = len(tokens)
                node_for_prefix[key] = node
                tokens.append(tok)
                parents.append(prev_node)
                depths.append(d + 1)
            seq_nodes[i][d] = node
            prev_node = node
    return (
        torch.tensor(tokens, dtype=torch.long),
        parents,
        depths,
        seq_nodes,
    )

def _reorder_tree(
    *,
    flat_tokens: torch.Tensor,
    parents: List[int],
    seq_nodes: List[List[int]],
) -> Tuple[torch.Tensor, List[int], List[int], List[List[int]], List[List[int]]]:
    # Borrow the same ordering strategy used by EAGLE's generate_tree_buffers:
    # sort by (depth, path of sibling-ranks).
    if not parents:
        return flat_tokens, parents, [], seq_nodes, []
    node_count = len(parents)
    children: List[List[int]] = [[] for _ in range(node_count)]
    roots: List[int] = []
    for idx, parent in enumerate(parents):
        if parent < 0:
            roots.append(idx)
        else:
            children[parent].append(idx)
    for idx in range(node_count):
        children[idx].sort()
    roots.sort()
    root_rank = {idx: rank for rank, idx in enumerate(roots)}
    child_rank: List[Dict[int, int]] = []
    for idx in range(node_count):
        ranks = {child: rank for rank, child in enumerate(children[idx])}
        child_rank.append(ranks)
    paths: List[List[int]] = [[] for _ in range(node_count)]
    for idx in range(node_count):
        cur = idx
        path_rev: List[int] = []
        parent = parents[cur]
        if parent < 0:
            path_rev.append(root_rank[cur])
        else:
            while parent >= 0:
                path_rev.append(child_rank[parent][cur])
                cur = parent
                parent = parents[cur]
            path_rev.append(root_rank[cur])
        paths[idx] = list(reversed(path_rev))
    order = sorted(range(node_count), key=lambda i: (len(paths[i]), paths[i]))
    new_index = {old: new for new, old in enumerate(order)}
    new_flat = flat_tokens[order]
    new_depths = [len(paths[i]) for i in order]
    new_parents = [new_index[p] if p >= 0 else -1 for p in (parents[i] for i in order)]
    new_seq_nodes: List[List[int]] = []
    for path in seq_nodes:
        new_seq_nodes.append([new_index[idx] for idx in path])
    tree_choices_sorted = [paths[i] for i in order]
    return new_flat, new_parents, new_depths, new_seq_nodes, tree_choices_sorted

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


# --- Acceptance logic---
from specdiff_acceptor import (  # type: ignore
    FrozenCache,
    PreallocCache,
    _accept_greedy,
    _accept_leviathan,
    _copy_cache_range,
    _debug_prefix_alignment,
    _log_root_candidates,
    _log_topk,
    _reuse_tree_cache_for_accepted,
    _truncate_cache,
)



def main() -> None:
    normalized_args = _apply_aliases(sys.argv[1:])
    sys.argv = [sys.argv[0]] + normalized_args
    p = argparse.ArgumentParser(description="Speculative Diffusion Generator")
    p.add_argument("--diffusion_model_name", type=str, default="apple/DiffuCoder-7B-cpGRPO")
    p.add_argument("--verify_model_name", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    p.add_argument("--prompt", type=str, required=False)
    p.add_argument("--prompts_file", type=str, help="Path to newline or JSONL file containing prompts.")
    p.add_argument("--prompt_field", type=str, default=None, help="When --prompts_file is JSONL, use this field for the prompt text.")
    p.add_argument("--max_prompts", type=int, default=None, help="Limit how many prompts to read from --prompts_file.")
    p.add_argument("--batch_stats_output", type=str, default=None, help="Optional JSON file to write aggregated batch stats.")
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
    # Default to top-k for speed: draft distribution can be arbitrary as long as acceptance
    # uses the same draft distribution parameters.
    p.add_argument("--diffusion_top_k", type=int, default=64)
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
    p.add_argument("--verify_max_cache_len", type=int, default=4096*32)
    p.add_argument("--verify_freeze_cache", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument(
        "--warmup_steps",
        type=int,
        default=1,
        help=(
            "Pre-loop warmup iterations to reduce first-step spikes. "
            "Runs a small end-to-end dry-run (diff->sample->verify->accept->(fake)update) without mutating the true "
            "prefix caches (verification uses the tree cache view)."
        ),
    )
    p.add_argument(
        "--sync_step_boundary",
        action="store_true",
        default=False,
        help="Synchronize CUDA at the end of every step to attribute hidden queueing/stalls to a [Sync] bucket.",
    )
    p.add_argument(
        "--debug_tree_compare_every",
        type=int,
        default=1,
        help="Run --debug_tree_compare only every N steps (reduces overhead).",
    )
    p.add_argument(
        "--debug_other_threshold_ms",
        type=float,
        default=20.0,
        help="If [Other] exceeds this threshold, print an extra hint line.",
    )
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
    
    
    p.add_argument(
        "--tree_buffers_on_cpu",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate tree buffers on CPU and copy into a preallocated GPU buffer (avoids per-step cudaMalloc).",
    )

    p.add_argument(
        "--tree_build_mode",
        type=str,
        default="k_chains",
        choices=("k_chains", "collapsed"),
        help=(
            "How to construct the token tree passed to the verifier. "
            "`k_chains` is a fixed K-path tree (no collapsing) and enables precomputed buffers; "
            "`collapsed` collapses repeated tokens into shared nodes but requires per-step buffer generation."
        ),
    )
    p.add_argument(
        "--profile_tree_breakdown",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Print a per-step breakdown of the Tree stage (CPU build/reorder/buffers/copies).",
    )
    args = p.parse_args()
    if not args.prompt and not args.prompts_file:
        p.error("Either --prompt or --prompts_file must be provided.")
    if args.prompts_file:
        _run_batch_mode(args, normalized_args)
        return

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    torch.set_grad_enabled(False)

    device = torch.device(args.device)
    tree_copy_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
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

    generate_tree_buffers = _load_generate_tree_buffers()

    Qwen2ForCausalLM = _load_qwen2_verifier()

    KVCacheCls, initialize_past_key_values = _load_kv_cache()

    diff_model, diff_tokenizer = _load_diffusion_model_and_tokenizer(args)
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

    verify_model = Qwen2ForCausalLM.from_pretrained(args.verify_model_name, torch_dtype=dtype)

    diff_model.to(device)
    verify_model.to(device)
    diff_model.eval()
    verify_model.eval()
    if bool(args.diffusion_fuse_mlp) and _phase1_fuse_mlps is not None:
        _phase1_fuse_mlps(diff_model, verbose=True)
    if bool(args.diffusion_fuse_qkv) and _phase1_fuse_qkv is not None:
        _phase1_fuse_qkv(diff_model, verbose=True)
    use_num_logits_to_keep = _supports_num_logits_to_keep(diff_model)
    print(f"[Info] verifier={args.verify_model_name}")

    mask_id = _maybe_add_mask_token(diff_model, diff_tokenizer)

    diff_vocab_size = getattr(diff_model.config, "vocab_size", None)
    verify_vocab_size = getattr(verify_model.config, "vocab_size", None)
    diff_to_verify, verify_to_diff, missing_diff, missing_verify = _build_token_id_maps(
        diff_tokenizer,
        verify_tokenizer,
        diff_size=diff_vocab_size,
        verify_size=verify_vocab_size,
    )
    diff_to_verify = diff_to_verify.to(device)
    verify_to_diff = verify_to_diff.to(device)
    if missing_diff or missing_verify:
        print(f"[Info] token id map missing: diff_missing={missing_diff} verify_missing={missing_verify}.")

    # IMPORTANT:
    # - `diff_to_verify` (with -1 holes) is used by the acceptance math: missing diffusion tokens must NOT
    #   be remapped to any verifier token (that would reassign probability mass).
    # - For per-step sampling/verification, we want a branchless gather. We therefore build a second tensor
    #   `diff_to_verify_for_samples` that fills holes with `unk` (those holes should never be sampled anyway
    #   because we mask them out using `diff_valid_mask`).
    verify_unk = verify_tokenizer.unk_token_id
    if verify_unk is None:
        verify_unk = 0
    diff_valid_mask = (diff_to_verify >= 0).to(device=device)
    diff_to_verify_for_samples = diff_to_verify.clone()
    diff_to_verify_for_samples[diff_to_verify_for_samples < 0] = int(verify_unk)

    verify_size = int(verify_vocab_size) if verify_vocab_size is not None else len(verify_tokenizer)
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

    # Preallocate prefix buffers to avoid per-step `torch.cat` allocations/copies (can dominate step time).
    # We keep a simple in-place growing prefix as long as we stay within `--verify_max_cache_len`.
    max_cache_len = int(args.verify_max_cache_len)
    init_verify_len = int(verify_prefix_ids.shape[1])
    init_diff_len = int(diff_prefix_ids.shape[1])
    if init_verify_len > max_cache_len or init_diff_len > max_cache_len:
        raise ValueError(
            f"Initial prefix longer than verify_max_cache_len={max_cache_len}: "
            f"verify_len={init_verify_len} diff_len={init_diff_len}"
        )
    if init_verify_len + int(args.max_new_tokens) > max_cache_len:
        print(
            f"[Warn] verify_max_cache_len={max_cache_len} may be too small for "
            f"prefix_len={init_verify_len} + max_new_tokens={int(args.max_new_tokens)}; "
            "prefix buffer will overflow and fall back to slow cat path."
        )

    verify_buf = torch.empty((1, max_cache_len), device=device, dtype=torch.long)
    verify_buf[:, :init_verify_len].copy_(verify_prefix_ids)
    verify_len = init_verify_len
    verify_prefix_ids = verify_buf[:, :verify_len]

    # Diffusion input buffer: reserve extra `gamma` slots for in-place mask tokens (avoid cat/copy each step).
    diff_buf = torch.empty((1, max_cache_len + int(args.gamma)), device=device, dtype=torch.long)
    diff_buf[:, :init_diff_len].copy_(diff_prefix_ids)
    diff_len = init_diff_len
    diff_prefix_ids = diff_buf[:, :diff_len]

    # Diffusion prefix next logits (no cache unless approx_kv_cache).
    diff_past = None
    diff_prefix_next_logits = None
    diff_shift = -1 if bool(args.diffusion_next_token) else 0
    diff_shift_locked = False
    warned_approx_shift = False
    approx_shift = 0
    approx_shift_locked = False
    approx_kv_cache = bool(args.approx_kv_cache)
    if approx_kv_cache:
        diff_past = DynamicCache()
        prefill_kwargs = {}
        if use_num_logits_to_keep:
            keep = 0
            if bool(args.diffusion_next_token):
                keep = 1
            prefill_kwargs["num_logits_to_keep"] = int(keep)
        with torch.inference_mode(), _diff_sdp_ctx():
            out = diff_model(
                input_ids=diff_prefix_ids,
                attention_mask=None,
                use_cache=True,
                past_key_values=diff_past,
                return_dict=True,
                **prefill_kwargs,
            )
        if not hasattr(out, "past_key_values") or out.past_key_values is None:
            raise RuntimeError(
                "approx_kv_cache requested, but the diffusion model did not return past_key_values. "
                "Use a checkpoint/class that supports KV caching for fast decoding."
            )
        if bool(args.diffusion_next_token):
            if hasattr(out, "logits") and out.logits is not None and out.logits.numel() > 0:
                diff_prefix_next_logits = out.logits[:, -1, :].squeeze(0)
            else:
                print(
                    "[Warn] diffusion_next_token requested, but diffusion prefill returned empty logits; "
                    "alignment may be degraded."
                )
        print("[ApproxCache] diffusion approx_kv_cache enabled.")
    if (not approx_kv_cache) and bool(args.diffusion_next_token):
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

    # Verifier cache.
    if int(args.verify_max_cache_len) <= 0:
        raise ValueError("--verify_max_cache_len must be > 0")

    verify_past, verify_past_data_list, verify_current_length = initialize_past_key_values(
            verify_model, max_length=int(args.verify_max_cache_len)
    )
    verify_current_length.zero_()

    with torch.inference_mode(), _verify_sdp_ctx():
        out = verify_model(
            input_ids=verify_prefix_ids,
            past_key_values=verify_past,
            use_cache=True,
            return_dict=True,
        )
    prefix_next_logits = out.logits[:, -1, :].squeeze(0)

    # Tree verification uses a separate "view" with an independent current_length tensor to avoid mutating prefix cache.
    tree_current_length = verify_current_length.clone()

    tree_past = _make_past_view_single_device(
        verify_model, verify_past_data_list, tree_current_length, KVCacheCls
    )


    # Precompute fixed buffers for the K-chain tree (fast path).
    # generate_tree_buffers defines the *tree structure* via `tree_choices` only; token ids are provided separately.
    tree_build_mode = str(args.tree_build_mode)
    if tree_build_mode == "collapsed":
        print(
            "[Warn] tree_build_mode=collapsed can be much slower (CPU trie build + per-step buffer generation). "
            "For throughput benchmarking, prefer `--tree_build_mode k_chains`."
        )
    chain_seq_nodes: Optional[List[List[int]]] = None

    chain_tree_mask: Optional[torch.Tensor] = None
    chain_tree_pos: Optional[torch.Tensor] = None

    if tree_build_mode == "k_chains":
        k_val = int(args.k)
        gamma_val = int(args.gamma)
        chain_seq_nodes = [[d * k_val + k for d in range(gamma_val)] for k in range(k_val)]
        # One node per (depth, root). Use a single-child chain per root by padding with zeros.
        chain_tree_choices = [[root] + [0] * (depth - 1) for depth in range(1, gamma_val + 1) for root in range(k_val)]
        chain_tree_buffers = generate_tree_buffers(chain_tree_choices, device=device, minimal=True)

        chain_tree_mask = chain_tree_buffers["tree_attn_mask"][:, :, 1:, 1:]
        chain_tree_pos = chain_tree_buffers["tree_position_ids"][1:]


    # Preallocate GPU buffers for tree mask/positions to avoid per-step cudaMalloc syncs.
    # Worst-case nodes after collapsing is <= k * gamma (full non-overlapping tree).
    max_tree_nodes = max(1, int(args.k) * int(args.gamma))
    max_tree_len = max_tree_nodes + 1  # +1 synthetic root slot in buffers.
    tree_mask_gpu = torch.empty((1, 1, max_tree_len, max_tree_len), device=device, dtype=torch.float32)
    tree_pos_gpu = torch.empty((max_tree_len,), device=device, dtype=torch.long)
    tree_mask_cpu_pinned = None
    tree_pos_cpu_pinned = None
    if bool(args.tree_buffers_on_cpu) and device.type == "cuda":
        tree_mask_cpu_pinned = torch.empty(
            (1, 1, max_tree_len, max_tree_len), device="cpu", dtype=torch.float32, pin_memory=True
        )
        tree_pos_cpu_pinned = torch.empty((max_tree_len,), device="cpu", dtype=torch.long, pin_memory=True)

    # Main loop.
    total_generated = 0
    max_new_tokens = int(args.max_new_tokens)
    max_steps = int(args.steps)
    if max_steps <= 0:
        max_steps = (max_new_tokens + int(args.gamma) - 1) // int(args.gamma)

    # Warmup (diff + verify only; no accept/update) to avoid first-iteration spikes.
    warmup_steps = max(0, int(args.warmup_steps))
    if warmup_steps > 0:
        print(f"[Warmup] steps={warmup_steps}")
        # Preserve determinism: restore RNG state after warmup.
        warm_rng_state = generator.get_state()
        for _ in range(warmup_steps):
            step_gamma = int(args.gamma)
            mask_tokens = torch.full((1, step_gamma), mask_id, device=device, dtype=torch.long)

            model_kwargs = {}
            if use_num_logits_to_keep:
                keep = int(step_gamma)
                if bool(args.diffusion_next_token) and not bool(args.approx_kv_cache):
                    keep += 1
                model_kwargs["num_logits_to_keep"] = int(keep)

            # Diffusion warmup.
            if bool(args.approx_kv_cache):
                with torch.inference_mode(), _diff_sdp_ctx():
                    out = diff_model(
                        input_ids=mask_tokens,
                        attention_mask=None,
                        past_key_values=diff_past,
                        use_cache=True,
                        return_dict=True,
                        **model_kwargs,
                    )
            else:
                if diff_len + step_gamma > diff_buf.shape[1]:
                    raise RuntimeError(
                        f"diff_buf overflow in warmup: diff_len={diff_len} gamma={step_gamma} cap={diff_buf.shape[1]}"
                    )
                diff_buf[:, diff_len : diff_len + step_gamma].fill_(mask_id)
                input_ids = diff_buf[:, : diff_len + step_gamma]
                attn_mask = None
                if not bool(args.diffusion_omit_attention_mask):
                    attn_mask = torch.ones_like(input_ids, dtype=torch.bool)
                with torch.inference_mode(), _diff_sdp_ctx():
                    out = diff_model(
                        input_ids=input_ids,
                        attention_mask=attn_mask,
                        use_cache=False,
                        return_dict=True,
                        **model_kwargs,
                    )
            logits = out.logits
            if logits.dim() == 3 and logits.shape[1] > step_gamma:
                logits = logits[:, -step_gamma:, :]
            elif logits.dim() == 2:
                logits = logits.unsqueeze(1).expand(-1, step_gamma, -1)
            logits = logits[:, :step_gamma, :]
            if diff_valid_mask is not None:
                mask = diff_valid_mask
                if mask.numel() != logits.shape[-1]:
                    tmp = torch.zeros(logits.shape[-1], device=device, dtype=torch.bool)
                    size = min(tmp.numel(), mask.numel())
                    tmp[:size] = mask[:size]
                    mask = tmp
                logits = logits.masked_fill(~mask, float("-inf"))

            # Sample + map warmup.
            warm_samples = _sample_k_sequences(
                logits,
                int(k_val),
                temperature=float(args.diffusion_temperature),
                top_k=int(args.diffusion_top_k),
                top_p=float(args.diffusion_top_p),
                generator=generator,
            )
            warm_samples_verify = diff_to_verify_for_samples[warm_samples]

            # Verifier warmup with the real tree path used in the main loop.
            if tree_build_mode == "k_chains":
                warm_flat = warm_samples_verify.transpose(0, 1).contiguous().view(-1)
                warm_node_count = int(warm_flat.numel())
                warm_seq_nodes = chain_seq_nodes

                warm_mask = chain_tree_mask
                warm_pos = chain_tree_pos
            else:
                (warm_flat_tokens, warm_parents, _warm_depths, warm_seq_nodes) = _build_tree_from_sequences(
                    warm_samples_verify
                )
                warm_node_count = int(len(warm_parents))

                warm_flat, _warm_parents2, _warm_depths2, warm_seq_nodes, warm_choices = _reorder_tree(
                    flat_tokens=warm_flat_tokens, parents=warm_parents, seq_nodes=warm_seq_nodes
                )

                warm_buffers = generate_tree_buffers(warm_choices, device=device, minimal=True)

                warm_mask = warm_buffers["tree_attn_mask"][:, :, 1:, 1:]
                warm_pos = warm_buffers["tree_position_ids"][1:]

            if warm_mask is None or warm_pos is None or warm_seq_nodes is None:
                raise RuntimeError("warmup tree buffers missing")

            warm_prefix_len = int(verify_len)
            warm_pos_ids = (warm_pos + (warm_prefix_len - 1)).unsqueeze(0)
            verify_model.model.tree_mask = warm_mask
            with torch.inference_mode(), _verify_sdp_ctx():
                tree_current_length.copy_(verify_current_length)
                out_tree = verify_model(
                    input_ids=warm_flat.to(device=device).unsqueeze(0),
                    past_key_values=tree_past,
                    position_ids=warm_pos_ids,
                    use_cache=True,
                    return_dict=True,
                )
            verify_model.model.tree_mask = None
            warm_tree_logits = out_tree.logits[:, -warm_node_count:, :].squeeze(0)

            # Acceptance warmup (exact acceptance logic; do not mutate true prefixes).
            _accepted, _acc_len, _best_idx, _replacement = _accept_leviathan(
                warm_samples_verify,
                warm_seq_nodes,
                warm_tree_logits,
                prefix_next_logits,
                logits,
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

            # Incremental verifier warmup (uses tree cache view, so prefix cache isn't mutated).
            warm_add = torch.as_tensor([int(_replacement)], device=device, dtype=torch.long).unsqueeze(0)
            with torch.inference_mode(), _verify_sdp_ctx():
                tree_current_length.copy_(verify_current_length)
                _ = verify_model(
                    input_ids=warm_add,
                    past_key_values=tree_past,
                    use_cache=True,
                    return_dict=True,
                )

            if torch.cuda.is_available():
                torch.cuda.synchronize()
        generator.set_state(warm_rng_state)

    t_start = time.perf_counter()
    total_steps = 0
    total_accept_len = 0
    total_added = 0
    total_nodes = 0

    for step in range(max_steps):
        if total_generated >= max_new_tokens:
            break
        step_t0 = time.perf_counter()
        debug_compare_wall_ms = 0.0
        retok_wall_ms = 0.0
        sample_wall_ms = 0.0
        map_wall_ms = 0.0
        prefix_wall_ms = 0.0
        print_wall_ms = 0.0
        sync_wall_ms = 0.0

        # Diffusion forward: context + gamma [MASK]
        def _diff_forward():
            nonlocal warned_approx_shift
            nonlocal diff_shift
            nonlocal diff_shift_locked
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
                if diff_len + step_gamma > diff_buf.shape[1]:
                    raise RuntimeError(
                        f"diff_buf overflow: diff_len={diff_len} gamma={step_gamma} cap={diff_buf.shape[1]}"
                    )
                diff_buf[:, diff_len : diff_len + step_gamma].fill_(mask_id)
                input_ids = diff_buf[:, : diff_len + step_gamma]
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
                        if not approx_shift_locked:
                            if logits.shape[1] >= 2:
                                left_logits = logits[:, 0, :].squeeze(0)
                                right_logits = logits[:, 1, :].squeeze(0)
                                diff_left = float((left_logits - diff_prefix_next_logits).abs().max().item())
                                diff_right = float((right_logits - diff_prefix_next_logits).abs().max().item())
                                approx_shift = 0 if diff_left <= diff_right else -1
                                approx_shift_locked = True
                                print(
                                    "[Info] diffusion_shift_auto approx "
                                    f"shift={approx_shift} diff_left={diff_left:.4e} diff_right={diff_right:.4e}"
                                )
                            else:
                                diff_left = float((logits[:, 0, :] - diff_prefix_next_logits).abs().max().item())
                                approx_shift = 0 if diff_left <= 1e-4 else -1
                                approx_shift_locked = True
                                print(
                                    "[Info] diffusion_shift_auto approx "
                                    f"shift={approx_shift} diff_left={diff_left:.4e}"
                                )
                        if approx_shift == -1:
                            prefix_log = diff_prefix_next_logits.unsqueeze(0).unsqueeze(1)
                            logits = torch.cat([prefix_log, logits[:, :-1, :]], dim=1)
                    elif not warned_approx_shift:
                        print(
                            "[Warn] diffusion_next_token with approx_kv_cache could not align; "
                            "prefill prefix_next_logits missing."
                        )
                        warned_approx_shift = True
                else:
                    # Auto-detect whether DiffuCoder's marginal is shifted by 1 vs prefix_next_logits.
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
                        start = max(0, prefix_len + int(diff_shift))
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
        smp_t0 = time.perf_counter()
        samples = _sample_k_sequences(
            diff_logits,
            int(k_val),
            temperature=float(args.diffusion_temperature),
            top_k=int(args.diffusion_top_k),
            top_p=float(args.diffusion_top_p),
            generator=generator,
        )
        sample_wall_ms = (time.perf_counter() - smp_t0) * 1000.0

        mp_t0 = time.perf_counter()
        # NOTE: avoid `.any()` / `.item()` in the hot path (GPU->CPU sync).
        # Invalid diffusion ids are masked out in `diff_logits`, so this should be a clean gather.
        samples_verify = diff_to_verify_for_samples[samples]
        map_wall_ms = (time.perf_counter() - mp_t0) * 1000.0

        # Build collapsed token tree (+ buffers).
        tree_t0 = time.perf_counter()
        build_wall_ms = 0.0
        reorder_wall_ms = 0.0
        gen_wall_ms = 0.0
        cpu_prep_wall_ms = 0.0
        h2d_wall_ms = 0.0

        # Fast path: fixed K-chain tree (precomputed buffers, GPU-only).
        if tree_build_mode == "k_chains":
            if chain_seq_nodes is None or chain_tree_mask is None or chain_tree_pos is None:
                raise RuntimeError("k_chains tree buffers are not initialized")

            # Flatten in (depth, root) order to match (len(path), path) sorting.
            flat_tokens = samples_verify.transpose(0, 1).contiguous().view(-1)
            tree_node_count = int(flat_tokens.numel())
            seq_nodes = chain_seq_nodes

            
            tree_mask = chain_tree_mask
            tree_pos = chain_tree_pos

            total_nodes += tree_node_count
        else:
            t_build0 = time.perf_counter()
            (flat_tokens, parents, depths, seq_nodes) = _build_tree_from_sequences(samples_verify)
            build_wall_ms = (time.perf_counter() - t_build0) * 1000.0
            tree_node_count = int(len(parents))
            total_nodes += tree_node_count

        # Reorder for buffer generation.
        if tree_build_mode != "k_chains":
            tree_mask = None
            tree_pos = None

            if tree_node_count > 0:
                t_reorder0 = time.perf_counter()
                flat_tokens, parents, depths, seq_nodes, tree_choices = _reorder_tree(
                    flat_tokens=flat_tokens,
                    parents=parents,
                    seq_nodes=seq_nodes,
                )
                reorder_wall_ms = (time.perf_counter() - t_reorder0) * 1000.0
                if bool(args.tree_buffers_on_cpu):
                    t_gen0 = time.perf_counter()
                    tree_buffers = generate_tree_buffers(tree_choices, device="cpu", minimal=True)
                    gen_wall_ms = (time.perf_counter() - t_gen0) * 1000.0
                    tree_len = int(tree_buffers["tree_position_ids"].shape[0])
                    if tree_len > tree_mask_gpu.shape[-1]:
                        raise RuntimeError(
                            f"tree_len overflow: tree_len={tree_len} cap={tree_mask_gpu.shape[-1]} "
                            f"(k={int(args.k)} gamma={int(args.gamma)})"
                        )
                    if tree_mask_cpu_pinned is None or tree_pos_cpu_pinned is None:
                        raise RuntimeError("expected pinned CPU tree buffers to be initialized")
                    t_cpu0 = time.perf_counter()
                    tree_mask_cpu_pinned[:, :, :tree_len, :tree_len].copy_(
                        tree_buffers["tree_attn_mask"], non_blocking=False
                    )
                    tree_pos_cpu_pinned[:tree_len].copy_(tree_buffers["tree_position_ids"], non_blocking=False)
                    cpu_prep_wall_ms = (time.perf_counter() - t_cpu0) * 1000.0
                    t_h2d0 = time.perf_counter()
                    if tree_copy_stream is None:
                        tree_mask_gpu[:, :, :tree_len, :tree_len].copy_(
                            tree_mask_cpu_pinned[:, :, :tree_len, :tree_len], non_blocking=False
                        )
                        tree_pos_gpu[:tree_len].copy_(tree_pos_cpu_pinned[:tree_len], non_blocking=False)
                    else:
                        with torch.cuda.stream(tree_copy_stream):
                            tree_mask_gpu[:, :, :tree_len, :tree_len].copy_(
                                tree_mask_cpu_pinned[:, :, :tree_len, :tree_len], non_blocking=True
                            )
                            tree_pos_gpu[:tree_len].copy_(tree_pos_cpu_pinned[:tree_len], non_blocking=True)
                    h2d_wall_ms = (time.perf_counter() - t_h2d0) * 1000.0

                    tree_mask = tree_mask_gpu[:, :, 1:tree_len, 1:tree_len]
                    tree_pos = tree_pos_gpu[1:tree_len]

                else:
                    t_gen0 = time.perf_counter()

                    tree_buffers = eagle_generate_tree_buffers(tree_choices, device=device, minimal=True)
                    tree_buffers = generate_tree_buffers(tree_choices, device=device, minimal=True)

                    gen_wall_ms = (time.perf_counter() - t_gen0) * 1000.0
                    tree_mask = tree_buffers["tree_attn_mask"][:, :, 1:, 1:]
                    tree_pos = tree_buffers["tree_position_ids"][1:]
            else:
                tree_mask = None
                tree_pos = None

        tree_wall_ms = (time.perf_counter() - tree_t0) * 1000.0
        if bool(args.profile_tree_breakdown):
            print(f"[TreeDbg] mode={tree_build_mode} nodes={tree_node_count} build_ms={build_wall_ms:.2f} "
                  f"reorder_ms={reorder_wall_ms:.2f} eagle_ms={gen_wall_ms:.2f} cpu_ms={cpu_prep_wall_ms:.2f} "
                  f"h2d_ms={h2d_wall_ms:.2f} total_ms={tree_wall_ms:.2f}")

        # Verify tree with buffers applied to the verifier.
        def _verify_stage():
            if tree_node_count <= 0:
                return torch.empty((0, verify_size), device=device, dtype=dtype)
            flat_tokens_device = flat_tokens.to(device=device)
            prefix_len = int(verify_len)
            # Reset the tree view to the prefix cache length (do NOT touch verify_current_length).
            if tree_build_mode != "k_chains" and tree_copy_stream is not None and bool(args.tree_buffers_on_cpu):
                torch.cuda.current_stream().wait_stream(tree_copy_stream)

            tree_current_length.copy_(verify_current_length)
            #  tree_position_ids is 1-based depth; align with LM absolute positions:
            # root depth=1 -> position_id = prefix_len (same as depths + prefix_len - 1).
            if tree_pos is None or tree_mask is None:
                raise RuntimeError("Missing tree buffers for verify stage.")

            pos_ids = (tree_pos + (prefix_len - 1)).unsqueeze(0)

            verify_model.model.tree_mask = tree_mask

            with torch.inference_mode(), _verify_sdp_ctx():
                out_tree = verify_model(
                    input_ids=flat_tokens_device.unsqueeze(0),
                    past_key_values=tree_past,
                    position_ids=pos_ids,
                    use_cache=True,
                    return_dict=True,
                )
            verify_model.model.tree_mask = None
            return out_tree.logits[:, -tree_node_count:, :].squeeze(0)

        tree_logits, verify_cuda_ms, verify_wall_ms = _measure_stage(
            start_event, end_event, bool(args.sync_timing), _verify_stage
        )

        # Accept tokens.
        prefix_next_logits_used = prefix_next_logits
        accept_argmax_prefix: Optional[int] = None
        accept_root_tokens: Optional[torch.Tensor] = None
        accept_prefix_group = None

        # Avoid per-step CPU work / GPU->CPU sync in the hot path:
        # - Leviathan acceptance does not need argmax or token string conversion.
        # - Greedy acceptance and qualitative debug do.
        if str(args.accept_mode) != "leviathan" or int(args.debug_nodes) > 0:
            accept_argmax_prefix = int(torch.argmax(prefix_next_logits_used).item())
            argmax_tok = verify_tokenizer.convert_ids_to_tokens(int(accept_argmax_prefix))
            accept_prefix_group = verify_token_groups.get(argmax_tok)
        if int(args.debug_nodes) > 0:
            accept_root_tokens = samples_verify[:, 0]

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
                accept_prefix_group or set(),
                verify_temperature=float(args.verify_temperature),
                verify_top_k=int(args.verify_top_k),
                verify_top_p=float(args.verify_top_p),
                generator=generator,
            )

        (accepted, accept_len, best_idx, replacement), _, accept_wall_ms = _measure_stage(
            None, None, False, _accept_stage
        )

        if (
            bool(args.debug_tree_compare)
            and samples_verify.numel() > 0
            and int(args.debug_tree_compare_every) > 0
            and (step % int(args.debug_tree_compare_every) == 0)
        ):
            dbg_t0 = time.perf_counter()
            cmp_idx = min(int(best_idx), int(samples_verify.shape[0] - 1))
            path = samples_verify[cmp_idx].unsqueeze(0)
            full_ids = torch.cat([verify_buf[:, :verify_len], path], dim=1)
            with torch.inference_mode(), _verify_sdp_ctx():
                out_ar = verify_model(input_ids=full_ids, use_cache=False, return_dict=True)
            ar_logits = out_ar.logits.squeeze(0)
            prefix_len = int(verify_len)
            ar_pos = max(0, prefix_len - 1)
            diff0 = (prefix_next_logits_used - ar_logits[ar_pos]).abs().max().item()
            print(f"[Debug] tree_compare prefix max_abs_diff={diff0:.4e}")
            max_depth = min(int(args.debug_tree_compare_depths), int(path.shape[1]) - 1)
            for d in range(max_depth):
                node_idx = seq_nodes[cmp_idx][d]
                ar_pos = prefix_len + d
                if ar_pos >= ar_logits.shape[0]:
                    break
                diffd = (tree_logits[node_idx] - ar_logits[ar_pos]).abs().max().item()
                print(f"[Debug] tree_compare depth={d+1} max_abs_diff={diffd:.4e}")
            debug_compare_wall_ms = (time.perf_counter() - dbg_t0) * 1000.0

        tokens_to_add = accepted + [int(replacement)]
        remaining = max_new_tokens - total_generated
        if remaining < len(tokens_to_add):
            tokens_to_add = tokens_to_add[:remaining]

        add_tensor = torch.as_tensor(tokens_to_add, device=device, dtype=torch.long).unsqueeze(0)

        # Update verifier cache & prefix logits.
        def _update_stage():
            with torch.inference_mode(), _verify_sdp_ctx():
                return verify_model(
                    input_ids=add_tensor,
                    past_key_values=verify_past,
                    use_cache=True,
                    return_dict=True,
                )

        out_upd, _, update_wall_ms = _measure_stage(None, None, False, _update_stage)
        prefix_next_logits = out_upd.logits[:, -1, :].squeeze(0)

        # Update token prefixes (for printing / next diffusion call) without `torch.cat` in the common case.
        pf_t0 = time.perf_counter()
        add_len = int(add_tensor.shape[1])
        if verify_len + add_len <= verify_buf.shape[1]:
            verify_buf[:, verify_len : verify_len + add_len].copy_(add_tensor)
            verify_len += add_len
            verify_prefix_ids = verify_buf[:, :verify_len]
        else:
            verify_prefix_ids = torch.cat([verify_prefix_ids, add_tensor], dim=1)
            verify_len = int(verify_prefix_ids.shape[1])
            verify_buf[:, :verify_len].copy_(verify_prefix_ids[:, :verify_len])

        mapped = verify_to_diff[add_tensor.squeeze(0)]
        # If every verifier token id maps to a diffusion id, avoid the `(mapped < 0).any()` sync entirely.
        if int(missing_verify) == 0:
            if diff_len + add_len <= max_cache_len:
                diff_buf[:, diff_len : diff_len + add_len].copy_(mapped.unsqueeze(0))
                diff_len += add_len
                diff_prefix_ids = diff_buf[:, :diff_len]
            else:
                diff_prefix_ids = torch.cat([diff_prefix_ids, mapped.unsqueeze(0)], dim=1)
                diff_len = int(diff_prefix_ids.shape[1])
        else:
            # Fallback: retokenize on mismatch.
            # NOTE: this branch checks `.any()` and will sync, but should be rare when verify_missing is small.
            if (mapped < 0).any():
                rt_t0 = time.perf_counter()
                new_text = verify_tokenizer.decode(tokens_to_add, skip_special_tokens=False)
                diff_ids = diff_tokenizer(new_text, return_tensors="pt", add_special_tokens=False).input_ids
                diff_ids = diff_ids.to(device=device, dtype=torch.long)
                if diff_len + int(diff_ids.shape[1]) <= max_cache_len:
                    diff_buf[:, diff_len : diff_len + int(diff_ids.shape[1])].copy_(diff_ids)
                    diff_len += int(diff_ids.shape[1])
                    diff_prefix_ids = diff_buf[:, :diff_len]
                else:
                    diff_prefix_ids = torch.cat([diff_prefix_ids, diff_ids], dim=1)
                    diff_len = int(diff_prefix_ids.shape[1])
                retok_wall_ms = (time.perf_counter() - rt_t0) * 1000.0
            else:
                if diff_len + add_len <= max_cache_len:
                    diff_buf[:, diff_len : diff_len + add_len].copy_(mapped.unsqueeze(0))
                    diff_len += add_len
                    diff_prefix_ids = diff_buf[:, :diff_len]
                else:
                    diff_prefix_ids = torch.cat([diff_prefix_ids, mapped.unsqueeze(0)], dim=1)
                    diff_len = int(diff_prefix_ids.shape[1])
        prefix_wall_ms = (time.perf_counter() - pf_t0) * 1000.0

        if bool(args.sync_step_boundary) and device.type == "cuda":
            st = time.perf_counter()
            torch.cuda.synchronize()
            sync_wall_ms = (time.perf_counter() - st) * 1000.0

        total_generated += int(len(tokens_to_add))
        total_steps += 1
        total_accept_len += int(accept_len)
        total_added += int(len(tokens_to_add))

        if int(args.log_every) > 0 and (step + 1) % int(args.log_every) == 0:
            total_step_ms = (time.perf_counter() - step_t0) * 1000.0
            accounted_ms = (
                float(diff_wall_ms)
                + float(tree_wall_ms)
                + float(verify_wall_ms)
                + float(accept_wall_ms)
                + float(update_wall_ms)
                + float(prefix_wall_ms)
                + float(debug_compare_wall_ms)
                + float(retok_wall_ms)
                + float(sample_wall_ms)
                + float(map_wall_ms)
                + float(sync_wall_ms)
            )
            pr_t0 = time.perf_counter()
            print(
                f"[Step {step+1:4d}] prefix_len={int(verify_len)} nodes={tree_node_count} mode={tree_build_mode}"
            )
            print(
                f"  [Diff] wall_ms={diff_wall_ms:.2f}  [Tree] wall_ms={tree_wall_ms:.2f} "
                f"  [Verify] wall_ms={verify_wall_ms:.2f}  [Accept] wall_ms={accept_wall_ms:.2f} "
                f"  [Update] wall_ms={update_wall_ms:.2f}  [Prefix] wall_ms={prefix_wall_ms:.2f} "
                f"  [Sample] wall_ms={sample_wall_ms:.2f} "
                f"  [Map] wall_ms={map_wall_ms:.2f}  [DbgCmp] wall_ms={debug_compare_wall_ms:.2f} "
                f"  [Retok] wall_ms={retok_wall_ms:.2f}  [Sync] wall_ms={sync_wall_ms:.2f}"
            )
            print(
                f"  [Accept] mode={args.accept_mode} best_idx={best_idx} "
                f"accept_len={accept_len} add={len(tokens_to_add)}"
            )
            print_wall_ms = (time.perf_counter() - pr_t0) * 1000.0
            other_ms = total_step_ms - accounted_ms - float(print_wall_ms)
            other_ms = max(0.0, other_ms)
            print(f"  [Print] wall_ms={print_wall_ms:.2f}  [Other] wall_ms={other_ms:.2f}")
            if other_ms >= float(args.debug_other_threshold_ms):
                hint: List[str] = []
                if bool(args.debug_tree_compare):
                    hint.append("debug_tree_compare can force large syncs")
                if not bool(args.sync_step_boundary):
                    hint.append("enable --sync_step_boundary to attribute hidden stalls")
                if hint:
                    print("  [OtherHint] " + "; ".join(hint))
            if int(args.debug_nodes) > 0:
                _log_root_candidates(
                    accept_root_tokens,
                    accept_argmax_prefix,
                    verify_tokenizer,
                    int(args.debug_topk),
                    accept_prefix_group,
                )
                if len(tree_logits) > 0:
                    _log_topk(tree_logits, verify_tokenizer, "tree", int(args.debug_topk), int(args.debug_nodes))

    elapsed = time.perf_counter() - t_start
    if device.type == "cuda":
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t_start
    tok_s = float(total_generated) / max(1e-9, elapsed)
    print(f"[Info] generated={total_generated} tokens in {elapsed:.2f}s ({tok_s:.2f} tok/s)")
    if total_steps > 0:
        avg_accept = total_accept_len / float(total_steps)
        avg_add = total_added / float(total_steps)
        accept_rate = total_accept_len / float(max(1, total_steps * int(args.gamma)))
        avg_nodes = total_nodes / float(total_steps)
        summary = (
            f"steps={total_steps} avg_accept_len={avg_accept:.2f} "
            f"avg_add={avg_add:.2f} accept_rate={accept_rate:.3f} "
            f"avg_nodes={avg_nodes:.1f} expected_streak={avg_accept:.2f}"
        )
        print(f"[Summary] {summary}")

    final_text = verify_tokenizer.decode(verify_prefix_ids.squeeze(0), skip_special_tokens=False)
    print("[Final] text=")
    print(final_text)


if __name__ == "__main__":
    main()
