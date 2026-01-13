#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import contextlib
import inspect
import os
import time
from typing import Optional, Tuple

import torch
from transformers import (
    AutoModel,
    AutoModelForCausalLM,
    AutoModelForMaskedLM,
    AutoTokenizer,
)
from transformers.cache_utils import DynamicCache


def _get_dtype(name: str) -> torch.dtype:
    name = name.lower()
    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    raise ValueError(f"Unsupported dtype: {name}")


def _get_logits(outputs, model, hidden_states: torch.Tensor) -> torch.Tensor:
    if hasattr(outputs, "logits") and outputs.logits is not None:
        return outputs.logits
    lm_head = getattr(model, "lm_head", None)
    if lm_head is None:
        lm_head = model.get_output_embeddings()
    if lm_head is None:
        raise RuntimeError("Model does not expose logits or output embeddings.")
    return lm_head(hidden_states)


def _supports_num_logits_to_keep(model) -> bool:
    try:
        return "num_logits_to_keep" in inspect.signature(model.forward).parameters
    except (TypeError, ValueError):
        return False


class _FusedDreamMLP(torch.nn.Module):
    def __init__(self, gate_proj, up_proj, down_proj, act_fn):
        super().__init__()
        self.hidden_size = gate_proj.in_features
        self.intermediate_size = gate_proj.out_features
        self.fused_proj = torch.nn.Linear(self.hidden_size, 2 * self.intermediate_size, bias=False)
        self.down_proj = down_proj
        self.act_fn = act_fn

        with torch.no_grad():
            fused_w = torch.cat([gate_proj.weight, up_proj.weight], dim=0)
            self.fused_proj.weight.copy_(fused_w)

    def forward(self, hidden_state):
        fused = self.fused_proj(hidden_state)
        gate, up = fused.chunk(2, dim=-1)
        return self.down_proj(self.act_fn(gate) * up)


def _is_dream_mlp(module: torch.nn.Module) -> bool:
    return (
        hasattr(module, "gate_proj")
        and hasattr(module, "up_proj")
        and hasattr(module, "down_proj")
        and hasattr(module, "act_fn")
    )


def _fuse_dream_mlps(model: torch.nn.Module, *, verbose: bool = True) -> int:
    targets = []
    for name, module in model.named_modules():
        if isinstance(module, _FusedDreamMLP):
            continue
        if _is_dream_mlp(module):
            targets.append((name, module))

    fused = 0
    for name, module in targets:
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        fused_mlp = _FusedDreamMLP(module.gate_proj, module.up_proj, module.down_proj, module.act_fn)
        param = next(module.parameters())
        fused_mlp.to(device=param.device, dtype=param.dtype)
        setattr(parent, parts[-1], fused_mlp)
        fused += 1

    if verbose:
        print(f"[Fuse] fused_mlps={fused}")
    return fused


class _FusedQKVLinear(torch.nn.Module):
    def __init__(self, q_proj, k_proj, v_proj):
        super().__init__()
        in_features = q_proj.in_features
        q_out = q_proj.out_features
        k_out = k_proj.out_features
        v_out = v_proj.out_features
        has_bias = q_proj.bias is not None
        if (k_proj.bias is not None) != has_bias or (v_proj.bias is not None) != has_bias:
            raise ValueError("q/k/v bias mismatch; cannot fuse")
        self.fused_proj = torch.nn.Linear(in_features, q_out + k_out + v_out, bias=has_bias)
        with torch.no_grad():
            self.fused_proj.weight.copy_(torch.cat([q_proj.weight, k_proj.weight, v_proj.weight], dim=0))
            if has_bias:
                self.fused_proj.bias.copy_(torch.cat([q_proj.bias, k_proj.bias, v_proj.bias], dim=0))
        self._cache_key = None
        self._cache_out = None

    def _key(self, x):
        return (x.data_ptr(), x.shape, x.stride(), x.dtype, x.device)

    def forward(self, x):
        key = self._key(x)
        if self._cache_key == key and self._cache_out is not None:
            return self._cache_out
        out = self.fused_proj(x)
        self._cache_key = key
        self._cache_out = out
        return out


class _FusedQKVSlice(torch.nn.Module):
    def __init__(self, fused: _FusedQKVLinear, start: int, end: int):
        super().__init__()
        self.fused = fused
        self.start = int(start)
        self.end = int(end)

    def forward(self, x):
        fused = self.fused(x)
        return fused[..., self.start : self.end]


def _fuse_qkv_projections(model: torch.nn.Module, *, verbose: bool = True) -> int:
    fused = 0
    for module in model.modules():
        if not (hasattr(module, "q_proj") and hasattr(module, "k_proj") and hasattr(module, "v_proj")):
            continue
        q_proj = module.q_proj
        k_proj = module.k_proj
        v_proj = module.v_proj
        if not isinstance(q_proj, torch.nn.Linear):
            continue
        if not isinstance(k_proj, torch.nn.Linear):
            continue
        if not isinstance(v_proj, torch.nn.Linear):
            continue
        q_out = q_proj.out_features
        k_out = k_proj.out_features
        v_out = v_proj.out_features
        fused_linear = _FusedQKVLinear(q_proj, k_proj, v_proj)
        param = q_proj.weight
        fused_linear.to(device=param.device, dtype=param.dtype)
        module.q_proj = _FusedQKVSlice(fused_linear, 0, q_out)
        module.k_proj = _FusedQKVSlice(fused_linear, q_out, q_out + k_out)
        module.v_proj = _FusedQKVSlice(fused_linear, q_out + k_out, q_out + k_out + v_out)
        fused += 1
    if verbose:
        print(f"[Fuse] fused_qkv={fused}")
    return fused


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


def _sample_from_logits(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    generator: Optional[torch.Generator],
) -> torch.Tensor:
    orig_shape = logits.shape[:-1]
    if logits.ndim > 2:
        logits = logits.reshape(-1, logits.shape[-1])
    if temperature <= 0:
        out = torch.argmax(logits, dim=-1)
        return out.reshape(orig_shape)
    logits = logits / float(temperature)
    logits = logits.to(dtype=torch.float32)
    vocab = logits.shape[-1]
    if top_k > 0:
        k = min(int(top_k), vocab)
        topk_vals, topk_idx = torch.topk(logits, k=k, dim=-1)
        if top_p < 1.0:
            probs = torch.softmax(topk_vals, dim=-1)
            sorted_probs, sorted_idx = torch.sort(probs, dim=-1, descending=True)
            cumprobs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumprobs > float(top_p)
            cutoff[..., 0] = False
            sorted_probs = sorted_probs.masked_fill(cutoff, 0.0)
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_local = torch.multinomial(sorted_probs, num_samples=1, generator=generator)
            chosen = sorted_idx.gather(dim=-1, index=next_local)
            out = topk_idx.gather(dim=-1, index=chosen).squeeze(-1)
            return out.reshape(orig_shape)
        probs = torch.softmax(topk_vals, dim=-1)
        next_local = torch.multinomial(probs, num_samples=1, generator=generator)
        out = topk_idx.gather(dim=-1, index=next_local).squeeze(-1)
        return out.reshape(orig_shape)
    if top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = torch.softmax(sorted_logits, dim=-1)
        cumprobs = torch.cumsum(probs, dim=-1)
        cutoff = cumprobs > float(top_p)
        cutoff[..., 0] = False
        probs = probs.masked_fill(cutoff, 0.0)
        probs = probs / probs.sum(dim=-1, keepdim=True)
        next_local = torch.multinomial(probs, num_samples=1, generator=generator)
        out = sorted_idx.gather(dim=-1, index=next_local).squeeze(-1)
        return out.reshape(orig_shape)
    probs = torch.softmax(logits, dim=-1)
    next_local = torch.multinomial(probs, num_samples=1, generator=generator)
    return next_local.squeeze(-1).reshape(orig_shape)


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


def _parse_profile_activities(spec: str):
    acts = []
    for part in (spec or "").split(","):
        key = part.strip().lower()
        if not key:
            continue
        if key == "cpu":
            acts.append(torch.profiler.ProfilerActivity.CPU)
        elif key == "cuda":
            if torch.cuda.is_available():
                acts.append(torch.profiler.ProfilerActivity.CUDA)
        else:
            raise ValueError(f"Unknown profiler activity: {part}")
    if not acts:
        raise ValueError("No valid profiler activities configured.")
    return acts


@torch.inference_mode()
def semi_autoregressive_generate(
    *,
    model,
    tokenizer,
    prompt: str,
    gamma: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int,
    top_p: float,
    eos_token_id: Optional[int],
    stop_on_eos: bool,
    generator: Optional[torch.Generator],
    log_step_timing: bool,
    log_every: int,
    log_spike_ms: float,
    warmup_steps: int,
    sync_timing: bool,
    omit_attention_mask: bool,
    use_num_logits_to_keep: bool,
    profile_steps: int,
    profile_start_step: int,
    profile_dir: str,
    profile_activities,
    profile_record_shapes: bool,
    profile_memory: bool,
    profile_stack: bool,
    profile_row_limit: int,
    profile_sort: str,
    approx_kv_cache: bool,
) -> Tuple[torch.Tensor, int]:
    device = next(model.parameters()).device
    enc = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)
    prompt_ids = enc.input_ids.to(device=device, dtype=torch.long)
    prompt_len = int(prompt_ids.shape[1])

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id
    if pad_id is None:
        pad_id = 0
    mask_id = _maybe_add_mask_token(model, tokenizer)

    total_len = prompt_len + int(max_new_tokens) + int(gamma)
    input_ids = torch.full((1, total_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((1, total_len), dtype=torch.bool, device=device)
    input_ids[:, :prompt_len] = prompt_ids
    attention_mask[:, :prompt_len] = 1

    cur_len = prompt_len
    generated = 0
    step_idx = 0

    use_cuda = device.type == "cuda"
    start_event = None
    end_event = None
    if use_cuda:
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

    past_key_values = None
    if approx_kv_cache:
        past_key_values = DynamicCache()
        if log_step_timing:
            print("[ApproxCache] enabled (quality may degrade; cache uses masked tokens).")
        _ = model(
            input_ids=input_ids[:, :prompt_len],
            attention_mask=None,
            use_cache=True,
            past_key_values=past_key_values,
            return_dict=True,
            num_logits_to_keep=0,
        )

    profiling = False
    prof = None
    prof_step = 0
    if int(profile_steps) > 0:
        import torch.profiler as tprof

    while generated < max_new_tokens:
        if int(profile_steps) > 0 and (not profiling) and step_idx == int(profile_start_step):
            os.makedirs(profile_dir, exist_ok=True)
            activities = profile_activities
            if activities is None:
                activities = [tprof.ProfilerActivity.CPU]
                if use_cuda:
                    activities.append(tprof.ProfilerActivity.CUDA)
            prof = tprof.profile(
                activities=activities,
                record_shapes=bool(profile_record_shapes),
                profile_memory=bool(profile_memory),
                with_stack=bool(profile_stack),
            )
            prof.__enter__()
            profiling = True
            prof_step = 0
            if log_step_timing:
                print(f"[Profiler] start step={step_idx} steps={int(profile_steps)} dir={profile_dir}")
        step_gamma = min(int(gamma), int(max_new_tokens) - generated)
        mask_slice = slice(cur_len, cur_len + step_gamma)
        input_ids[:, mask_slice] = mask_id
        attention_mask[:, : cur_len + step_gamma] = True

        seq_len = cur_len + step_gamma
        step_total_start = time.perf_counter()
        if use_cuda and sync_timing:
            torch.cuda.synchronize()
        if use_cuda:
            start_event.record()
        else:
            forward_start = time.perf_counter()
        step_input_ids = input_ids[:, :seq_len]
        if approx_kv_cache:
            step_input_ids = input_ids[:, mask_slice]
        attn_mask = None if (omit_attention_mask or approx_kv_cache) else attention_mask[:, :seq_len]
        model_kwargs = {}
        if use_num_logits_to_keep:
            model_kwargs["num_logits_to_keep"] = int(step_gamma)
        if approx_kv_cache:
            model_kwargs["use_cache"] = True
            model_kwargs["past_key_values"] = past_key_values
        else:
            model_kwargs["use_cache"] = False
        outputs = model(
            input_ids=step_input_ids,
            attention_mask=attn_mask,
            return_dict=True,
            **model_kwargs,
        )
        if use_cuda:
            end_event.record()
            if sync_timing:
                torch.cuda.synchronize()
            forward_ms = float(start_event.elapsed_time(end_event))
        else:
            forward_ms = (time.perf_counter() - forward_start) * 1000.0
        step_total_ms = (time.perf_counter() - step_total_start) * 1000.0
        if log_step_timing and step_idx >= warmup_steps:
            should_log = (step_idx % max(1, int(log_every)) == 0)
            if log_spike_ms > 0 and forward_ms >= float(log_spike_ms):
                should_log = True
            if should_log:
                print(
                    f"[Step {step_idx:4d}] seq_len={seq_len} gamma={step_gamma} "
                    f"forward_ms={forward_ms:.2f} total_ms={step_total_ms:.2f}"
                )
        hidden_states = getattr(outputs, "last_hidden_state", None)
        logits = _get_logits(outputs, model, hidden_states)
        if logits.shape[1] == step_gamma or approx_kv_cache:
            step_logits = logits
        else:
            step_logits = logits[:, cur_len:seq_len, :]
        next_ids = _sample_from_logits(
            step_logits,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            generator=generator,
        )
        input_ids[:, mask_slice] = next_ids

        if stop_on_eos and eos_token_id is not None:
            eos_positions = (next_ids == int(eos_token_id)).nonzero(as_tuple=False)
            if eos_positions.numel() > 0:
                first_eos = int(eos_positions[0].item())
                cur_len += first_eos + 1
                generated += first_eos + 1
                break

        cur_len = seq_len
        generated += step_gamma
        step_idx += 1
        if profiling:
            prof.step()
            prof_step += 1
            if prof_step >= int(profile_steps):
                if use_cuda:
                    torch.cuda.synchronize()
                prof.__exit__(None, None, None)
                trace_path = os.path.join(
                    profile_dir, f"trace_step{int(profile_start_step)}_len{seq_len}.json"
                )
                prof.export_chrome_trace(trace_path)
                if log_step_timing:
                    print(f"[Profiler] trace saved to {trace_path}")
                    print(prof.key_averages().table(sort_by=profile_sort, row_limit=int(profile_row_limit)))
                profiling = False

    return input_ids[:, :cur_len], prompt_len


def main() -> None:
    p = argparse.ArgumentParser(description="Semi-autoregressive inference for DiffuCoder-style diffusion models.")
    p.add_argument("--model_name", type=str, default="apple/DiffuCoder-7B-cpGRPO")
    p.add_argument("--prompt", type=str, required=True)
    p.add_argument("--gamma", type=int, default=8, help="Number of masked tokens to draft per step.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_k", type=int, default=0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--dtype", type=str, default="bfloat16", choices=("float16", "bfloat16", "float32"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--trust_remote_code", action="store_true", default=True)
    p.add_argument("--compile", action="store_true", default=False)
    p.add_argument("--compile_mode", type=str, default="reduce-overhead")
    p.add_argument("--compile_fullgraph", action="store_true", default=False)
    p.add_argument("--compile_dynamic", action="store_true", default=True)
    p.add_argument("--compile_backend", type=str, default="inductor")
    p.add_argument("--stop_on_eos", action="store_true", default=True)
    p.add_argument("--log_step_timing", action="store_true", default=True)
    p.add_argument("--log_every", type=int, default=1)
    p.add_argument("--log_spike_ms", type=float, default=0.0)
    p.add_argument("--warmup_steps", type=int, default=2)
    p.add_argument("--sync_timing", action="store_true", default=True)
    p.add_argument("--sdp_backend", type=str, default="auto", choices=("auto", "flash", "mem_efficient", "math"))
    p.add_argument("--log_attention_backend", action="store_true", default=False)
    p.add_argument("--omit_attention_mask", action="store_true", default=False)
    p.add_argument("--fuse_mlp", action="store_true", default=False)
    p.add_argument("--fuse_qkv", action="store_true", default=False)
    p.add_argument("--profile_steps", type=int, default=0)
    p.add_argument("--profile_start_step", type=int, default=0)
    p.add_argument("--profile_dir", type=str, default="profile")
    p.add_argument("--profile_activities", type=str, default="cpu,cuda")
    p.add_argument("--profile_record_shapes", action="store_true", default=False)
    p.add_argument("--profile_memory", action="store_true", default=False)
    p.add_argument("--profile_stack", action="store_true", default=False)
    p.add_argument("--profile_row_limit", type=int, default=20)
    p.add_argument("--profile_sort", type=str, default="cuda_time_total")
    p.add_argument("--approx_kv_cache", action="store_true", default=False)
    args = p.parse_args()

    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    device = torch.device(args.device)
    dtype = _get_dtype(args.dtype)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(args.seed))

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=bool(args.trust_remote_code))
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token

    model = None
    load_errors = []
    for factory in (AutoModelForCausalLM, AutoModelForMaskedLM, AutoModel):
        try:
            model = factory.from_pretrained(
                args.model_name,
                torch_dtype=dtype,
                trust_remote_code=bool(args.trust_remote_code),
            )
            break
        except Exception as exc:
            load_errors.append(str(exc))
    if model is None:
        raise RuntimeError("Failed to load model. Errors:\n" + "\n".join(load_errors))

    model.to(device)
    model.eval()
    if args.fuse_mlp:
        _fuse_dream_mlps(model, verbose=True)
    if args.fuse_qkv:
        _fuse_qkv_projections(model, verbose=True)
    use_num_logits_to_keep = _supports_num_logits_to_keep(model)
    if int(args.profile_start_step) < 0:
        args.profile_start_step = 0
    profile_activities = None
    if int(args.profile_steps) > 0:
        profile_activities = _parse_profile_activities(args.profile_activities)

    if args.compile:
        try:
            model = torch.compile(
                model,
                mode=str(args.compile_mode),
                fullgraph=bool(args.compile_fullgraph),
                dynamic=bool(args.compile_dynamic),
                backend=str(args.compile_backend),
            )
        except TypeError:
            model = torch.compile(model, mode=str(args.compile_mode), fullgraph=bool(args.compile_fullgraph))

    sdp_ctx = _build_sdp_context(args.sdp_backend, verbose=bool(args.log_attention_backend))

    t0 = time.time()
    with sdp_ctx:
        output_ids, prompt_len = semi_autoregressive_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=args.prompt,
            gamma=int(args.gamma),
            max_new_tokens=int(args.max_new_tokens),
            temperature=float(args.temperature),
            top_k=int(args.top_k),
            top_p=float(args.top_p),
            eos_token_id=tokenizer.eos_token_id,
            stop_on_eos=bool(args.stop_on_eos),
            generator=generator,
            log_step_timing=bool(args.log_step_timing),
            log_every=int(args.log_every),
            log_spike_ms=float(args.log_spike_ms),
            warmup_steps=int(args.warmup_steps),
            sync_timing=bool(args.sync_timing),
            omit_attention_mask=bool(args.omit_attention_mask),
            use_num_logits_to_keep=use_num_logits_to_keep,
            profile_steps=int(args.profile_steps),
            profile_start_step=int(args.profile_start_step),
            profile_dir=str(args.profile_dir),
            profile_activities=profile_activities,
            profile_record_shapes=bool(args.profile_record_shapes),
            profile_memory=bool(args.profile_memory),
            profile_stack=bool(args.profile_stack),
            profile_row_limit=int(args.profile_row_limit),
            profile_sort=str(args.profile_sort),
            approx_kv_cache=bool(args.approx_kv_cache),
        )
    dt = time.time() - t0
    text = tokenizer.decode(output_ids[0, prompt_len:], skip_special_tokens=True)
    print(text)
    print(f"[Info] generated={output_ids.shape[1] - prompt_len} tokens in {dt:.2f}s")


if __name__ == "__main__":
    main()
