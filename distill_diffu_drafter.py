#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Finetune a diffusion drafter (DiffuCoder/Dream-style) on verifier generations using γ-window masking,
optimizing E[streak] directly (with Dream's left-shifted logits + position indices), VRAM-safe.

Stage A (optional): generate verifier completions -> JSONL
Stage B: finetune drafter via γ-length masked blocks:
  For each start offset s: feed [prompt || completion[:s] || <MASK>×γ], get drafter logits,
  apply Dream left-shift, compute E[streak] = Σ_{j=1..γ} Π_{i<=j} q_i where q_i is prob on verifier token i,
  and maximize E[streak] (we minimize -E[streak]).

Key implementation details (important for correct E[streak]):
- Bidirectional attention via 4D additive mask over non-pad pairs.
- Dream-style position_ids `tok_idx` computed from non-pad cumsum and passed to the model.
- Request `num_logits_to_keep = γ + 1`, then apply a one-step left-shift of logits and keep the last γ steps.
- Softmax temperature matches drafter_temp used at inference for consistent probabilities.

This version adds **minimal** Stage A support for the same benchmarks used by the
benchmarking script: alpaca | MT-Bench | math-500 | gpqa | gsm8k | livecodebench.
Use `--benchmark` (and optional `--lcb_version_tag`) to switch prompt sources.
If `--benchmark` is omitted/empty, the original dataset_name/config/split path is used unchanged.
"""

import os
import json
import time
import random
import argparse
import math
import re
import subprocess
import sys
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForCausalLM,
    GenerationConfig,
    get_linear_schedule_with_warmup
)

# ----- LoRA (optional) -----
_HAS_PEFT = False
try:
    from peft import LoraConfig, get_peft_model, PeftModel
    _HAS_PEFT = True
except Exception:
    _HAS_PEFT = False
    class _Dummy: pass
    PeftModel = _Dummy  # placeholder for isinstance guard

# ----- bitsandbytes (optional) -----
_HAS_BNB = False
try:
    import bitsandbytes as bnb  # noqa: F401
    _HAS_BNB = True
except Exception:
    _HAS_BNB = False


# -------------------------- Utils --------------------------

def seed_all(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def log(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def jsonl_write(path: str, obj: dict):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -------------------------- Stage A prompt builders (NEW) --------------------------

def parse_int_list(value: str) -> List[int]:
    if not value:
        return []
    items = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        items.append(int(part))
    return items


def load_eval_prompts(jsonl_path: str, num_prompts: int, seed: int, prompt_field: str) -> List[str]:
    prompts: List[str] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            prompt = obj.get(prompt_field)
            if isinstance(prompt, str) and prompt.strip():
                prompts.append(prompt.strip())
    if not prompts:
        return []
    if num_prompts <= 0 or num_prompts >= len(prompts):
        return prompts
    rng = random.Random(seed)
    return rng.sample(prompts, num_prompts)


@dataclass
class RunMetrics:
    tok_s: float
    elapsed_s: float
    generated: int
    accept_rate: Optional[float] = None
    avg_nodes: Optional[float] = None
    avg_accept_len: Optional[float] = None
    avg_add: Optional[float] = None
    gpu_util: Optional[float] = None
    tok_s_std: Optional[float] = None


class GpuUtilSampler:
    def __init__(self, interval_s: float = 0.1):
        self._interval_s = float(interval_s)
        self._stop = False
        self._vals: List[float] = []
        self._thread = None
        self._nvml = None
        self._nvml_handle = None
        self._last_smi_t = 0.0
        self._last_smi_val: Optional[float] = None

    def __enter__(self):
        try:
            import pynvml  # type: ignore

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None
            self._nvml_handle = None
            self._interval_s = max(self._interval_s, 0.25)

        def query() -> Optional[float]:
            if self._nvml is None or self._nvml_handle is None:
                now = time.perf_counter()
                if now - self._last_smi_t < 0.75:
                    return self._last_smi_val
                self._last_smi_t = now
                try:
                    out = subprocess.check_output(
                        [
                            "nvidia-smi",
                            "--query-gpu=utilization.gpu",
                            "--format=csv,noheader,nounits",
                        ],
                        text=True,
                        stderr=subprocess.DEVNULL,
                    ).strip()
                    if not out:
                        return self._last_smi_val
                    line = out.splitlines()[0].strip()
                    self._last_smi_val = float(line)
                    return self._last_smi_val
                except Exception:
                    return self._last_smi_val
            try:
                return float(self._nvml.nvmlDeviceGetUtilizationRates(self._nvml_handle).gpu)
            except Exception:
                return None

        def run():
            while not self._stop:
                util = query()
                if util is not None:
                    self._vals.append(float(util))
                time.sleep(self._interval_s)

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self._stop = True
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._nvml is not None:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass
        return False

    def mean(self) -> Optional[float]:
        if not self._vals:
            return None
        return sum(self._vals) / len(self._vals)


def _parse_metrics(output: str) -> RunMetrics:
    info_re = re.compile(r"generated=(\d+) tokens in ([\d.]+)s \(([\d.]+) tok/s\)")
    summary_re = re.compile(
        r"avg_accept_len=([\d.]+)\s+avg_add=([\d.]+)\s+accept_rate=([\d.]+)\s+avg_nodes=([\d.]+)"
    )
    batch_re = re.compile(
        r"prompt_tok_s_mean=([\d.]+)\s+prompt_tok_s_std=([\d.]+)\s+prompt_tok_s_n=(\d+)"
    )
    generated = 0
    elapsed_s = 0.0
    tok_s = 0.0
    accept_rate = None
    avg_nodes = None
    avg_accept_len = None
    avg_add = None
    tok_s_std = None

    for line in output.splitlines():
        m = info_re.search(line)
        if m:
            generated = int(m.group(1))
            elapsed_s = float(m.group(2))
            tok_s = float(m.group(3))
        s = summary_re.search(line)
        if s:
            avg_accept_len = float(s.group(1))
            avg_add = float(s.group(2))
            accept_rate = float(s.group(3))
            avg_nodes = float(s.group(4))
        b = batch_re.search(line)
        if b:
            tok_s = float(b.group(1))
            tok_s_std = float(b.group(2))

    if tok_s <= 0.0:
        raise RuntimeError("Could not parse throughput from specdiff_decode output.")
    return RunMetrics(
        tok_s=tok_s,
        elapsed_s=elapsed_s,
        generated=generated,
        accept_rate=accept_rate,
        avg_nodes=avg_nodes,
        avg_accept_len=avg_accept_len,
        avg_add=avg_add,
        tok_s_std=tok_s_std,
    )


def _maybe_write_metrics(path: Optional[str], payload: Dict[str, Any]) -> None:
    if not path:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def _optimizer_to(optimizer: Optional[torch.optim.Optimizer], device: torch.device) -> None:
    if optimizer is None:
        return
    for state in optimizer.state.values():
        for key, value in state.items():
            if torch.is_tensor(value):
                state[key] = value.to(device=device)


def _run_specdiff(
    prompt_file: str,
    gamma: int,
    k: int,
    draft_temp: float,
    args: argparse.Namespace,
    run_idx: int,
) -> RunMetrics:
    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "specdiff_decode.py"),
        "--drafter",
        args.eval_drafter,
        "--verifier",
        args.eval_verify_model_id,
        "--prompt_file",
        prompt_file,
        "--n_prompts",
        str(int(args.eval_num_prompts)),
        "--gamma",
        str(gamma),
        "--k",
        str(k),
        "--max_new_tokens",
        str(int(args.eval_max_new_tokens)),
        "--diffusion_temperature",
        str(draft_temp),
        "--verify_temperature",
        str(float(args.eval_verify_temperature)),
        "--seed",
        str(int(args.eval_seed) + int(run_idx)),
        "--no-print_final",
    ]
    if args.eval_steps is not None:
        cmd.extend(["--steps", str(int(args.eval_steps))])
    if args.eval_warmup_steps is not None:
        cmd.extend(["--warmup_steps", str(int(args.eval_warmup_steps))])
    if args.eval_log_every is not None:
        cmd.extend(["--log_every", str(int(args.eval_log_every))])
    if args.eval_diffusion_top_k is not None:
        cmd.extend(["--diffusion_top_k", str(int(args.eval_diffusion_top_k))])
    if args.eval_diffusion_top_p is not None:
        cmd.extend(["--diffusion_top_p", str(float(args.eval_diffusion_top_p))])
    if args.eval_dtype:
        cmd.extend(["--dtype", str(args.eval_dtype)])
    if args.eval_device:
        cmd.extend(["--device", str(args.eval_device)])
    if args.eval_tree_build_mode:
        cmd.extend(["--tree_build_mode", str(args.eval_tree_build_mode)])
    if args.eval_diffusion_sdp_backend:
        cmd.extend(["--diffusion_sdp_backend", str(args.eval_diffusion_sdp_backend)])
    if args.eval_verify_sdp_backend:
        cmd.extend(["--verify_sdp_backend", str(args.eval_verify_sdp_backend)])
    if bool(args.approx_kv_cache):
        cmd.append("--approx_kv_cache")
    if bool(args.eval_diffusion_next_token):
        cmd.append("--diffusion_next_token")
    else:
        cmd.append("--no-diffusion_next_token")
    if bool(args.eval_diffusion_fuse_mlp):
        cmd.append("--diffusion_fuse_mlp")
    if bool(args.eval_diffusion_fuse_qkv):
        cmd.append("--diffusion_fuse_qkv")
    if bool(args.eval_trust_remote_code):
        cmd.append("--trust_remote_code")

    env = None
    if bool(args.local_files_only):
        env = dict(os.environ)
        env["HF_HUB_OFFLINE"] = "1"
        env["TRANSFORMERS_OFFLINE"] = "1"

    with GpuUtilSampler(interval_s=float(args.eval_util_sample_interval)) as sampler:
        proc = subprocess.run(cmd, text=True, capture_output=True, env=env)
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    if proc.returncode != 0:
        raise RuntimeError(f"specdiff_decode failed:\n{output}")
    metrics = _parse_metrics(output)
    metrics.gpu_util = sampler.mean()
    return metrics


def _prepare_eval_model(
    model,
    tokenizer,
    output_dir: str,
    base_model_id: str,
    local_files_only: bool,
    bf16: bool,
    model_revision: Optional[str] = None,
) -> str:
    if is_lora_model(model):
        adapter_dir = os.path.join(output_dir, "adapter")
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        merge_dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float32
        base = AutoModel.from_pretrained(
            base_model_id,
            trust_remote_code=True,
            torch_dtype=merge_dtype,
            local_files_only=local_files_only,
            device_map="cpu",
            revision=model_revision,
        )
        if tokenizer is not None:
            base.resize_token_embeddings(len(tokenizer))
        base = PeftModel.from_pretrained(base, adapter_dir)
        merged = base.merge_and_unload()
        merged.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    else:
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
    return output_dir


def run_live_eval(
    *,
    model,
    tokenizer,
    step: int,
    epoch: int,
    args,
    eval_prompts: List[str],
    eval_k_list: List[int],
) -> None:
    if not eval_prompts or not eval_k_list:
        log("[Eval] no prompts or k values available; skipping.")
        return

    specdiff_path = os.path.join(os.path.dirname(__file__), "specdiff_decode.py")
    if not os.path.exists(specdiff_path):
        raise FileNotFoundError(f"specdiff_decode.py not found at {specdiff_path}")

    log(f"[Eval] preparing merged model at step={step} (prompts={len(eval_prompts)})")
    tmp_dir = None
    if bool(args.eval_keep_merged):
        eval_dir = os.path.join(args.output_dir, f"eval-merged-step-{step}")
        os.makedirs(eval_dir, exist_ok=True)
    else:
        tmp_dir = tempfile.TemporaryDirectory(dir=args.output_dir, prefix=f"eval-merged-step-{step}-")
        eval_dir = tmp_dir.name

    eval_dir = _prepare_eval_model(
        model=model,
        tokenizer=tokenizer,
        output_dir=eval_dir,
        base_model_id=args.drafter_model_id,
        local_files_only=args.local_files_only,
        bf16=args.bf16,
        model_revision=(getattr(args, "eval_model_revision", None) or None),
    )

    eval_dtype = args.eval_dtype
    if not eval_dtype:
        eval_dtype = "bfloat16" if args.bf16 else "float32"
    eval_max_new_tokens = args.eval_max_new_tokens if args.eval_max_new_tokens is not None else args.max_new_tokens
    eval_diffusion_temperature = (
        args.eval_diffusion_temperature if args.eval_diffusion_temperature is not None else args.drafter_temp
    )
    args.eval_max_new_tokens = int(eval_max_new_tokens)
    args.eval_diffusion_temperature = float(eval_diffusion_temperature)
    args.eval_dtype = str(eval_dtype)
    args.eval_drafter = str(eval_dir)
    args.eval_num_prompts = int(len(eval_prompts))
    eval_start = time.perf_counter()
    prompt_file = os.path.join(eval_dir, "eval_prompts.txt")
    with open(prompt_file, "w", encoding="utf-8") as f:
        for item in eval_prompts:
            f.write(item.replace("\n", " ").strip() + "\n")

    try:
        k_pbar = tqdm(eval_k_list, desc="Eval K sweep", leave=False)
        run_idx = 0
        for k_val in k_pbar:
            log(f"[Eval] k={k_val} starting ({len(eval_prompts)} prompts)")
            k_start = time.perf_counter()
            try:
                metrics = _run_specdiff(
                    prompt_file=prompt_file,
                    gamma=int(args.gamma),
                    k=int(k_val),
                    draft_temp=float(args.eval_diffusion_temperature),
                    args=args,
                    run_idx=run_idx,
                )
            except Exception as exc:
                log(f"[Eval] specdiff_decode failed (k={k_val}): {exc}")
                continue
            run_idx += 1

            tok_s = float(metrics.tok_s)
            tok_s_std = float(metrics.tok_s_std) if metrics.tok_s_std is not None else 0.0
            n_prompts = int(len(eval_prompts))
            se_tok = float(tok_s_std) / math.sqrt(n_prompts) if n_prompts > 1 else 0.0
            k_elapsed = time.perf_counter() - k_start
            log(
                f"[Eval] k={k_val} tok/s mean={tok_s:.2f} se={se_tok:.2f} "
                f"n={n_prompts} util={metrics.gpu_util if metrics.gpu_util is not None else 'n/a'}"
            )
            _maybe_write_metrics(
                args.metrics_jsonl,
                {
                    "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "epoch": epoch,
                    "step": step,
                    "split": "eval",
                    "k": int(k_val),
                    "tok_s_mean": tok_s,
                    "tok_s_std": tok_s_std,
                    "tok_s_se": se_tok,
                    "gpu_util": metrics.gpu_util,
                    "generated": metrics.generated,
                    "elapsed_s": metrics.elapsed_s,
                    "accept_rate": metrics.accept_rate,
                    "avg_nodes": metrics.avg_nodes,
                    "avg_accept_len": metrics.avg_accept_len,
                    "avg_add": metrics.avg_add,
                    "num_prompts": n_prompts,
                    "eval_elapsed_s": float(k_elapsed),
                    "eval_max_new_tokens": int(eval_max_new_tokens),
                    "gamma": int(args.gamma),
                    "approx_kv_cache": bool(args.approx_kv_cache),
                    "distill_objective": str(args.distill_objective),
                },
            )
    finally:
        eval_elapsed = time.perf_counter() - eval_start
        log(f"[Eval] done in {eval_elapsed:.2f}s")
        if tmp_dir is not None:
            tmp_dir.cleanup()

def build_prompts_from_benchmark(
    benchmark: Optional[str],
    num_samples: int,
    lcb_version_tag: Optional[str] = None,
) -> Optional[List[str]]:
    """Minimal, self-contained prompt loader that mirrors the benchmarking script.
    Returns a list of prompts for known benchmarks, or None if `benchmark` is empty/unsupported.
    Supported: alpaca | MT-Bench | math-500 | gpqa | gsm8k | livecodebench
    """
    if not benchmark:
        return None
    bm = benchmark.strip().lower()

    if bm in ("alpaca", "alpaca-cleaned"):
        ds = load_dataset("tatsu-lab/alpaca", split="train").select(range(num_samples))
        def _mk(row):
            instr = (row.get("instruction") or "").strip()
            inp = (row.get("input") or "").strip()
            if inp:
                return f"Instruction: {instr}\n\nInput: {inp}\n\nAnswer:"
            return f"Instruction: {instr}\n\nAnswer:"
        return [_mk(r) for r in ds]

    if bm in ("mt-bench", "mt_bench", "mtbench"):
        ds = load_dataset("HuggingFaceH4/mt_bench_prompts", split="train").select(range(num_samples))
        def _mk(row):
            turns = row.get("prompt")
            if isinstance(turns, list) and len(turns) >= 1:
                fst = (turns[0] or "").strip()
                return f"{fst}\n\nAnswer:"
            return f"{str(row.get('prompt'))}\n\nAnswer:"
        return [_mk(r) for r in ds]

    if bm in ("math-500", "math500", "math"):
        ds = load_dataset("HuggingFaceH4/MATH-500", split="test").select(range(num_samples))
        def _mk(row):
            prob = (row.get("problem") or "").strip()
            return f"Problem: {prob}\nAnswer:"
        return [_mk(r) for r in ds]

    if bm in ("gpqa",):
        ds = load_dataset("casimiir/gpqa", split="test").select(range(num_samples))
        letters = ["A","B","C","D","E","F","G","H"]
        def _mk(row):
            q = (row.get("question") or "").strip()
            choices = row.get("choices") or []
            lines = [f"{letters[i]}. {choices[i]}" for i in range(min(len(choices), len(letters)))]
            return "Question: " + q + "\nChoices:\n" + "\n".join(lines) + "\nAnswer (A/B/C/D):"
        return [_mk(r) for r in ds]

    if bm in ('gsm8k', ):
        ds = load_dataset("openai/gsm8k", "main", split="test").select(range(num_samples))
        prompts = [f"Question: {row['question'].strip()}\nAnswer:" for row in ds]
        return prompts

    if bm in ("livecodebench", "lcb", "lcb-codegen", "lcb_codegen", "livecodebench-codegen", "livecodebench-code_generation"):
        vtag = (lcb_version_tag or "release_latest").strip()
        ds = load_dataset("livecodebench/code_generation_lite", version_tag=vtag, split="test", trust_remote_code=True).select(range(num_samples))

        def _mk(row):
            title = (row.get("question_title") or "").strip()
            q = (row.get("question_content") or "").strip()
            starter = (row.get("starter_code") or "").strip()
            header = f"### LiveCodeBench Problem: {title}\n\n### Question:\n{q}\n"
            if starter:
                header += f"\n### Starter code:\n{starter}\n"
            header += "\n### Answer:\n"
            return header
        return [_mk(r) for r in ds]

    return None  # fall back to dataset_name/config/split path


# -------------------------- Stage A: data generation --------------------------

@torch.no_grad()
def generate_with_verifier(
    model_id: str,
    num_prompts: int,
    out_jsonl: str,
    max_new_tokens: int = 128,
    verifier_num_completions: int = 4,
    temperature: float = 1.0,
    top_p: float = 0.95,
    local_files_only: bool = False,
    dataset_name: str = "openai/gsm8k",
    dataset_config: str = "main",
    dataset_split: str = "test",
    seed: int = 1235,
    # NEW: benchmark prompt source (mirrors verify script)
    benchmark: str = "",
    lcb_version_tag: str = "release_latest",
):
    log(f"Stage A: loading verifier: {model_id}")
    tok_v = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True,
                                          local_files_only=local_files_only)
    if tok_v.pad_token_id is None and tok_v.eos_token_id is not None:
        tok_v.pad_token_id = tok_v.eos_token_id

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    verifier = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", trust_remote_code=True,
        torch_dtype=dtype, local_files_only=local_files_only
    ).eval()

    # ---- NEW: try benchmark prompts first ----
    prompts = build_prompts_from_benchmark(benchmark, num_prompts, lcb_version_tag)

    # Fallback to the original dataset path (unchanged behaviour)
    if prompts is None:
        ds = load_dataset(dataset_name, dataset_config, split=dataset_split).select(range(num_prompts))
        if "question" in ds.column_names:
            prompts = [f"Question: {row['question'].strip()}\nAnswer:" for row in ds]
        else:
            col = ds.column_names[0]
            prompts = [str(row[col]).strip() for row in ds]

    gcfg = GenerationConfig(
        do_sample=True, temperature=temperature, top_p=top_p,
        max_new_tokens=max_new_tokens, pad_token_id=tok_v.eos_token_id,
        eos_token_id=tok_v.eos_token_id, return_dict_in_generate=True
    )

    log(f"Generating {verifier_num_completions} samples per prompt (total prompts={len(prompts)})")
    pbar = tqdm(total=len(prompts) * verifier_num_completions, desc="verifier: sampling", leave=False)
    for prompt in prompts:
        enc = tok_v(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(verifier.device)
        attn = enc.get("attention_mask", torch.ones_like(input_ids)).to(verifier.device)

        for _ in range(verifier_num_completions):
            out = verifier.generate(input_ids=input_ids, attention_mask=attn, generation_config=gcfg)
            seq = out.sequences[0]
            full_text = tok_v.decode(seq, skip_special_tokens=True)
            prompt_text = tok_v.decode(input_ids[0], skip_special_tokens=True)

            comp = full_text[len(prompt_text):]
            for s in (tok_v.all_special_tokens or []):
                if s and s in comp:
                    comp = comp.replace(s, "")

            row = {"prompt": prompt, "completion": comp}
            jsonl_write(out_jsonl, row)
            pbar.update(1)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    pbar.close()

    del verifier
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    log(f"Stage A done. JSONL: {out_jsonl}")


# -------------------------- Stage B: dataset (windows) --------------------------

@dataclass
class WindowIndex:
    row_idx: int
    start: int

class WindowedMaskedDataset(Dataset):
    """
    Builds γ-length windows over each (prompt, completion) pair from the JSONL.
    For start offset s:
      input_ids = [prompt_tokens || completion[:s] || <MASK>×γ]
      labels    = [-100 ... -100 || verifier_tokens[s : s+γ]]
    """
    def __init__(
        self,
        jsonl_path: str,
        tok,
        gamma: int,
        max_seq_len: int = 2048,
        trim_prefix: bool = True,
    ):
        super().__init__()
        self.tok = tok
        self.gamma = gamma
        self.max_seq_len = max_seq_len
        self.trim_prefix = trim_prefix

        self.rows: List[Dict[str, str]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if not obj.get("prompt") or not obj.get("completion"):
                        continue
                    self.rows.append({"prompt": obj["prompt"], "completion": obj["completion"]})
                except Exception:
                    continue

        self.special_ids = set(self.tok.all_special_ids or [])
        self._diagnostics = {"filtered_comp_tokens": 0}

        self.pre_enc: List[Tuple[List[int], List[int]]] = []
        self.index: List[WindowIndex] = []
        for i, row in enumerate(tqdm(self.rows, desc="Tokenizing", leave=False)):
            p_ids = self.tok(row["prompt"], add_special_tokens=False)["input_ids"]
            c_ids = self.tok(row["completion"], add_special_tokens=False)["input_ids"]

            # Remove special tokens from completion
            c_ids_f = [int(t) for t in c_ids if int(t) not in self.special_ids]
            self._diagnostics["filtered_comp_tokens"] += (len(c_ids) - len(c_ids_f))

            self.pre_enc.append((p_ids, c_ids_f))
            if len(c_ids_f) >= self.gamma:
                for start in range(0, len(c_ids_f) - self.gamma + 1):
                    self.index.append(WindowIndex(i, start))

        if not self.index:
            raise RuntimeError("No training windows produced. Check JSONL and gamma.")

        self.pad_id = self.tok.pad_token_id
        self.mask_id = self.tok.mask_token_id
        if self.pad_id is None: raise RuntimeError("Tokenizer pad_token_id is None.")
        if self.mask_id is None: raise RuntimeError("Tokenizer mask_token_id is None.")
        log(f"[dataset] Filtered {self._diagnostics['filtered_comp_tokens']} special tokens. "
            f"Created {len(self.index)} training windows.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        wi = self.index[idx]
        p_ids, c_ids_f = self.pre_enc[wi.row_idx]
        start = wi.start

        # Prefix includes completion[:start]
        prefix_ids = p_ids + c_ids_f[:start]

        # Ensure prefix + gamma ≤ max_seq_len
        if self.trim_prefix and len(prefix_ids) + self.gamma > self.max_seq_len:
            keep = max(0, self.max_seq_len - self.gamma)
            prefix_ids = prefix_ids[-keep:]

        input_ids = prefix_ids + [self.mask_id] * self.gamma
        labels    = ([-100] * len(prefix_ids)) + c_ids_f[start:start + self.gamma]

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pad_collate(batch: List[Dict[str, torch.Tensor]], pad_id: int) -> Dict[str, torch.Tensor]:
    maxlen = max(x["input_ids"].shape[0] for x in batch)
    B = len(batch)
    input_ids = torch.full((B, maxlen), pad_id, dtype=torch.long)
    labels = torch.full((B, maxlen), -100, dtype=torch.long)
    for i, ex in enumerate(batch):
        L = ex["input_ids"].shape[0]
        input_ids[i, :L] = ex["input_ids"]
        labels[i,   :L] = ex["labels"]
    return {"input_ids": input_ids, "labels": labels}


# -------------------------- LoRA helpers --------------------------

def build_lora(model, r: int = 16, alpha: int = 32, dropout: float = 0.05):
    if not _HAS_PEFT:
        raise RuntimeError("peft is not installed but LoRA was requested.")
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    cfg = LoraConfig(
        r=r, lora_alpha=alpha, lora_dropout=dropout, bias="none",
        task_type="CAUSAL_LM", target_modules=target_modules
    )
    return get_peft_model(model, cfg)


def is_lora_model(m) -> bool:
    if not _HAS_PEFT:
        return False
    try:
        return isinstance(m, PeftModel)
    except Exception:
        return hasattr(m, "peft_config")


def save_lora_or_full(model, tokenizer, path: str, strategy: str = "adapters"):
    """
    Saver:
      - If LoRA and strategy == 'adapters': save ONLY adapters.
      - If LoRA and strategy == 'merged': merge_and_unload() then save full.
      - If not LoRA: save full model.
    Tokenizer saved alongside.
    """
    os.makedirs(path, exist_ok=True)
    if is_lora_model(model):
        if strategy == "merged":
            merged = model.merge_and_unload()
            merged.save_pretrained(path)
        else:
            model.save_pretrained(path)
    else:
        model.save_pretrained(path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path)


# -------------------------- Dream-style attention & positions --------------------------

def build_additive_mask_from_attn2d(
    attn2d: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    *,
    labels: Optional[torch.Tensor] = None,
    approx_kv_cache_mode: bool = False,
) -> torch.Tensor:
    """
    attn2d: (B, L) with 1.0 for non-pad tokens and 0.0 for pad tokens (float or bool/long acceptable).
    Returns a 4D additive attention mask (B, 1, L, L) with 0 on valid pairs and −inf otherwise.
    """
    if attn2d.dtype != torch.bool:
        valid = attn2d > 0
    else:
        valid = attn2d
    pair = torch.logical_and(valid.unsqueeze(1).unsqueeze(-2),  # (B,1,L,1)
                             valid.unsqueeze(1).unsqueeze(-1))  # (B,1,1,L)
    if approx_kv_cache_mode and labels is not None:
        prefix = valid & labels.eq(-100)
        masked = valid & labels.ne(-100)
        block = prefix.unsqueeze(1).unsqueeze(-1) & masked.unsqueeze(1).unsqueeze(-2)
        pair = pair & ~block
    zero = torch.zeros(1, dtype=dtype, device=device)
    neg  = torch.full((1,), torch.finfo(dtype).min, dtype=dtype, device=device)
    return torch.where(pair, zero, neg).expand(pair.shape).contiguous()


def make_tok_idx(attn2d: torch.Tensor) -> torch.Tensor:
    """
    Build Dream-style position_ids ("tok_idx"):
      - cumulative sum over non-pad tokens (1s), minus 1
      - for padded positions, set to 1 (or any constant) as they're masked anyway
    attn2d must be (B, L) with 1.0 (or True) for valid tokens else 0.
    """
    valid = (attn2d > 0).long()
    tok_idx = valid.cumsum(-1) - 1
    tok_idx.masked_fill_(valid.eq(0), 1)
    return tok_idx


# -------------------------- E[streak] objective with Dream alignment --------------------------

def estreak_loss_dream_aligned(
    logits: torch.Tensor,
    labels_last: torch.Tensor,
    drafter_temp: float,
    objective: str = "streak",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Inputs:
      logits: (B, T, V) for the *last T* time steps, where T should be >= γ+1.
              We will apply Dream's left-shift and then align to the last γ.
      labels_last: (B, γ) integer target IDs for the γ masked positions (verifier tokens).
      drafter_temp: temperature to match deployment probabilities.

    Returns:
      loss: scalar tensor to minimize
      e_streak_mean: scalar objective value over the batch (E[streak] for streak, weighted log for CE-streak)
    """
    B, T, V = logits.shape
    gamma = labels_last.size(1)

    # Dream left-shift: prob for position t comes from logits at t-1 (except first slot).
    logits = torch.cat([logits[:, :1, :], logits[:, :-1, :]], dim=1)

    # Keep the last γ positions after shift
    if T != gamma:
        logits = logits[:, -gamma:, :]

    # Softmax with deployment temperature
    if drafter_temp is not None and drafter_temp > 0:
        probs = torch.softmax(logits / float(drafter_temp), dim=-1)
    else:
        probs = torch.softmax(logits, dim=-1)

    # q_i = prob assigned to the verifier token i
    p_correct = probs.gather(-1, labels_last.unsqueeze(-1)).squeeze(-1)  # (B, γ)

    # Safety for any -100 slips (shouldn't be present in last γ, but be robust)
    p_correct = torch.where(labels_last.eq(-100), torch.ones_like(p_correct), p_correct).clamp_min(1e-8)

    prefix_one = torch.ones((B, 1), dtype=p_correct.dtype, device=p_correct.device)
    cumprod = torch.cumprod(torch.cat([prefix_one, p_correct], dim=1), dim=1)[:, 1:]  # (B, γ)

    if objective == "ce_streak":
        log_p = p_correct.clamp_min(1e-8).log()
        weights = torch.cat([prefix_one, cumprod[:, :-1]], dim=1).detach()  # (B, γ)
        e_vals = (weights * log_p).sum(dim=-1)
    else:
        e_vals = cumprod.sum(dim=-1)

    e_mean = e_vals.mean()
    loss = -e_mean
    return loss, e_mean


# -------------------------- Training loop --------------------------

def train_on_windows(
    jsonl_path: str,
    drafter_model_id: str,
    output_dir: str,
    gamma: int = 11,
    drafter_temp: float = 1.0,
    max_seq_len: int = 2048,
    epochs: int = 3,
    batch_size: int = 1,
    grad_accum: int = 8,
    lr: float = 2e-5,
    weight_decay: float = 0.01,
    warmup_steps: int = 100,
    log_every: int = 50,
    save_every: int = 1000,
    bf16: bool = True,
    no_lora: bool = False,
    lora_r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    local_files_only: bool = False,
    val_ratio: float = 0.05,
    num_workers: int = 2,
    metrics_jsonl: Optional[str] = None,
    seed: int = 1234,
    save_strategy: str = "adapters",
    approx_kv_cache_mode: bool = False,
    distill_objective: str = "streak",
    cli_args: Optional[argparse.Namespace] = None,
):
    os.makedirs(output_dir, exist_ok=True)
    seed_all(seed)

    log(f"Stage B: loading drafter: {drafter_model_id}")
    tok_d = AutoTokenizer.from_pretrained(drafter_model_id, trust_remote_code=True,
                                          local_files_only=local_files_only)
    if tok_d.pad_token_id is None:
        raise RuntimeError("Drafter tokenizer pad_token_id is None.")
    if tok_d.mask_token_id is None:
        raise RuntimeError("Drafter tokenizer mask_token_id is None.")
    pad_id = tok_d.pad_token_id

    model_dtype = torch.bfloat16 if (bf16 and torch.cuda.is_available()) else torch.float32
    model = AutoModel.from_pretrained(
        drafter_model_id, trust_remote_code=True, torch_dtype=model_dtype, local_files_only=local_files_only
    )

    # Keep vocab synced to tokenizer
    model.resize_token_embeddings(len(tok_d))
    log(f"Synced model vocabulary size to tokenizer size: {len(tok_d)}")

    model = model.to(get_device())
    model.train()

    # LoRA
    if not no_lora:
        model = build_lora(model, r=lora_r, alpha=lora_alpha, dropout=lora_dropout)
        model.train()
        try:
            model.print_trainable_parameters()
        except Exception:
            pass

    # Optimizer
    if _HAS_BNB:
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    # Data
    ds = WindowedMaskedDataset(jsonl_path=jsonl_path, tok=tok_d, gamma=gamma, max_seq_len=max_seq_len)
    val_len = max(1, int(len(ds) * val_ratio))
    train_len = len(ds) - val_len
    train_ds, val_ds = random_split(ds, [train_len, val_len], generator=torch.Generator().manual_seed(seed))

    collate_fn = lambda b: pad_collate(b, pad_id)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                              pin_memory=True, collate_fn=collate_fn, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=max(1, batch_size), shuffle=False, num_workers=num_workers,
                            pin_memory=True, collate_fn=collate_fn, drop_last=False)

    # Scheduler
    total_steps = max(1, (len(train_loader) // max(1, grad_accum)) * epochs)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=total_steps)

    step = 0
    if metrics_jsonl:
        open(metrics_jsonl, "a").close()

    log(f"Training windows: {len(train_ds)} | Val windows: {len(val_ds)} | γ={gamma} | temp={drafter_temp}")
    device = get_device()
    param_dtype = next(model.parameters()).dtype
    eval_prompts: List[str] = []
    eval_k_list: List[int] = []
    if cli_args is not None and int(getattr(cli_args, "eval_every", 0)) > 0:
        eval_k_list = parse_int_list(str(getattr(cli_args, "eval_k_list", "")))
        eval_prompts = load_eval_prompts(
            jsonl_path,
            int(getattr(cli_args, "eval_num_prompts", 0)),
            int(getattr(cli_args, "eval_seed", seed)),
            str(getattr(cli_args, "eval_prompt_field", "prompt")),
        )
        if not eval_k_list or not eval_prompts:
            log("[Eval] disabled: missing prompts or k list.")
            eval_k_list = []
            eval_prompts = []

    for ep in range(1, epochs + 1):
        model.train()
        pbar = tqdm(train_loader, desc=f"Train Epoch {ep}/{epochs}", leave=False)
        for it, batch in enumerate(pbar, start=1):
            step += 1

            input_ids = batch["input_ids"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)

            # Build Dream-style mask & positions
            attn2d = (input_ids != pad_id).to(torch.float32)                # (B, L) 1.0 for non-pad
            tok_idx = make_tok_idx(attn2d).to(device)                       # (B, L) long
            attn4d = build_additive_mask_from_attn2d(
                attn2d,
                param_dtype,
                device,
                labels=labels,
                approx_kv_cache_mode=approx_kv_cache_mode,
            )  # (B,1,L,L)

            try:
                # Request γ+1 logits so that after left-shift we have γ aligned outputs
                out = model(
                    input_ids=input_ids,
                    attention_mask=attn4d,
                    position_ids=tok_idx,
                    use_cache=False,
                    num_logits_to_keep=gamma + 1
                )
                logits = out.logits                           # (B, ≥γ+1, V) – repo returns last T steps
                labels_last = labels[:, -gamma:]              # (B, γ)

                loss, e_streak_mean = estreak_loss_dream_aligned(
                    logits=logits,
                    labels_last=labels_last,
                    drafter_temp=drafter_temp,
                    objective=distill_objective,
                )

                if not torch.isnan(loss):
                    loss.backward()
            except torch.cuda.OutOfMemoryError as e:
                log(f"OOM at step {step}; skipping batch. {e}")
                optimizer.zero_grad(set_to_none=True)
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                continue

            if it % grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            if step % log_every == 0:
                train_loss = float(loss.detach().item())             # == -E[streak]
                train_e = float(e_streak_mean.detach().item())
                if metrics_jsonl:
                    rec = {
                        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "epoch": ep,
                        "step": step,
                        "iter": it,
                        "split": "train",
                        "loss": train_loss,
                        "e_streak": train_e,
                    }
                    with open(metrics_jsonl, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                pbar.set_postfix({"E[streak]": f"{train_e:.3f}"})

            if (save_every > 0) and (step % save_every == 0):
                sp = os.path.join(output_dir, f"step-{step}")
                save_lora_or_full(model, tok_d, sp, strategy="adapters")
                log(f"Saved checkpoint: {sp}")
            if (
                cli_args is not None
                and eval_k_list
                and int(getattr(cli_args, "eval_every", 0)) > 0
                and (step % int(getattr(cli_args, "eval_every", 0)) == 0)
            ):
                log(f"[Eval] starting live eval at step={step}")
                model.eval()
                orig_device = next(model.parameters()).device
                if orig_device.type == "cuda":
                    torch.cuda.synchronize()
                    log(f"[Eval] vram before offload: {torch.cuda.memory_allocated() / (1024**2):.1f} MiB")
                    _optimizer_to(optimizer, torch.device("cpu"))
                    model.to("cpu")
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    log(f"[Eval] vram after offload: {torch.cuda.memory_allocated() / (1024**2):.1f} MiB")
                run_live_eval(
                    model=model,
                    tokenizer=tok_d,
                    step=step,
                    epoch=ep,
                    args=cli_args,
                    eval_prompts=eval_prompts,
                    eval_k_list=eval_k_list,
                )
                if orig_device.type == "cuda":
                    model.to(orig_device)
                    _optimizer_to(optimizer, orig_device)
                    torch.cuda.empty_cache()
                model.train()
        pbar.close()

        # ---- Validation at end of epoch ----
        model.eval()
        val_estreaks: List[float] = []
        with torch.no_grad():
            for vb in val_loader:
                vin = vb["input_ids"].to(device, non_blocking=True)
                vlabels = vb["labels"].to(device, non_blocking=True)

                vattn2d = (vin != pad_id).to(torch.float32)
                vtok_idx = make_tok_idx(vattn2d).to(device)
                vattn4d = build_additive_mask_from_attn2d(
                    vattn2d,
                    param_dtype,
                    device,
                    labels=vlabels,
                    approx_kv_cache_mode=approx_kv_cache_mode,
                )

                vout = model(
                    input_ids=vin,
                    attention_mask=vattn4d,
                    position_ids=vtok_idx,
                    use_cache=False,
                    num_logits_to_keep=gamma + 1
                )
                vlogits = vout.logits
                vlabels_last = vlabels[:, -gamma:]

                vneg, vestreak = estreak_loss_dream_aligned(
                    logits=vlogits,
                    labels_last=vlabels_last,
                    drafter_temp=drafter_temp,
                    objective=distill_objective,
                )
                val_estreaks.append(float(vestreak.detach().item()))

        val_e = float(sum(val_estreaks) / max(1, len(val_estreaks))) if val_estreaks else float('nan')
        log(f"[val] epoch={ep} E[streak]={val_e:.4f} over {len(val_loader)} batches")
        if metrics_jsonl:
            rec = {
                "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                "epoch": ep,
                "step": step,
                "split": "val",
                "e_streak": val_e,
                "loss": float(-val_e),  # keep 'loss' channel for dashboards
            }
            with open(metrics_jsonl, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        model.train()

    # ---- Final save (adapters-only by default) ----
    fp = os.path.join(output_dir, "final")
    save_lora_or_full(model, tok_d, fp, strategy=save_strategy or "adapters")
    log(f"Saved final checkpoint to: {fp}")



# -------------------------- CLI --------------------------

def parse_args():
    ap = argparse.ArgumentParser(description="Finetune DiffuCoder/Dream drafter on verifier γ-window generations (E[streak] objective, Dream-aligned)")
    # Stage selector
    ap.add_argument("--do_generate", action="store_true", help="Run Stage A (verifier sampling)")
    ap.add_argument("--do_train", action="store_true", help="Run Stage B (drafter finetune)")
    # Common
    ap.add_argument("--seed", type=int, default=1236)
    ap.add_argument("--local_files_only", action="store_true")
    # Stage A
    ap.add_argument("--verifier_model_id", type=str, default="Qwen/Qwen2.5-14B-Instruct")
    ap.add_argument("--num_prompts", type=int, default=64)
    ap.add_argument("--max_new_tokens", type=int, default=2048)
    ap.add_argument("--verifier_num_completions", type=int, default=32)
    ap.add_argument("--out_jsonl", type=str, default="Distillation-Files-Qwen2.5-14B/Qwen2.5-14B_distill_data.jsonl")
    ap.add_argument("--temperature", type=float, default=0.5)
    ap.add_argument("--top_p", type=float, default=0.95)
    # NEW: unified benchmark switch (mirrors verify script)
    ap.add_argument("--benchmark", type=str, default="gsm8k", help="alpaca | MT-Bench | math-500 | gpqa | gsm8k | livecodebench (leave empty to use dataset_name/config/split)")
    ap.add_argument("--lcb_version_tag", type=str, default="release_latest", help="LiveCodeBench version tag (when --benchmark livecodebench)")
    # Original dataset path (unchanged, used when --benchmark is empty)
    ap.add_argument("--dataset_name", type=str, default="openai/gsm8k")
    ap.add_argument("--dataset_config", type=str, default="main")
    ap.add_argument("--dataset_split", type=str, default="test")
    # Stage B
    ap.add_argument("--drafter_model_id", type=str, default="apple/DiffuCoder-7B-cpGRPO")
    ap.add_argument("--gamma", type=int, default=16)
    ap.add_argument("--drafter_temp", type=float, default=1.0, help="Temperature used to compute q_i; match deployment.")
    ap.add_argument("--max_seq_len", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--weight_decay", type=float, default=0.005)
    ap.add_argument("--warmup_steps", type=int, default=100)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--save_every", type=int, default=1000)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--no_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=32)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--output_dir", type=str, default="SpecDiffu-Qwen2.5-14B")
    ap.add_argument("--metrics_jsonl", type=str, default="SpecDiffu-Qwen2.5-14B-metrics.jsonl")
    ap.add_argument(
        "--approx-kv-cache-mode",
        "--approx_kv_cache",
        dest="approx_kv_cache",
        action="store_true",
        default=False,
    )
    obj_group = ap.add_mutually_exclusive_group()
    obj_group.add_argument(
        "--streak-distill",
        dest="distill_objective",
        action="store_const",
        const="streak",
        default="streak",
    )
    obj_group.add_argument(
        "--ce-streak-distill",
        dest="distill_objective",
        action="store_const",
        const="ce_streak",
    )
    ap.add_argument("--eval_every", type=int, default=9999999, help="Run specdiff decode eval every N steps (0 disables).")
    ap.add_argument("--eval_num_prompts", type=int, default=50)
    ap.add_argument("--eval_k_list", type=str, default="8")
    ap.add_argument("--eval_prompt_field", type=str, default="prompt")
    ap.add_argument("--eval_seed", type=int, default=1234)
    ap.add_argument("--eval_verify_model_id", type=str, default="Qwen/Qwen2.5-1.5B-Instruct")
    ap.add_argument("--eval_verify_temperature", type=float, default=0.0)
    ap.add_argument("--eval_max_new_tokens", type=int, default=None)
    ap.add_argument("--eval_diffusion_temperature", type=float, default=None)
    ap.add_argument("--eval_diffusion_top_k", type=int, default=64)
    ap.add_argument("--eval_diffusion_top_p", type=float, default=1.0)
    ap.add_argument("--eval_diffusion_next_token", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--eval_diffusion_fuse_mlp", action="store_true", default=False)
    ap.add_argument("--eval_diffusion_fuse_qkv", action="store_true", default=False)
    ap.add_argument("--eval_diffusion_sdp_backend", type=str, default="")
    ap.add_argument("--eval_verify_sdp_backend", type=str, default="")
    ap.add_argument("--eval_steps", type=int, default=0)
    ap.add_argument("--eval_warmup_steps", type=int, default=1)
    ap.add_argument("--eval_log_every", type=int, default=0)
    ap.add_argument(
        "--eval_tree_build_mode",
        type=str,
        default="k_chains",
        choices=("k_chains", "collapsed"),
    )
    ap.add_argument("--eval_device", type=str, default="cuda")
    ap.add_argument("--eval_dtype", type=str, default="")
    ap.add_argument("--eval_keep_merged", action="store_true", default=False)
    ap.add_argument("--eval_trust_remote_code", action="store_true", default=True)
    ap.add_argument("--eval_util_sample_interval", type=float, default=0.1)
    ap.add_argument("--eval_model_revision", type=str, default="")
    # Final checkpoint policy
    ap.add_argument("--save_strategy", choices=["adapters", "merged"], default="adapters")
    return ap.parse_args()


def main():
    args = parse_args()
    seed_all(args.seed)

    if args.do_generate:
        generate_with_verifier(
            model_id=args.verifier_model_id,
            num_prompts=args.num_prompts,
            out_jsonl=args.out_jsonl,
            max_new_tokens=args.max_new_tokens,
            verifier_num_completions=args.verifier_num_completions,
            temperature=args.temperature,
            top_p=args.top_p,
            local_files_only=args.local_files_only,
            dataset_name=args.dataset_name,
            dataset_config=args.dataset_config,
            dataset_split=args.dataset_split,
            seed=args.seed,
            benchmark=args.benchmark,
            lcb_version_tag=args.lcb_version_tag,
        )

    if args.do_train:
        if not os.path.exists(args.out_jsonl):
            raise FileNotFoundError(f"JSONL not found: {args.out_jsonl}. Run --do_generate first, or supply your own.")
        train_on_windows(
            jsonl_path=args.out_jsonl,
            drafter_model_id=args.drafter_model_id,
            output_dir=args.output_dir,
            gamma=args.gamma,
            drafter_temp=args.drafter_temp,
            max_seq_len=args.max_seq_len,
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum=args.grad_accum,
            lr=args.lr,
            weight_decay=args.weight_decay,
            warmup_steps=args.warmup_steps,
            log_every=args.log_every,
            save_every=args.save_every,
            bf16=args.bf16,
            no_lora=args.no_lora,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            local_files_only=args.local_files_only,
            val_ratio=0.05,
            num_workers=2,
            metrics_jsonl=args.metrics_jsonl,
            seed=args.seed,
            save_strategy=args.save_strategy,
            approx_kv_cache_mode=bool(args.approx_kv_cache),
            distill_objective=str(args.distill_objective),
            cli_args=args,
        )

if __name__ == "__main__":
    main()
