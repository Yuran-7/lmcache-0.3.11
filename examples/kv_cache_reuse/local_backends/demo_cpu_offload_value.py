# SPDX-License-Identifier: Apache-2.0
"""
Demo: LMCache CPU offload value
================================
This script demonstrates the value of LMCache's CPU KV-cache offloading.

Key design:
  - vLLM's built-in APC (Automatic Prefix Caching) is DISABLED.
    Without APC, vLLM does NOT retain any KV blocks across requests.
    Every request would normally require a full prefill.
  - LMCache stores KV caches to CPU after each request.
  - On a repeated prefix, LMCache loads the KV from CPU back to GPU,
    saving the expensive GPU prefill computation.

Three requests are sent:
  [A] Baseline  – unique content,  no cache hit at all  → full prefill
  [B] Store     – shared prefix,   no cache hit yet     → full prefill + store to CPU
  [C] CPU Load  – shared prefix,   LMCache CPU hit      → load from CPU + tiny prefill

Expected output:
  Request A: ~same as B (full prefill, no caching benefit)
  Request B: ~same as A (full prefill, but KV is saved to CPU at the end)
  Request C: significantly faster than B  ← this is the value of CPU offloading
"""

# Standard
from dataclasses import asdict
import argparse
import contextlib
import os
import time

# Third Party
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig
from vllm.engine.arg_utils import EngineArgs

# First Party
from lmcache.integration.vllm.utils import ENGINE_NAME
from lmcache.v1.cache_engine import LMCacheEngineBuilder


# ---------------------------------------------------------------------------
# Environment setup
# ---------------------------------------------------------------------------

def setup_environment(use_disk: bool = False):
    os.environ["LMCACHE_CHUNK_SIZE"] = "256"

    if use_disk:
        os.environ["LMCACHE_LOCAL_CPU"] = "False"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "5"
        os.environ["LMCACHE_LOCAL_DISK"] = "file://local_disk/"
        os.environ["LMCACHE_MAX_LOCAL_DISK_SIZE"] = "20"
    else:
        os.environ["LMCACHE_LOCAL_CPU"] = "True"
        os.environ["LMCACHE_MAX_LOCAL_CPU_SIZE"] = "10"


@contextlib.contextmanager
def build_llm(model: str, use_disk: bool):
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )
    llm_args = EngineArgs(
        model=model,
        kv_transfer_config=ktc,
        max_model_len=16000,
        gpu_memory_utilization=0.7,
        # *** KEY: disable vLLM's own prefix caching ***
        # With APC on, vLLM retains GPU blocks across requests,
        # so LMCache's CPU load is never triggered (GPU hit happens first).
        # With APC off, vLLM relinquishes all blocks after each request,
        # forcing the next request to reload from LMCache CPU.
        enable_prefix_caching=False,
    )
    llm = LLM(**asdict(llm_args))
    try:
        yield llm
    finally:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run_request(llm: LLM, prompt: str, sampling_params: SamplingParams,
                label: str) -> float:
    """Run a single request and return wall-clock time in seconds."""
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")
    t0 = time.perf_counter()
    outputs = llm.generate([prompt], sampling_params)
    elapsed = time.perf_counter() - t0
    text = outputs[0].outputs[0].text
    print(f"  Output : {text!r}")
    print(f"  Time   : {elapsed:.3f}s")
    return elapsed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="LMCache CPU offload value demo")
    parser.add_argument("--model", type=str,
                        default="/NV1/ykw/models/Meta-Llama-3.1-8B-Instruct/",
                        help="Model to use")
    parser.add_argument("--use-disk", action="store_true",
                        help="Use disk backend instead of CPU memory")
    parser.add_argument("--repeat", type=int, default=1,
                        help="Repeat measurement C this many times (default 1)")
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment(args.use_disk)

    # ------------------------------------------------------------------
    # Build prompts
    # ------------------------------------------------------------------
    # Shared prefix – long enough to make CPU offloading worthwhile.
    # ~5000 tokens ≈ 10x "Hello, how are you?" repetitions of 500.
    shared_prefix = "Hello, how are you? " * 500        # ~5000 tokens

    # Unique baseline content (completely different from shared_prefix)
    baseline_content = "The quick brown fox jumps over the lazy dog. " * 500

    prompt_A = baseline_content + "What is the first word of this text?"
    prompt_B = shared_prefix   + "Hello, my name is"
    prompt_C = shared_prefix   + "Tell me a very long story"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    with build_llm(args.model, args.use_disk) as llm:

        # [A] Baseline: completely different content, no cache anywhere
        time_A = run_request(llm, prompt_A, sampling_params,
                             "[A] Baseline  — unique content, no cache anywhere")
        time.sleep(1)

        # [B] Store: shared prefix, cold start — full prefill, KV stored to CPU/Disk
        time_B = run_request(llm, prompt_B, sampling_params,
                             "[B] Store     — shared prefix, cold miss → full prefill "
                             "+ LMCache stores to CPU")
        time.sleep(1)

        # [C] Load from CPU: same shared prefix, APC is OFF so GPU has no blocks;
        #     LMCache loads KV from CPU, only tiny suffix needs prefill on GPU.
        time_C_list = []
        for i in range(args.repeat):
            label = (f"[C] CPU Load  — shared prefix hit via LMCache CPU "
                     f"(run {i+1}/{args.repeat})")
            t = run_request(llm, prompt_C, sampling_params, label)
            time_C_list.append(t)
            time.sleep(0.5)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    time_C = min(time_C_list)
    speedup = time_B / time_C if time_C > 0 else float("inf")

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    print(f"  [A] Baseline  (full prefill, no LMCache)  : {time_A:.3f}s")
    print(f"  [B] Store     (full prefill + CPU store)   : {time_B:.3f}s")
    print(f"  [C] CPU Load  (LMCache CPU hit, best run)  : {time_C:.3f}s")
    print(f"  Speedup  B → C  : {speedup:.2f}x")
    print()
    print("  Interpretation:")
    print("    • A vs B: both do full prefill — similar times, confirming no GPU")
    print("              APC benefit in this script.")
    print("    • B vs C: C is faster because LMCache loaded the shared-prefix KV")
    print("              from CPU instead of recomputing it on GPU.")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
