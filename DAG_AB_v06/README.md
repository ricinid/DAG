# DAG v0.6 — experimental llama.cpp CPU inference A/B

> **Status:** executable native Q4_0/Q4_K GEMV kernel and GGML CPU hook source **implemented**, tested on synthetic GGUF slices. **Not yet built or validated against a full llama.cpp checkout or run on Qwen 3B.** No speedup, RAM/VRAM improvement, or bit-identical llama.cpp inference is claimed.

The existing `weights.adagstream` is **not** a stand-alone GGUF and is not loadable by upstream llama.cpp. This experiment compiles a custom llama.cpp whose GGML CPU `MUL_MAT` path can run the existing DAG on single-token F32 matrix-vector operations and falls back to GGML on unsupported tensors/shapes.

## Experiment matrix

All values are cache limits on **decoded DAG data**, computed from the **original GGUF file size**, not fractions of the larger `.adagstream`. On the completed `qwen2.5-coder:3b` snapshot (1,929,903,072 bytes):

| Mode | DAG cache budget | What executes |
| --- | ---: | --- |
| `baseline` | none | Original llama.cpp CPU kernels, entire GGUF |
| `resident` | unbounded cache | DAG Q4_0/Q4_K GEMV (eligible, batch=1), ordinary GGML fallback |
| `stream50` | 964,951,536 bytes | Same DAG GEMV, 50% memory-budget cache |
| `stream25` | 482,475,768 bytes | Same DAG GEMV, 25% cache |
| `stream10` | 192,990,307 bytes | Same DAG GEMV, 10% cache |

**A cache budget is not a memory limit for the llama-server process.** The original GGUF tensors remain loaded, and the cache, index, transient ZIP/zlib decompression, activations and KV cache contribute additional RAM. GPU offload is disabled (`-ngl 0`). Consequently this is a **CPU inference/streaming feasibility experiment**, not yet a valid test of lowering GPU VRAM or eliminating resident original weights. The runtime is a C++ interpreter for DAG IR rather than ahead-of-time native code generation.

The `resident` mode may consume more than the 3.58 GB compressed DAG stream because the runtime caches *decompressed* DAG IR. If memory is limited, omit `resident` initially. The current indexed format comprises ~288,000 tiny ZIP entries and may be **extremely slow** at full-model inference.

## Preparation — Windows / PowerShell

Requirements: Python 3.10+, `psutil`, CMake, Visual Studio C++ Build Tools, zlib development library (`vcpkg` is one way), an installed GGUF from Ollama, and the completed `compiled/` output containing `weights.adagstream`, `weights.index.jsonl`, `full_model_report.json`.

```powershell
python -m pip install psutil numpy pytest
# Install zlib using your existing vcpkg, or prepare ZLIB_ROOT manually.
& C:\tools\vcpkg\vcpkg.exe install zlib:x64-windows

git clone https://github.com/ggml-org/llama.cpp.git llama-dag
cd llama-dag
git checkout d38bea34227bbf9069d62fdc239c98ab6e05435a
cd ..

python DAG_AB_v06\patch_llama_cpp.py --llama-src .\llama-dag
cmake -S .\llama-dag -B .\llama-dag\build-dag -A x64 `
  -DGGML_DAG_RUNTIME=ON `
  -DGGML_CUDA=OFF `
  -DLLAMA_BUILD_SERVER=ON `
  -DCMAKE_TOOLCHAIN_FILE=C:\tools\vcpkg\scripts\buildsystems\vcpkg.cmake
cmake --build .\llama-dag\build-dag --config Release --target llama-server
```

`C:\tools\vcpkg` is an **example installation path**; use the location of your real vcpkg or zlib installation. The `patch_llama_cpp.py` script refuses to edit an upstream version with unexpected source anchors, instead of installing an unvalidated hook. If CMake fails, preserve its error log rather than trying unrelated revisions.

## Run baseline first (real full-model generation)

Use *the same patched binary* for every mode. Baseline is activated by the absence of `DAG_STREAM_DIR`.

```powershell
$compiled = Resolve-Path .\dag_results\qwen_ab_02\compiled
$report = Get-Content "$compiled\full_model_report.json" -Raw | ConvertFrom-Json
$gguf = $report.source_path
$server = ".\llama-dag\build-dag\bin\Release\llama-server.exe"

python DAG_AB_v06\run_llama_ab.py `
  --server $server --gguf $gguf --compiled $compiled `
  --out-dir dag_results\native_baseline `
  --modes baseline --trials 3 --num-predict 128
```

The harness checks the full GGUF SHA256 against `full_model_report.json` before running. It starts a fresh server for each phase, uses temperature zero, seed 42, CPU threads 1, GPU layers 0, `cache_prompt=false`, and records streamed TTFT plus server-reported decode tok/s, peak process RSS, and aggregate GPU memory (global device numbers **are not per process**).

## Smoke test the **hook** before trying full DAG inference

This test generates real tokens with the full model, **but only one matrix is intercepted by DAG**; it is an integration/correctness check, **not full-DAG throughput**.

```powershell
python DAG_AB_v06\run_llama_ab.py `
  --server $server --gguf $gguf --compiled $compiled `
  --out-dir dag_results\native_hook_smoke `
  --modes baseline stream50 `
  --dag-tensors blk.0.ffn_gate.weight `
  --trials 1 --num-predict 8
```

The run fails if no DAG GEMV calls are recorded. Check `stream50.server.log` for `[DAG native] STATS` lines, then compare generated token IDs. A matching token sequence is a smoke check only; **full logits numerical parity is still required before qualifying the backend**.

## Full DAG streaming, after a successful smoke

⚠️ The current ZIP/zlib unit interpreter may be many orders slower than llama.cpp. Begin with **1–4 generated tokens** and monitor CPU/RAM before increasing prompt or token count. Different modes start separate processes; command writes to a fresh directory.

```powershell
python DAG_AB_v06\run_llama_ab.py `
  --server $server --gguf $gguf --compiled $compiled `
  --out-dir dag_results\native_full_dag_01 `
  --modes baseline resident stream50 stream25 stream10 `
  --dag-tensors all --trials 1 --num-predict 2 `
  --request-timeout 3600
```

`resident` is unlimited and may exhaust RAM, so omitting it is preferable on machines without substantial free physical memory. Once stability and numerical behavior are established, repeat with `--trials 3 --num-predict 128` only if latency is acceptable.

Output: `full_inference_ab.json` (updated after each successful phase), `baseline.server.log`, `resident.server.log`, `stream50.server.log`, `stream25.server.log`, `stream10.server.log`. A nonzero DAG GEMV counter is mandatory. `relative_to_baseline` reports median decode speedup, TTFT ratio, process RSS ratio and whether the first trial produced the same token IDs.

## What is *not* solved by v0.6

1. **Original GGUF residency:** the patched GGML still loads the entire quantized model, including all Q4 tensors. No claimed net VRAM/RAM reduction, even if the DAG cache is small.
2. **GPU inference:** GGML CPU hook cannot replace CUDA matmul. GPU A/B needs separate CUDA backend and loader support for DAG-only weights.
3. **Numerical semantics:** the kernel evaluates exact Q4 integer codes against F32 activations using FP64 intermediates, then casts to F32; conventional GGML Q4×Q8 quantizes activations and rounds/accumulates differently. The scalar *quantized-weight mapping* is correct within the synthetic test tolerance, but logits and token-level parity are not guaranteed.
4. **Graph coverage:** only `Q4_0` and `Q4_K` `GGML_OP_MUL_MAT` for batch=1, F32 input, single CPU thread is intercepted. Unsupported types (e.g. Q6_K and embeddings) remain original GGML ops. `source_gguf_sha256` is checked by the Python harness; C++ does not independently authenticate every artifact record.
5. **Performance:** existing v1 artifact ZIPs have heavy decompression/parse overhead; native IR, vectorized and AOT kernels must be benchmarked. A subset DAG that silently falls back to GGML is not a full-DAG A/B.
6. **Cache semantics:** budgets constrain the decoded DAG cache payload; decompression scratch, source weights, KV cache, index, process and OS file cache are outside it. Resident is a cache strategy, not a promise that all DAGs fit in memory.

## Validation gates for production-quality A/B

- Synthetic Q4_0/Q4_K projection equivalence (implemented, `test_native_dag.py`).
- Successful custom llama.cpp build at a **pinned** upstream commit (must be done in a full checkout).
- One patched GEMV actually executes during token generation (harness enforces nonzero calls).
- Compare logits across representative tokens, not just same generated string (not yet implemented).
- Measure per-mode peak RAM, VRAM, decode tok/s, TTFT and I/O; confirm memory improvements with elimination of original quantized weight allocation (not yet implemented).

This is a deliberately small, measurable native entry point. A good streaming cache policy cannot compensate for re-reading/decompressing multiple gigabytes of IR on every generated token, so optimize IR and memory traffic **after observing the first real measurements**, not before.

## Troubleshooting: zero DAG GEMV calls on Qwen

The harness correctly rejects a DAG run if no actual DAG GEMV was executed.
A normal-looking tok/s in this case is NOT evidence of DAG throughput: operations
may have silently fallen back to GGML. The current hook accepts only one CPU
thread, Q4_0/Q4_K weights, F32 activations, contiguous single-token GEMV, and
an exactly matching tensor name.

No need to recompile the DAG snapshot. To inspect the real GGML shapes and types,
install the one-shot diagnostic into the already patched llama-dag checkout,
rebuild only llama-server and run a short diagnostic smoke:

~~~powershell
git pull --ff-only
Select-String ..\llama-dag\ggml\src\ggml-cpu\ggml-cpu.c -Pattern DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK
Select-String ..\llama-dag\build-dag\CMakeCache.txt -Pattern '^GGML_DAG_RUNTIME:BOOL=ON'

python DAG_AB_v06\diagnose_hook.py --llama-src ..\llama-dag
cmake --build ..\llama-dag\build-dag --config Release --target llama-server

$server = (Resolve-Path ..\llama-dag\build-dag\bin\Release\llama-server.exe).Path
$compiled = (Resolve-Path .\dag_results\qwen_ab_02\compiled).Path
$report = Get-Content "$compiled\full_model_report.json" -Raw | ConvertFrom-Json
$env:DAG_HOOK_DIAGNOSTIC = "1"
python DAG_AB_v06\run_llama_ab.py --server $server --gguf $report.source_path --compiled $compiled --out-dir dag_results\native_hook_diag_01 --modes stream50 --dag-tensors blk.0.ffn_gate.weight --trials 1 --num-predict 2
Remove-Item Env:DAG_HOOK_DIAGNOSTIC -ErrorAction SilentlyContinue
Select-String dag_results\native_hook_diag_01\stream50.server.log -Pattern '\[DAG hook diag\]|\[DAG native\]'
~~~

The diagnostic prints up to 6 first MUL_MAT calls and 16 FFN-gate operations.
It reports tensor name, types, threads, shapes and contiguity. It does not
change computation.

- No [DAG hook diag] lines: confirm the rebuilt CPU backend contains the
  patched source and GGML_DAG_RUNTIME is ON.
- Observed src1_type != f32, threads != 1, or multi-token input: the current
  hook's narrow guard rejects the operation. Do not reinterpret Q8 data as F32.
- Actual tensor name differs: reconcile it against the DAG index.
- Shape and type appear eligible but no DAG-native STATS: inspect the allowlist,
  artifact index and dimension matching before changing kernel semantics.

The diagnostic refuses a second installation in the same checkout. This is a
diagnostic-only experiment, NOT proof of full-model DAG inference or acceleration.


## Q4 matrices bypassed by CPU_REPACK: relocate the installed hook

The original v0.6 hook was installed inside ggml_compute_forward_mul_mat().
However, in pinned llama.cpp ggml_compute_forward() calls
ggml_cpu_extra_compute_forward() BEFORE dispatching to standard MUL_MAT.
CPU_REPACK may execute the Q4 matrices on this earlier path. Thus the original
diagnostic can report only Q6_K fallback matrices and zero DAG GEMV calls.

For an EXISTING checkout where patch_llama_cpp.py and diagnose_hook.py were
already applied, migrate the installed hook once; DO NOT run patch_llama_cpp.py
again and DO NOT clone/recompile the existing DAG snapshot.

~~~powershell
git pull --ff-only
python -m pytest -q DAG_AB_v06/test_relocate_hook.py DAG_AB_v06/test_hook_install.py

python DAG_AB_v06\relocate_hook.py --llama-src ..\llama-dag
cmake --build ..\llama-dag\build-dag --config Release --target llama-server

Select-String ..\llama-dag\ggml\src\ggml-cpu\ggml-cpu.c -Pattern 'DAG_NATIVE_EARLY_MUL_MAT_DISPATCH'
$env:DAG_HOOK_DIAGNOSTIC = "1"
$server = (Resolve-Path ..\llama-dag\build-dag\bin\Release\llama-server.exe).Path
$compiled = (Resolve-Path .\dag_results\qwen_ab_02\compiled).Path
$report = Get-Content "$compiled\full_model_report.json" -Raw | ConvertFrom-Json
python DAG_AB_v06\run_llama_ab.py --server $server --gguf $report.source_path --compiled $compiled --out-dir dag_results\native_hook_diag_02 --modes stream50 --dag-tensors blk.0.ffn_gate.weight --trials 1 --num-predict 2
Remove-Item Env:DAG_HOOK_DIAGNOSTIC -ErrorAction SilentlyContinue
Select-String dag_results\native_hook_diag_02\stream50.server.log -Pattern '\[DAG hook diag\]|\[DAG native\]'
~~~

The newly updated patch_llama_cpp.py installs the hook directly at the
EARLY_MUL_MAT dispatch point on future fresh checkouts, before CPU_REPACK.
The relocation script preserves previously enabled diagnostic instrumentation.
Supported hook scope remains Q4_0/Q4_K, F32 single-token CPU GEMV at one thread.

If the relocated hook still reports zero calls, provide the full stream50 log.
The first GGML call can have multiple input columns due to prompt prefill;
decoding should use a one-column GEMV, but it must be verified in the log.
