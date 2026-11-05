# DAG — lossless snapshot-aware algebraic compiler (research)

**Current version: v0.4, whole-snapshot compiler + honest A/B harness.**

DAG factors Q4_0 and Q4_K weight **codes** into a shared arithmetic DAG,
without changing their stored values. The compiler processes **every eligible
rank-2 Q4_0/Q4_K tensor** in a source GGUF, in bounded-memory units. Every
unit is verified for exact recovery from the DAG. Other GGUF tensors are left
as **passthrough references to the original GGUF**; the resulting `.adagstream`
**is not an Ollama/llama.cpp-loadable model**.

**This project does not currently have an integrated DAG inference runtime.**
That is the gating requirement for a real before/after full-model tokens/s or
RAM/VRAM comparison. The A/B harness does **not** compare a baseline Ollama
run against the same original GGUF masquerading as "optimized". It records
baseline and compilation measurements, and reports a blocked optimized phase.
It can exercise a future DAG-aware runtime after verifying its identity via an
explicit capabilities endpoint.

## Setup

Requires Python 3.10+ and a running Ollama server. CUDA-enabled PyTorch is
optional for separate slice microbenchmarks; the full-model A/B harness reads
NVIDIA usage through `nvidia-smi`, if installed.

```powershell
py -m pip install -r requirements.txt
ollama pull qwen2.5-coder:3b
```

## One-command whole-model experiment

```powershell
py ab_experiment.py `
  --model qwen2.5-coder:3b `
  --prompt-file prompt.txt `
  --out-dir dag_results/qwen_full_ab `
  --rows 32 --cols 256 --max-nodes 16
```

Without `prompt.txt`, omit `--prompt-file` to use the built-in prompt.
Add `--pull` to download the model automatically. The default is to process
**all eligible tensors**, not an arbitrary sample. Compilation may be
extremely CPU-intensive; there is no guarantee that algebraic factorization
will be useful on actual LLM weights. Use `--max-work-units 3` **only for an
explicit smoke test**; the A/B harness then reports `compile_incomplete`.

Exit codes: `0` = genuine A/B complete, `2` = incomplete / no verified DAG
runtime, `1` = failure. Default current outcome after a complete compile is
`blocked_no_dag_inference_runtime` (exit 2), as intended.

### Experiment outputs

```text
dag_results/qwen_full_ab/
  ab_report.json                # single shared report, updated after each phase
  events.jsonl                 # timestamped phase/event log
  logs/
    prompt.txt                 # exact prompt used
    baseline.txt               # decoded baseline response
    baseline.json              # latency, throughput, model residency, RAM/VRAM
    baseline.memory.jsonl      # time-series memory samples
    compile.stdout.log
    compile.stderr.log
    compile.memory.jsonl
  compiled/
    full_model_report.json     # full coverage, completion, checksum, sizes
    compiler_state.json        # strict resume fingerprint
    compile_events.jsonl
    weights.adagstream         # DAG data, one length-prefixed artifact per unit
    weights.index.jsonl       # tensor and source-region index
```

### Memory measurement details

`baseline` records Ollama's `/api/ps` `size_vram` value, peak **aggregate**
RSS of detected Ollama processes, system RAM usage, and `nvidia-smi` per-GPU
system-wide VRAM usage at the monitor's sampling interval. Global GPU usage
includes other processes. Client-observed TTFT and Ollama's generation timing
are both captured. A monitor sampling process can miss very short peaks.
`compile` records the peak RSS of the compiler process tree plus system RAM/VRAM during the separate compiler subprocess;
this is not optimized inference memory. For the original CUDA allocator slice
microbenchmarks see `README_ollama_dag.md`.

`ab_report.json` is written atomically after each phase to survive partial
failures. Prompts and full model responses are intentionally retained locally,
so avoid publishing `dag_results/` to GitHub.

### A/B contract for a future optimized DAG inference runtime

To run the second pass, pass BOTH:

```powershell
  --optimized-url http://127.0.0.1:PORT `
  --optimized-model qwen-dag
```

Before calling `/api/generate`, the harness requires that the endpoint expose
`GET /dag/capabilities` with at least:

```json
{
  "backend": "algebraic_dag",
  "ready": true,
  "artifact_format": "adagstream-v1",
  "source_gguf_sha256": "<exact SHA256 of source GGUF>"
}
```

The DAG runtime is **not implemented in v0.4**. Advertised capabilities are
an integration contract, not independent attestation; subsequently we must
also compare logits and check that the runtime actually consumes the compiled
arithmetic graph. Reusing ordinary Ollama with renamed GGUF is invalid.

### Compiling separately, with resume

```powershell
py whole_model_dag.py --gguf C:\\models\\qwen.gguf `
  --out-dir dag_results/full_compiled --rows 32 --cols 256
```

Re-run the same command to continue (the index and checkpoint are validated).
The original GGUF is needed for unsupported tensors and exact provenance; the
compiler output is not standalone. Flags changing compiler parameters while
resuming are rejected.

## Existing slice tool

`ollama_dag_pipeline.py`, `algebraic_dag.py`, and `memory_bench.py` remain
available for small targeted correctness/performance experiments. See the other
README files. They are not substitutes for the whole-model study.

## Tests

```powershell
py -m pytest -q
```

## Limitations and next milestone

- Current 4-bit compiler operates on stored integer quantization codes and
  retains scale/min metadata. It checks byte-for-byte recovery, not bit-exact
  llama.cpp floating-point accumulation.
- Does not algebraically reduce BF16/F16, Q6_K or non-linear ops; those tensors
  are referenced unchanged, so optimized inference requires pass-through.
- Current factorization is bounded by row/column work units. Global CSE
  across *all* tensors/layers is not yet implemented. The resulting stream
  is a collection of per-unit DAGs, not one globally factored algebraic graph.
- The only credible next milestone for end-to-end claims is a real llama.cpp
  GGML matrix-operation backend that executes `.adagstream` plus passthrough
  tensor ops, with correctness validation and full-process memory metrics.

## Ollama model resolution on Windows

Model names such as `qwen2.5-coder:3b` contain a version dot, which must NOT
be treated as a registry hostname. The resolver now checks the standard
manifest and uses the running Ollama service `/api/show` as a fallback for
custom `OLLAMA_MODELS` locations.

If the model is pulled but Python still cannot find a locally readable GGUF:

```powershell
ollama list
ollama show --modelfile qwen2.5-coder:3b
```

Copy the absolute path after `FROM` and pass it as `--gguf` to
`ab_experiment.py`, or pass the service's models root with `--models-dir`.
The model file must be accessible on the machine running the compiler.
Use a fresh `--out-dir` for each experiment: reports are not overwritten.

After `git pull`, rerun without `--pull` if the model is already installed.


## CSE acceleration (v0.5)

The greedy CSE candidate search now uses the cached Numba JIT in `cse_accelerator.py` when installed. Its compact open-addressing table preserves the original **pair-window occurrences**, first-seen tie-breaking and NAF cost model. The DAG and `.adagstream` format are unchanged, and the existing compiler checkpoint can be resumed.

Set `$env:DAG_CSE_ENGINE = 'python'` to force the reference implementation or `'numba'` to require JIT. The default is `'auto'`. First Numba invocation incurs a JIT startup cost. Run `python bench_cse.py --repeat 50 --out dag_results/cse_benchmark.json` to measure CSE-only speedup on this computer. Results on synthetic tiles do not establish full-model RAM/VRAM savings.

The A/B orchestrator now prints periodic compiler progress and ETA. `Ctrl+C` terminates its child compiler, leaving the already committed index and stream resumable using `whole_model_dag.py` with the **same source GGUF and compiler options**. The original inference A/B remains blocked until an executable DAG runtime exists.
