# Ollama Snapshot → Algebraic DAG (research PoC v0.2)

## What it does

The CLI reads **a real Ollama GGUF blob**, selects one 2D quantized weight tensor region,
extracts signed 4-bit code groups, runs global-greedy arithmetic common-subexpression
elimination **before** arithmetic lowering, and serializes a compact algebraic DAG.
It reports **measured byte sizes**, validates **bit-for-bit reconstruction of the selected
packed GGUF bytes**, and benchmarks an independent original-weights NumPy reference
against the DAG evaluator. Optionally it builds a **fully reversible .dagpack** of the
entire original GGUF, and benchmarks unmodified Ollama token generation separately.

**Critical scope:** it does NOT yet optimize an entire LLM or replace Ollama's executor.
Full-snapshot .dagpack is our research archive, **not a runnable GGUF or an Ollama model**.
Restore it to GGUF to load it with Ollama. The full archive only replaces one
selected tensor **slice**; its size is an actual on-disk byte count, but must not be
interpreted as whole-model compression performance. The numerical microbenchmark
is **not** LLM tokens/second; it uses pure Python/NumPy and does not implement a
fused CUDA kernel.

Input support: GGUF `Q4_0` and `Q4_K` (including individual Q4_K tensors in
mixed `Q4_K_M` models). Other GGUF quantization types are **not silently converted**.
Quantization metadata (half-precision scales, 6-bit scales/minima) is preserved
byte-for-byte. The immutable quantized snapshot is the reference; nothing is
fine-tuned or requantized. Operations are mathematically exact over integer codes
with sufficient accumulator width. Floating output comparison uses float64 and
**does not claim bit-exact equivalence with llama.cpp's floating-point inference**.

## Install

Tested with Python 3.11+, NumPy, pytest. Ollama must be installed **on the user's
computer** for `--model` / `--pull`; no Ollama or GPU required to run the synthetic tests.

```powershell
py -m pip install numpy pytest
py -m pytest -q test_algebraic_dag.py test_ollama_dag_pipeline.py
```

## Run an actual Ollama snapshot

Start Ollama (if it is not already running). The CLI uses the local server
`http://localhost:11434`, Ollama's official `/api/pull` endpoint and the local
`manifests` → `blobs/sha256-*` cache. It **does not download from this ChatGPT environment**.

The official Qwen2.5 Coder options include **3B and 7B**, not 5B.

```powershell
# Inspect the model's tensor names / quantization types first:
py ollama_dag_pipeline.py inspect --model qwen2.5-coder:3b --pull --limit 30

# Single-tensor-slice DAG, exact packed-byte restoration, real size measurements,
# NumPy microbenchmarks and original Ollama generation baseline:
py ollama_dag_pipeline.py run --model qwen2.5-coder:3b --rows 32 --cols 256 --pack-full --bench-ollama --out-dir dag_results

# To target a specific tensor and a larger experimental region:
py ollama_dag_pipeline.py run --model qwen2.5-coder:3b --tensor ffn_gate --rows 128 --cols 1024 --pack-full --out-dir dag_large
```

You can use `--spool` as an alias for `--pull`. You may also bypass Ollama:

```powershell
py ollama_dag_pipeline.py run --gguf C:\path\to\model.gguf --rows 32 --cols 256 --pack-full
```

If Ollama runs as a Windows service or stores files on another drive:

```powershell
py ollama_dag_pipeline.py run --model qwen2.5-coder:3b --models-dir D:\OllamaModels --rows 32 --cols 256
```

Or set the `OLLAMA_MODELS` environment variable to the local model directory.

## Outputs

- `dag_results/selected.adag`: **actual** zipped compressed DAG for the selected tensor slice, quantization metadata and hash.
- `dag_results/report.json`: measured byte sizes (GGUF original, raw slice, `.adag`, actual full `.dagpack` if requested), DAG node/level counts, compile time and matrix-vector benchmarks.
- `dag_results/model.dagpack` when `--pack-full`: **actual full snapshot archive**, containing unchanged bytes plus DAG-replaced slice. The archive has its own container overhead; it may be **larger** than original GGUF.

Restore a full package to its original GGUF (SHA-256 checked):

```powershell
py ollama_dag_pipeline.py restore --pack dag_results/model.dagpack --out restored.gguf
```

The restored file can be used as input to Ollama/llama.cpp as appropriate. The archive itself cannot.

## What the benchmarks mean

Three CPU microbenchmarks run on the *same* selected tensor slice and input vector:

1. `predecoded_dense_numpy_matvec_ms` — NumPy matvec after unpacking GGUF weights once into dense float64; excludes decoding cost, higher RAM usage.
2. `packed_gguf_decode_plus_numpy_matvec_ms` — unpack GGUF quantization and perform NumPy matvec at every invocation; includes Python decoding overhead.
3. `dag_numpy_matvec_ms` — run integer-code DAG groups and original per-block scale/offset correction, without building the dense weight matrix.

These are **not optimized production baselines**; in particular they are not
comparisons with llama.cpp's AVX/CUDA quantized matvec implementations. Do not infer
tokens/sec improvement from their ratios.

If `--bench-ollama` is selected, `ollama_inference_baseline` reports **only the unmodified
original model's** generation tokens/sec from Ollama. The DAG is not yet integrated
into its runtime. A proper whole-model A/B benchmark requires an executor integration.

## Important engineering constraints

- Quantization block boundaries: 32 input columns for Q4_0, 256 for Q4_K.
- Reversible `.adag` contains the **original code values and original header bits**.
- DAG optimization is greedy and tiled; no claim of global optimum or consistent speedup.
- `max_nodes` is per 32-column group. High row counts / large slices can compile slowly.
- Back-end `algebraic_dag.py` supports **integer codes**, not arbitrary FP16/BF16 tensors.
- Only **one selected tensor slice** is changed per `.dagpack`, so the baseline whole-model size usually dominates; not representative of an optimized 3B snapshot.
- `--pack-full` consumes roughly another original-model-sized amount of disk, plus the DAG artifact; restore requires additional space.
- `Q4_K` per-group scales and minima are preserved; numeric execution combines integer DAGs with these scales and offsets, which changes floating-point summation order relative to some runtimes.

## Next implementation gates

1. Batch all Q4_0/Q4_K matrix slices across an entire model, with bounded-memory processing.
2. Compact DAG codegen and a fused CPU/CUDA backend (benchmark against llama.cpp/ggml).
3. Full-model correctness checks: layer outputs/logits, numerical drift, token-level A/B.
4. Real model-wide original-vs-optimized snapshot-size and latency reporting.