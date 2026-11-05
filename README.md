# DAG v0.3 — RAM / VRAM experiment

This prototype runs `Ollama GGUF -> quantized tensor slice -> algebraic DAG -> profiles`.
It currently compiles **one slice**, not the whole model, and is not installed in
Ollama. No end-to-end VRAM or generation performance savings are claimed.

## Install and run (Windows PowerShell)

```powershell
py -m pip install -r requirements_dag.txt
# Optional for GPU profile: install CUDA-enabled PyTorch from https://pytorch.org/get-started/locally/
py ollama_dag_pipeline.py run --model qwen2.5-coder:3b --pull `
  --rows 128 --cols 1024 --bench-ollama --profile-memory --profile-cuda `
  --out-dir dag_results
```

You may use `--gguf C:\\path\\to\\model.gguf` instead of `--model` to test
without Ollama. For a fast synthetic smoke test use the provided unit tests.

## RAM / VRAM metrics

`dag_results/report.json` contains:

- `memory_profile.metrics.compile.peak_rss_bytes`: compile-time RAM RSS.
- `memory_profile.metrics.dense.peak_rss_bytes`: predecoded dense NumPy baseline RAM RSS.
- `memory_profile.metrics.packed.peak_rss_bytes`: original packed-code decoding RAM RSS.
- `memory_profile.metrics.dag.peak_rss_bytes`: executable DAG CPU RAM RSS.
- `memory_profile.metrics.dense_cuda.cuda_allocator.torch_peak_reserved_bytes`: dense CUDA reference PyTorch allocator VRAM (if CUDA available).
- `memory_profile.metrics.dag_cuda.cuda_allocator.torch_peak_reserved_bytes`: unfused DAG CUDA PyTorch allocator VRAM (if CUDA available).
- `ollama_model_residency.ollama_reported_vram_bytes`: **original** model allocation via `/api/ps`, not DAG peak VRAM.

Each mode executes in a new process so the dense baseline does not contaminate
DAG peak RAM. RSS includes interpreter and NumPy initialization; CUDA allocator
metrics omit some driver/context overhead and unrelated processes. If CUDA is
unavailable, it is marked `unavailable`, never fabricated as zero. Do not
compare the slice measurements to whole-model Ollama GPU allocation as a proof
of reduced inference VRAM. This requires integration with llama.cpp/Ollama.

`--no-profile-memory` skips all isolated benchmarks;
`--no-profile-cuda` skips CUDA workers. `--memory-repeat` increases the number of
executions; `--memory-interval-ms` controls RSS sampling.

Original snapshot quantization is preserved; the selected Q4_0 / Q4_K slice
can be restored byte-for-byte, but arithmetic output comparisons use float64
and are not bit-exact to llama.cpp floating-point execution.