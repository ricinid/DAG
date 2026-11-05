# Algebraic Snapshot DAG — prototype v0.1

A **research PoC**, not a production LLM backend. It implements:

1. Import of an **unpacked signed INT4** matrix (`.npy` or explicitly named rank-2 integer tensor in `.safetensors`).
2. Greedy common algebraic expression extraction **before** arithmetic lowering. Nodes can be shared across output rows; identical nodes across row tiles are interned globally.
3. Serializable DAG (`.npz`) with exact integer weights recoverable for small test matrices.
4. Dependency scheduling into independent levels; sparse COO representation for each level and for the output reducer (`.json`).
5. Scalar Python and levelized NumPy executors; optional PyTorch executor that can run on CUDA, with constant DAG indices staged once (`TorchDAGExecutor`). **Not a fused CUDA kernel.**
6. Completely unrolled C99 code generation with constant weights embedded in source, no dense matrix at runtime.
7. Exactness checks on randomized integer input vectors and small-matrix coefficient reconstruction.

**Scope warning:** mathematically exact integer calculations only, assuming signed int64 never overflows. BF16/FP16/FP32, GPTQ/AWQ bitpacked tensors, per-group scales, bias, activation, and full Qwen layer semantics are not implemented. There is no silent quantization. Compiling from an actual BF16 Qwen snapshot currently fails safely with an explanatory message.

## Quick start

```bash
python -m pip install numpy pytest
python algebraic_dag.py demo --case both --rows 32 --cols 64
python -m pytest -q test_algebraic_dag.py
```

Optional dependency for GPU functional execution (requires your own NVIDIA/CUDA-capable PyTorch installation):

```bash
python -m pip install torch safetensors
```

Generate DAG, all level matrices, Graphviz source, and hardwired C source from unpacked INT4 weights:

```bash
python algebraic_dag.py compile --matrix dag_example_weights.npy --out dag_example.npz --export-layers dag_example_layers.json --emit-c dag_example.c --emit-dot dag_example.dot --torch-device cpu
python algebraic_dag.py verify --dag dag_example.npz --matrix dag_example_weights.npy
```

Compile C source using GCC or Clang (Linux):

```bash
cc -O3 -shared -fPIC -std=c99 dag_example.c -o dag_example.so
```

Optional Graphviz graph rendering:

```bash
dot -Tsvg dag_example.dot -o dag_example.svg
```

## Future Qwen adapter

The *official* Qwen2.5 line has 3B and 7B, **not 5B**. For a BF16 Qwen snapshot we must first implement a float-semantic backend or handle a compatible pre-quantized snapshot *with correct decoding of its packing/scales*. Treating GPTQ/AWQ `qweight` integer storage as unpacked signed INT4 is **wrong**.

The following syntax is already valid for **unpacked integer** `.safetensors` tensors, but not for ordinary BF16 Qwen weights:

```bash
python algebraic_dag.py compile --safetensors weights.safetensors --tensor model.layers.0.mlp.gate_proj.weight --rows 32 --cols 64 --out sample.dag.npz
```

To fully compile the Qwen MLP we should extract `gate_proj` and `up_proj` as **one logical linear region with the same input** (stack rows), and stop before SiLU/gating. This region-combination frontend is not yet implemented.

## Limitations of this version

- Optimization is limited to `tile_rows` × `tile_cols` and a bounded greedy pair search. It does not guarantee optimal algebraic factoring, and misses patterns spanning column tiles.
- `plain_ir_estimated_bytes` is the size of a naive uncompressed representation. It is **not** the actual `.npz` size or a claim of lossless compression.
- Algorithmic addition counts are not speedups. Real performance must be compared against packed INT4 GEMV/GEMM, CPU JIT/unrolled code, GPU kernel and actual hardware constraints.
- Scalar C source can grow explosively for large matrices; compilation and CPU I-cache effects need separate measurement.
- Current PyTorch CUDA path launches many operations and is only a functional reference for future Triton/CUDA fusion.
- Output reducer is a sparse linear transform after shared DAG nodes; each stage can also be represented as sparse COO rather than reconstructing W.