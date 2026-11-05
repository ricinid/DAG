"""Synthetic exact-Q4 kernel tests; do not require a real LLM."""
from __future__ import annotations
import ctypes
import json
import os
import subprocess
import sys
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
if (REPO / 'whole_model_dag.py').is_file():
    sys.path.insert(0, str(REPO))
else:
    # Local test workspace fallback. Not used when installed inside the DAG repo.
    sys.path.insert(0, '/mnt/data/dag_v04')
    sys.path.insert(0, '/mnt/data/dag_cse_fast')
from test_ollama_dag_pipeline import fake_gguf
from whole_model_dag import compile_full
from ollama_dag_pipeline import GGUFReader, decode_raw_selection


def load_lib():
    if sys.platform == 'win32':
        libpath = ROOT / 'native' / 'dag_native.dll'
    elif sys.platform == 'darwin':
        libpath = ROOT / 'native' / 'libdag_native.dylib'
    else:
        libpath = ROOT / 'native' / 'libdag_native.so'
    if not libpath.exists():
        pytest.skip(f'Compile native library first: {libpath}')
    lib = ctypes.CDLL(str(libpath))
    lib.dag_context_open.argtypes = [ctypes.c_char_p, ctypes.c_uint64]
    lib.dag_context_open.restype = ctypes.c_void_p
    lib.dag_context_close.argtypes = [ctypes.c_void_p]
    lib.dag_context_gemv.argtypes = [ctypes.c_void_p, ctypes.c_char_p,
                                     ctypes.POINTER(ctypes.c_float), ctypes.c_uint64,
                                     ctypes.POINTER(ctypes.c_float), ctypes.c_uint64]
    lib.dag_context_gemv.restype = ctypes.c_int
    lib.dag_context_stats.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.dag_context_stats.restype = ctypes.c_int
    return lib


@pytest.mark.parametrize('dtype', [2, 12])
@pytest.mark.parametrize('rows', [2, 7, 32])
@pytest.mark.parametrize('cache', [0, 1024, 128*1024])
def test_native_gemv_matches_raw_quantized_weights(tmp_path, dtype, rows, cache):
    model, raw = fake_gguf(tmp_path, dtype, rows=rows, cols=512, structured=False)
    compiled = tmp_path / 'compiled'
    compile_full(model, compiled, rows=4, cols=256, max_nodes=2)
    lib = load_lib()
    handle = lib.dag_context_open(str(compiled).encode(), cache)
    assert handle
    try:
        rng = np.random.default_rng(12)
        for _ in range(3):
            x = rng.normal(size=512).astype(np.float32)
            y = np.zeros(rows, dtype=np.float32)
            rc = lib.dag_context_gemv(handle, b'blk.0.ffn_gate.weight',
                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x),
                    y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(y))
            assert rc == 1
            expected = decode_raw_selection(raw, dtype, rows, 512) @ x.astype(np.float64)
            np.testing.assert_allclose(y, expected, rtol=3e-5, atol=3e-5)
        out = ctypes.create_string_buffer(1000)
        assert lib.dag_context_stats(handle, out, len(out)) > 0
        stats = json.loads(out.value)
        assert stats['gemv_calls'] == 3
        assert stats['cache_budget_bytes'] == cache
        if cache:
            assert stats['cache_payload_bytes'] <= cache
        assert lib.dag_context_gemv(handle, b'unknown',
                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(x),
                    y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), len(y)) == 0
    finally:
        lib.dag_context_close(handle)