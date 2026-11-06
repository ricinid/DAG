"""Compare optional decoded-plan execution to v0.6 binary interpreter exactly."""
from __future__ import annotations

import ctypes
import json

import numpy as np
import pytest

from test_native_dag import load_lib, fake_gguf, compile_full


@pytest.mark.parametrize('dtype', [2, 12])
@pytest.mark.parametrize('rows', [2, 7, 32])
@pytest.mark.parametrize('cache_bytes', [0, 1024, 1024 * 1024])
def test_decoded_plan_matches_interpreter(tmp_path, monkeypatch, dtype, rows, cache_bytes):
    gguf, _ = fake_gguf(tmp_path, dtype, rows=rows, cols=512, structured=False)
    compiled = tmp_path / 'compiled'
    compile_full(gguf, compiled, rows=4, cols=256, max_nodes=2)
    lib = load_lib()
    outputs = {}
    for plan in (0, 1):
        monkeypatch.setenv('DAG_EXEC_PLAN', str(plan))
        handle = lib.dag_context_open(str(compiled).encode(), cache_bytes)
        assert handle
        try:
            samples = []
            rng = np.random.default_rng(338)
            for _ in range(3):
                x = rng.normal(size=512).astype(np.float32)
                y = np.full(rows, np.nan, dtype=np.float32)
                assert lib.dag_context_gemv(
                    handle, b'blk.0.ffn_gate.weight',
                    x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x.size,
                    y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y.size
                ) == 1
                samples.append(y.copy())
            buf = ctypes.create_string_buffer(4096)
            assert lib.dag_context_stats(handle, buf, len(buf)) > 0
            stats = json.loads(buf.value)
            assert stats['exec_plan'] == plan
            assert stats['gemv_calls'] == 3
            if cache_bytes:
                assert stats['cache_payload_bytes'] <= cache_bytes
            outputs[plan] = samples
        finally:
            lib.dag_context_close(handle)
    for raw, decoded in zip(outputs[0], outputs[1]):
        np.testing.assert_array_equal(raw.view('uint32'), decoded.view('uint32'))


def test_exec_plan_rejects_invalid_mode(tmp_path, monkeypatch):
    gguf, _ = fake_gguf(tmp_path, 2, rows=2, cols=256)
    compiled = tmp_path / 'compiled'
    compile_full(gguf, compiled, rows=2, cols=256, max_nodes=2)
    monkeypatch.setenv('DAG_EXEC_PLAN', 'yes')
    lib = load_lib()
    assert not lib.dag_context_open(str(compiled).encode(), 0)
