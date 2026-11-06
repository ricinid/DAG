"""Ensure per-GEMV native snapshots survive Windows forced server termination."""
from __future__ import annotations
import ctypes
import json
import numpy as np

from test_native_dag import load_lib, fake_gguf, compile_full


def test_per_gemv_snapshots_and_cache_hits(tmp_path, monkeypatch, capfd):
    model, _ = fake_gguf(tmp_path, 2, rows=2, cols=512, structured=False)
    compiled = tmp_path / 'compiled'
    compile_full(model, compiled, rows=2, cols=512, max_nodes=2)
    monkeypatch.setenv('DAG_STATS_EVERY', '1')
    lib = load_lib()
    handle = lib.dag_context_open(str(compiled).encode(), 0)
    assert handle
    try:
        x = np.arange(512, dtype=np.float32)
        y = np.empty(2, dtype=np.float32)
        for _ in range(3):
            assert lib.dag_context_gemv(
                handle, b'blk.0.ffn_gate.weight',
                x.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), x.size,
                y.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), y.size
            ) == 1
    finally:
        lib.dag_context_close(handle)
    raw = capfd.readouterr().err
    stats = [json.loads(line.split('[DAG native] STATS ', 1)[1])
             for line in raw.splitlines() if '[DAG native] STATS ' in line]
    assert [s['gemv_calls'] for s in stats] == [1, 2, 3]
    assert all(s['stats_every'] == 1 for s in stats)
    assert stats[0]['cache_misses'] > 0
    assert stats[-1]['cache_hits'] > 0
    assert stats[0]['gemv_cold_calls'] == 1
    assert stats[0]['gemv_warm_calls'] == 0
    assert stats[-1]['gemv_cold_calls'] == 1
    assert stats[-1]['gemv_warm_calls'] == 2
    assert stats[-1]['gemv_first_ns'] > 0
    assert stats[-1]['gemv_last_ns'] > 0
    assert stats[-1]['gemv_warm_total_ns'] > 0
    assert stats[-1]['gemv_total_ns'] >= stats[-1]['gemv_warm_total_ns']
    assert stats[-1]['gemv_execute_total_ns'] > 0
    assert stats[-1]['gemv_load_total_ns'] > 0
    assert stats[-1]['stream_io_total_ns'] > 0
    assert stats[-1]['unit_materialize_total_ns'] > 0
    assert stats[0]['gemv_warm_total_ns'] == 0
    assert stats[-1]['gemv_last_execute_ns'] <= stats[-1]['gemv_last_ns']
    assert stats[-1]['gemv_last_load_ns'] <= stats[-1]['gemv_last_ns']
