"""Regression checks for one-shot DAG hook diagnostics on a patched checkout."""
from __future__ import annotations
from pathlib import Path

import pytest

from diagnose_hook import install, TRACE_MARKER


def source_tree(tmp_path):
    cpu = tmp_path / 'ggml/src/ggml-cpu/ggml-cpu.c'
    cpu.parent.mkdir(parents=True)
    cpu.write_text('''\n#ifdef GGML_DAG_RUNTIME
    // DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK
    if (dag_try_ggml_gemv(src0->name, 0, x, n, y, m)) return;
#endif
''')
    return cpu


def test_insert_and_refuse_double_insert(tmp_path):
    cpu = source_tree(tmp_path)
    assert install(tmp_path) == cpu
    txt = cpu.read_text()
    assert TRACE_MARKER in txt
    assert 'getenv("DAG_HOOK_DIAGNOSTIC")' in txt
    assert 'ggml_type_name(src1->type)' in txt
    assert txt.count('dag_try_ggml_gemv(') == 1
    assert txt.count(TRACE_MARKER) == 1
    with pytest.raises(ValueError, match='already installed'):
        install(tmp_path)


def test_refuses_untouched_upstream(tmp_path):
    cpu = source_tree(tmp_path)
    cpu.write_text('void ggml_compute_forward_mul_mat() {}')
    with pytest.raises(RuntimeError, match='hook not found'):
        install(tmp_path)
