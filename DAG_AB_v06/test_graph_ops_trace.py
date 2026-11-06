"""Check graph diagnostics install location and failure isolation."""
from pathlib import Path
import pytest

from trace_graph_ops import install, ANCHOR, MARKER


def make_source(tmp_path, contents):
    p = tmp_path / 'ggml/src/ggml-cpu/ggml-cpu.c'
    p.parent.mkdir(parents=True)
    p.write_text(contents)
    return p


def test_graph_trace_injects_before_dispatch_and_only_once(tmp_path):
    source = make_source(
        tmp_path,
        'void f() {\n'
        '    // DAG_NATIVE_EARLY_MUL_MAT_DISPATCH\n'
        '    if (ggml_cpu_extra_compute_forward(params, tensor)) return;\n'
        + ANCHOR + '\n}\n'
    )
    assert install(tmp_path) == source
    new = source.read_text()
    assert new.count(MARKER) == 1
    assert new.index(MARKER) < new.index('ggml_cpu_try_fuse_ops(cgraph')
    assert 'ggml_op_name(node->op)' in new
    assert 'getenv("DAG_STREAM_DIR")' in new
    assert 'dag_gate_count' in new
    with pytest.raises(ValueError, match='already installed'):
        install(tmp_path)


def test_graph_trace_rejects_old_hook(tmp_path):
    source = make_source(tmp_path, ANCHOR + '\n')
    original = source.read_text()
    with pytest.raises(RuntimeError, match='Early DAG hook absent'):
        install(tmp_path)
    assert source.read_text() == original


def test_graph_trace_rejects_unknown_graph_loop(tmp_path):
    source = make_source(tmp_path, 'DAG_NATIVE_EARLY_MUL_MAT_DISPATCH\nunknown')
    original = source.read_text()
    with pytest.raises(RuntimeError, match='anchor missing'):
        install(tmp_path)
    assert source.read_text() == original
