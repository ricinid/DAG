"""Regression test for the pinned GGML shadow Q8 diagnostic source transformation."""
from pathlib import Path
import pytest

from extend_shadow_q8 import install, MARKER


def make_source(root: Path, text: str) -> Path:
    path = root / 'ggml' / 'src' / 'ggml-cpu' / 'ggml-cpu.c'
    path.parent.mkdir(parents=True)
    path.write_text(text, encoding='utf-8')
    return path


def minimal_existing_shadow() -> str:
    return '''
#include "quants.h"
void f() {
    // DAG_NATIVE_SHADOW_REFERENCE_COMPARE
    const char * p = "reference";
    if (p) {
            // Restore the native DAG values. No GGML reference result enters
            memcpy(dst->data, dag_saved_output, dag_shadow_bytes);
    }
}
'''


def test_extension_injected_once_and_restores_default_result(tmp_path):
    path = make_source(tmp_path, minimal_existing_shadow())
    assert install(tmp_path) == path
    code = path.read_text()
    assert code.count(MARKER) == 1
    assert code.count('#include "ggml-quants.h"') == 1
    assert code.count('#include "quants.h"') == 1
    assert code.index(MARKER) < code.index(
        'memcpy(dst->data, dag_saved_output, dag_shadow_bytes);')
    assert 'type_traits_cpu[dag_qtype].from_float(' in code
    assert 'dequantize_row_q8_K(' in code
    assert 'dequantize_row_q8_0(' in code
    assert 'dag_try_ggml_gemv(' in code
    assert 'dag_rmse / dag_q8_rmse' in code
    old = path.read_bytes()
    with pytest.raises(ValueError, match="already installed"):
        install(tmp_path)
    assert path.read_bytes() == old


def test_fail_closed_without_existing_shadow(tmp_path):
    path = make_source(tmp_path, '#include "quants.h"\n')
    old = path.read_bytes()
    with pytest.raises(RuntimeError, match="original shadow"):
        install(tmp_path)
    assert path.read_bytes() == old


def test_fail_closed_on_duplicate_restore_anchor(tmp_path):
    src = minimal_existing_shadow()
    src += '            // Restore the native DAG values. No GGML reference result enters\n'
    path = make_source(tmp_path, src)
    original = path.read_bytes()
    with pytest.raises(RuntimeError, match="original shadow"):
        install(tmp_path)
    assert path.read_bytes() == original
