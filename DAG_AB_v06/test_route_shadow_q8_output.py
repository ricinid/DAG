"""Idempotent source installer for token-parity Q8 shadow routing."""
from pathlib import Path

import pytest

from route_shadow_q8_output import ROUTE, install


SOURCE = '''
#include "quants.h"
static void test_shadow(void) {
    // DAG_NATIVE_SHADOW_REFERENCE_COMPARE
            // DAG_NATIVE_SHADOW_Q8_ROUNDTRIP
            if (dag_q8_rc == 1) {
                const float * dag_q8_out = (const float *) dst->data;
                // compare reference
            }
            // Restore the native DAG values. No GGML reference result enters
            memcpy(dst->data, dag_saved_output, dag_shadow_bytes);
}
'''


def make_source(root: Path, source: str) -> Path:
    path = root / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    path.parent.mkdir(parents=True)
    path.write_text(source, encoding="utf-8")
    return path


def test_q8_route_opt_in_restores_selected_result_and_refuses_second_apply(tmp_path):
    path = make_source(tmp_path, SOURCE)
    assert install(tmp_path) == path
    installed = path.read_text(encoding="utf-8")
    assert installed.count(ROUTE) == 1
    assert installed.index(ROUTE) < installed.index("const float * dag_q8_out")
    assert installed.index("const float * dag_q8_out") < installed.index(
        "if (dag_route_requested) {")
    assert installed.index("dag_route_applied = 1;") < installed.index(
        "if (dag_route_requested && !dag_route_applied)")
    assert installed.index("if (dag_route_requested && !dag_route_applied)") < installed.index(
        "memcpy(dst->data, dag_saved_output, dag_shadow_bytes);")
    assert "DAG_SHADOW_Q8_OUTPUT" in installed
    assert "[DAG shadow q8 route]" in installed
    assert "abort();" in installed
    assert installed.count('memcpy(dst->data, dag_saved_output, dag_shadow_bytes);') == 1

    original = path.read_bytes()
    with pytest.raises(ValueError, match="already installed"):
        install(tmp_path)
    assert path.read_bytes() == original


def test_q8_route_refuses_missing_original_shadow(tmp_path):
    path = make_source(tmp_path, SOURCE.replace(
        "// DAG_NATIVE_SHADOW_REFERENCE_COMPARE", "original shadow missing"))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="Requires installed"):
        install(tmp_path)
    assert path.read_bytes() == before


def test_q8_route_refuses_duplicate_marker_or_misordered_source(tmp_path):
    path = make_source(tmp_path, SOURCE.replace(
        "// DAG_NATIVE_SHADOW_Q8_ROUNDTRIP",
        "// DAG_NATIVE_SHADOW_Q8_ROUNDTRIP\n"
        "            // DAG_NATIVE_SHADOW_Q8_ROUNDTRIP"))
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="Requires installed"):
        install(tmp_path)
    assert path.read_bytes() == before
    path.write_text(SOURCE.replace(
        "            // Restore the native DAG values. No GGML reference result enters",
        "            // Restore the native DAG values. No GGML reference result enters\n"
        "            // Restore the native DAG values. No GGML reference result enters"), encoding="utf-8")
    before = path.read_bytes()
    with pytest.raises(RuntimeError, match="exactly one"):
        install(tmp_path)
    assert path.read_bytes() == before
