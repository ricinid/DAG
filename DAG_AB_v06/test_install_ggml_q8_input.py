"""Fail-closed Q8 input bridge installer; raw and shadow branches remain intact."""
from pathlib import Path

import pytest

from install_ggml_q8_input import EARLY, HOOK, MARKER, install
from patch_llama_cpp import HOOK as EARLY_HOOK


def make_source(root: Path, body: str) -> Path:
    target=root/"ggml"/"src"/"ggml-cpu"/"ggml-cpu.c"
    target.parent.mkdir(parents=True)
    target.write_text(body, encoding="utf-8")
    return target


def patched_source() -> str:
    return '#include "quants.h"\nvoid work(void) {\n' + EARLY_HOOK + '''
    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }
}
'''


def test_single_pass_installs_and_keeps_native_f32(tmp_path):
    target=make_source(tmp_path,patched_source())
    assert install(tmp_path)==target
    compiled=target.read_text(encoding="utf-8")
    assert compiled.count(MARKER)==1
    assert compiled.count(EARLY)==1 and compiled.count(HOOK)==1
    assert compiled.count("dag_try_ggml_gemv(")==2
    assert "DAG_INPUT_SEMANTICS" in compiled
    assert 'strcmp(dag_input_semantics, "ggml-q8")' in compiled
    assert 'strcmp(dag_input_semantics, "native-f32")' in compiled
    assert "type_traits_cpu[dag_qtype].from_float(" in compiled
    assert "dequantize_row_q8_K(" in compiled
    assert "dequantize_row_q8_0(" in compiled
    assert 'const int dag_rc = dag_try_ggml_gemv(' in compiled
    assert compiled.index(MARKER)<compiled.index("if (dag_try_ggml_gemv(")
    assert compiled.index("} else\n")<compiled.index("if (dag_try_ggml_gemv(")
    assert compiled.count('#include "ggml-quants.h"')==1
    previous=target.read_bytes()
    with pytest.raises(ValueError,match="already installed"):
        install(tmp_path)
    assert target.read_bytes()==previous


def test_works_on_tree_with_shadow_installed(tmp_path):
    source=patched_source().replace(
        "            return;\n        }",
        '            // DAG_NATIVE_SHADOW_REFERENCE_COMPARE\n'
        '            // Original F32 and Q8 shadow paths must remain inside raw mode.\n'
        '            return;\n        }',1)
    target=make_source(tmp_path,source)
    install(tmp_path)
    text=target.read_text()
    assert "DAG_NATIVE_SHADOW_REFERENCE_COMPARE" in text
    assert text.index(MARKER)<text.index("DAG_NATIVE_SHADOW_REFERENCE_COMPARE")
    assert text.count("DAG_NATIVE_SHADOW_REFERENCE_COMPARE")==1


def test_refuses_missing_hook_without_mutation(tmp_path):
    original=patched_source().replace(HOOK,"NO_HOOK")
    target=make_source(tmp_path,original)
    with pytest.raises(RuntimeError,match="early DAG hook"):
        install(tmp_path)
    assert target.read_text()==original


def test_refuses_relocated_after_extra_dispatch(tmp_path):
    source=patched_source()
    source=source.replace(EARLY,"NOT_HERE").replace(
        "    // extra_buffer op?",
        "    // extra_buffer op?\n    // DAG_NATIVE_EARLY_MUL_MAT_DISPATCH")
    target=make_source(tmp_path,source)
    with pytest.raises(RuntimeError,match="before GGML CPU-extra"):
        install(tmp_path)
