"""The shadow diagnostic must never alter the default GGML hook path."""
from pathlib import Path

import pytest

from install_shadow_gemv import EARLY, HOOK, SENTINEL, install


def installed_cpu_source() -> str:
    return '''
#include <stdint.h>
void ggml_compute_forward(void) {
#ifdef GGML_DAG_RUNTIME
    // DAG_NATIVE_EARLY_MUL_MAT_DISPATCH
    if (tensor->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor * src0 = tensor->src[0];
        const struct ggml_tensor * src1 = tensor->src[1];
        struct ggml_tensor * dst = tensor;
    // DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK
    if (params->nth == 1) {
        if (dag_try_ggml_gemv(src0->name, params->ith,
                              (const float *) src1->data, (uint64_t) src1->ne[0],
                              (float *) dst->data, (uint64_t) src0->ne[1])) {
            return;
        }
    }
    }
#endif
    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }
}
'''


def cpu_file(tmp_path: Path, contents: str) -> Path:
    path = tmp_path / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    path.parent.mkdir(parents=True)
    path.write_text(contents, encoding="utf-8")
    return path


def test_shadow_is_in_successful_early_hook_only(tmp_path):
    path = cpu_file(tmp_path, installed_cpu_source())
    result = install(tmp_path)
    assert result == path
    actual = path.read_text(encoding="utf-8")
    assert actual.count(SENTINEL) == 1
    assert actual.count(EARLY) == 1
    assert actual.count(HOOK) == 1
    assert actual.count("dag_try_ggml_gemv(") == 1
    assert actual.count("ggml_cpu_extra_compute_forward(params, tensor)") == 2
    assert actual.index(EARLY) < actual.index(SENTINEL) < actual.index("    // extra_buffer op?")
    assert actual.index("memcpy(dst->data, dag_saved_output, dag_shadow_bytes);") < actual.index("return;", actual.index(SENTINEL))
    assert 'strcmp(dag_shadow_env, "1") == 0' in actual
    assert "DAG_SHADOW_MAX" in actual
    assert 'reference=%s' in actual
    original_len = len(actual)
    with pytest.raises(ValueError, match="already installed"):
        install(tmp_path)
    assert len(path.read_text(encoding="utf-8")) == original_len


def test_refuses_unpatched_or_unrelocated_source(tmp_path):
    path = cpu_file(tmp_path, installed_cpu_source().replace(EARLY, "NO EARLY HOOK"))
    old = path.read_text()
    with pytest.raises(RuntimeError, match="early"):
        install(tmp_path)
    assert path.read_text() == old


def test_refuses_missing_handled_dag_call(tmp_path):
    path = cpu_file(tmp_path, installed_cpu_source().replace("dag_try_ggml_gemv(", "no_such_hook("))
    old = path.read_text()
    with pytest.raises(RuntimeError, match="handled DAG GEMV call"):
        install(tmp_path)
    assert path.read_text() == old


def test_rejects_dag_hook_installed_behind_extra_dispatch(tmp_path):
    text = installed_cpu_source()
    text = text.replace(EARLY, "REPLACED")
    text = text.replace("    // extra_buffer op?", "    // extra_buffer op?\n    // DAG_NATIVE_EARLY_MUL_MAT_DISPATCH")
    path = cpu_file(tmp_path, text)
    with pytest.raises(RuntimeError, match="before CPU"):
        install(tmp_path)
