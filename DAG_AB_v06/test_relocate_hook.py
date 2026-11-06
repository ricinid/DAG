"""Check dispatch order and migration safety without requiring a llama.cpp build."""
from pathlib import Path
import pytest
from relocate_hook import migrate, HOOK, EARLY

STUB = '''#include "dag_runtime.h"
void ggml_compute_forward_mul_mat(void *params, void *dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];
#ifdef GGML_DAG_RUNTIME
    // DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK
    if (dag_try_ggml_gemv(src0->name, params->ith, x, n, y, m)) return;
#endif
    int hint = 0;
}
static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    GGML_ASSERT(params);
    if (tensor->op == GGML_OP_NONE || ggml_is_empty(tensor)) {
        return;
    }
    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }
    switch (tensor->op) { default: return; }
}
'''
DIAG = '''    // DAG_NATIVE_EXPERIMENTAL_DIAGNOSTIC
    if (getenv("DAG_HOOK_DIAGNOSTIC")) fprintf(stderr,"[DAG hook diag]\\n");
'''


def setup(tmp_path, text=STUB):
    file = tmp_path / 'ggml/src/ggml-cpu/ggml-cpu.c'
    file.parent.mkdir(parents=True)
    file.write_text(text)
    return file


@pytest.mark.parametrize('diag', [False, True])
def test_move_before_extra_and_preserve_diagnostic(tmp_path, diag):
    original = STUB
    if diag:
        original = original.replace(HOOK+'\n', HOOK+'\n'+DIAG)
    f=setup(tmp_path, original)
    migrate(tmp_path)
    modified=f.read_text()
    assert modified.count(HOOK) == 1
    assert modified.count(EARLY) == 1
    assert modified.index(EARLY) < modified.index('ggml_cpu_extra_compute_forward')
    assert modified.index(EARLY) > modified.index('static void ggml_compute_forward')
    assert modified.index(EARLY) > modified.index('void ggml_compute_forward_mul_mat')
    assert modified.count('dag_try_ggml_gemv(') == 1
    assert ('DAG_NATIVE_EXPERIMENTAL_DIAGNOSTIC' in modified) == diag
    assert 'if (tensor->op == GGML_OP_MUL_MAT)' in modified
    with pytest.raises(ValueError, match='already'):
        migrate(tmp_path)


def test_refuse_unpatched(tmp_path):
    file=setup(tmp_path, STUB.replace('DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK','UNPATCHED'))
    saved=file.read_text()
    with pytest.raises(RuntimeError,match='Old DAG hook not found'):
        migrate(tmp_path)
    assert file.read_text() == saved


def test_refuse_changed_dispatch(tmp_path):
    original = STUB.replace('    // extra_buffer op?', '    // changed upstream')
    file=setup(tmp_path, original)
    with pytest.raises(RuntimeError, match='CPU extra-buffer dispatch'):
        migrate(tmp_path)
    assert file.read_text() == original
