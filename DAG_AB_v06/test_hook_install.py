from pathlib import Path
import pytest
from patch_llama_cpp import patch, SENTINEL

def test_ggml_source_patch_and_reject_double_insertion(tmp_path):
    p = tmp_path/'ggml/src/ggml-cpu'
    p.mkdir(parents=True)
    (p/'ggml-cpu.c').write_text('''void ggml_compute_forward_mul_mat(const struct ggml_compute_params * params,
              struct ggml_tensor * dst) {
    const struct ggml_tensor * src0 = dst->src[0];
    const struct ggml_tensor * src1 = dst->src[1];

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
}''')
    (p/'CMakeLists.txt').write_text('    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)')
    patch(tmp_path)
    assert SENTINEL in (p/'ggml-cpu.c').read_text()
    assert (p/'dag_runtime.cpp').is_file()
    assert 'find_package(ZLIB REQUIRED)' in (p/'CMakeLists.txt').read_text()
    with pytest.raises(ValueError, match='Already patched'):
        patch(tmp_path)


def test_patch_fails_closed_on_unknown_api(tmp_path):
    p = tmp_path/'ggml/src/ggml-cpu'
    p.mkdir(parents=True)
    (p/'ggml-cpu.c').write_text('incorrect API')
    (p/'CMakeLists.txt').write_text('    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)')
    with pytest.raises(RuntimeError,match='could not safely find'):
        patch(tmp_path)