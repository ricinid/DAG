#!/usr/bin/env python3
"""Install the experimental *CPU-only single-thread GEMV* DAG bridge in a llama.cpp source checkout.

The patch is intentionally explicit: if current upstream signatures have changed,
refuse to patch instead of installing an unverified hook. Run on a clean checkout.
"""
from __future__ import annotations
import argparse
import shutil
from pathlib import Path

SENTINEL = "// DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK"
HEADER = '''\n#ifdef GGML_DAG_RUNTIME\n#include "dag_runtime.h"\n#endif\n'''
HOOK = '''\n#ifdef GGML_DAG_RUNTIME
    // DAG_NATIVE_EARLY_MUL_MAT_DISPATCH
    // Intercept before ggml_cpu_extra_compute_forward (CPU_REPACK etc.).
    if (tensor->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor * src0 = tensor->src[0];
        const struct ggml_tensor * src1 = tensor->src[1];
        struct ggml_tensor * dst = tensor;
    // DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK
    // Only contiguous CPU F32 GEMV (batch=1, one thread). Unsupported tensor
    // types/shapes use the existing GGML path unchanged. No GPU offload here.
    if (params->nth == 1 && !params->use_ref && src0 && src1 && dst &&
        (src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 &&
        src0->ne[2] == 1 && src0->ne[3] == 1 &&
        src1->ne[1] == 1 && src1->ne[2] == 1 && src1->ne[3] == 1 &&
        src0->ne[0] == src1->ne[0] && dst->ne[0] == src0->ne[1] &&
        src1->data && dst->data &&
        ggml_is_contiguous(src1) && ggml_is_contiguous(dst)) {
        if (dag_try_ggml_gemv(src0->name, params->ith,
                              (const float *) src1->data, (uint64_t) src1->ne[0],
                              (float *) dst->data, (uint64_t) src0->ne[1])) {
            return;
        }
    }
    }
#endif
'''
CMAKE = '''\n    if (GGML_DAG_RUNTIME)
        find_package(ZLIB REQUIRED)
        list(APPEND GGML_CPU_SOURCES
            ggml-cpu/dag_runtime.cpp
            ggml-cpu/dag_runtime.h)
        target_link_libraries(${GGML_CPU_NAME} PRIVATE ZLIB::ZLIB)
        target_compile_definitions(${GGML_CPU_NAME} PRIVATE GGML_DAG_RUNTIME=1)
    endif()
'''


def patch(root: Path) -> dict:
    src = Path(__file__).resolve().parent / 'native'
    cpu = root / 'ggml/src/ggml-cpu/ggml-cpu.c'
    cmake = root / 'ggml/src/ggml-cpu/CMakeLists.txt'
    assert cpu.is_file() and cmake.is_file(), 'Expected a llama.cpp checkout with ggml-cpu'
    cpu_text = cpu.read_text(encoding='utf-8')
    cmake_text = cmake.read_text(encoding='utf-8')
    if SENTINEL in cpu_text:
        raise ValueError('Already patched: do not inject this hook twice')
    anchor = '''    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }'''
    if cpu_text.count(anchor) != 1:
        raise RuntimeError('ggml-cpu.c changed: could not safely find early CPU extra-buffer dispatch')
    target = '    target_compile_features(${GGML_CPU_NAME} PRIVATE c_std_11 cxx_std_17)'
    if cmake_text.count(target) != 1:
        raise RuntimeError('ggml-cpu CMake signature changed; patch refused')
    if 'GGML_DAG_RUNTIME' in cmake_text:
        raise RuntimeError('Already modified CMake; patch refused')
    updated_cpu = cpu_text.replace(anchor, HOOK+'\n'+anchor)
    updated_cpu = HEADER + updated_cpu
    updated_cmake = 'option(GGML_DAG_RUNTIME "Experimental streaming DAG CPU GEMV" OFF)\n' + \
        cmake_text.replace(target,CMAKE+'\n'+target)
    assert updated_cpu.count(SENTINEL)==1
    cpu.write_text(updated_cpu,encoding='utf-8')
    cmake.write_text(updated_cmake,encoding='utf-8')
    shutil.copy2(src/'dag_runtime.cpp',cpu.parent/'dag_runtime.cpp')
    shutil.copy2(src/'dag_runtime.h',cpu.parent/'dag_runtime.h')
    return {'ggml_cpu':str(cpu),'cmake':str(cmake),'hooked':1}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-src',required=True,type=Path)
    a = p.parse_args()
    print(patch(a.llama_src.resolve()))


if __name__=='__main__': main()