#!/usr/bin/env python3
"""Move an installed DAG v0.6 GGML hook before the CPU_REPACK extra dispatch.

The old hook was inside ggml_compute_forward_mul_mat(), which Q4 extra-buffer
kernels can bypass. This migration works on an ALREADY PATCHED llama.cpp tree;
retains optional diagnostic instrumentation and leaves unrelated GGML code intact.
Refuses unknown/ambiguous sources or repeated migrations.
"""
from __future__ import annotations
import argparse
from pathlib import Path

HOOK = '    // DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK'
EARLY = '    // DAG_NATIVE_EARLY_MUL_MAT_DISPATCH'
START = '\n#ifdef GGML_DAG_RUNTIME\n' + HOOK + '\n'
END = '\n#endif\n'
DISPATCH = '''    // extra_buffer op?
    if (ggml_cpu_extra_compute_forward(params, tensor)) {
        return;
    }
'''


def migrate(root: Path) -> Path:
    source = root / 'ggml' / 'src' / 'ggml-cpu' / 'ggml-cpu.c'
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding='utf-8')
    if EARLY in text:
        raise ValueError('DAG hook is already at early MUL_MAT dispatch')
    if text.count(START) != 1:
        raise RuntimeError('Old DAG hook not found exactly once; source left unchanged')
    if text.count(DISPATCH) != 1:
        raise RuntimeError('Expected CPU extra-buffer dispatch not found exactly once')
    start = text.index(START)
    stop = text.find(END, start + len(START))
    if stop < 0:
        raise RuntimeError('Cannot find closing #endif for old hook')
    stop += len(END)
    block = text[start:stop]
    if block.count(HOOK) != 1 or block.count('dag_try_ggml_gemv(') != 1:
        raise RuntimeError('Unexpected contents of old hook')
    # The installed diagnostic, if present, lives after the hook marker.
    # Both it and the dispatch guard are moved to the new call site.
    inner = block[len('\n#ifdef GGML_DAG_RUNTIME\n'):-len(END)]
    if 'DAG_NATIVE_EXPERIMENTAL_DIAGNOSTIC' in inner and inner.count('DAG_NATIVE_EXPERIMENTAL_DIAGNOSTIC') != 1:
        raise RuntimeError('Ambiguous DAG diagnostic instrumentation')
    replacement = '''#ifdef GGML_DAG_RUNTIME
''' + EARLY + '''
    if (tensor->op == GGML_OP_MUL_MAT) {
        const struct ggml_tensor * src0 = tensor->src[0];
        const struct ggml_tensor * src1 = tensor->src[1];
        struct ggml_tensor * dst = tensor;
''' + inner + '''
    }
#endif

'''
    updated = text[:start] + text[stop:]
    updated = updated.replace(DISPATCH, replacement + DISPATCH, 1)
    if updated.count(HOOK) != 1 or updated.count(EARLY) != 1:
        raise AssertionError('DAG hook moved more or fewer than once')
    if updated.index(EARLY) > updated.index('    // extra_buffer op?'):
        raise AssertionError('DAG hook is still behind extra dispatch')
    source.write_text(updated, encoding='utf-8')
    return source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--llama-src', type=Path, required=True)
    args = ap.parse_args()
    print(f'Migrated DAG GEMV hook before CPU extra-buffer dispatch: {migrate(args.llama_src.resolve())}')


if __name__ == '__main__':
    main()
