#!/usr/bin/env python3
"""Add opt-in, bounded MUL_MAT diagnostics to an already patched llama.cpp tree.

Does not change inference semantics. For the DAG_AB_v06 experimental CPU hook only.
"""
from __future__ import annotations
import argparse
from pathlib import Path

HOOK_MARKER = '    // DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK\n'
TRACE_MARKER = '    // DAG_NATIVE_EXPERIMENTAL_DIAGNOSTIC\n'
TRACE = r'''    // DAG_NATIVE_EXPERIMENTAL_DIAGNOSTIC
    // Diagnostic only: hard cap printing to avoid flooding logs on long decodes.
    // DAG_HOOK_DIAGNOSTIC is enabled only for an explicit smoke run.
    if (getenv("DAG_HOOK_DIAGNOSTIC") && getenv("DAG_STREAM_DIR")) {
        static atomic_int dag_trace_other = 0;
        static atomic_int dag_trace_gate = 0;
        const int is_gate = src0 && strstr(src0->name, "ffn_gate") != NULL;
        const int trace_index = is_gate ? atomic_fetch_add(&dag_trace_gate, 1)
                                        : atomic_fetch_add(&dag_trace_other, 1);
        if (trace_index < (is_gate ? 16 : 6)) {
            fprintf(stderr,
                "[DAG hook diag] src0=%s src0_type=%s src1_type=%s dst_type=%s "
                "threads=%d ith=%d use_ref=%d "
                "weight=%lldx%lld input=%lldx%lld output=%lldx%lld "
                "contig_src1=%d contig_dst=%d\\n",
                src0 ? src0->name : "<null>",
                src0 ? ggml_type_name(src0->type) : "<null>",
                src1 ? ggml_type_name(src1->type) : "<null>",
                dst ? ggml_type_name(dst->type) : "<null>",
                params->nth, params->ith, params->use_ref,
                src0 ? (long long) src0->ne[0] : -1ll,
                src0 ? (long long) src0->ne[1] : -1ll,
                src1 ? (long long) src1->ne[0] : -1ll,
                src1 ? (long long) src1->ne[1] : -1ll,
                dst ? (long long) dst->ne[0] : -1ll,
                dst ? (long long) dst->ne[1] : -1ll,
                src1 ? (int) ggml_is_contiguous(src1) : -1,
                dst ? (int) ggml_is_contiguous(dst) : -1);
            fflush(stderr);
        }
    }
'''.replace('\\\\n', '\\n')


def install(root: Path) -> Path:
    source = root / 'ggml' / 'src' / 'ggml-cpu' / 'ggml-cpu.c'
    if not source.is_file():
        raise FileNotFoundError(f'Expected patched llama.cpp GGML source: {source}')
    text = source.read_text(encoding='utf-8')
    if TRACE_MARKER in text:
        raise ValueError('Diagnostics already installed; refusing double insertion')
    if text.count(HOOK_MARKER) != 1 or 'dag_try_ggml_gemv(' not in text:
        raise RuntimeError('DAG v0.6 hook not found; patch llama.cpp before running diagnostics')
    updated = text.replace(HOOK_MARKER, HOOK_MARKER + TRACE)
    source.write_text(updated, encoding='utf-8')
    return source


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-src', type=Path, required=True)
    args = p.parse_args()
    print(f'DAG hook diagnostics added to: {install(args.llama_src.resolve())}')


if __name__ == '__main__':
    main()
