#!/usr/bin/env python3
"""Trace Q4 FFN graph nodes before GGML CPU chooses a fused/repack kernel.

Install in an already patched, pinned llama.cpp checkout. Diagnostic only:
the instrumentation does not change dispatch or mathematics, and it is
enabled only when DAG_HOOK_DIAGNOSTIC and DAG_STREAM_DIR are both present.
"""
from __future__ import annotations

import argparse
from pathlib import Path

MARKER = '    // DAG_NATIVE_GRAPH_OP_TRACE'
ANCHOR = '''        // TODO: move fused-op detection into ggml_graph_plan so fusion decisions are made once at planning time
        // Try fused ops, fall back to normal compute
        const int n_fused = ggml_cpu_try_fuse_ops(cgraph, node_n, &params, cplan);'''
TRACE = r'''        // DAG_NATIVE_GRAPH_OP_TRACE
        // Captures nodes BEFORE CPU_REPACK, GLU or other fused-op dispatch.
        // Print a bounded sample of Q4, FFN-gate and GLU nodes.
#ifdef GGML_DAG_RUNTIME
        if (state->ith == 0 &&
            getenv("DAG_HOOK_DIAGNOSTIC") && getenv("DAG_STREAM_DIR")) {
            const struct ggml_tensor * dag_w = node->src[0];
            const struct ggml_tensor * dag_x = node->src[1];
            const bool dag_q4 = dag_w && (dag_w->type == GGML_TYPE_Q4_0 ||
                                         dag_w->type == GGML_TYPE_Q4_K);
            const bool dag_gate =
                (dag_w && strstr(dag_w->name, "ffn_gate") != NULL) ||
                (dag_x && strstr(dag_x->name, "ffn_gate") != NULL) ||
                strstr(node->name, "ffn_gate") != NULL;
            const bool dag_glu = node->op == GGML_OP_GLU;
            static atomic_int dag_q4_count = 0;
            static atomic_int dag_gate_count = 0;
            static atomic_int dag_glu_count = 0;
            bool dag_print = false;
            const char * dag_class = "other";
            if (dag_gate) {
                dag_class = "gate";
                dag_print = atomic_fetch_add(&dag_gate_count, 1) < 16;
            } else if (dag_q4) {
                dag_class = "q4";
                dag_print = atomic_fetch_add(&dag_q4_count, 1) < 16;
            } else if (dag_glu) {
                dag_class = "glu";
                dag_print = atomic_fetch_add(&dag_glu_count, 1) < 4;
            }
            if (dag_print) {
                fprintf(stderr,
                    "[DAG graph diag] class=%s op=%s node=%s "
                    "src0=%s src0_type=%s src1=%s src1_type=%s "
                    "weight=%lldx%lld input=%lldx%lld out=%lldx%lld "
                    "threads=%d\n",
                    dag_class, ggml_op_name(node->op), node->name,
                    dag_w ? dag_w->name : "<null>",
                    dag_w ? ggml_type_name(dag_w->type) : "<null>",
                    dag_x ? dag_x->name : "<null>",
                    dag_x ? ggml_type_name(dag_x->type) : "<null>",
                    dag_w ? (long long) dag_w->ne[0] : -1ll,
                    dag_w ? (long long) dag_w->ne[1] : -1ll,
                    dag_x ? (long long) dag_x->ne[0] : -1ll,
                    dag_x ? (long long) dag_x->ne[1] : -1ll,
                    (long long) node->ne[0], (long long) node->ne[1],
                    params.nth);
                fflush(stderr);
            }
        }
#endif
'''


def install(root: Path) -> Path:
    source = root / 'ggml/src/ggml-cpu/ggml-cpu.c'
    if not source.is_file():
        raise FileNotFoundError(source)
    text = source.read_text(encoding='utf-8')
    if MARKER in text:
        raise ValueError('Graph diagnostics already installed')
    if text.count('DAG_NATIVE_EARLY_MUL_MAT_DISPATCH') != 1:
        raise RuntimeError('Early DAG hook absent: apply relocate_hook.py first')
    if text.count(ANCHOR) != 1:
        raise RuntimeError('Pinned llama.cpp graph-execution anchor missing')
    updated = text.replace(ANCHOR, TRACE + '\n' + ANCHOR)
    assert updated.count(MARKER) == 1
    source.write_text(updated, encoding='utf-8')
    return source


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--llama-src', type=Path, required=True)
    args = p.parse_args()
    print(f'Installed bounded pre-dispatch graph trace in {install(args.llama_src.resolve())}')


if __name__ == '__main__':
    main()
