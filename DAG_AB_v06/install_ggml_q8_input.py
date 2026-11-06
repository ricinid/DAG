#!/usr/bin/env python3
"""Install the opt-in SINGLE-PASS GGML-Q8 numerical compatibility mode.

Use on an ALREADY PATCHED llama.cpp pinned by DAG_AB_v06, including trees
with the optional shadow diagnostics. It modifies only the successful early
DAG hook. The normal native-f32 and existing shadow path remain unchanged.

Runtime env DAG_INPUT_SEMANTICS=native-f32 (default) or ggml-q8.
For ggml-q8, CPU's own Q8 quantizer round-trips the contiguous F32 input,
then runs native DAG GEMV ONCE, without computing the GGML reference.
Rejects unknown modes, missing markers, and duplicate installation.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

MARKER = "// DAG_NATIVE_GGML_Q8_SINGLE_PASS"
EARLY = "// DAG_NATIVE_EARLY_MUL_MAT_DISPATCH"
HOOK = "// DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK"
EXTRA = "    // extra_buffer op?"
INCLUDE = '#include "ggml-quants.h"'

# The call, in its existing early hook. Any shadow instrumentation is inside
# this call's body and MUST remain attached to native-f32 mode only.
CALL = re.compile(
    r"(?m)^(?P<indent>[ \t]*)if[ \t]*\("
    r"dag_try_ggml_gemv\(src0->name,[ \t]*params->ith,[ \t]*"
    r"\(const float \*\)[ \t]*src1->data,[ \t]*"
    r"\(uint64_t\)[ \t]*src1->ne\[0\],[ \t]*"
    r"\(float \*\)[ \t]*dst->data,[ \t]*"
    r"\(uint64_t\)[ \t]*src0->ne\[1\]\)\)[ \t]*\{",
    re.MULTILINE,
)

SNIPPET = r'''// DAG_NATIVE_GGML_Q8_SINGLE_PASS
// The GGML-Q8 branch uses native DAG exactly once, not the shadow path.
// The normal original-F32 path (and its opt-in shadow instrumentation)
// remains attached to the else-if immediately below this block.
const char * dag_input_semantics = getenv("DAG_INPUT_SEMANTICS");
if (dag_input_semantics &&
    strcmp(dag_input_semantics, "native-f32") != 0 &&
    strcmp(dag_input_semantics, "ggml-q8") != 0) {
    fprintf(stderr, "[DAG q8 normal] invalid DAG_INPUT_SEMANTICS=%s\n",
            dag_input_semantics);
    abort();
}
if (dag_input_semantics &&
    strcmp(dag_input_semantics, "ggml-q8") == 0) {
    // Avoid quantizing every unsupported matrix when a name allowlist is set.
    // dag_try_ggml_gemv performs the authoritative scope validation again.
    int dag_name_selected = 1;
    const char * dag_allow = getenv("DAG_TENSOR_ALLOW");
    if (dag_allow && strcmp(dag_allow, "all") != 0 &&
        strcmp(dag_allow, "*") != 0) {
        dag_name_selected = 0;
        const char * item = dag_allow;
        while (*item) {
            const char * sep = strchr(item, ',');
            const size_t n = sep ? (size_t)(sep - item) : strlen(item);
            if (strlen(src0->name) == n && memcmp(src0->name, item, n) == 0) {
                dag_name_selected = 1;
                break;
            }
            if (!sep) break;
            item = sep + 1;
        }
    }
    if (dag_name_selected && getenv("DAG_STREAM_DIR")) {
        const enum ggml_type dag_qtype =
            src0->type == GGML_TYPE_Q4_K ? GGML_TYPE_Q8_K : GGML_TYPE_Q8_0;
        const int64_t dag_cols = src1->ne[0];
        const int64_t dag_block = ggml_blck_size(dag_qtype);
        if (dag_cols <= 0 || dag_cols % dag_block != 0 ||
            (uint64_t)dag_cols > (uint64_t)SIZE_MAX/sizeof(float)) {
            fprintf(stderr, "[DAG q8 normal] invalid quantized input dimensions\n");
            abort();
        }
        const size_t dag_qsize = ggml_row_size(dag_qtype, dag_cols);
        void * dag_qbuffer = malloc(dag_qsize);
        float * dag_x = (float *)malloc((size_t)dag_cols * sizeof(float));
        if (!dag_qbuffer || !dag_x) {
            fprintf(stderr, "[DAG q8 normal] OOM during input Q8 roundtrip\n");
            abort();
        }
        type_traits_cpu[dag_qtype].from_float(
            (const float *)src1->data, dag_qbuffer, dag_cols);
        if (dag_qtype == GGML_TYPE_Q8_K) {
            dequantize_row_q8_K(
                (const block_q8_K *)dag_qbuffer, dag_x, dag_cols);
        } else {
            dequantize_row_q8_0(
                (const block_q8_0 *)dag_qbuffer, dag_x, dag_cols);
        }
        const int dag_rc = dag_try_ggml_gemv(
            src0->name, params->ith, dag_x, (uint64_t)dag_cols,
            (float *)dst->data, (uint64_t)src0->ne[1]);
        free(dag_x);
        free(dag_qbuffer);
        if (dag_rc == 1) {
            static unsigned long long dag_q8_normal_calls = 0;
            ++dag_q8_normal_calls;
            if (dag_q8_normal_calls == 1 ||
                (dag_q8_normal_calls % 1000ULL) == 0) {
                fprintf(stderr,
                        "[DAG q8 normal] tensor=%s quant_type=%s call=%llu\n",
                        src0->name, ggml_type_name(dag_qtype),
                        dag_q8_normal_calls);
                fflush(stderr);
            }
            return;
        }
    }
} else'''


def install(root: Path) -> Path:
    source = root / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    if not source.is_file():
        raise FileNotFoundError(source)
    contents = source.read_text(encoding="utf-8")
    if MARKER in contents:
        raise ValueError("GGML-Q8 single-pass mode already installed")
    if contents.count(EARLY) != 1 or contents.count(HOOK) != 1:
        raise RuntimeError("Expected exactly one early DAG hook")
    if contents.count(EXTRA) != 1:
        raise RuntimeError("Expected one GGML CPU extra-buffer dispatch")
    begin, end = contents.index(EARLY), contents.index(EXTRA)
    if begin >= end:
        raise RuntimeError("DAG hook must run before GGML CPU-extra dispatch")
    candidates = [m for m in CALL.finditer(contents) if begin < m.start() < end]
    if len(candidates) != 1:
        raise RuntimeError("Expected exactly one original DAG(F32) call in early hook")
    m = candidates[0]
    indentation = m.group("indent")
    insertion = "\n".join(indentation + line if line else ""
                          for line in SNIPPET.splitlines())
    # Preserve the rest of the original DAG if body verbatim, including any
    # existing shadow instrumentation. Only change "if" -> "if" after "else".
    modified = contents[:m.start()] + insertion + "\n" + contents[m.start():]
    if INCLUDE not in modified:
        if modified.count('#include "quants.h"') != 1:
            raise RuntimeError("Pinned ggml-cpu quant header was not found")
        modified = modified.replace('#include "quants.h"',
                                    '#include "quants.h"\n' + INCLUDE, 1)
    if modified.count(MARKER) != 1 or modified.count(HOOK) != 1:
        raise AssertionError("Unexpected source mutation")
    source.write_text(modified, encoding="utf-8")
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-src", required=True, type=Path)
    args = parser.parse_args()
    print(f"Installed GGML-Q8 single-pass mode: {install(args.llama_src.resolve())}")


if __name__ == "__main__":
    main()
