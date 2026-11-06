#!/usr/bin/env python3
"""Extend an EXISTING DAG shadow hook with Q8-roundtrip input diagnosis.

On DAG_SHADOW_CHECK=1, existing diagnostic computes DAG(original F32) and
the ordinary GGML CPU reference for the identical tensor and activation.
This optional second comparison computes DAG(dequantize(quantize(F32))),
using the same GGML Q8_K / Q8_0 quantization routines as GGML CPU
type_traits. It then restores the first DAG(F32) output.

Adds no new default-mode runtime cost. Requires install_shadow_gemv.py
already applied. A second application is refused.
"""
from __future__ import annotations
import argparse
from pathlib import Path

MARKER = "// DAG_NATIVE_SHADOW_Q8_ROUNDTRIP"
EXISTING = "// DAG_NATIVE_SHADOW_REFERENCE_COMPARE"
ANCHOR = "            // Restore the native DAG values. No GGML reference result enters"
QUANTS_INCLUDE = '#include "ggml-quants.h"'

SNIPPET = r'''// DAG_NATIVE_SHADOW_Q8_ROUNDTRIP
// Diagnostic: compare DAG on the EXACT GGML Q8-roundtripped input to
// the GGML normal CPU result already stored in dst->data.
// Restore the original DAG(F32) result below before graph execution resumes.
const enum ggml_type dag_qtype =
    src0->type == GGML_TYPE_Q4_K ? GGML_TYPE_Q8_K : GGML_TYPE_Q8_0;
const int64_t dag_n_input = src1->ne[0];
const int64_t dag_qblock = ggml_blck_size(dag_qtype);
if (dag_n_input > 0 && dag_n_input % dag_qblock == 0 &&
    (uint64_t) dag_n_input <= (uint64_t) SIZE_MAX / sizeof(float)) {
    const size_t dag_qbytes = ggml_row_size(dag_qtype, dag_n_input);
    void * dag_qbytes_buffer = malloc(dag_qbytes);
    float * dag_qfloat = (float *) malloc((size_t) dag_n_input * sizeof(float));
    float * dag_reference_copy = (float *) malloc(dag_shadow_bytes);
    if (dag_qbytes_buffer && dag_qfloat && dag_reference_copy) {
        // Capture the reference BEFORE the second native DAG invocation.
        memcpy(dag_reference_copy, dst->data, dag_shadow_bytes);
        type_traits_cpu[dag_qtype].from_float(
            (const float *) src1->data, dag_qbytes_buffer, dag_n_input);
        if (dag_qtype == GGML_TYPE_Q8_K) {
            dequantize_row_q8_K(
                (const block_q8_K *) dag_qbytes_buffer, dag_qfloat, dag_n_input);
        } else {
            dequantize_row_q8_0(
                (const block_q8_0 *) dag_qbytes_buffer, dag_qfloat, dag_n_input);
        }
        const int dag_q8_rc = dag_try_ggml_gemv(
            src0->name, params->ith, dag_qfloat, (uint64_t) dag_n_input,
            (float *) dst->data, (uint64_t) dag_shadow_rows);
        if (dag_q8_rc == 1) {
            const float * dag_q8_out = (const float *) dst->data;
            double dag_q8_sq=0.0, dag_q8_max=0.0;
            size_t dag_q8_nonfinite=0, dag_q8_diff_bits=0;
            for (size_t dag_i=0; dag_i<dag_shadow_rows; ++dag_i) {
                if (memcmp(dag_q8_out+dag_i, dag_reference_copy+dag_i,
                           sizeof(float)) != 0) ++dag_q8_diff_bits;
                if (!isfinite(dag_q8_out[dag_i]) ||
                    !isfinite(dag_reference_copy[dag_i])) {
                    ++dag_q8_nonfinite;
                    continue;
                }
                const double delta = fabs(
                    (double)dag_q8_out[dag_i] - (double)dag_reference_copy[dag_i]);
                dag_q8_sq += delta * delta;
                if (delta > dag_q8_max) dag_q8_max=delta;
            }
            const double dag_q8_rmse = sqrt(dag_q8_sq / (double)dag_shadow_rows);
            fprintf(stderr,
                "[DAG shadow q8] tensor=%s call=%llu quant_type=%s"
                " rmse_f32=%.9g rmse_q8=%.9g max_abs_q8=%.9g"
                " reduction=%.9g differing_f32=%zu nonfinite=%zu\n",
                src0->name, dag_shadow_call, ggml_type_name(dag_qtype),
                dag_rmse, dag_q8_rmse, dag_q8_max,
                dag_rmse > 0.0 ? dag_rmse / dag_q8_rmse : 0.0,
                dag_q8_diff_bits, dag_q8_nonfinite);
            fflush(stderr);
        } else {
            fprintf(stderr,
                "[DAG shadow q8] call=%llu native DAG Q8 probe returned %d\n",
                dag_shadow_call, dag_q8_rc);
            fflush(stderr);
        }
    } else {
        fprintf(stderr,
            "[DAG shadow q8] call=%llu allocation failed\n", dag_shadow_call);
        fflush(stderr);
    }
    free(dag_reference_copy);
    free(dag_qfloat);
    free(dag_qbytes_buffer);
}
'''


def install(root: Path) -> Path:
    path = root / 'ggml' / 'src' / 'ggml-cpu' / 'ggml-cpu.c'
    if not path.is_file():
        raise FileNotFoundError(path)
    src=path.read_text(encoding='utf-8')
    if MARKER in src:
        raise ValueError("Q8 shadow extension already installed")
    if src.count(EXISTING)!=1 or src.count(ANCHOR)!=1:
        raise RuntimeError("Requires exactly one installed original shadow comparison")
    if src.index(ANCHOR) < src.index(EXISTING):
        raise RuntimeError("Original shadow block misplaced")
    if src.count('#include "quants.h"')!=1:
        raise RuntimeError("Pinned ggml-cpu quants.h include not found")
    # The public GGML quantization/dequantization interface is in ggml-quants.h.
    if QUANTS_INCLUDE not in src:
        src=src.replace('#include "quants.h"',
                        '#include "quants.h"\n'+QUANTS_INCLUDE,1)
    indented='\n'.join(('            '+line if line else '')
                      for line in SNIPPET.splitlines())
    src=src.replace(ANCHOR, indented+'\n'+ANCHOR,1)
    if src.count(MARKER)!=1:
        raise AssertionError("Expected one new shadow Q8 block")
    path.write_text(src,encoding='utf-8')
    return path


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--llama-src',type=Path,required=True)
    args=ap.parse_args()
    print(f'Extended shadow comparison with GGML Q8 roundtrip: {install(args.llama_src.resolve())}')


if __name__=='__main__':
    main()
