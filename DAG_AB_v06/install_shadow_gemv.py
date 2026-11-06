#!/usr/bin/env python3
"""Install a bounded, opt-in DAG-vs-GGML shadow comparison in patched ggml-cpu.c.

Requires existing EARLY DAG GEMV hook, pinned llama.cpp tree. The shadow path:
  1. saves the completed native DAG output;
  2. runs the ordinary GGML extra-backend dispatch (and CPU fallback);
  3. compares GGML result and DAG result for the SAME src0/src1;
  4. restores DAG output before the next graph op.

Enable via DAG_SHADOW_CHECK=1, optionally DAG_SHADOW_MAX=12.
This is a deliberately slow correctness experiment, not a benchmark.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

EARLY = "// DAG_NATIVE_EARLY_MUL_MAT_DISPATCH"
HOOK = "// DAG_NATIVE_EXPERIMENTAL_GEMV_HOOK"
EXTRA = "    // extra_buffer op?"
SENTINEL = "// DAG_NATIVE_SHADOW_REFERENCE_COMPARE"

# Match only the handled DAG call with an immediate return, not any other
# GGML return path. Retain indentation and the original arguments verbatim.
CALL = re.compile(
    r"(?P<indent>^[ \t]*)if[ \t]*\(dag_try_ggml_gemv\("
    r"(?P<args>.{0,480}?)\)\)[ \t]*\{[ \t]*\n"
    r"(?P<rindent>[ \t]+)return;[ \t]*\n"
    r"(?P=indent)\}",
    re.MULTILINE | re.DOTALL,
)

# This snippet is inserted ONLY inside the existing successful DAG dispatch.
# The standard GGML extra-backend predispatch is the exact branch which would
# have run without the DAG hook. ggml_compute_forward_mul_mat is the fallback.
SHADOW = r'''// DAG_NATIVE_SHADOW_REFERENCE_COMPARE
// The hook already verified nth==1, F32 contiguous destination and GEMV.
const char * dag_shadow_env = getenv("DAG_SHADOW_CHECK");
if (dag_shadow_env && strcmp(dag_shadow_env, "1") == 0) {
    static unsigned long long dag_shadow_calls = 0;
    const unsigned long long dag_shadow_call = ++dag_shadow_calls;
    const char * dag_shadow_max_env = getenv("DAG_SHADOW_MAX");
    const unsigned long long dag_shadow_max =
        dag_shadow_max_env ? strtoull(dag_shadow_max_env, NULL, 10) : 12ULL;
    if (dag_shadow_call <= dag_shadow_max && src0->ne[1] > 0 &&
        (uint64_t) src0->ne[1] <= (uint64_t) SIZE_MAX / sizeof(float)) {
        const size_t dag_shadow_rows = (size_t) src0->ne[1];
        const size_t dag_shadow_bytes = dag_shadow_rows * sizeof(float);
        float * dag_saved_output = (float *) malloc(dag_shadow_bytes);
        if (!dag_saved_output) {
            fprintf(stderr, "[DAG shadow] allocation failed for call=%llu\n",
                    dag_shadow_call);
        } else {
            memcpy(dag_saved_output, dst->data, dag_shadow_bytes);

            // Reference is the actual normal CPU dispatch for this op.
            // The DAG hook is not called recursively: we call the ordinary
            // backend dispatcher directly, then its original CPU fallback.
            const bool dag_ref_used_extra = ggml_cpu_extra_compute_forward(params, tensor);
            if (!dag_ref_used_extra) {
                ggml_compute_forward_mul_mat(params, tensor);
            }
            const float * dag_ref = (const float *) dst->data;

            double dag_abs_max = 0.0;
            double dag_squared_error = 0.0;
            double dag_max_ref_mag = 0.0;
            size_t dag_worst_row = 0;
            size_t dag_different_bits = 0;
            size_t dag_nonfinite = 0;
            for (size_t dag_row = 0; dag_row < dag_shadow_rows; ++dag_row) {
                const double a = dag_saved_output[dag_row];
                const double b = dag_ref[dag_row];
                if (memcmp(dag_saved_output + dag_row,
                           dag_ref + dag_row, sizeof(float)) != 0) {
                    ++dag_different_bits;
                }
                if (!isfinite(a) || !isfinite(b)) {
                    ++dag_nonfinite;
                    continue;
                }
                const double delta = fabs(a - b);
                if (delta > dag_abs_max) {
                    dag_abs_max = delta;
                    dag_worst_row = dag_row;
                }
                dag_squared_error += delta * delta;
                if (fabs(b) > dag_max_ref_mag) dag_max_ref_mag = fabs(b);
            }
            const double dag_rmse = sqrt(dag_squared_error / (double) dag_shadow_rows);
            fprintf(stderr,
                    "[DAG shadow] tensor=%s call=%llu rows=%zu reference=%s"
                    " max_abs=%.9g rmse=%.9g max_abs_over_max_ref=%.9g"
                    " differing_f32=%zu nonfinite=%zu worst_row=%zu"
                    " dag_value=%.9g ggml_value=%.9g\n",
                    src0->name, dag_shadow_call, dag_shadow_rows,
                    dag_ref_used_extra ? "cpu-extra" : "cpu-mul-mat",
                    dag_abs_max, dag_rmse,
                    dag_abs_max / (dag_max_ref_mag > 1e-30 ? dag_max_ref_mag : 1.0),
                    dag_different_bits, dag_nonfinite, dag_worst_row,
                    (double) dag_saved_output[dag_worst_row],
                    (double) dag_ref[dag_worst_row]);
            fflush(stderr);

            // Restore the native DAG values. No GGML reference result enters
            // the remainder of the model. Even if reference overwrites dst,
            // generation continues along the same DAG inference path.
            memcpy(dst->data, dag_saved_output, dag_shadow_bytes);
            free(dag_saved_output);
        }
    }
}'''


def install(llama_src: Path) -> Path:
    src = llama_src / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    if not src.is_file():
        raise FileNotFoundError(src)
    text = src.read_text(encoding="utf-8")
    if SENTINEL in text:
        raise ValueError("DAG shadow compare already installed")
    if text.count(EARLY) != 1 or text.count(HOOK) != 1:
        raise RuntimeError("Requires exactly one installed early DAG GEMV hook")
    if text.count(EXTRA) != 1:
        raise RuntimeError("Cannot identify the GGML extra-buffer dispatch")
    early = text.index(EARLY)
    dispatch = text.index(EXTRA)
    if not early < dispatch:
        raise RuntimeError("DAG hook not before CPU extra-buffer dispatch")
    matches = list(CALL.finditer(text))
    matches = [m for m in matches if early < m.start() < dispatch]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one handled DAG GEMV call; source unchanged")

    m = matches[0]
    indent = m.group("indent")
    inner_indent = m.group("rindent")
    snippet = "\n".join((inner_indent + line if line else "")
                        for line in SHADOW.splitlines())
    original_call = text[m.start():m.end()]
    new_call = original_call.replace(inner_indent + "return;", snippet + "\n" + inner_indent + "return;", 1)
    if new_call == original_call or new_call.count(SENTINEL) != 1:
        raise AssertionError("Could not inject bounded DAG shadow comparison")
    updated = text[:m.start()] + new_call + text[m.end():]
    if updated.count(SENTINEL) != 1 or updated.count(HOOK) != 1:
        raise AssertionError("Unexpected source mutation")
    src.write_text(updated, encoding="utf-8")
    return src


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--llama-src", type=Path, required=True)
    args = p.parse_args()
    print(f"Installed DAG-vs-GGML shadow comparison: {install(args.llama_src.resolve())}")


if __name__ == "__main__":
    main()
