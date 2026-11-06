#!/usr/bin/env python3
"""Opt-in routing of an existing Q8-roundtrip shadow result into the GGML graph.

This is a CORRECTNESS DIAGNOSTIC, not a performance feature. After the
original shadow path has computed and compared:
    DAG(F32) vs GGML CPU-extra and DAG(Q8->F32) vs GGML CPU-extra,
DAG_SHADOW_Q8_OUTPUT=1 selects the latter output for downstream ops.
DAG_SHADOW_CHECK=1 must also be set; without it nothing is routed.

Requires install_shadow_gemv.py and extend_shadow_q8.py already installed.
Refuses a double install or a source layout it cannot recognize.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROUTE = "// DAG_NATIVE_SHADOW_Q8_OUTPUT_ROUTE"
SHADOW = "// DAG_NATIVE_SHADOW_REFERENCE_COMPARE"
Q8 = "// DAG_NATIVE_SHADOW_Q8_ROUNDTRIP"

RULES = [
    (
        r"(?m)^(?P<indent>[ \t]*)// DAG_NATIVE_SHADOW_Q8_ROUNDTRIP[^\r\n]*",
        "after",
        '''// DAG_NATIVE_SHADOW_Q8_OUTPUT_ROUTE
const char * dag_route_env = getenv("DAG_SHADOW_Q8_OUTPUT");
const int dag_route_requested = dag_route_env && strcmp(dag_route_env, "1") == 0;
int dag_route_applied = 0;''',
    ),
    (
        r"(?m)^(?P<indent>[ \t]*)const float \* dag_q8_out = \(const float \*\) dst->data;[^\r\n]*",
        "after",
        '''if (dag_route_requested) {
    memcpy(dag_saved_output, dag_q8_out, dag_shadow_bytes);
    dag_route_applied = 1;
    fprintf(stderr, "[DAG shadow q8 route] tensor=%s call=%llu q8_output=1\\n",
            src0->name, dag_shadow_call);
    fflush(stderr);
}''',
    ),
    (
        r"(?m)^(?P<indent>[ \t]*)// Restore the native DAG values\. No GGML reference result enters[^\r\n]*",
        "before",
        '''if (dag_route_requested && !dag_route_applied) {
    fprintf(stderr, "[DAG shadow q8 route] FAIL: Q8 output not available at call=%llu\\n",
            dag_shadow_call);
    fflush(stderr);
    abort();
}''',
    ),
]


def install(root: Path) -> Path:
    path = root / "ggml" / "src" / "ggml-cpu" / "ggml-cpu.c"
    if not path.is_file():
        raise FileNotFoundError(path)
    source = path.read_text(encoding="utf-8")
    if ROUTE in source:
        raise ValueError("Shadow Q8 output routing already installed")
    if source.count(Q8) != 1 or source.count(SHADOW) != 1:
        raise RuntimeError("Requires installed Q8-roundtrip and reference shadow diagnostics")

    edits = []
    for pattern, where, snippet in RULES:
        matches = list(re.finditer(pattern, source))
        if len(matches) != 1:
            raise RuntimeError("Expected exactly one Q8 shadow routing anchor; source unchanged")
        match = matches[0]
        indent = match.group("indent")
        indented = "\n".join(indent + line if line else "" for line in snippet.splitlines())
        offset = match.end() if where == "after" else match.start()
        addition = "\n" + indented if where == "after" else indented + "\n"
        edits.append((offset, addition))

    if not (edits[0][0] < edits[1][0] < edits[2][0]):
        raise RuntimeError("Q8 shadow anchor order is unexpected; source unchanged")

    for offset, addition in reversed(edits):
        source = source[:offset] + addition + source[offset:]
    if source.count(ROUTE) != 1:
        raise AssertionError("Q8 shadow route inserted incorrectly")
    path.write_text(source, encoding="utf-8")
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llama-src", required=True, type=Path)
    args = parser.parse_args()
    print(f"Installed opt-in Q8 shadow output routing: {install(args.llama_src.resolve())}")


if __name__ == "__main__":
    main()
