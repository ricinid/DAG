#ifndef DAG_RUNTIME_H
#define DAG_RUNTIME_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* Context methods return 1 on full successful DAG matvec, 0 if tensor is not present,
 * -1 for an error (no partial results are committed). Requires contiguous F32 x,y.
 * Only GEMV, batch=1, on CPU; caller must retain original GGUF for fallback.
 */
void * dag_context_open(const char * compiled_dir, uint64_t cache_budget_bytes);
void dag_context_close(void * context);
int dag_context_gemv(void * context, const char * tensor_name, const float * x,
                     uint64_t n, float * y, uint64_t m);
/* Caller-supplied text buffer; returns length excluding NUL, or -1 on error. */
int dag_context_stats(void * context, char * dst, size_t capacity);
/* GGML hook. 1=handled; 0=fall through. A matched execution error
 * terminates process rather than pretending the DAG A/B succeeded.
 * Must be called only when GGML thread count is exactly one. */
int dag_try_ggml_gemv(const char * tensor_name, int ith, const float * x,
                      uint64_t n, float * y, uint64_t m);
#ifdef __cplusplus
}
#endif
#endif