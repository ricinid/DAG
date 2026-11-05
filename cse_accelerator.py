"""Numba accelerated greedy INT4 CSE; exact match to Python reference."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True

if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _gcd64(a, b):
        a = abs(a)
        b = abs(b)
        while b:
            a, b = b, a % b
        return a

    @njit(cache=True)
    def _naf_adds(c):
        n = abs(c)
        digits = 0
        while n:
            if n & 1:
                digit = 2 - (n & 3)
                n -= digit
                digits += 1
            n >>= 1
        return max(0, digits - 1)

    @njit(cache=True)
    def _extract_core(w, max_nodes, pair_window, min_gain):
        m, n = w.shape
        coeffs = np.zeros((m, n + max_nodes), dtype=np.int64)
        coeffs[:, :n] = w
        nodes = np.zeros((max_nodes, 4), dtype=np.int64)
        active = np.empty(n + max_nodes, dtype=np.int64)
        capacity = m * (n + max_nodes) * pair_window
        table_size = 1
        while table_size < max(16, capacity * 2):
            table_size <<= 1
        mask = table_size - 1
        slots = np.empty(table_size, dtype=np.int64)
        stamps = np.zeros(table_size, dtype=np.int64)
        pa = np.empty(capacity, dtype=np.int64)
        pb = np.empty(capacity, dtype=np.int64)
        pca = np.empty(capacity, dtype=np.int64)
        pcb = np.empty(capacity, dtype=np.int64)
        count = np.empty(capacity, dtype=np.int64)
        saving_terms = np.empty(capacity, dtype=np.int64)
        first_occurrence = np.empty(capacity, dtype=np.int64)
        next_occurrence = np.empty(capacity, dtype=np.int64)
        occurrence_row = np.empty(capacity, dtype=np.int64)
        occurrence_scale = np.empty(capacity, dtype=np.int64)
        used_nodes = 0
        for _ in range(max_nodes):
            stamp = used_nodes + 1
            n_patterns = 0
            n_occurrences = 0
            for r in range(m):
                width = 0
                for j in range(n + used_nodes):
                    if coeffs[r, j] != 0:
                        active[width] = j
                        width += 1
                for pos in range(width):
                    a = active[pos]
                    ca = coeffs[r, a]
                    end = min(width, pos + 1 + pair_window)
                    for other in range(pos + 1, end):
                        b = active[other]
                        cb = coeffs[r, b]
                        g = _gcd64(ca, cb)
                        scale = g if ca > 0 else -g
                        norm_ca = ca // scale
                        norm_cb = cb // scale
                        h = (a * 1315423911 ^ b * 2654435761 ^
                             (norm_ca + 16) * 97531 ^
                             (norm_cb + 16) * 16777619) & mask
                        while stamps[h] == stamp:
                            old = slots[h]
                            if (pa[old] == a and pb[old] == b and
                                pca[old] == norm_ca and pcb[old] == norm_cb):
                                break
                            h = (h + 1) & mask
                        if stamps[h] != stamp:
                            idx = n_patterns
                            stamps[h] = stamp
                            slots[h] = idx
                            pa[idx] = a
                            pb[idx] = b
                            pca[idx] = norm_ca
                            pcb[idx] = norm_cb
                            count[idx] = 0
                            saving_terms[idx] = 0
                            first_occurrence[idx] = -1
                            n_patterns += 1
                        else:
                            idx = slots[h]
                        count[idx] += 1
                        saving_terms[idx] += (1 + _naf_adds(ca) + _naf_adds(cb)
                                              - _naf_adds(scale))
                        occurrence_row[n_occurrences] = r
                        occurrence_scale[n_occurrences] = scale
                        next_occurrence[n_occurrences] = first_occurrence[idx]
                        first_occurrence[idx] = n_occurrences
                        n_occurrences += 1
            best_idx = -1
            best_gain = 0
            for k in range(n_patterns):
                if count[k] < 2:
                    continue
                gain = (saving_terms[k] - 1 - _naf_adds(pca[k])
                        - _naf_adds(pcb[k]))
                if gain > best_gain:
                    best_gain = gain
                    best_idx = k
            if best_idx < 0 or best_gain < min_gain:
                break
            a = pa[best_idx]
            b = pb[best_idx]
            ca = pca[best_idx]
            cb = pcb[best_idx]
            nodes[used_nodes, 0] = a
            nodes[used_nodes, 1] = b
            nodes[used_nodes, 2] = ca
            nodes[used_nodes, 3] = cb
            new_var = n + used_nodes
            # Only replace pair-window occurrences. Replacing all proportional
            # pairs would alter the reference DAG's greedy selection semantics.
            occ = first_occurrence[best_idx]
            while occ >= 0:
                r = occurrence_row[occ]
                scale = occurrence_scale[occ]
                coeffs[r, a] = 0
                coeffs[r, b] = 0
                coeffs[r, new_var] = scale
                occ = next_occurrence[occ]
            used_nodes += 1
        return nodes[:used_nodes], coeffs[:, :n + used_nodes]


def extract_numba(w: np.ndarray, max_nodes: int, pair_window: int, min_gain: int):
    """Return node tuples and final coefficient matrix, same semantics as Python CSE."""
    if not NUMBA_AVAILABLE:
        raise RuntimeError('Numba is unavailable; pip install numba')
    return _extract_core(np.ascontiguousarray(w, dtype=np.int64),
                         max_nodes, pair_window, min_gain)
