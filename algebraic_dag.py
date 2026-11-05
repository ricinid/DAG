#!/usr/bin/env python3
"""Immutable-weight, exact-integer Algebraic DAG Compiler (research prototype).

Pipeline: INT4 matrix -> tiled greedy shared-expression extraction -> global DAG
          -> dependency levels -> CPU scalar / vectorized NumPy / optional Torch
          -> native scalar C source or portable .npz IR.

Input coefficients must be INTEGER -8..7; inputs and intermediate values must
fit signed int64. This is NOT a bit-exact BF16/FP32 or GPTQ/AWQ compiler.
The optimizer preserves mathematical integers, not the rounding behavior of a
floating-point network. No silent quantization is performed.

Examples:
    python algebraic_dag.py demo --case both
    python algebraic_dag.py compile --matrix weights_int4.npy --out model.dag.npz
    python algebraic_dag.py verify --dag model.dag.npz --matrix weights_int4.npy
    python algebraic_dag.py compile --matrix weights_int4.npy --emit-c module.c
    python algebraic_dag.py compile --safetensors model.safetensors --tensor model.layers.0.mlp.gate_proj.weight --rows 32 --cols 64

Compatible with the snapshot compiler idea; self-contained (numpy required).
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from math import gcd
from pathlib import Path
from time import perf_counter
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass(frozen=True)
class Node:
    """t_i = ca * value[a] + cb * value[b], a,b < node's global variable ID."""
    a: int
    b: int
    ca: int
    cb: int


@dataclass
class LocalDAG:
    inputs: int
    nodes: List[Node]
    outputs: List[Dict[int, int]]


def _naf(x: int) -> List[Tuple[int, int]]:
    n = abs(int(x)); result = []; bit = 0
    while n:
        if n & 1:
            digit = 2 - (n & 3)
            result.append((bit, -digit if x < 0 else digit))
            n -= digit
        n >>= 1
        bit += 1
    return result


def _adds_for_coeff(c: int) -> int:
    return max(0, len(_naf(c)) - 1)


def _canonical_pair(a: int, b: int, ca: int, cb: int):
    g = gcd(abs(ca), abs(cb))
    scale = g if ca > 0 else -g
    return (a, b, ca // scale, cb // scale), scale


def extract_shared_expressions(w: np.ndarray, max_nodes: int = 16,
                               pair_window: int = 4, min_gain: int = 1) -> LocalDAG:
    """Greedy multioutput pair CSE, before shift-add lowering. Exact for Z.

    Bounded pair window / max nodes deliberately make this a heuristic, not an
    optimizer claiming global minimality. Matrix values must already be INT4.
    """
    if w.ndim != 2 or not np.issubdtype(w.dtype, np.integer):
        raise TypeError('W must be a rank-2 integer matrix')
    if w.size and (w.min() < -8 or w.max() > 7):
        raise ValueError('Only signed INT4 coefficients [-8, 7] are supported')
    m, n = w.shape
    rows = [{j: int(c) for j, c in enumerate(w[i]) if c} for i in range(m)]
    nodes: List[Node] = []
    for _ in range(max_nodes):
        uses: Dict[Tuple[int,int,int,int], List[Tuple[int,int]]] = defaultdict(list)
        for ri, row in enumerate(rows):
            keys = sorted(row)
            for pos, a in enumerate(keys):
                for b in keys[pos + 1:pos + 1 + pair_window]:
                    pat, scale = _canonical_pair(a, b, row[a], row[b])
                    uses[pat].append((ri, scale))
        best_pat = None
        best_uses = None
        best_gain = 0
        for pat, occurrences in uses.items():
            if len(occurrences) < 2:
                continue
            _, _, ca, cb = pat
            original = sum(1 + _adds_for_coeff(ca * s) + _adds_for_coeff(cb * s)
                           for _, s in occurrences)
            proposed = (1 + _adds_for_coeff(ca) + _adds_for_coeff(cb) +
                        sum(_adds_for_coeff(s) for _, s in occurrences))
            saving = original - proposed
            if saving > best_gain:
                best_pat, best_uses, best_gain = pat, occurrences, saving
        if best_pat is None or best_gain < min_gain:
            break
        a, b, ca, cb = best_pat
        new_var = n + len(nodes)
        nodes.append(Node(a, b, ca, cb))
        for ri, scale in best_uses:
            row = rows[ri]
            assert row[a] == ca * scale and row[b] == cb * scale
            del row[a]; del row[b]
            row[new_var] = scale
    return LocalDAG(n, nodes, rows)


@dataclass
class AlgebraicDAG:
    n_inputs: int
    nodes: List[Node]
    outputs: List[Dict[int, int]]

    @property
    def n_outputs(self):
        return len(self.outputs)

    def validate(self) -> None:
        assert self.n_inputs >= 0
        for i, node in enumerate(self.nodes):
            assert 0 <= node.a < self.n_inputs+i
            assert 0 <= node.b < self.n_inputs+i
            assert node.a != node.b
            assert node.ca and node.cb
        for row in self.outputs:
            for k, c in row.items():
                assert 0 <= k < self.n_inputs + len(self.nodes) and c != 0

    def levels(self) -> List[List[int]]:
        """Independent DAG node IDs grouped by topological depth (inputs level 0)."""
        depth = [0] * self.n_inputs
        groups: Dict[int, List[int]] = defaultdict(list)
        for i, node in enumerate(self.nodes):
            d = 1 + max(depth[node.a], depth[node.b])
            depth.append(d)
            groups[d].append(self.n_inputs + i)
        return [groups[k] for k in sorted(groups)]

    def sparse_layers(self) -> List[dict]:
        """Equivalent layer-wise sparse linear operators over the live DAG state.

        Each stage computes only new variables, and reads from the accumulated
        state (all inputs + earlier nodes). No dense original W is reconstructed.
        """
        layers = []
        for depth, ids in enumerate(self.levels(), 1):
            entries = []
            for row_id, global_id in enumerate(ids):
                node = self.nodes[global_id-self.n_inputs]
                entries.extend([(row_id,node.a,node.ca), (row_id,node.b,node.cb)])
            layers.append({'depth':depth, 'outputs':list(ids),
                           'input_state_size':max((max(self.nodes[v-self.n_inputs].a, self.nodes[v-self.n_inputs].b) + 1 for v in ids), default=self.n_inputs),
                           'shape':[len(ids),self.n_inputs+len(self.nodes)],
                           'coo':entries})
        r,c,v=self.output_coo()
        layers.append({'depth':'outputs', 'outputs':list(range(self.n_outputs)),
                       'input_state_size':self.n_inputs+len(self.nodes),
                       'shape':[self.n_outputs,self.n_inputs+len(self.nodes)],
                       'coo':list(zip(r.tolist(),c.tolist(),v.tolist()))})
        return layers

    def emit_dot(self, path: str, max_nodes: int = 128) -> None:
        """Graphviz dot preview of first max_nodes shared expression nodes."""
        lines=['digraph AlgebraicDAG {','  rankdir=LR;','  node [shape=box, fontsize=10];']
        shown=min(len(self.nodes),max_nodes)
        used_inputs=set()
        for i, node in enumerate(self.nodes[:shown]):
            dst=f't{i}'
            lines.append(f'  {dst} [label="t{i} = {node.ca}a + {node.cb}b",style=filled,fillcolor=lightcyan];')
            for v in (node.a,node.b):
                src=f'x{v}' if v < self.n_inputs else f't{v-self.n_inputs}'
                if v < self.n_inputs: used_inputs.add(v)
                lines.append(f'  {src} -> {dst};')
        for i in sorted(used_inputs):
            lines.append(f'  x{i} [shape=oval];')
        lines.append('}')
        Path(path).write_text('\n'.join(lines)+'\n',encoding='utf-8')

    def output_coo(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        rr, cc, vv = [], [], []
        for i, row in enumerate(self.outputs):
            for j, val in sorted(row.items()):
                rr.append(i); cc.append(j); vv.append(val)
        return (np.asarray(rr, dtype=np.int64), np.asarray(cc, dtype=np.int64),
                np.asarray(vv, dtype=np.int64))

    def summary(self) -> dict:
        depths = self.levels()
        term_count = sum(len(row) for row in self.outputs)
        adds = sum(1 + _adds_for_coeff(n.ca) + _adds_for_coeff(n.cb) for n in self.nodes)
        adds += sum(max(0, len(r)-1) + sum(_adds_for_coeff(c) for c in r.values())
                    for r in self.outputs)
        # Upper-bound style plain-array estimate, not actual entropy coding.
        ir_bytes = 32 * len(self.nodes) + 24 * term_count + 8 * (len(self.outputs)+1)
        return {'inputs': self.n_inputs, 'outputs': self.n_outputs,
                'shared_nodes': len(self.nodes), 'dag_levels': len(depths),
                'max_parallel_nodes': max(map(len, depths), default=0),
                'output_terms': term_count, 'est_shift_add_additions': adds,
                'plain_ir_estimated_bytes': ir_bytes,
                'packed_int4_reference_bytes': (self.n_inputs*self.n_outputs+1)//2}

    def _get_x(self, x: np.ndarray) -> Tuple[np.ndarray, bool]:
        x = np.asarray(x)
        if x.ndim == 1:
            x = x.reshape(1, -1)
            squeeze = True
        elif x.ndim == 2:
            squeeze = False
        else:
            raise ValueError('x must be shape [inputs] or [batch, inputs]')
        if x.shape[1] != self.n_inputs:
            raise ValueError(f'expected {self.n_inputs} input components, got {x.shape[1]}')
        if not np.issubdtype(x.dtype, np.integer):
            raise TypeError('exact integer executor only; float inputs are not supported')
        return x.astype(np.int64, copy=False), squeeze

    def eval_scalar(self, x: np.ndarray) -> np.ndarray:
        x, squeeze = self._get_x(x)
        out = np.empty((x.shape[0], self.n_outputs), dtype=np.int64)
        for batch_idx in range(x.shape[0]):
            values = list(map(int, x[batch_idx]))
            for node in self.nodes:
                values.append(node.ca * values[node.a] + node.cb * values[node.b])
            for ri, row in enumerate(self.outputs):
                out[batch_idx, ri] = sum(c * values[v] for v,c in row.items())
        return out[0] if squeeze else out

    def eval_numpy(self, x: np.ndarray) -> np.ndarray:
        """Levelized vector executor, uses NumPy array ops, not dense W."""
        x, squeeze = self._get_x(x)
        batch = x.shape[0]
        state = np.zeros((self.n_inputs+len(self.nodes), batch), dtype=np.int64)
        state[:self.n_inputs] = x.T
        for lvl in self.levels():
            node_indices = np.asarray(lvl, dtype=np.int64) - self.n_inputs
            a = np.fromiter((self.nodes[k].a for k in node_indices), np.int64)
            b = np.fromiter((self.nodes[k].b for k in node_indices), np.int64)
            ca = np.fromiter((self.nodes[k].ca for k in node_indices), np.int64)
            cb = np.fromiter((self.nodes[k].cb for k in node_indices), np.int64)
            state[np.asarray(lvl, dtype=np.int64)] = (state[a] * ca[:,None] +
                                                        state[b] * cb[:,None])
        rr, cc, coeff = self.output_coo()
        out = np.zeros((self.n_outputs, batch), dtype=np.int64)
        if len(rr):
            np.add.at(out, rr, state[cc] * coeff[:,None])
        return out.T[0] if squeeze else out.T

    def eval_torch(self, x, device: Optional[str] = None):
        """Optional Torch levelized tensor executor; works on CPU or CUDA.

        This is a FUNCTIONAL GPU baseline, NOT a fused Triton/CUDA kernel.
        Multi-level graph may incur many launches and slow GPU performance.
        """
        import torch
        x = torch.as_tensor(x, dtype=torch.int64, device=device)
        squeeze = x.ndim == 1
        if squeeze:
            x = x.unsqueeze(0)
        if x.ndim != 2 or x.shape[1] != self.n_inputs:
            raise ValueError('incorrect Torch input shape')
        state = torch.zeros((self.n_inputs+len(self.nodes), x.shape[0]),
                            dtype=torch.int64, device=x.device)
        state[:self.n_inputs] = x.T
        for lvl in self.levels():
            indices = [v-self.n_inputs for v in lvl]
            ids = torch.tensor(lvl, dtype=torch.int64, device=x.device)
            a = torch.tensor([self.nodes[i].a for i in indices], dtype=torch.int64, device=x.device)
            b = torch.tensor([self.nodes[i].b for i in indices], dtype=torch.int64, device=x.device)
            ca = torch.tensor([self.nodes[i].ca for i in indices], dtype=torch.int64, device=x.device)[:,None]
            cb = torch.tensor([self.nodes[i].cb for i in indices], dtype=torch.int64, device=x.device)[:,None]
            state[ids] = state[a]*ca + state[b]*cb
        rr, cc, c = self.output_coo()
        y = torch.zeros((self.n_outputs, x.shape[0]), dtype=torch.int64, device=x.device)
        if len(rr):
            rows = torch.as_tensor(rr, dtype=torch.int64, device=x.device)
            cols = torch.as_tensor(cc, dtype=torch.int64, device=x.device)
            vals = torch.as_tensor(c, dtype=torch.int64, device=x.device)[:,None]
            y.index_add_(0, rows, state[cols]*vals)
        return y.T[0] if squeeze else y.T

    def save(self, path: str) -> None:
        self.validate()
        no = np.asarray([[n.a,n.b,n.ca,n.cb] for n in self.nodes], dtype=np.int64).reshape(-1,4)
        rr, cc, vv = self.output_coo()
        np.savez_compressed(path, schema=np.array(1, dtype=np.int32),
                            n_inputs=np.array(self.n_inputs,dtype=np.int64),
                            n_outputs=np.array(self.n_outputs,dtype=np.int64),
                            nodes=no, out_rows=rr, out_vars=cc, out_coeff=vv)

    @classmethod
    def load(cls, path: str) -> 'AlgebraicDAG':
        with np.load(path, allow_pickle=False) as d:
            if int(d['schema']) != 1:
                raise ValueError('unsupported DAG format')
            n = int(d['n_inputs']); m = int(d['n_outputs'])
            nodes = [Node(*map(int,r)) for r in d['nodes']]
            rows = [dict() for _ in range(m)]
            for i, v, c in zip(d['out_rows'], d['out_vars'], d['out_coeff']):
                rows[int(i)][int(v)] = int(c)
        result = cls(n, nodes, rows)
        result.validate()
        return result

    def emit_c(self, path: str, fn_name: str = 'dag_kernel') -> None:
        """Generate a *fully unrolled* C99 scalar kernel with fixed constants.
        Use a native optimizing compiler to lower constant multiplies. Do not
        interpret large source size as evidence of better runtime performance.
        """
        if not fn_name.isidentifier():
            raise ValueError('invalid C identifier')
        def ref(var):
            return f'x[{var}]' if var < self.n_inputs else f't{var-self.n_inputs}'
        def mul(c, expr):
            if c == 1: return expr
            if c == -1: return f'(-{expr})'
            return f'({c}LL*{expr})'
        lines = ['#include <stdint.h>',
                 '/* Exact only while all intermediate results fit int64_t. */',
                 f'void {fn_name}(const int64_t* x, int64_t* y) {{']
        for ni, node in enumerate(self.nodes):
            lines.append(f'  const int64_t t{ni} = {mul(node.ca, ref(node.a))} + {mul(node.cb, ref(node.b))};')
        for oi, row in enumerate(self.outputs):
            terms = [mul(c, ref(v)) for v,c in sorted(row.items())]
            lines.append(f"  y[{oi}] = {' + '.join(terms) if terms else '0'};")
        lines.append('}')
        Path(path).write_text('\n'.join(lines)+'\n', encoding='utf-8')

    def reconstruct(self) -> np.ndarray:
        """For small test matrices only; scales as [outputs, inputs] memory."""
        ident = np.eye(self.n_inputs, dtype=np.int64)
        return self.eval_numpy(ident).T


class TorchDAGExecutor:
    """Preload immutable DAG indices/coefficients once for repeated Torch calls.

    This is level-wise tensor code. GPU support is functional, not fused CUDA.
    """
    def __init__(self, dag: AlgebraicDAG, device: str = 'cpu'):
        import torch
        self.torch=torch
        self.dag=dag
        self.device=device
        self.stages=[]
        for ids in dag.levels():
            indices=[i-dag.n_inputs for i in ids]
            self.stages.append(tuple(torch.tensor(vals,dtype=torch.int64,device=device)
                for vals in [ids,
                             [dag.nodes[i].a for i in indices],
                             [dag.nodes[i].b for i in indices],
                             [dag.nodes[i].ca for i in indices],
                             [dag.nodes[i].cb for i in indices]]))
        rr,cc,v=dag.output_coo()
        self.output_tensors=tuple(torch.as_tensor(vals,dtype=torch.int64,device=device)
                                  for vals in (rr,cc,v))

    def __call__(self,x):
        torch=self.torch
        x=torch.as_tensor(x,dtype=torch.int64,device=self.device)
        squeeze=x.ndim==1
        if squeeze: x=x[None,:]
        if x.ndim!=2 or x.shape[1]!=self.dag.n_inputs:
            raise ValueError('incorrect input shape')
        values=torch.zeros((self.dag.n_inputs+len(self.dag.nodes),x.shape[0]),
                           device=self.device,dtype=torch.int64)
        values[:self.dag.n_inputs]=x.T
        for ids,a,b,ca,cb in self.stages:
            values[ids]=values[a]*ca[:,None]+values[b]*cb[:,None]
        rr,cc,c=self.output_tensors
        out=torch.zeros((self.dag.n_outputs,x.shape[0]),device=self.device,dtype=torch.int64)
        if rr.numel(): out.index_add_(0,rr,values[cc]*c[:,None])
        return out.T[0] if squeeze else out.T


def compile_tiled(w: np.ndarray, tile_rows: int = 32, tile_cols: int = 64,
                  max_nodes: int = 12, pair_window: int = 4) -> AlgebraicDAG:
    """Bounded heuristic search within tiles, with global hash-consing of nodes.

    Limitations: does not discover expressions crossing different column tiles;
    global hash-consing *can* share identical definitions across row tiles.
    """
    w = np.asarray(w)
    if w.ndim != 2 or not np.issubdtype(w.dtype, np.integer):
        raise TypeError('W must be 2D integer matrix')
    if w.size and (w.min() < -8 or w.max() > 7):
        raise ValueError('Only INT4 signed [-8,7] weight matrices; no quantization')
    m,n = w.shape
    if min(tile_rows,tile_cols) < 1 or max_nodes < 0 or pair_window < 1:
        raise ValueError('invalid tile or optimizer parameters')
    out: List[Dict[int,int]] = [dict() for _ in range(m)]
    nodes: List[Node] = []
    intern = {}
    for c0 in range(0,n,tile_cols):
        c1 = min(n,c0+tile_cols)
        for r0 in range(0,m,tile_rows):
            r1 = min(m,r0+tile_rows)
            local = extract_shared_expressions(w[r0:r1,c0:c1],
                                               max_nodes=max_nodes, pair_window=pair_window)
            remap = {i:c0+i for i in range(c1-c0)}
            for i, node in enumerate(local.nodes):
                a,b = remap[node.a],remap[node.b]
                # Canonicalize sign of operands by ordering ID for interning.
                if a>b:
                    a,b = b,a
                    ca,cb = node.cb,node.ca
                else:
                    ca,cb = node.ca,node.cb
                key = (a,b,ca,cb)
                if key not in intern:
                    intern[key] = n+len(nodes)
                    nodes.append(Node(*key))
                remap[local.inputs+i] = intern[key]
            for i, row in enumerate(local.outputs):
                dst = out[r0+i]
                for v, c in row.items():
                    gv = remap[v]
                    dst[gv] = dst.get(gv,0)+c
                    if dst[gv] == 0:
                        del dst[gv]
    result = AlgebraicDAG(n, nodes, out)
    result.validate()
    return result


def generated_example(case: str, rows: int, cols: int, rng: np.random.Generator):
    if case == 'random':
        return rng.integers(-8, 8, size=(rows,cols), dtype=np.int64)
    if case != 'structured':
        raise ValueError(case)
    w = rng.integers(-2,3,size=(rows,cols),dtype=np.int64)
    # Inject a common sub-function (2*x0+3*x1), scaled across all rows.
    if cols >= 2:
        s = np.asarray([1,2,-1,-2],dtype=np.int64)
        for i in range(rows):
            q = int(s[i%len(s)])
            w[i,0], w[i,1] = 2*q,3*q
    # Repeat additional column-pair patterns to create measurable DAG reuse.
    for j in range(2, cols-1, 2):
        for i in range(rows):
            q = [1,-1,2][(i+j)%3]
            w[i,j],w[i,j+1] = q,-q
    return w


def read_matrix(args) -> np.ndarray:
    if args.matrix:
        w = np.load(args.matrix, allow_pickle=False, mmap_mode='r')
        if w.ndim != 2:
            raise ValueError('Only two-dimensional .npy matrix accepted')
        r0=args.row_start; c0=args.col_start
        w = np.asarray(w[r0:r0+args.rows if args.rows else None,
                         c0:c0+args.cols if args.cols else None])
    elif args.safetensors:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError('pip install safetensors') from exc
        if not args.tensor:
            raise ValueError('--tensor NAME is required with --safetensors')
        with safe_open(args.safetensors, framework='pt', device='cpu') as f:
            if args.tensor not in f.keys():
                raise KeyError(f'{args.tensor} missing; examples: '+', '.join(list(f.keys())[:8]))
            sl = f.get_slice(args.tensor)
            shape=sl.get_shape()
            if len(shape) != 2:
                raise ValueError('Selected tensor must have rank 2')
            r0=args.row_start; c0=args.col_start
            if sl.get_dtype() not in ('I8','U8','I16','U16','I32','U32','I64','U64'):
                raise ValueError(f'Unsupported snapshot tensor dtype={sl.get_dtype()}: only pre-quantized, unpacked integer tensors; no implicit lossy conversion')
            tensor = sl[r0:r0+args.rows if args.rows else shape[0],
                        c0:c0+args.cols if args.cols else shape[1]]
            w = tensor.numpy()
    else:
        raise ValueError('Choose --matrix or --safetensors')
    if not np.issubdtype(w.dtype, np.integer):
        raise ValueError(f'weights dtype={w.dtype}; this prototype refuses to silently '
                         'quantize BF16/FP16/FP32. Use an existing integer snapshot '
                         'or a separate explicitly labeled lossy quantization study.')
    if w.size and (w.min() < -8 or w.max() > 7):
        raise ValueError('Integer storage is not necessarily unpacked INT4 weights; '
                         'quantized GPTQ/AWQ packed tensors require format-aware unpacking.')
    return np.asarray(w, dtype=np.int64)


def verify(dag: AlgebraicDAG, w: np.ndarray, cases: int = 20, seed: int = 1,
           torch_device: Optional[str] = None) -> None:
    if w.shape != (dag.n_outputs, dag.n_inputs):
        raise ValueError(f'W shape {w.shape} does not match DAG')
    rng = np.random.default_rng(seed)
    for i in range(cases):
        batch = 1 if i % 3 == 0 else 4
        x = rng.integers(-64,65,size=(batch,dag.n_inputs), dtype=np.int64)
        ref = x @ w.T.astype(np.int64)
        a = dag.eval_scalar(x)
        b = dag.eval_numpy(x)
        if not np.array_equal(ref,a) or not np.array_equal(ref,b):
            raise AssertionError(f'integer execution mismatch in test {i}')
        if torch_device:
            c = dag.eval_torch(x,torch_device).cpu().numpy()
            if not np.array_equal(ref,c):
                raise AssertionError(f'torch execution mismatch in test {i}')
    # For small matrices, verify exact coefficient reconstruction too.
    if dag.n_inputs*dag.n_outputs <= 1_000_000:
        if not np.array_equal(dag.reconstruct(),w):
            raise AssertionError('matrix reconstruction mismatch')


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest='command',required=True)
    demo = sub.add_parser('demo',help='Generate synthetic INT4 matrices, compile and validate')
    demo.add_argument('--case', choices=['random','structured','both'],default='both')
    demo.add_argument('--rows', type=int,default=32)
    demo.add_argument('--cols', type=int,default=64)
    demo.add_argument('--seed', type=int,default=42)
    for cmd in ('compile','verify'):
        pp = sub.add_parser(cmd)
        pp.add_argument('--matrix',help='2D integer .npy matrix (unpacked INT4 values)')
        pp.add_argument('--safetensors',help='Safetensors shard with an integer-valued 2D tensor')
        pp.add_argument('--tensor',help='Name of 2D tensor in safetensors shard')
        pp.add_argument('--row-start',type=int,default=0)
        pp.add_argument('--col-start',type=int,default=0)
        pp.add_argument('--rows',type=int,help='Limit number of output rows')
        pp.add_argument('--cols',type=int,help='Limit number of input columns')
    comp = sub.choices['compile']
    comp.add_argument('--out',help='Save portable DAG as .npz')
    comp.add_argument('--emit-c',help='Write fully expanded scalar C99 source')
    comp.add_argument('--emit-dot',help='Write Graphviz .dot preview')
    comp.add_argument('--export-layers',help='Write sparse level-wise matrix COO JSON')
    comp.add_argument('--tile-rows',type=int,default=32)
    comp.add_argument('--tile-cols',type=int,default=64)
    comp.add_argument('--max-nodes',type=int,default=12)
    comp.add_argument('--pair-window',type=int,default=4)
    comp.add_argument('--torch-device',help='Optional torch validation: cpu or cuda')
    test=sub.choices['verify']
    test.add_argument('--dag',required=True,help='Stored .npz graph')
    test.add_argument('--torch-device',help='cpu or cuda')
    args=p.parse_args()
    if args.command=='verify':
        dag=AlgebraicDAG.load(args.dag)
        w=read_matrix(args)
        verify(dag,w,torch_device=args.torch_device)
        print('PASS: exact integer outputs + weight reconstruction (when small)')
        print(json.dumps(dag.summary(),indent=2))
        return
    if args.command=='demo':
        rng=np.random.default_rng(args.seed)
        cases=['random','structured'] if args.case=='both' else [args.case]
        for case in cases:
            w=generated_example(case,args.rows,args.cols,rng)
            start=perf_counter()
            dag=compile_tiled(w)
            ms=(perf_counter()-start)*1000
            verify(dag,w)
            base=sum(max(0,np.count_nonzero(row)-1)+sum(_adds_for_coeff(int(v)) for v in row if v) for row in w)
            optimized=dag.summary()['est_shift_add_additions']
            print(f'[{case}] compile_ms={ms:.1f} exact_correctness=PASS; baseline_adds={base} dag_adds={optimized}; reduction={(100*(1-optimized/base) if base else 0):.1f}%')
            print(json.dumps(dag.summary(),indent=2))
        return
    w=read_matrix(args)
    start=perf_counter()
    dag=compile_tiled(w,args.tile_rows,args.tile_cols,args.max_nodes,args.pair_window)
    ms=(perf_counter()-start)*1000
    verify(dag,w,torch_device=args.torch_device)
    print(f'Compiled {w.shape[0]}x{w.shape[1]} INT4 in {ms:.1f} ms; verification PASS')
    print(json.dumps(dag.summary(),indent=2))
    if args.out:
        dag.save(args.out)
        print('Saved',args.out)
    if args.emit_c:
        dag.emit_c(args.emit_c)
        print('C99 source',args.emit_c)
    if args.emit_dot:
        dag.emit_dot(args.emit_dot)
        print('DOT preview',args.emit_dot)
    if args.export_layers:
        Path(args.export_layers).write_text(json.dumps(dag.sparse_layers(),indent=2)+'\n',encoding='utf-8')
        print('Sparse layer COO JSON',args.export_layers)


if __name__=='__main__':
    main()