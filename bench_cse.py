#!/usr/bin/env python3
"""Steady-state reference vs Numba greedy CSE microbenchmark (JIT warmup excluded)."""
import argparse, json, statistics, time
from pathlib import Path
import numpy as np
from algebraic_dag import extract_shared_expressions
from cse_accelerator import NUMBA_AVAILABLE

def med(fn,n):
    times=[]
    for _ in range(n):
        s=time.perf_counter();fn();times.append((time.perf_counter()-s)*1000)
    return statistics.median(times)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--rows',type=int,default=32)
    p.add_argument('--cols',type=int,default=32)
    p.add_argument('--max-nodes',type=int,default=16)
    p.add_argument('--pair-window',type=int,default=4)
    p.add_argument('--repeat',type=int,default=30)
    p.add_argument('--out')
    a=p.parse_args()
    if not NUMBA_AVAILABLE:p.error('Install numba first')
    if min(a.rows,a.cols,a.pair_window,a.repeat)<1 or a.max_nodes<0:p.error('Invalid arguments')
    rng=np.random.default_rng(42)
    report={}
    for kind in ('random_int4','structured_int4'):
        w=rng.integers(-8,8,size=(a.rows,a.cols),dtype=np.int64)
        if kind=='structured_int4':
            w[:,1::4]=w[:,0::4][:,:w[:,1::4].shape[1]]
        opts={'max_nodes':a.max_nodes,'pair_window':a.pair_window}
        ref=lambda:extract_shared_expressions(w,engine='python',**opts)
        fast=lambda:extract_shared_expressions(w,engine='numba',**opts)
        fast()
        if ref()!=fast():raise RuntimeError('DAG mismatch')
        t1=med(ref,a.repeat);t2=med(fast,a.repeat)
        report[kind]={'reference_ms':round(t1,3),'numba_ms':round(t2,3),
                      'speedup':round(t1/t2,2) if t2 else None,
                      'shared_nodes':len(fast().nodes)}
    result={'tiles':[a.rows,a.cols],'max_nodes':a.max_nodes,'repeat':a.repeat,
            'results':report,'note':'CSE-only steady-state, not end-to-end model speed'}
    output=json.dumps(result,indent=2)
    print(output)
    if a.out:
        path=Path(a.out);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(output+'\n',encoding='utf-8')

if __name__=='__main__':main()
