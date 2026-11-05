#!/usr/bin/env python3
"""Isolated RAM + optional CUDA allocator footprint measurements for a GGUF slice.

Each candidate runs in a fresh subprocess. These are microbenchmarks of ONE
selected tensor slice, not whole-model inference or an Ollama A/B comparison.
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def worker(a):
    import numpy as np
    import psutil
    from ollama_dag_pipeline import GGUFReader, compile_slice, load_artifact, artifact_bytes, decode_raw_selection
    x = np.random.default_rng(17).integers(-16, 17, size=a.cols, dtype=np.int64)
    base = psutil.Process().memory_info().rss
    cuda = a.mode.endswith('_cuda')
    if cuda:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable')
        torch.cuda.init()
        torch.cuda.reset_peak_memory_stats()
    with GGUFReader(a.gguf) as gguf:
        t = next(t for t in gguf.tensors if t.name == a.tensor)
        regions = gguf.regions(t,a.row_start,a.rows,a.col_start,a.cols)
        if a.mode == 'compile':
            compiled, _, _ = compile_slice(gguf,t,a.row_start,a.rows,a.col_start,a.cols,
                                           a.tile_rows,a.max_nodes,a.pair_window)
            blob = artifact_bytes(compiled)
            for _ in range(a.repeat): len(blob)
        elif a.mode in ('dag','dag_cuda'):
            compiled = load_artifact(Path(a.artifact).read_bytes())
            for _ in range(a.repeat):
                y = compiled.eval_grouped_torch(x,device='cuda') if cuda else compiled.eval_grouped(x)
        else:
            raw=b''.join(gguf.mm[p:p+n] for p,n in regions)
            if a.mode in ('dense','dense_cuda'):
                mat=decode_raw_selection(raw,t.dtype,a.rows,a.cols)
                if cuda:
                    import torch
                    mat=torch.as_tensor(mat,device='cuda',dtype=torch.float64)
                    x=torch.as_tensor(x,device='cuda',dtype=torch.float64)
                for _ in range(a.repeat): y=mat@x.astype(np.float64) if not cuda else mat@x
            elif a.mode == 'packed':
                for _ in range(a.repeat):
                    mat=decode_raw_selection(raw,t.dtype,a.rows,a.cols)
                    y=mat@x.astype(np.float64)
                    del mat
    if cuda:
        torch.cuda.synchronize()
        gpu={'torch_peak_allocated_bytes':int(torch.cuda.max_memory_allocated()),
             'torch_peak_reserved_bytes':int(torch.cuda.max_memory_reserved()),
             'note':'PyTorch CUDA allocator only; excludes some context/driver and other processes.'}
    else:
        gpu=None
    time.sleep(.07)
    peak=None
    info=psutil.Process().memory_info()
    if getattr(info,'peak_wset',None) is not None:peak=int(info.peak_wset)
    elif os.name!='nt':
        try:
            import resource,platform
            peak=int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if platform.system()!='Darwin':peak*=1024
        except (ImportError,OSError): pass
    print(json.dumps({'baseline_rss_bytes':base,'end_rss_bytes':int(info.rss),
                      'os_peak_rss_bytes':peak,'cuda':gpu}))


def measure(a,mode,interval_ms):
    import psutil
    cmd=[sys.executable,str(Path(__file__).resolve()),'_worker','--mode',mode,
         '--gguf',str(a.gguf),'--artifact',str(a.artifact),'--tensor',a.tensor,
         '--row-start',str(a.row_start),'--rows',str(a.rows),'--col-start',str(a.col_start),
         '--cols',str(a.cols),'--tile-rows',str(a.tile_rows),'--max-nodes',str(a.max_nodes),
         '--pair-window',str(a.pair_window),'--repeat',str(a.repeat)]
    proc=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True)
    peak=0; count=0; started=time.monotonic(); psproc=psutil.Process(proc.pid)
    while proc.poll() is None:
        try: peak=max(peak,psproc.memory_info().rss);count+=1
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass
        time.sleep(interval_ms/1000)
    out,err=proc.communicate()
    if proc.returncode:raise RuntimeError(f'isolated {mode} failed: {err[-2000:]}')
    result=json.loads(out)
    peak=max(peak,result.get('os_peak_rss_bytes') or 0)
    return {'peak_rss_bytes':peak,'peak_rss_mib':round(peak/1048576,3),
            'baseline_rss_bytes':result['baseline_rss_bytes'],
            'peak_minus_baseline_rss_bytes':max(0,peak-result['baseline_rss_bytes']),
            'end_rss_bytes':result['end_rss_bytes'],
            'cuda_allocator':result['cuda'],'samples':count,
            'elapsed_seconds_including_startup':round(time.monotonic()-started,3)}


def profile_slice(gguf,artifact,tensor,row_start,rows,col_start,cols,
                  tile_rows,max_nodes,pair_window,repeat=8,interval_ms=20,profile_cuda=True):
    if interval_ms<5:raise ValueError('interval must be at least 5 ms')
    a=argparse.Namespace(gguf=gguf,artifact=artifact,tensor=tensor,row_start=row_start,
        rows=rows,col_start=col_start,cols=cols,tile_rows=tile_rows,max_nodes=max_nodes,
        pair_window=pair_window,repeat=repeat)
    metrics={m:measure(a,m,interval_ms) for m in ('compile','dense','packed','dag')}
    cuda_status='disabled'
    if profile_cuda:
        try:
            import torch
            if torch.cuda.is_available():
                for m in ('dense_cuda','dag_cuda'):metrics[m]=measure(a,m,interval_ms)
                cuda_status='measured via torch CUDA allocator'
            else:cuda_status='unavailable: CUDA-enabled PyTorch not detected'
        except ImportError:cuda_status='unavailable: torch not installed'
    return {'scope':'One selected GGUF tensor slice, not whole-model inference',
            'isolation':'Fresh process for every mode; absolute RSS includes interpreter startup',
            'sampling_interval_ms':interval_ms,'metrics':metrics,'cuda_status':cuda_status,
            'warning':'Do not compare slice RAM/VRAM to whole Ollama model as a claimed optimization.'}


def main():
    p=argparse.ArgumentParser();sub=p.add_subparsers(dest='command',required=True)
    w=sub.add_parser('_worker')
    for name in ('mode','gguf','artifact','tensor'):w.add_argument('--'+name,required=True)
    for name in ('row-start','rows','col-start','cols','tile-rows','max-nodes','pair-window','repeat'):
        w.add_argument('--'+name,type=int,required=True)
    a=p.parse_args()
    if a.command=='_worker':worker(a)

if __name__=='__main__':main()
