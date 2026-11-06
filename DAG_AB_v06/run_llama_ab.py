#!/usr/bin/env python3
"""Full-token-inference A/B for vanilla llama.cpp vs experimental CPU DAG hook.

This does NOT measure a reduced model weight allocation: GGML still loads original
GGUF tensors. It compares native DAG execution and cache strategies, reporting
observed CPU RSS and aggregate GPU utilization separately. CPU-only, threads=1.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

try:
    import psutil
except ImportError as exc:
    raise SystemExit('Install psutil: python -m pip install psutil') from exc

MODES = ('baseline', 'resident', 'stream50', 'stream25', 'stream10')
FRACTIONS = {'stream50': .50, 'stream25': .25, 'stream10': .10}


def ratio_or_none(numerator, denominator):
    """Return a finite ratio or JSON null when baseline cannot be measured.

    In fast HTTP mock tests, client TTFT and sampled process RSS may be zero
    (especially on Windows). A zero baseline is not evidence of a speedup.
    """
    if numerator is None or denominator is None:
        return None
    try:
        n = float(numerator)
        d = float(denominator)
    except (TypeError, ValueError, OverflowError):
        return None
    if not math.isfinite(n) or not math.isfinite(d) or d <= 0:
        return None
    result = n / d
    return result if math.isfinite(result) else None


def now(): return time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def atomic_json(path: Path, obj):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    tmp.replace(path)


def hash_file(path:Path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''):h.update(part)
    return h.hexdigest()


def preflight(source: Path, compiled: Path):
    if not source.is_file(): raise FileNotFoundError(source)
    report_path=compiled/'full_model_report.json'
    if not report_path.is_file():raise FileNotFoundError(report_path)
    report=json.loads(report_path.read_text(encoding='utf-8'))
    if not report.get('complete'):raise ValueError('Incomplete DAG stream: no full-model A/B allowed')
    if source.stat().st_size!=report['original_gguf_bytes']:
        raise ValueError('Original GGUF size differs from DAG source')
    if hash_file(source)!=report['source_sha256']:
        raise ValueError('Original GGUF SHA256 differs from DAG source')
    for f in ('weights.index.jsonl','weights.adagstream'):
        if not (compiled/f).is_file():raise FileNotFoundError(compiled/f)
    return report


class Monitor:
    def __init__(self,pid:int,interval:float):
        self.pid=pid;self.interval=interval;self.done=threading.Event()
        self.t=threading.Thread(target=self.loop,daemon=True)
        self.rss_peak=0;self.system_used_peak=0;self.gpu_used_peak={}
        self.n_samples=0
    def __enter__(self):self.t.start();return self
    def __exit__(self,*args):self.done.set();self.t.join(timeout=3)
    def loop(self):
        while not self.done.is_set():
            try:
                p=psutil.Process(self.pid)
                allp=[p]+p.children(recursive=True)
                rss=0
                for child in allp:
                    try:rss+=child.memory_info().rss
                    except (psutil.NoSuchProcess,psutil.AccessDenied):pass
                self.rss_peak=max(self.rss_peak,rss)
                self.system_used_peak=max(self.system_used_peak,psutil.virtual_memory().used)
                self.n_samples+=1
                # nvidia-smi may be unavailable; overall device memory, NOT model-only.
                try:
                    r=subprocess.run(['nvidia-smi','--query-gpu=index,memory.used',
                                      '--format=csv,noheader,nounits'],capture_output=True,
                                     text=True,timeout=3)
                    if r.returncode==0:
                        for line in r.stdout.splitlines():
                            parts=line.split(',')
                            if len(parts)==2:
                                ix=int(parts[0].strip()); mib=int(parts[1].strip())
                                self.gpu_used_peak[ix]=max(self.gpu_used_peak.get(ix,0),mib)
                except (FileNotFoundError,subprocess.TimeoutExpired,ValueError):pass
            except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            self.done.wait(self.interval)
    def summary(self):return {
        'peak_server_process_tree_rss_bytes':self.rss_peak,
        'peak_system_used_ram_bytes':self.system_used_peak,
        'nvidia_global_peak_used_mib':self.gpu_used_peak,
        'n_samples':self.n_samples,
        'note':'GPU usage is global, including other processes. Process RSS includes original GGUF backing.'}


def server_health(base:str,proc:subprocess.Popen,timeout:float):
    start=time.monotonic()
    while time.monotonic()-start<timeout:
        if proc.poll() is not None: raise RuntimeError(f'llama-server exited during load ({proc.returncode})')
        try:
            with urllib.request.urlopen(base+'/health',timeout=2) as resp:
                status=json.loads(resp.read())
                if status.get('status') in ('ok','ready') or status.get('status') is None:
                    return
        except (urllib.error.URLError,TimeoutError,ValueError,ConnectionError):pass
        time.sleep(.5)
    raise TimeoutError(f'Could not start llama-server in {timeout}s')


def compare_token_sequences(reference:dict, candidate:dict):
    """Compare actual streamed token IDs, not only hashes.

    Do not claim parity if either capture was incomplete or unavailable:
    sha256([]) == sha256([]) is not evidence of equivalent inference.
    Indices in divergence metadata are zero-based.
    """
    a=reference.get('token_ids')
    b=candidate.get('token_ids')
    if not isinstance(a,list) or not isinstance(b,list):
        return {'status':'unknown','reason':'token IDs missing from one or both reports'}
    if not reference.get('token_ids_complete') or not candidate.get('token_ids_complete'):
        return {'status':'unknown','reason':'stream token capture incomplete',
                'baseline_ids_captured':len(a),'candidate_ids_captured':len(b)}
    i=next((i for i,(x,y) in enumerate(zip(a,b)) if x!=y),min(len(a),len(b)))
    if i==len(a)==len(b):
        return {'status':'equal','same_token_ids':True,'matching_prefix_tokens':i,
                'baseline_count':len(a),'candidate_count':len(b)}
    return {'status':'different','same_token_ids':False,
            'first_mismatch_index':i,'first_mismatch_position':i+1,
            'matching_prefix_tokens':i,'baseline_count':len(a),'candidate_count':len(b),
            'baseline_token_id':a[i] if i<len(a) else None,
            'candidate_token_id':b[i] if i<len(b) else None}


def completion(base:str,prompt:str,predict:int,seed:int,timeout:float):
    body={'prompt':prompt,'stream':True,'n_predict':predict,'temperature':0,
          'seed':seed,'cache_prompt':False,'return_tokens':True,
          'repeat_penalty':1.0,'top_k':1}
    request=urllib.request.Request(base+'/completion',
        data=json.dumps(body).encode('utf-8'),headers={'Content-Type':'application/json'})
    started=time.monotonic();first=None;final=None;output=[];token_ids=[]
    with urllib.request.urlopen(request,timeout=timeout) as response:
        for binary in response:
            line=binary.decode('utf-8').strip()
            if not line.startswith('data:'):continue
            payload=line[5:].strip()
            if payload=='[DONE]':continue
            data=json.loads(payload)
            if 'error' in data:raise RuntimeError(str(data['error']))
            tokens=data.get('tokens') or []
            if tokens:token_ids.extend(tokens)
            chunk=data.get('content','')
            if chunk:output.append(chunk)
            if first is None and (tokens or chunk):first=time.monotonic()-started
            if data.get('stop',False):final=data
    if final is None:raise RuntimeError('SSE stream ended without stop/timings')
    total=time.monotonic()-started
    timings=final.get('timings',{})
    if not timings:raise RuntimeError('llama-server did not return timings')
    text=''.join(output)
    predicted=timings.get('predicted_n')
    capture_complete=(isinstance(predicted,int) and len(token_ids)==predicted)
    return {'client_ttft_s':first,'client_total_s':total,
            'decode_tokens':timings.get('predicted_n'),
            'decode_tps':timings.get('predicted_per_second'),
            'prompt_tokens':timings.get('prompt_n'),
            'prompt_tps':timings.get('prompt_per_second'),
            'timings':timings,
            'text_sha256':hashlib.sha256(text.encode()).hexdigest(),
            'token_ids_sha256':hashlib.sha256(json.dumps(token_ids).encode()).hexdigest(),
            'token_ids':token_ids,
            'token_ids_captured':len(token_ids),
            'token_ids_complete':capture_complete,
            'stop_type':final.get('stop_type'),
            'text':text}


def extract_dag_stats(logpath:Path):
    raw=logpath.read_text(encoding='utf-8',errors='replace')
    arr=re.findall(r'\[DAG native\] STATS (\{[^\r\n]+\})',raw)
    return json.loads(arr[-1]) if arr else None


def phase(mode:str,args,report:dict,source_bytes:int):
    base=f'http://127.0.0.1:{args.port}'
    env=os.environ.copy()
    env.pop('DAG_STREAM_DIR',None)
    env.pop('DAG_CACHE_BYTES',None)
    env.pop('DAG_TENSOR_ALLOW',None)
    env.pop('DAG_STATS_EVERY',None)
    env.pop('DAG_EXEC_PLAN',None)
    env.pop('DAG_INPUT_SEMANTICS',None)
    budget=None
    stats_every=None
    if mode!='baseline':
        env['DAG_STREAM_DIR']=str(args.compiled.resolve())
        env['DAG_TENSOR_ALLOW']=args.dag_tensors
        budget = 0 if mode=='resident' else int(source_bytes*FRACTIONS[mode])
        env['DAG_CACHE_BYTES']=str(budget)
        # Windows Popen.terminate() kills the server without C++ atexit.
        # A targeted smoke test needs a snapshot after EVERY completed GEMV.
        stats_every=(args.dag_stats_every if args.dag_stats_every is not None
                     else (1 if args.dag_tensors != 'all' else 1000))
        env['DAG_STATS_EVERY']=str(stats_every)
        env['DAG_EXEC_PLAN']='1' if args.dag_execution=='decoded' else '0'
        env['DAG_INPUT_SEMANTICS']=args.dag_input_semantics
    logfile=args.out_dir / (mode+'.server.log')
    # A .py mock server is launched through the active interpreter on every OS.
    # Production llama-server binaries remain direct executables, so an invalid
    # native binary still fails rather than silently switching to a mock.
    server_prefix = ([sys.executable, str(args.server)]
                     if args.server.suffix.lower() == '.py' else [str(args.server)])
    cmd=server_prefix+['-m',str(args.gguf),'--host','127.0.0.1',
         '--port',str(args.port),'-t','1','-ngl','0',
         '-c',str(args.ctx),'--parallel','1']
    obj={'status':'started','mode':mode,'cache_budget_bytes':budget,
         'started_utc':now(),'server_command':cmd,'trials':[],
         'dag_tensor_scope':args.dag_tensors if mode!='baseline' else None,
         'dag_execution':args.dag_execution if mode!='baseline' else None,
         'dag_input_semantics':args.dag_input_semantics if mode!='baseline' else None,
         'dag_native_stats_sample_interval':stats_every,
         'dag_native_stats_exact_after_completed_trials':stats_every==1}
    with logfile.open('w',encoding='utf-8') as log:
        proc=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT,env=env)
        try:
            with Monitor(proc.pid,args.interval) as mon:
                server_health(base,proc,args.load_timeout)
                for j in range(args.trials):
                    result=completion(base,args.prompt,args.num_predict,args.seed,args.request_timeout)
                    result['trial']=j+1
                    obj['trials'].append(result)
                    print(f'[{mode}] trial={j+1} TTFT={result["client_ttft_s"]:.3f}s '
                          f' decode={result["decode_tps"]} tok/s',flush=True)
            obj['memory']=mon.summary()
        finally:
            if proc.poll() is None:
                proc.terminate()
                try:proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill();proc.wait(timeout=5)
    obj['dag_native_stats']=extract_dag_stats(logfile)
    # Prove the requested compatibility bridge was actually executed.
    # Old patched GGML binaries must fail, not silently report native F32.
    obj['dag_q8_bridge_confirmed']=None
    if mode!='baseline' and args.dag_input_semantics=='ggml-q8':
        obj['dag_q8_bridge_confirmed']='[DAG q8 normal] tensor=' in logfile.read_text(encoding='utf-8',errors='replace')
        if not obj['dag_q8_bridge_confirmed']:
            raise RuntimeError(f'{mode}: ggml-q8 requested but single-pass GGML bridge did not run. '
                               'Install DAG_AB_v06/install_ggml_q8_input.py into llama-dag, '
                               'rebuild llama-server and check the tensor allowlist.')
    obj['finished_utc']=now()
    obj['server_returncode']=proc.returncode
    if mode!='baseline':
        stats=obj['dag_native_stats']
        if stats and (args.dag_execution=='decoded' and stats.get('exec_plan') != 1 or
                      args.dag_execution=='raw' and stats.get('exec_plan') not in (None, 0)):
            raise RuntimeError(f'{mode}: native execution plan mismatch; update dag_runtime.cpp and rebuild llama-server')
        if stats and stats.get('stats_every') != stats_every:
            raise RuntimeError(f'{mode}: DAG stats interval mismatch: runtime reports '
                               f'{stats.get("stats_every")}, harness expects {stats_every}. '
                               'Re-copy DAG_AB_v06/native/dag_runtime.cpp into llama-dag '
                               'and rebuild llama-server.')
        if not stats or stats['gemv_calls']<1:
            raise RuntimeError(f'{mode}: zero DAG GEMV calls. No genuine DAG A/B. See {logfile}')
    else:
        if obj['dag_native_stats'] and obj['dag_native_stats']['gemv_calls']:
            raise RuntimeError('Baseline unexpectedly executed DAG hooks')
    trials=obj['trials']
    obj['median_client_ttft_s']=statistics.median(t['client_ttft_s'] for t in trials if t['client_ttft_s'] is not None)
    obj['median_decode_tps']=statistics.median(t['decode_tps'] for t in trials if t['decode_tps'] is not None)
    obj['status']='completed'
    return obj


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--server',required=True,type=Path,help='Patched llama-server executable')
    p.add_argument('--gguf',required=True,type=Path)
    p.add_argument('--compiled',required=True,type=Path)
    p.add_argument('--out-dir',required=True,type=Path)
    p.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    p.add_argument('--dag-tensors',default='all',help='all OR exact comma-separated GGUF tensor names (for smoke)')
    p.add_argument('--dag-input-semantics',choices=('native-f32','ggml-q8'),
                   default='native-f32',
                   help='Use original F32 activations (default) or GGML Q8-quantize/dequantize input before one DAG GEMV')
    p.add_argument('--dag-execution',choices=('raw','decoded'),default='raw',
                   help='Raw bytecode interpreter (default) vs opt-in predecoded execution plan')
    p.add_argument('--dag-stats-every',type=int,default=None,
                   help='Log native stats every N GEMVs (default: 1 for targeted smoke, 1000 for all). '
                        'Logging each full-model GEMV distorts benchmark timings.')
    p.add_argument('--prompt',default='Explain Java happens-before and volatile in 80 words.')
    p.add_argument('--prompt-file',type=Path)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-predict',type=int,default=128)
    p.add_argument('--ctx',type=int,default=4096)
    p.add_argument('--trials',type=int,default=3)
    p.add_argument('--port',type=int,default=19081)
    p.add_argument('--interval',type=float,default=.25)
    p.add_argument('--load-timeout',type=float,default=300)
    p.add_argument('--request-timeout',type=float,default=1800)
    a=p.parse_args(argv)
    if a.prompt_file:a.prompt=a.prompt_file.read_text(encoding='utf-8')
    if not a.server.is_file():p.error(f'No llama-server: {a.server}')
    if a.trials<1:p.error('--trials must be >=1')
    if a.dag_stats_every is not None and a.dag_stats_every<1:
        p.error('--dag-stats-every must be >=1')
    if len(a.modes)!=len(set(a.modes)):p.error('Duplicate modes')
    a.gguf=a.gguf.resolve();a.compiled=a.compiled.resolve()
    meta=preflight(a.gguf,a.compiled)
    a.out_dir.mkdir(parents=True,exist_ok=True)
    path=a.out_dir/'full_inference_ab.json'
    if path.exists():p.error(f'Report already exists: {path}; select new --out-dir')
    report={'schema_version':1,'status':'running','started_utc':now(),
            'source_sha256':meta['source_sha256'],'source_bytes':meta['original_gguf_bytes'],
            'compiled':str(a.compiled),'prompt':a.prompt,
            'settings':{'temperature':0,'seed':a.seed,'num_predict':a.num_predict,
                        'ctx':a.ctx,'trials':a.trials,'cpu_threads':1,'gpu_layers':0,
                        'dag_input_semantics':a.dag_input_semantics},
            'scope':{'full_model_token_generation':True,'dag_exec':'CPU Q4_0/Q4_K GEMV only',
                'original_weight_allocation_eliminated':False,
                'dag_tensor_scope':a.dag_tensors,
                'all_supported_matrices_requested':a.dag_tensors=='all',
                'memory_budget_is':'decoded DAG cache only; not a hard server RSS/VRAM cap',
                'baseline_vs_dag_numeric_bit_exact':False},
            'phases':{}}
    atomic_json(path,report)
    for mode in a.modes:
        print(f'[{mode}] starting server...',flush=True)
        try:
            report['phases'][mode]=phase(mode,a,report,meta['original_gguf_bytes'])
        except Exception as e:
            report['phases'][mode]={'status':'failed','error':str(e)}
            report['status']='failed'
            atomic_json(path,report)
            print(f'[{mode}] FAILED: {e}',file=sys.stderr)
            return 1
        atomic_json(path,report)
    if 'baseline' in report['phases']:
        ref=report['phases']['baseline']
        for mode,pdata in report['phases'].items():
            if mode=='baseline':continue
            cmp=compare_token_sequences(ref['trials'][0],pdata['trials'][0])
            pdata['relative_to_baseline']={
                'decode_speedup':ratio_or_none(pdata['median_decode_tps'], ref['median_decode_tps']),
                'ttft_ratio':ratio_or_none(pdata['median_client_ttft_s'], ref['median_client_ttft_s']),
                'peak_rss_ratio':ratio_or_none(
                    pdata['memory']['peak_server_process_tree_rss_bytes'],
                    ref['memory']['peak_server_process_tree_rss_bytes']),
                'same_token_ids_first_trial':cmp.get('same_token_ids'),
                'token_comparison_first_trial':cmp,
                'same_text_first_trial':pdata['trials'][0]['text_sha256']==
                                        ref['trials'][0]['text_sha256']}
    report['status']='complete';report['finished_utc']=now();atomic_json(path,report)
    print(f'Full-model report: {path}',flush=True)
    return 0

if __name__=='__main__':sys.exit(main())