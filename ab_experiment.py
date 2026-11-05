#!/usr/bin/env python3
"""Reproducible Ollama baseline + whole-snapshot DAG compile + gated true A/B.

The compiled .adagstream is not runnable by Ollama. This orchestrator NEVER
pretends to execute it by feeding the restored GGUF back to baseline Ollama.
An optimized pass requires a separately implemented, verified DAG-aware runtime
with the /dag/capabilities and /api/generate interfaces documented in README.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from ollama_dag_pipeline import post_json, resolve_ollama_blob
from whole_model_dag import atomic_json


def iso_now():
    return datetime.now(timezone.utc).isoformat()


def http_get_json(base: str, path: str, *, timeout: int = 10):
    with urllib.request.urlopen(base.rstrip('/')+path, timeout=timeout) as response:
        return json.load(response)


def model_residency(base, model):
    try:
        models=http_get_json(base,'/api/ps').get('models',[])
        return next((m for m in models if m.get('model')==model or m.get('name')==model),None)
    except (OSError,ValueError):
        return None


class MemoryMonitor:
    """Sample system and Ollama RSS plus system-wide NVIDIA VRAM usage.

    Global Nvidia memory is *not* per-model VRAM: measurements include unrelated
    GPU processes. Ollama /api/ps provides per-loaded-model size_vram separately.
    """
    def __init__(self, interval: float = .4, target_pid: int | None = None):
        self.interval=interval
        self.target_pid=target_pid
        self.samples=[]
        self._stop=threading.Event()
        self._thread=None
        self.nvidia=shutil.which('nvidia-smi')

    def capture(self):
        import psutil
        vm=psutil.virtual_memory()
        ollama_rss=0
        for p in psutil.process_iter(['name','memory_info']):
            try:
                name=(p.info['name'] or '').lower()
                if 'ollama' in name or 'llama-server' in name or 'llama_server' in name:
                    ollama_rss+=p.info['memory_info'].rss
            except (psutil.AccessDenied,psutil.NoSuchProcess,TypeError):
                continue
        target_rss=None
        if self.target_pid is not None:
            try:
                proc=psutil.Process(self.target_pid)
                members=[proc]+proc.children(recursive=True)
                target_rss=sum(p.memory_info().rss for p in members if p.is_running())
            except (psutil.AccessDenied,psutil.NoSuchProcess,psutil.ZombieProcess):
                pass
        item={'elapsed_s':round(time.monotonic()-self.start,3),
              'system_used_ram_bytes':int(vm.used),
              'system_available_ram_bytes':int(vm.available),
              'ollama_process_rss_bytes':int(ollama_rss),
              'target_process_tree_rss_bytes':target_rss}
        if self.nvidia:
            try:
                p=subprocess.run([self.nvidia,'--query-gpu=index,memory.used,memory.total',
                     '--format=csv,noheader,nounits'],capture_output=True,text=True,
                     timeout=max(2.,self.interval*3))
                if p.returncode == 0:
                    item['nvidia_gpus']=[dict(zip(('index','used_mib','total_mib'),
                                  map(int,line.split(','))))
                              for line in p.stdout.splitlines() if line.strip()]
            except (ValueError,TypeError,subprocess.SubprocessError,OSError):
                pass
        return item

    def _run(self):
        while not self._stop.is_set():
            try:self.samples.append(self.capture())
            except (OSError,ValueError):pass
            self._stop.wait(self.interval)

    def __enter__(self):
        self.start=time.monotonic()
        self._thread=threading.Thread(target=self._run,daemon=True)
        self._thread.start()
        return self

    def __exit__(self,*args):
        self._stop.set()
        self._thread.join(timeout=5)
        try:self.samples.append(self.capture())
        except (OSError,ValueError):pass

    def summary(self):
        if not self.samples:
            return {'status':'unavailable'}
        def val(k):return [x[k] for x in self.samples if k in x]
        out={'samples':len(self.samples),'peak_ollama_process_rss_bytes':max(val('ollama_process_rss_bytes')),
             'peak_system_used_ram_bytes':max(val('system_used_ram_bytes')),
             'initial_system_used_ram_bytes':self.samples[0]['system_used_ram_bytes'],
             'peak_system_ram_increase_bytes':max(0,max(val('system_used_ram_bytes'))-self.samples[0]['system_used_ram_bytes'])}
        per_gpu={}
        for sample in self.samples:
            for gpu in sample.get('nvidia_gpus',[]):
                ix=gpu['index']
                if ix not in per_gpu:per_gpu[ix]={'initial_mib':gpu['used_mib'],'peak_mib':gpu['used_mib'],'total_mib':gpu['total_mib']}
                per_gpu[ix]['peak_mib']=max(per_gpu[ix]['peak_mib'],gpu['used_mib'])
        for g in per_gpu.values():
            g['increase_mib']=max(0,g['peak_mib']-g['initial_mib'])
        targets=[s['target_process_tree_rss_bytes'] for s in self.samples if s.get('target_process_tree_rss_bytes') is not None]
        if targets:out['peak_target_process_tree_rss_bytes']=max(targets)
        out['nvidia_system_vram_mib_by_gpu']=per_gpu
        out['vram_note']='Global device usage (other processes included), not allocated GPU memory exclusively for this model.'
        return out


def streamed_generation(base: str, model: str, prompt: str, options: dict, timeout: int = 3600):
    """Ollama-compatible streamed generate: accurate client-observed TTFT."""
    body={'model':model,'prompt':prompt,'stream':True,'keep_alive':'5m', 'options':options}
    request=urllib.request.Request(base.rstrip('/')+'/api/generate',
        data=json.dumps(body).encode('utf-8'),headers={'Content-Type':'application/json'})
    start=time.monotonic()
    chunks=[]; final=None; first_token_s=None
    with urllib.request.urlopen(request,timeout=timeout) as response:
        for line in response:
            if not line.strip():continue
            obj=json.loads(line)
            if 'error' in obj:raise RuntimeError(obj['error'])
            token=obj.get('response','')
            if token:
                chunks.append(token)
                if first_token_s is None:first_token_s=time.monotonic()-start
            if obj.get('done',False):final=obj
    if final is None:raise RuntimeError('Generation stream terminated without final metrics')
    wall=time.monotonic()-start
    count=final.get('eval_count')
    edur=final.get('eval_duration')
    return {'text':''.join(chunks),'metrics':{
        'wall_seconds':wall,'client_ttft_seconds':first_token_s,
        'eval_count':count,'eval_duration_ns':edur,
        'tokens_per_second':count*1e9/edur if count and edur else None,
        'prompt_eval_count':final.get('prompt_eval_count'),
        'prompt_eval_duration_ns':final.get('prompt_eval_duration'),
        'load_duration_ns':final.get('load_duration'),
        'total_duration_ns':final.get('total_duration'),
        'done_reason':final.get('done_reason')}}


class Experiment:
    def __init__(self,out:Path,config:dict):
        self.out=out
        out.mkdir(parents=True,exist_ok=True)
        self.logs=out/'logs';self.logs.mkdir(exist_ok=True)
        self.events=out/'events.jsonl'
        self.path=out/'ab_report.json'
        if self.path.exists():
            raise FileExistsError(f'{self.path} exists: choose a new experiment output folder')
        self.report={'schema_version':1,'started_utc':iso_now(),'status':'running',
                     'configuration':config,'phases':{},'interpretation':{
                        'memory':'System-wide metrics include unrelated processes; Ollama /api/ps adds size_vram per loaded model.',
                        'ab':'Only a genuine DAG runtime may be identified as optimized; normal GGUF restoration is not an optimization.'}}
        self.save()

    def event(self,name,**fields):
        with self.events.open('a',encoding='utf-8') as f:
            f.write(json.dumps({'timestamp_utc':iso_now(),'event':name,**fields},ensure_ascii=False)+'\n')
        self.save()

    def save(self):
        atomic_json(self.path,self.report)

    def phase(self,name,info):
        self.report['phases'][name]=info
        self.event('phase_completed',phase=name,status=info.get('status'))


def run_generation(exp:Experiment, name:str,base:str,model:str,prompt:str,options:dict,interval:float):
    exp.event('generation_started',phase=name,model=model)
    with MemoryMonitor(interval) as monitor:
        resp=streamed_generation(base,model,prompt,options)
    result={'status':'completed','model':model,'endpoint':base,
            'metrics':resp['metrics'],'memory':monitor.summary(),
            'ollama_model_residency':model_residency(base,model),
            'text_sha256':hashlib.sha256(resp['text'].encode('utf-8')).hexdigest()}
    (exp.logs/f'{name}.txt').write_text(resp['text'],encoding='utf-8')
    (exp.logs/f'{name}.json').write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    with (exp.logs/f'{name}.memory.jsonl').open('w',encoding='utf-8') as f:
        for record in monitor.samples:
            f.write(json.dumps(record)+'\n')
    exp.phase(name,result)
    return result


def _latest_compile_event(path: Path):
    """Read only the tail of progress logs to bound RAM usage."""
    try:
        with path.open('rb') as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 16384), os.SEEK_SET)
            lines = f.read().splitlines()
        for line in reversed(lines):
            try:
                item = json.loads(line)
            except (ValueError, UnicodeDecodeError):
                continue
            if item.get('event') in ('compile_started','compile_progress','compile_finished','compile_checkpoint'):
                return item
    except OSError:
        pass
    return None


def _emit_compile_progress(item, elapsed):
    done = int(item.get('completed', item.get('already_completed', 0)))
    total = int(item.get('total', item.get('total_units', 0)))
    duration = float(item.get('elapsed_seconds', elapsed))
    if total:
        rate = done/duration if duration > 0 else 0
        eta = (total-done)/rate/3600 if rate > 0 else None
        eta_label = f'{eta:.1f}h' if eta is not None else 'unknown'
        print(f'[DAG CSE] {done:,}/{total:,} ({done/total:.2%}) | {rate:.2f} units/s | ETA {eta_label}', flush=True)
    else:
        print('[DAG CSE] compiler active, awaiting first checkpoint',flush=True)


def run_compile(exp:Experiment,gguf:Path,args):
    exp.event('compile_started',gguf=str(gguf))
    log=exp.logs/'compile.stdout.log';err=exp.logs/'compile.stderr.log'
    cmd=[sys.executable,str(Path(__file__).with_name('whole_model_dag.py')),
         '--gguf',str(gguf),'--out-dir',str(exp.out/'compiled'),
         '--rows',str(args.rows),'--cols',str(args.cols),
         '--max-nodes',str(args.max_nodes)]
    if args.max_work_units is not None:cmd.extend(['--max-work-units',str(args.max_work_units)])
    start=time.monotonic()
    with log.open('w',encoding='utf-8') as stdout, err.open('w',encoding='utf-8') as stderr:
        process=subprocess.Popen(cmd,stdout=stdout,stderr=stderr)
        with MemoryMonitor(args.interval,target_pid=process.pid) as monitor:
            last_print=time.monotonic()
            last_count=-1
            try:
                while True:
                    try:
                        returncode=process.wait(timeout=5)
                        break
                    except subprocess.TimeoutExpired:
                        now=time.monotonic()
                        if now-last_print < 15:
                            continue
                        event=_latest_compile_event(exp.out/'compiled'/'compile_events.jsonl')
                        done=event.get('completed',event.get('already_completed',0)) if event else 0
                        if event and (done != last_count or now-last_print >= 30):
                            _emit_compile_progress(event,now-start)
                            last_count=done
                        elif now-last_print >= 30:
                            print('[DAG CSE] compiler active; awaiting next checkpoint',flush=True)
                        last_print=now
            except KeyboardInterrupt:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                raise
    summary_path=exp.out/'compiled'/'full_model_report.json'
    data=json.loads(summary_path.read_text(encoding='utf-8')) if summary_path.exists() else {}
    info={'status':'completed' if returncode==0 else 'failed',
          'returncode':returncode,'wall_seconds':round(time.monotonic()-start,3),
          'compiler_report':data,'host_memory_during_compile':monitor.summary(),
          'stdout_path':str(log),'stderr_path':str(err)}
    with (exp.logs/'compile.memory.jsonl').open('w',encoding='utf-8') as f:
        for record in monitor.samples:f.write(json.dumps(record)+'\n')
    exp.phase('compile',info)
    return info


def verify_dag_runtime(base:str,source_sha:str):
    try:
        cap=http_get_json(base,'/dag/capabilities',timeout=10)
    except (OSError,ValueError,urllib.error.URLError) as exc:
        raise RuntimeError(f'Optimized endpoint does not advertise a DAG runtime: {exc}') from exc
    if (cap.get('backend') != 'algebraic_dag' or not cap.get('ready') or
        cap.get('source_gguf_sha256')!=source_sha or
        cap.get('artifact_format')!='adagstream-v1'):
        raise RuntimeError('Optimized endpoint lacks verified DAG identity or source snapshot checksum')
    return cap


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',default='qwen2.5-coder:3b')
    p.add_argument('--gguf',help='Use direct GGUF path instead of Ollama model resolution')
    p.add_argument('--pull',action='store_true')
    p.add_argument('--ollama-url',default='http://127.0.0.1:11434')
    p.add_argument('--models-dir')
    p.add_argument('--out-dir',default='dag_results/ab_run')
    p.add_argument('--prompt',default='Explain Java volatile and happens-before in 80 words.')
    p.add_argument('--prompt-file')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--num-predict',type=int,default=128)
    p.add_argument('--num-ctx',type=int,default=4096)
    p.add_argument('--interval',type=float,default=.5)
    p.add_argument('--rows',type=int,default=32)
    p.add_argument('--cols',type=int,default=256)
    p.add_argument('--max-nodes',type=int,default=16)
    p.add_argument('--max-work-units',type=int,help='Explicit PARTIAL smoke test; default compiles all eligible tensors')
    p.add_argument('--optimized-url',help='DAG-aware runtime base URL, NOT ordinary Ollama')
    p.add_argument('--optimized-model',help='Model ID served by DAG-aware runtime')
    args=p.parse_args(argv)
    if args.interval<.1 or args.num_predict<1 or args.num_ctx<1:p.error('Invalid monitoring/generation settings')
    if bool(args.optimized_url)!=bool(args.optimized_model):p.error('Specify both --optimized-url and --optimized-model')
    prompt=Path(args.prompt_file).read_text(encoding='utf-8') if args.prompt_file else args.prompt
    options={'temperature':0,'seed':args.seed,'num_predict':args.num_predict,'num_ctx':args.num_ctx}
    out=Path(args.out_dir)
    exp=Experiment(out,{'model':args.model,'prompt':prompt,'generation_options':options,
                        'compile':{'rows':args.rows,'cols':args.cols,'max_nodes':args.max_nodes,
                                   'max_work_units':args.max_work_units},
                        'baseline_url':args.ollama_url,'optimized_url':args.optimized_url})
    try:
        if args.pull:
            exp.event('pull_started',model=args.model)
            post_json(args.ollama_url,'/api/pull',{'model':args.model,'stream':True})
            exp.event('pull_completed')
        gguf=Path(args.gguf) if args.gguf else resolve_ollama_blob(args.model,args.models_dir,args.ollama_url)
        (exp.logs/'prompt.txt').write_text(prompt,encoding='utf-8')
        exp.report['source_gguf']=str(gguf)
        exp.event('baseline_started')
        baseline=run_generation(exp,'baseline',args.ollama_url,args.model,prompt,options,args.interval)
        # Explicitly unload baseline before compiling or checking the optimized runtime.
        try:
            post_json(args.ollama_url,'/api/generate',{'model':args.model,'keep_alive':0,'stream':False},timeout=90)
            exp.event('baseline_unloaded')
        except (OSError,RuntimeError,ValueError) as exc:
            exp.event('baseline_unload_warning',message=str(exc))
        compilation=run_compile(exp,gguf,args)
        if compilation['status']!='completed':
            exp.report['status']='compile_failed'
        elif not compilation['compiler_report'].get('complete'):
            exp.report['status']='compile_incomplete'
        elif not args.optimized_url:
            exp.report['status']='blocked_no_dag_inference_runtime'
            exp.report['optimized_inference']={'status':'not_run','reason':
                'Compiler writes a DAG stream but Ollama cannot execute it. Run-time integration required.'}
        else:
            cap=verify_dag_runtime(args.optimized_url,compilation['compiler_report']['source_sha256'])
            exp.report['optimized_runtime_capabilities']=cap
            optimized=run_generation(exp,'optimized',args.optimized_url,args.optimized_model,prompt,options,args.interval)
            a=baseline['metrics'];b=optimized['metrics']
            exp.report['ab_comparison']={
                'baseline_tokens_per_second':a['tokens_per_second'],
                'optimized_tokens_per_second':b['tokens_per_second'],
                'throughput_speedup':b['tokens_per_second']/a['tokens_per_second'] if a['tokens_per_second'] and b['tokens_per_second'] else None,
                'baseline_client_ttft_seconds':a['client_ttft_seconds'],
                'optimized_client_ttft_seconds':b['client_ttft_seconds'],
                'same_text_sha256':baseline['text_sha256']==optimized['text_sha256'],
                'baseline_model_vram_bytes':(baseline.get('ollama_model_residency') or {}).get('size_vram'),
                'optimized_runtime_reported_model_vram_bytes':(optimized.get('ollama_model_residency') or {}).get('size_vram'),
                'note':'Same prompt and decode options; full numerical/logits equivalence NOT established merely by matching text.'}
            exp.report['status']='ab_completed'
    except KeyboardInterrupt:
        exp.report['status']='interrupted'
        exp.report['error']='Interrupted. Compiler checkpoint retained for resume.'
        exp.event('experiment_interrupted')
    except Exception as exc:
        exp.report['status']='failed'
        exp.report['error']=str(exc)
        exp.event('experiment_error',message=str(exc))
    finally:
        exp.report['finished_utc']=iso_now()
        exp.save()
        print(json.dumps({'status':exp.report['status'],'report':str(exp.path),
              'phases':list(exp.report['phases'])},indent=2))
    if exp.report['status']=='interrupted':
        return 130
    if exp.report['status'] in ('failed','compile_failed'):
        return 1
    if exp.report['status']!='ab_completed':
        return 2  # Incomplete genuine A/B is visible to CI/shell as non-success.
    return 0


if __name__=='__main__':
    sys.exit(main())
