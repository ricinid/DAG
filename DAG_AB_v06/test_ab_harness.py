"""End-to-end harness mechanics against a *mock* llama HTTP server, not llama.cpp inference."""
import json
import os
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parent.parent
if (REPO/'whole_model_dag.py').is_file():
    sys.path.insert(0, str(REPO))
else:
    sys.path.insert(0, '/mnt/data/dag_v04')
    sys.path.insert(0, '/mnt/data/dag_cse_fast')
from test_ollama_dag_pipeline import fake_gguf
from whole_model_dag import compile_full
from run_llama_ab import main

FAKE_SERVER = '''#!/usr/bin/env python3
import argparse,json,os,time,sys
from http.server import BaseHTTPRequestHandler,ThreadingHTTPServer
p=argparse.ArgumentParser(add_help=False)
p.add_argument('--port',type=int,default=19081)
a,_=p.parse_known_args()
class H(BaseHTTPRequestHandler):
 def log_message(self,*a):pass
 def do_GET(self):
  self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers()
  self.wfile.write(b'{"status":"ok"}')
 def do_POST(self):
  n=int(self.headers.get('Content-Length','0'))
  body=json.loads(self.rfile.read(n))
  self.send_response(200);self.send_header('Content-Type','text/event-stream');self.end_headers()
  self.wfile.write(b'data: {"content":"hello","tokens":[123],"stop":false}\\n\\n')
  self.wfile.write(b'data: {"content":" world","tokens":[124],"stop":false}\\n\\n')
  self.wfile.write(b'data: {"content":"","stop":true,"timings":{"predicted_n":2,"predicted_per_second":18,"prompt_n":5,"prompt_per_second":20}}\\n\\n')
  self.wfile.flush()
  if os.environ.get('DAG_STREAM_DIR'):
   print('[DAG native] STATS '+json.dumps({'gemv_calls':99,'cache_hits':10}),file=sys.stderr,flush=True)
ThreadingHTTPServer(('127.0.0.1',a.port),H).serve_forever()
'''


def test_full_four_stage_harness_and_ratios(tmp_path):
    gguf, _ = fake_gguf(tmp_path, 2, rows=3, cols=256)
    compiled = tmp_path/'compiled'
    compile_full(gguf, compiled, rows=3, cols=256, max_nodes=2)
    f = tmp_path/'fake-server.py'
    f.write_text(FAKE_SERVER)
    out = tmp_path/'ab'
    assert main(['--server',str(f),'--gguf',str(gguf),'--compiled',str(compiled),
                 '--out-dir',str(out),'--port','19987','--trials','2',
                 '--modes','baseline','stream50','stream25','stream10']) == 0
    report=json.loads((out/'full_inference_ab.json').read_text())
    assert report['status']=='complete'
    assert set(report['phases']) == {'baseline','stream50','stream25','stream10'}
    for mode in ['stream50','stream25','stream10']:
        d=report['phases'][mode]
        assert d['dag_native_stats']['gemv_calls']==99
        assert d['relative_to_baseline']['same_token_ids_first_trial']
    assert report['phases']['stream50']['cache_budget_bytes']==gguf.stat().st_size//2
    assert report['phases']['stream10']['cache_budget_bytes']==int(gguf.stat().st_size*.1)