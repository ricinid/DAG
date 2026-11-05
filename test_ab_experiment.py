import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from ab_experiment import main, streamed_generation
from test_ollama_dag_pipeline import fake_gguf


class FakeOllama(BaseHTTPRequestHandler):
    def log_message(self,fmt,*args):
        pass
    def do_GET(self):
        if self.path=='/api/ps':
            result={'models':[{'model':'qwen2.5-coder:3b','size_vram':1048576}]}
        else:
            self.send_error(404);return
        blob=json.dumps(result).encode()
        self.send_response(200);self.send_header('Content-Type','application/json');self.end_headers();self.wfile.write(blob)
    def do_POST(self):
        n=int(self.headers['Content-Length']);request=json.loads(self.rfile.read(n))
        if self.path!='/api/generate':self.send_error(404);return
        self.send_response(200);self.send_header('Content-Type','application/x-ndjson');self.end_headers()
        if request.get('keep_alive')==0:
            self.wfile.write(json.dumps({'done':True}).encode()+b'\n');return
        self.wfile.write(json.dumps({'response':'Java ','done':False}).encode()+b'\n')
        self.wfile.write(json.dumps({'response':'volatile','done':False}).encode()+b'\n')
        self.wfile.write(json.dumps({'response':'','done':True,'eval_count':2,
                          'eval_duration':100000000,'prompt_eval_count':5,
                          'prompt_eval_duration':50000000,'total_duration':160000000}).encode()+b'\n')


@pytest.fixture
def fake_ollama():
    server=ThreadingHTTPServer(('127.0.0.1',0),FakeOllama)
    thread=threading.Thread(target=server.serve_forever,daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}'
    finally:
        server.shutdown(); server.server_close();thread.join(timeout=2)


def test_streamed_generation(fake_ollama):
    result=streamed_generation(fake_ollama,'qwen2.5-coder:3b','prompt',
         {'temperature':0,'seed':42,'num_predict':2})
    assert result['text']=='Java volatile'
    assert result['metrics']['tokens_per_second']==20
    assert result['metrics']['client_ttft_seconds'] is not None


def test_ab_blocked_without_real_dag_backend(tmp_path,fake_ollama):
    src,_=fake_gguf(tmp_path,2,rows=2,cols=256)
    out=tmp_path/'exp'
    ret=main(['--model','qwen2.5-coder:3b','--gguf',str(src),
              '--ollama-url',fake_ollama,'--out-dir',str(out),
              '--rows','2','--cols','256','--max-nodes','1','--interval','.1'])
    assert ret==2
    summary=json.loads((out/'ab_report.json').read_text())
    assert summary['status']=='blocked_no_dag_inference_runtime'
    assert summary['phases']['baseline']['metrics']['tokens_per_second']==20
    assert summary['phases']['compile']['compiler_report']['complete'] is True
    assert not (out/'logs'/'optimized.txt').exists()
    assert (out/'logs'/'baseline.memory.jsonl').exists()
    assert (out/'events.jsonl').exists()
    assert (out/'logs'/'baseline.txt').read_text()=='Java volatile'


def test_ab_rejects_partial_compiler(tmp_path,fake_ollama):
    src,_=fake_gguf(tmp_path,12,rows=4,cols=256)
    out=tmp_path/'exp'
    ret=main(['--model','qwen2.5-coder:3b','--gguf',str(src),
              '--ollama-url',fake_ollama,'--out-dir',str(out),
              '--rows','2','--cols','256','--max-work-units','1','--max-nodes','1','--interval','.1'])
    assert ret==2
    assert json.loads((out/'ab_report.json').read_text())['status']=='compile_incomplete'
