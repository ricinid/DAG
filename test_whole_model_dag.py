import json
from pathlib import Path
import numpy as np
import pytest

from ollama_dag_pipeline import load_artifact
from whole_model_dag import compile_full, MAGIC
from test_ollama_dag_pipeline import fake_gguf

@pytest.mark.parametrize('dtype',[2,12])
def test_complete_stream_exact_and_checkpoint(tmp_path,dtype):
    src,raw=fake_gguf(tmp_path,dtype,rows=4,cols=256)
    out=tmp_path/'compiled'
    first=compile_full(src,out,rows=2,cols=256,max_nodes=2,max_work_units=1)
    assert first['complete'] is False
    assert first['work_units_completed']==1
    second=compile_full(src,out,rows=2,cols=256,max_nodes=2)
    assert second['complete'] is True
    assert second['work_units_total']==2
    assert second['work_units_completed']==2
    third=compile_full(src,out,rows=2,cols=256,max_nodes=2)
    assert third['complete'] is True
    assert third['new_work_units_this_invocation']==0
    entries=[json.loads(line) for line in (out/'weights.index.jsonl').read_text().splitlines()]
    assert len(entries)==2
    with (out/'weights.adagstream').open('rb') as f:
        assert f.read(len(MAGIC))==MAGIC
        for e in entries:
            f.seek(e['offset'])
            n=int.from_bytes(f.read(8),'little')
            assert n==e['size']
            assert load_artifact(f.read(n)).source_raw()==raw[e['row_start']*(len(raw)//4):
                           (e['row_start']+e['rows'])*(len(raw)//4)]


def test_refuse_conflicting_resume(tmp_path):
    src,_=fake_gguf(tmp_path,2,rows=2,cols=256)
    out=tmp_path/'compiled'
    compile_full(src,out,rows=2,cols=256,max_nodes=1)
    with pytest.raises(ValueError,match='changed'):
        compile_full(src,out,rows=1,cols=256,max_nodes=1)


def test_restore_bad_stream_fails_resume(tmp_path):
    src,_=fake_gguf(tmp_path,2,rows=2,cols=256)
    out=tmp_path/'compiled'
    compile_full(src,out,rows=2,cols=256,max_nodes=1)
    with (out/'weights.adagstream').open('r+b') as f:
        f.seek(-1,2)
        b=f.read(1)
        f.seek(-1,2)
        f.write(bytes([b[0]^1]))
    with pytest.raises(ValueError,match='checksum'):
        compile_full(src,out,rows=2,cols=256,max_nodes=1)
