"""The Numba path must generate the very same DAG as reference greedy CSE."""
import numpy as np
import pytest
from algebraic_dag import extract_shared_expressions, compile_tiled
from cse_accelerator import NUMBA_AVAILABLE

@pytest.mark.skipif(not NUMBA_AVAILABLE,reason='Numba unavailable')
@pytest.mark.parametrize('shape',[(1,1),(1,32),(3,8),(16,32),(32,32)])
@pytest.mark.parametrize('kind',['random','low_entropy','zero'])
@pytest.mark.parametrize('window',[1,4,8])
def test_numba_dag_identical(shape,kind,window):
    rng=np.random.default_rng(9012+shape[0]*97+window)
    if kind=='random': w=rng.integers(-8,8,size=shape,dtype=np.int64)
    elif kind=='low_entropy': w=rng.integers(-2,3,size=shape,dtype=np.int64)
    else: w=np.zeros(shape,dtype=np.int64)
    a=extract_shared_expressions(w,engine='python',pair_window=window,max_nodes=16)
    b=extract_shared_expressions(w,engine='numba',pair_window=window,max_nodes=16)
    assert a==b

@pytest.mark.skipif(not NUMBA_AVAILABLE,reason='Numba unavailable')
def test_repeated_substitutions_and_ties():
    rng=np.random.default_rng(17)
    for _ in range(25):
        w=rng.integers(-8,8,size=(16,32),dtype=np.int64)
        assert extract_shared_expressions(w,engine='python')==extract_shared_expressions(w,engine='numba')

@pytest.mark.skipif(not NUMBA_AVAILABLE,reason='Numba unavailable')
def test_full_tile_compiler_and_hashcons(monkeypatch):
    rng=np.random.default_rng(33)
    w=rng.integers(-8,8,size=(48,96),dtype=np.int64)
    monkeypatch.setenv('DAG_CSE_ENGINE','python')
    a=compile_tiled(w,tile_rows=24,tile_cols=32,max_nodes=16)
    monkeypatch.setenv('DAG_CSE_ENGINE','numba')
    b=compile_tiled(w,tile_rows=24,tile_cols=32,max_nodes=16)
    assert a.nodes==b.nodes and a.outputs==b.outputs
    assert np.array_equal(b.reconstruct(),w)

def test_python_fallback(monkeypatch):
    monkeypatch.setenv('DAG_CSE_ENGINE','python')
    w=np.array([[1,2],[2,4]],dtype=np.int64)
    assert np.array_equal(compile_tiled(w).reconstruct(),w)

def test_reject_bad_engine(monkeypatch):
    monkeypatch.setenv('DAG_CSE_ENGINE','typo')
    with pytest.raises(ValueError,match='Unknown CSE engine'):
        extract_shared_expressions(np.zeros((2,3),dtype=np.int64))

@pytest.mark.skipif(not NUMBA_AVAILABLE,reason='Numba unavailable')
def test_checkpoint_resume_python_to_numba(tmp_path,monkeypatch):
    from test_ollama_dag_pipeline import fake_gguf
    from whole_model_dag import compile_full
    source,_=fake_gguf(tmp_path,2,rows=4,cols=256)
    output=tmp_path/'compiled'
    monkeypatch.setenv('DAG_CSE_ENGINE','python')
    a=compile_full(source,output,rows=2,cols=256,max_nodes=8,max_work_units=1)
    assert a['work_units_completed']==1
    monkeypatch.setenv('DAG_CSE_ENGINE','numba')
    b=compile_full(source,output,rows=2,cols=256,max_nodes=8)
    assert b['complete'] and b['work_units_completed']==2

def test_compile_event_tail(tmp_path):
    import json
    from ab_experiment import _latest_compile_event
    f=tmp_path/'compile_events.jsonl'
    f.write_text('\n'.join(json.dumps(x) for x in [
        {'event':'compile_started','already_completed':0,'total_units':40},
        {'event':'compile_progress','completed':10,'total':40},
        {'event':'compile_progress','completed':20,'total':40}
    ])+'\n',encoding='utf-8')
    assert _latest_compile_event(f)['completed']==20
