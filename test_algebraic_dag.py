#!/usr/bin/env python3
"""Tests for algebraic_dag.py. Run: python -m pytest -q test_algebraic_dag.py"""
import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from algebraic_dag import AlgebraicDAG, TorchDAGExecutor, compile_tiled, generated_example


@pytest.mark.parametrize('shape', [(1,1),(8,16),(16,32),(33,65),(64,32)])
@pytest.mark.parametrize('kind', ['random','structured'])
def test_exact_integer_execution(shape, kind):
    rng=np.random.default_rng(17)
    w=generated_example(kind,*shape,rng)
    dag=compile_tiled(w,tile_rows=9,tile_cols=17,max_nodes=8)
    assert np.array_equal(dag.reconstruct(),w)
    for batch in [1,3,7]:
        x=rng.integers(-128,129,size=(batch,shape[1]),dtype=np.int64)
        ref=x@w.T
        assert np.array_equal(dag.eval_scalar(x),ref)
        assert np.array_equal(dag.eval_numpy(x),ref)


def test_scaling_common_expression():
    w=np.asarray([[1,2,1,0],[2,4,0,-1],[3,6,1,0]],dtype=np.int64)
    dag=compile_tiled(w,max_nodes=12,tile_rows=10,tile_cols=10)
    assert len(dag.nodes)>0
    assert np.array_equal(dag.reconstruct(),w)


def test_levels_and_serialization(tmp_path):
    w=np.array([[1,2,1,-3],[2,4,-1,2],[3,6,2,1]],dtype=np.int64)
    dag=compile_tiled(w,max_nodes=12)
    path=tmp_path/'small.npz'
    dag.save(str(path))
    loaded=AlgebraicDAG.load(str(path))
    assert loaded.summary()==dag.summary()
    assert loaded.nodes==dag.nodes
    assert loaded.outputs==dag.outputs
    assert np.array_equal(loaded.reconstruct(),w)
    levels=loaded.levels()
    seen=set(range(loaded.n_inputs))
    for level in levels:
        for v in level:
            n=loaded.nodes[v-loaded.n_inputs]
            assert n.a in seen and n.b in seen
        seen.update(level)


def test_hardwired_c99(tmp_path):
    w=np.asarray([[1,2,1,0],[2,4,0,-1],[3,6,1,0]],dtype=np.int64)
    dag=compile_tiled(w)
    source=tmp_path/'generated.c'
    library=tmp_path/'generated.so'
    dag.emit_c(str(source))
    try:
        subprocess.run(['cc','-O2','-shared','-fPIC','-std=c99',str(source),'-o',str(library)],
                       check=True,capture_output=True,text=True)
    except FileNotFoundError:
        pytest.skip('C compiler unavailable')
    f=ctypes.CDLL(str(library)).dag_kernel
    f.argtypes=[ctypes.POINTER(ctypes.c_int64),ctypes.POINTER(ctypes.c_int64)]
    rng=np.random.default_rng(3)
    for _ in range(20):
        x=rng.integers(-1000,1001,size=w.shape[1],dtype=np.int64)
        y=np.empty(w.shape[0],dtype=np.int64)
        f(x.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)),
          y.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)))
        assert np.array_equal(y,w@x)


def test_reject_float_or_out_of_range():
    with pytest.raises((TypeError,ValueError)):
        compile_tiled(np.array([[1.2,3.1]]))
    with pytest.raises(ValueError):
        compile_tiled(np.array([[12,3]],dtype=np.int64))


def test_optional_torch_cpu():
    torch=pytest.importorskip('torch')
    w=np.array([[1,2,1,0],[2,4,0,-1],[3,6,1,0]],dtype=np.int64)
    dag=compile_tiled(w)
    x=np.array([[1,2,3,4],[-3,5,0,2]],dtype=np.int64)
    assert np.array_equal(dag.eval_torch(x,device='cpu').cpu().numpy(),x@w.T)
    executor=TorchDAGExecutor(dag,device='cpu')
    assert np.array_equal(executor(x).numpy(),x@w.T)


def test_sparse_levels(tmp_path):
    w=np.array([[1,2,1,0],[2,4,0,-1],[3,6,1,0]],dtype=np.int64)
    dag=compile_tiled(w)
    layers=dag.sparse_layers()
    assert layers[-1]['depth']=='outputs'
    assert layers[-1]['shape']==[3,4+len(dag.nodes)]
    for layer in layers[:-1]:
        assert all(col<node for row,col,c in layer['coo']
                   for node in [layer['outputs'][row]])
    dst=tmp_path/'preview.dot'
    dag.emit_dot(str(dst))
    assert 'digraph AlgebraicDAG' in dst.read_text()