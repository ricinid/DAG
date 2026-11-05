"""Offline synthetic tests: real-format GGUF Q4_0 and Q4_K packing and whole-file restore."""
import json
import struct
from pathlib import Path
import numpy as np
import pytest

from ollama_dag_pipeline import (GGUFReader, Q_TYPES, unpack_block, pack_block,
    model_to_manifest, resolve_ollama_blob, compile_slice, artifact_bytes,
    load_artifact, benchmark, pack_full, restore_full, scale_offset, q4k_scale_min)


def fake_gguf(tmp_path, dtype: int, rows=8, cols=256, structured=True):
    rng=np.random.default_rng(83)
    nblock=cols//Q_TYPES[dtype][1]
    raw=[]
    for r in range(rows):
        for b in range(nblock):
            meta=(np.float16(0.125).tobytes() if dtype==2 else
                  np.float16(0.125).tobytes()+np.float16(0.0625).tobytes()+bytes(range(1,13)))
            if structured:
                q=np.tile(np.array([1,2,3,-4,5,6,7,-8],dtype=np.int64),Q_TYPES[dtype][1]//8)
                q=q.reshape(Q_TYPES[dtype][4],32)
            else:
                q=rng.integers(-8,8,size=(Q_TYPES[dtype][4],32),dtype=np.int64)
            raw.append(pack_block(meta,q,dtype))
    raw=b''.join(raw)
    s=b'blk.0.ffn_gate.weight'
    # GGUF v3, no metadata, 1 tensor, 2 dimensions, offset zero.
    header=(b'GGUF'+struct.pack('<IQQ',3,1,0)+struct.pack('<Q',len(s))+s+
            struct.pack('<IQQIQ',2,cols,rows,dtype,0))
    header+=bytes((-len(header))%32)
    p=tmp_path/'synthetic.gguf'
    p.write_bytes(header+raw+b'OTHER_NON_TENSOR_TAIL')
    return p,raw


@pytest.mark.parametrize('dtype',[2,12])
@pytest.mark.parametrize('structured',[True,False])
def test_format_and_exact_roundtrip(tmp_path,dtype,structured):
    path,raw=fake_gguf(tmp_path,dtype,structured=structured)
    with GGUFReader(path) as gguf:
        t=gguf.select('ffn_gate')
        assert t.dims==(256,8)
        compiled,regions,elapsed=compile_slice(gguf,t,0,8,0,256,8,8,4)
        assert compiled.source_raw()==raw
        blob=artifact_bytes(compiled)
        restored=load_artifact(blob)
        assert restored.source_raw()==raw
        assert benchmark(compiled,3,2,raw)['max_absolute_error_vs_original_gguf']<1e-8
    pack=tmp_path/'full.dagpack'
    pack_full(path,regions,blob,pack)
    restored_file=tmp_path/'restored.gguf'
    restore_full(pack,restored_file)
    assert restored_file.read_bytes()==path.read_bytes()


def test_q4k_codes_scale_reference():
    meta=(np.float16(0.25).tobytes()+np.float16(0.125).tobytes()+bytes(range(1,13)))
    for g in range(8):
        sc,m=q4k_scale_min(meta[4:],g)
        a,b=scale_offset(meta,g,12)
        assert a==0.25*sc
        assert b==8*a-0.125*m


def test_model_manifest_resolver(tmp_path):
    root=tmp_path/'models'; manifest=root/model_to_manifest('qwen2.5-coder:3b')
    manifest.parent.mkdir(parents=True)
    blob=root/'blobs'/('sha256-'+('a'*64))
    blob.parent.mkdir()
    blob.write_bytes(b'GGUFwhatever')
    manifest.write_text(json.dumps({'layers':[
        {'mediaType':'application/vnd.ollama.image.template','digest':'sha256:'+'b'*64,'size':1000},
        {'mediaType':'application/vnd.ollama.image.model','digest':'sha256:'+'a'*64,'size':17}]}))
    assert resolve_ollama_blob('qwen2.5-coder:3b',str(root))==blob


def test_reject_misaligned_slice(tmp_path):
    path,_=fake_gguf(tmp_path,12)
    with GGUFReader(path) as gguf:
        with pytest.raises(ValueError,match='align'):
            gguf.regions(gguf.select(None),0,8,32,128)


def test_dotted_model_name_is_not_a_registry():
    assert model_to_manifest('qwen2.5-coder:3b') == Path(
        'manifests/registry.ollama.ai/library/qwen2.5-coder/3b')
    assert model_to_manifest('phi3.5:latest') == Path(
        'manifests/registry.ollama.ai/library/phi3.5/latest')
    assert model_to_manifest('registry.ollama.ai/library/qwen2.5-coder:3b') == Path(
        'manifests/registry.ollama.ai/library/qwen2.5-coder/3b')


def test_ollama_show_fallback_without_manifest(tmp_path,monkeypatch):
    blob=tmp_path/'model.gguf'
    blob.write_bytes(b'GGUF'+b'fake-content')
    def show(base, endpoint, body, timeout=30):
        assert endpoint == '/api/show'
        assert body == {'model':'qwen2.5-coder:3b'}
        return {'modelfile':f'FROM {blob}\n'}
    monkeypatch.setattr('ollama_dag_pipeline.post_json',show)
    assert resolve_ollama_blob('qwen2.5-coder:3b',str(tmp_path/'missing'),
                               'http://127.0.0.1:11434') == blob


def test_ollama_show_legacy_colon_blob_name(tmp_path,monkeypatch):
    blob=tmp_path/('sha256-'+'a'*64)
    blob.write_bytes(b'GGUF')
    legacy=str(blob).replace('sha256-', 'sha256:')
    monkeypatch.setattr('ollama_dag_pipeline.post_json',
        lambda *a,**kw:{'modelfile':f'FROM {legacy}\n'})
    assert resolve_ollama_blob('qwen2.5-coder:3b',str(tmp_path/'missing'),
                               'http://127.0.0.1:11434') == blob


def test_ollama_show_rejects_non_gguf(tmp_path,monkeypatch):
    blob=tmp_path/'bad.bin'
    blob.write_bytes(b'NOT_GGUF')
    monkeypatch.setattr('ollama_dag_pipeline.post_json',
        lambda *a,**kw:{'modelfile':f'FROM {blob}\n'})
    with pytest.raises(FileNotFoundError,match='--gguf'):
        resolve_ollama_blob('qwen2.5-coder:3b',str(tmp_path/'missing'),
                            'http://127.0.0.1:11434')
