import numpy as np
from ollama_dag_pipeline import GGUFReader,compile_slice,artifact_bytes
from memory_bench import profile_slice
from test_ollama_dag_pipeline import fake_gguf

def test_isolated_memory_profile(tmp_path):
    gguf,_=fake_gguf(tmp_path,2,rows=4,cols=256)
    with GGUFReader(gguf) as reader:
        t=reader.select('ffn_gate')
        compiled,_,_=compile_slice(reader,t,0,4,0,256,4,4,4)
    artifact=tmp_path/'test.adag'
    artifact.write_bytes(artifact_bytes(compiled))
    report=profile_slice(str(gguf),str(artifact),t.name,0,4,0,256,4,4,4,
                         repeat=2,profile_cuda=False)
    assert report['cuda_status']=='disabled'
    for mode in ('compile','dense','packed','dag'):
        m=report['metrics'][mode]
        assert m['peak_rss_bytes']>=m['baseline_rss_bytes']>0
        assert m['cuda_allocator'] is None
