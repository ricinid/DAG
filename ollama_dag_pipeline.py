#!/usr/bin/env python3
"""Ollama GGUF -> exact packed-code DAG -> measured .adag (+ optional whole-model .dagpack).

This is a research microbenchmark, NOT an Ollama inference accelerator. It
compiles unchanged Q4_0 and Q4_K quantization *codes* into exact-integer DAGs,
retaining all original quantization metadata byte-for-byte. On restore, it
reproduces the original GGUF bytes, including the selected tensor region.
Numeric projection execution is mathematically equivalent over real numbers,
not necessarily bitwise equal to llama.cpp's floating-point accumulation.

Requirements: numpy, algebraic_dag.py in same directory. No gguf library needed.

Examples:
  python ollama_dag_pipeline.py run --model qwen2.5-coder:3b-instruct-q4_0 --pull --rows 16 --cols 256
  python ollama_dag_pipeline.py run --model qwen2.5-coder:3b --pull --rows 16 --cols 256 --pack-full
  python ollama_dag_pipeline.py inspect --model qwen2.5-coder:3b --limit 20
  python ollama_dag_pipeline.py restore --pack results/model.dagpack --out restored.gguf
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import mmap
import os
import struct
import sys
import time
import urllib.error
import urllib.request
import zipfile
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from algebraic_dag import AlgebraicDAG, Node, compile_tiled

Q_TYPES = {2: ('Q4_0', 32, 18, 2, 1), 12: ('Q4_K', 256, 144, 16, 8)}
GGUF_TYPES = {0:4, 1:2, 2:18, 12:144}


def jprint(d: dict) -> None:
    print(json.dumps(d, ensure_ascii=False, indent=2))


def post_json(base: str, endpoint: str, body: dict, *, timeout: int = 3600) -> Any:
    url = base.rstrip('/') + endpoint
    request = urllib.request.Request(url, json.dumps(body).encode('utf-8'),
                                     {'Content-Type': 'application/json'})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if endpoint != '/api/pull':
                return json.load(response)
            last_percentage = -10
            success = False
            for line in response:
                if not line.strip():
                    continue
                event = json.loads(line)
                if 'error' in event:
                    raise RuntimeError(event['error'])
                total = event.get('total', 0)
                done = event.get('completed', 0)
                if total:
                    percentage = int(done * 100 / total)
                    if percentage >= last_percentage + 10 or percentage == 100:
                        print(f'pull: {event.get("status", "download")} {percentage}%', flush=True)
                        last_percentage = percentage
                elif event.get('status') in ('pulling manifest','verifying sha256 digest','success'):
                    print('pull:', event['status'], flush=True)
                if event.get('status') == 'success':
                    success = True
            if not success:
                raise RuntimeError('Ollama pull stream ended without success')
            return {'status':'success'}
    except urllib.error.URLError as exc:
        raise RuntimeError(f'Ollama API unavailable at {url}: {exc}. Start `ollama serve`.') from exc


def model_to_manifest(model: str) -> Path:
    # Public registry default; supports explicit domain and namespaces.
    if '@' in model:
        raise ValueError('Use a tagged Ollama model name; digest notation not supported in local manifest resolver')
    base, tag = (model.rsplit(':', 1) if ':' in model else (model, 'latest'))
    parts = base.split('/')
    if not base or any(p in ('', '.', '..') for p in parts) or '/' in tag or tag in ('', '.', '..'):
        raise ValueError('Invalid model or tag')
    if '.' in parts[0] or ':' in parts[0]:
        registry = parts.pop(0)
    else:
        registry = 'registry.ollama.ai'
    if len(parts) == 1:
        parts.insert(0, 'library')
    return Path('manifests', registry, *parts, tag)


def model_roots(specified: str | None) -> list[Path]:
    if specified:
        return [Path(specified).expanduser()]
    env = os.environ.get('OLLAMA_MODELS')
    if env:
        return [Path(env).expanduser()]
    return [Path.home()/'.ollama'/'models', Path('/usr/share/ollama/.ollama/models')]


def resolve_ollama_blob(model: str, specified: str | None = None) -> Path:
    manifest = model_to_manifest(model)
    for root in model_roots(specified):
        path = root / manifest
        if not path.is_file():
            continue
        data = json.loads(path.read_text(encoding='utf-8'))
        model_layers = [l for l in data.get('layers', []) if 'image.model' in l.get('mediaType', '')]
        if not model_layers:
            raise RuntimeError(f'{path} has no model layer; multimodal or unsupported Ollama manifest')
        layer = max(model_layers, key=lambda l: l.get('size', 0))
        digest = layer['digest']
        if not digest.startswith('sha256:'):
            raise ValueError('Only SHA256 Ollama blob digests supported')
        blob = root/'blobs'/('sha256-'+digest.split(':',1)[1])
        if blob.is_file():
            return blob
        raise FileNotFoundError(f'Ollama manifest resolved but blob missing: {blob}')
    raise FileNotFoundError(f'Cannot find local Ollama manifest for {model!r}. Pull it first, '
                            'or set --models-dir / OLLAMA_MODELS to the service model root.')


@dataclass(frozen=True)
class Tensor:
    name: str
    dims: tuple[int, ...]  # GGUF convention: [ne0=inputs, ne1=outputs]
    dtype: int
    relative_offset: int


class GGUFReader:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.fd = self.path.open('rb')
        self.mm = mmap.mmap(self.fd.fileno(), 0, access=mmap.ACCESS_READ)
        self.pos = 0
        if self._read(4) != b'GGUF':
            self.close()
            raise ValueError('Not GGUF (first four bytes must be GGUF)')
        version = self._num('<I')
        if version not in (2,3):
            self.close()
            raise ValueError(f'Unsupported GGUF version {version}')
        count, kv_count = self._num('<Q'), self._num('<Q')
        if count > 2_000_000 or kv_count > 1_000_000:
            raise ValueError('Unreasonable GGUF record count')
        alignment = 32
        for _ in range(kv_count):
            key = self._string()
            typ = self._num('<I')
            val = self._value(typ, key == 'general.alignment')
            if key == 'general.alignment':
                alignment = int(val)
        if alignment < 1 or alignment > 1<<20 or alignment & (alignment-1):
            raise ValueError('Invalid GGUF tensor alignment')
        self.tensors = []
        for _ in range(count):
            name = self._string()
            nd = self._num('<I')
            if nd < 1 or nd > 8:
                raise ValueError('Invalid GGUF tensor rank')
            dims = tuple(self._num('<Q') for _ in range(nd))
            dtype = self._num('<I')
            offset = self._num('<Q')
            self.tensors.append(Tensor(name,dims,dtype,offset))
        self.data_offset = (self.pos + alignment - 1) // alignment * alignment
        if self.data_offset > len(self.mm):
            raise ValueError('GGUF data start out of file')

    def close(self):
        if getattr(self, 'mm', None) is not None:
            self.mm.close()
        self.fd.close()

    def __enter__(self): return self
    def __exit__(self, *args): self.close()

    def _read(self, n):
        if n < 0 or self.pos+n > len(self.mm):
            raise ValueError('Truncated or invalid GGUF')
        out = self.mm[self.pos:self.pos+n]
        self.pos += n
        return out

    def _num(self, fmt): return struct.unpack(fmt, self._read(struct.calcsize(fmt)))[0]
    def _string(self):
        n = self._num('<Q')
        if n > 1<<25:
            raise ValueError('GGUF string too long')
        return self._read(n).decode('utf-8')

    def _value(self, typ: int, keep: bool = False):
        formats = {0:'<B',1:'<b',2:'<H',3:'<h',4:'<I',5:'<i',6:'<f',7:'<?',10:'<Q',11:'<q',12:'<d'}
        if typ in formats:
            val = self._num(formats[typ]); return val if keep else None
        if typ == 8:
            val = self._string(); return val if keep else None
        if typ == 9:
            nested = self._num('<I'); length = self._num('<Q')
            if length > 100_000_000:
                raise ValueError('Unreasonable GGUF metadata array')
            if nested in formats:
                width = struct.calcsize(formats[nested]); self._read(length*width)
            elif nested == 8:
                for _ in range(length): self._string()
            else:
                raise ValueError(f'Unsupported nested GGUF type {nested}')
            return None
        raise ValueError(f'Unknown GGUF metadata type {typ}')

    def select(self, pattern: str | None) -> Tensor:
        eligible = [t for t in self.tensors if t.dtype in Q_TYPES and len(t.dims)==2 and
                    t.dims[0] % Q_TYPES[t.dtype][1] == 0]
        if pattern:
            matched = [t for t in eligible if pattern in t.name]
            if not matched:
                all_match = [t for t in self.tensors if pattern in t.name]
                raise ValueError(f'Tensor {pattern!r} not supported/found. Matching types: '+
                                 str([(t.name,t.dtype) for t in all_match[:12]]))
            return matched[0]
        # Prefer FFN projections over vocab/embedding so we exercise actual weights.
        return next((t for t in eligible if 'ffn_gate' in t.name),
                    next((t for t in eligible if 'attn_q' in t.name),
                         eligible[0] if eligible else self._no_supported()))

    @staticmethod
    def _no_supported():
        raise ValueError('No supported Q4_0/Q4_K rank-2 tensor found')

    def row_bytes(self, t: Tensor):
        _,block,bsize,_,_ = Q_TYPES[t.dtype]
        return (t.dims[0]//block)*bsize

    def regions(self, t: Tensor, row_start: int, rows: int, col_start: int, cols: int) -> list[tuple[int,int]]:
        _,block,bsize,_,_ = Q_TYPES[t.dtype]
        if not (0 <= row_start < t.dims[1] and rows > 0 and row_start+rows <= t.dims[1]):
            raise ValueError('Invalid row selection')
        if not (0 <= col_start < t.dims[0] and cols > 0 and col_start+cols <= t.dims[0]
                and col_start % block == 0 and cols % block == 0):
            raise ValueError(f'Column start/length must fit tensor and align to GGUF block size {block}')
        nblocks=cols//block
        rowbytes=self.row_bytes(t)
        regions = [(self.data_offset+t.relative_offset+(row_start+r)*rowbytes+(col_start//block)*bsize,
                    nblocks*bsize) for r in range(rows)]
        if regions and (regions[0][0] < self.data_offset or regions[-1][0]+regions[-1][1] > len(self.mm)):
            raise ValueError('Tensor byte range exceeds GGUF blob')
        return regions


def metadata_len(dtype): return Q_TYPES[dtype][3]
def groups_per_block(dtype): return Q_TYPES[dtype][4]


def unpack_block(raw: bytes, dtype: int) -> tuple[bytes, np.ndarray]:
    _,block,bsize,header,groups=Q_TYPES[dtype]
    if len(raw) != bsize: raise ValueError('Bad quantized block')
    meta = raw[:header]
    qbytes = np.frombuffer(raw[header:], dtype=np.uint8)
    q = np.empty(block, dtype=np.int64)
    if dtype == 2:
        q[:16] = (qbytes & 15).astype(np.int64) - 8
        q[16:] = (qbytes >> 4).astype(np.int64) - 8
    else:
        for k in range(4):
            chunk = qbytes[k*32:(k+1)*32]
            q[k*64:k*64+32] = (chunk & 15).astype(np.int64) - 8
            q[k*64+32:(k+1)*64] = (chunk >> 4).astype(np.int64) - 8
    return meta,q.reshape(groups, 32)


def pack_block(meta: bytes, centered: np.ndarray, dtype: int) -> bytes:
    _,block,bsize,header,groups=Q_TYPES[dtype]
    c = np.asarray(centered, dtype=np.int64).reshape(groups,32)
    if np.any((c < -8) | (c > 7)) or len(meta)!=header:
        raise ValueError('Invalid centered codes or block header')
    q = (c+8).astype(np.uint8)
    if dtype == 2:
        body = q[0,:16] | (q[0,16:] << 4)
    else:
        body = np.concatenate([q[2*k] | (q[2*k+1] << 4) for k in range(4)])
    result = meta + body.tobytes()
    assert len(result)==bsize
    return result


def q4k_scale_min(codes: bytes, group: int) -> tuple[int,int]:
    if group < 4:
        return codes[group]&63, codes[group+4]&63
    return ((codes[group+4]&15)|((codes[group-4]>>6)<<4),
            ((codes[group+4]>>4)|((codes[group]>>6)<<4)))


def scale_offset(meta: bytes, group: int, dtype: int) -> tuple[float,float]:
    if dtype == 2:
        return float(np.frombuffer(meta[:2], dtype='<f2')[0]), 0.0
    d = float(np.frombuffer(meta[:2],dtype='<f2')[0])
    dm = float(np.frombuffer(meta[2:4],dtype='<f2')[0])
    sc,m = q4k_scale_min(meta[4:16],group)
    scale = d*sc
    return scale, 8*scale-dm*m


def serialize_dag(dag: AlgebraicDAG) -> bytes:
    stream = io.BytesIO()
    stream.write(struct.pack('<III',dag.n_inputs,dag.n_outputs,len(dag.nodes)))
    for node in dag.nodes:
        stream.write(struct.pack('<IIii',node.a,node.b,node.ca,node.cb))
    for row in dag.outputs:
        stream.write(struct.pack('<I',len(row)))
        for v,c in sorted(row.items()):
            stream.write(struct.pack('<Ii',v,c))
    return stream.getvalue()


def deserialize_dag(stream: io.BytesIO) -> AlgebraicDAG:
    def read(n):
        d=stream.read(n)
        if len(d)!=n: raise ValueError('Truncated DAG stream')
        return d
    n,m,nn=struct.unpack('<III',read(12))
    nodes=[Node(*struct.unpack('<IIii',read(16))) for _ in range(nn)]
    outs=[]
    for _ in range(m):
        ct=struct.unpack('<I',read(4))[0]
        outs.append({a:b for a,b in (struct.unpack('<Ii',read(8)) for _ in range(ct))})
    dag=AlgebraicDAG(n,nodes,outs)
    dag.validate()
    return dag


@dataclass
class CompiledSlice:
    info: dict
    meta: bytes
    dags: list[AlgebraicDAG]

    def source_raw(self) -> bytes:
        # Restores selected packed bytes without referencing the original GGUF.
        rows, cols, dtype = self.info['rows'], self.info['cols'], self.info['dtype']
        groups=groups_per_block(dtype); bs=Q_TYPES[dtype][1]
        blocks=cols//bs; hsize=metadata_len(dtype)
        expanded=[dag.reconstruct() for dag in self.dags]
        parts=[]
        for r in range(rows):
            for b in range(blocks):
                header=self.meta[(r*blocks+b)*hsize:(r*blocks+b+1)*hsize]
                q=np.stack([expanded[b*groups+g][r] for g in range(groups)])
                parts.append(pack_block(header,q,dtype))
        out=b''.join(parts)
        if hashlib.sha256(out).hexdigest()!=self.info['selected_sha256']:
            raise AssertionError('Reconstructed quantized bytes did not match selected source')
        return out

    def eval_grouped(self, x: np.ndarray) -> np.ndarray:
        x=np.asarray(x,dtype=np.int64)
        if x.ndim!=1 or len(x)!=self.info['cols']:
            raise ValueError('x must be 1-D integer vector of selected cols')
        rows, cols, dtype=self.info['rows'],self.info['cols'],self.info['dtype']
        bs=Q_TYPES[dtype][1]; groups=groups_per_block(dtype); nblocks=cols//bs
        hsize=metadata_len(dtype)
        acc=np.zeros(rows,dtype=np.float64)
        for b in range(nblocks):
            for g in range(groups):
                start=b*bs+g*32
                xs=x[start:start+32]
                t=self.dags[b*groups+g].eval_numpy(xs).astype(np.float64)
                sx=float(np.sum(xs))
                for r in range(rows):
                    meta=self.meta[(r*nblocks+b)*hsize:(r*nblocks+b+1)*hsize]
                    scale, offset=scale_offset(meta,g,dtype)
                    acc[r] += scale*float(t[r]) + offset*sx
        return acc

    def eval_grouped_torch(self, x, device='cuda'):
        """Unfused Torch baseline for optional CUDA memory profiling, not optimized inference."""
        import torch
        x=torch.as_tensor(x,dtype=torch.int64,device=device)
        if x.ndim!=1 or x.numel()!=self.info['cols']: raise ValueError('bad input')
        rows,cols,dtype=self.info['rows'],self.info['cols'],self.info['dtype']
        bs=Q_TYPES[dtype][1]; groups=groups_per_block(dtype); nb=cols//bs
        hsize=metadata_len(dtype)
        acc=torch.zeros(rows,dtype=torch.float64,device=device)
        for b in range(nb):
            for g in range(groups):
                xs=x[b*bs+g*32:b*bs+(g+1)*32]
                t=self.dags[b*groups+g].eval_torch(xs,device=device).to(torch.float64)
                sx=xs.sum().to(torch.float64)
                scales=[]; offsets=[]
                for r in range(rows):
                    meta=self.meta[(r*nb+b)*hsize:(r*nb+b+1)*hsize]
                    scale,offset=scale_offset(meta,g,dtype)
                    scales.append(scale); offsets.append(offset)
                acc += t*torch.tensor(scales,dtype=torch.float64,device=device) + sx*torch.tensor(offsets,dtype=torch.float64,device=device)
        return acc

    def eval_reference(self, x: np.ndarray) -> np.ndarray:
        x=np.asarray(x,dtype=np.int64)
        rows,cols,dtype=self.info['rows'],self.info['cols'],self.info['dtype']
        bs=Q_TYPES[dtype][1]; groups=groups_per_block(dtype); nblocks=cols//bs
        hsize=metadata_len(dtype)
        dag_matrices=[dag.reconstruct() for dag in self.dags]
        y=np.zeros(rows,dtype=np.float64)
        for b in range(nblocks):
            for g in range(groups):
                start=b*bs+32*g
                for r in range(rows):
                    meta=self.meta[(r*nblocks+b)*hsize:(r*nblocks+b+1)*hsize]
                    scale,offset=scale_offset(meta,g,dtype)
                    coef=(dag_matrices[b*groups+g][r].astype(np.float64)*scale+offset)
                    y[r] += float(coef@x[start:start+32].astype(np.float64))
        return y


def compile_slice(gguf: GGUFReader, t: Tensor, row_start: int, rows: int,
                  col_start: int, cols: int, tile_rows: int, max_nodes: int,
                  pair_window: int) -> tuple[CompiledSlice, list[tuple[int,int]], float]:
    regions=gguf.regions(t,row_start,rows,col_start,cols)
    dtype=t.dtype; block=Q_TYPES[dtype][1]; bsize=Q_TYPES[dtype][2]
    ng=groups_per_block(dtype); nb=cols//block
    cube=np.empty((nb*ng,rows,32),dtype=np.int64)
    headers=[]; rawhash=hashlib.sha256()
    start=time.perf_counter()
    for r,(offset,length) in enumerate(regions):
        raw=gguf.mm[offset:offset+length]
        rawhash.update(raw)
        for b in range(nb):
            meta, groups=unpack_block(raw[b*bsize:(b+1)*bsize],dtype)
            headers.append(meta)
            for g in range(ng): cube[b*ng+g,r]=groups[g]
    dags=[]
    for g in range(nb*ng):
        dags.append(compile_tiled(cube[g],tile_rows=tile_rows,tile_cols=32,
                                  max_nodes=max_nodes,pair_window=pair_window))
    compile_seconds=time.perf_counter()-start
    info={'version':1,'model_tensor':t.name,'tensor_dims':list(t.dims),
          'dtype':dtype,'dtype_name':Q_TYPES[dtype][0],
          'rows':rows,'cols':cols,'row_start':row_start,'col_start':col_start,
          'original_region_bytes':sum(length for _,length in regions),
          'selected_sha256':rawhash.hexdigest(),
          'max_nodes_per_group':max_nodes,'total_shared_nodes':sum(len(d.nodes) for d in dags),
          'total_dag_levels':sum(len(d.levels()) for d in dags),
          'compile_seconds':compile_seconds}
    result=CompiledSlice(info,b''.join(headers),dags)
    return result,regions,compile_seconds


def artifact_bytes(result: CompiledSlice) -> bytes:
    stream=io.BytesIO()
    with zipfile.ZipFile(stream,'w') as z:
        z.writestr('info.json',json.dumps(result.info,separators=(',',':')).encode(),compress_type=zipfile.ZIP_DEFLATED)
        z.writestr('meta.bin',result.meta,compress_type=zipfile.ZIP_DEFLATED)
        payload=io.BytesIO()
        payload.write(struct.pack('<I',len(result.dags)))
        for dag in result.dags:
            raw=serialize_dag(dag)
            payload.write(struct.pack('<I',len(raw)))
            payload.write(raw)
        z.writestr('dags.bin',zlib.compress(payload.getvalue(),9),compress_type=zipfile.ZIP_STORED)
    return stream.getvalue()


def load_artifact(content: bytes) -> CompiledSlice:
    with zipfile.ZipFile(io.BytesIO(content)) as z:
        info=json.loads(z.read('info.json'))
        meta=z.read('meta.bin')
        stream=io.BytesIO(zlib.decompress(z.read('dags.bin')))
    count=struct.unpack('<I',stream.read(4))[0]
    dags=[]
    for _ in range(count):
        n=struct.unpack('<I',stream.read(4))[0]
        dags.append(deserialize_dag(io.BytesIO(stream.read(n))))
    if stream.read(1): raise ValueError('Trailing bytes in DAG stream')
    expected=(info['cols']//Q_TYPES[info['dtype']][1])*groups_per_block(info['dtype'])
    if len(dags)!=expected: raise ValueError('Unexpected number of DAG groups')
    if len(meta)!=(info['cols']//Q_TYPES[info['dtype']][1])*info['rows']*metadata_len(info['dtype']):
        raise ValueError('Unexpected quantization metadata length')
    return CompiledSlice(info,meta,dags)


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(8*1024*1024),b''): h.update(part)
    return h.hexdigest()


def copy_bytes(src, dst, size: int) -> None:
    while size:
        part=src.read(min(size,8*1024*1024))
        if not part: raise IOError('Truncated source stream')
        dst.write(part)
        size-=len(part)


def pack_full(source: Path, regions: list[tuple[int,int]], artifact: bytes, dest: Path) -> None:
    """Pack an *entire* GGUF as rest-of-original + optimized slice; restoration is exact."""
    ordered=sorted(regions)
    current=0
    for start,n in ordered:
        if start<current: raise ValueError('Selected GGUF regions overlap')
        current=start+n
    header={'schema':1,'source_size':source.stat().st_size,'source_sha256':sha256_file(source),
            'regions':ordered}
    with zipfile.ZipFile(dest,'w',allowZip64=True) as z:
        z.writestr('pack.json',json.dumps(header),compress_type=zipfile.ZIP_STORED)
        z.writestr('optimized.adag',artifact,compress_type=zipfile.ZIP_STORED)
        with z.open('unchanged.gguf.bytes','w',force_zip64=True) as output, source.open('rb') as inp:
            pos=0
            for offset,length in ordered:
                copy_bytes(inp,output,offset-pos)
                inp.seek(length,1)
                pos=offset+length
            copy_bytes(inp,output,source.stat().st_size-pos)


def restore_full(pack: Path, dest: Path) -> None:
    if dest.exists(): raise FileExistsError(f'Output exists: {dest}')
    try:
        with zipfile.ZipFile(pack) as z:
            header=json.loads(z.read('pack.json'))
            result=load_artifact(z.read('optimized.adag'))
            replacement=result.source_raw()
            regions=[(int(a),int(b)) for a,b in header['regions']]
            if sum(length for _,length in regions)!=len(replacement):
                raise ValueError('DAG restoration size differs from packed regions')
            with z.open('unchanged.gguf.bytes') as inp, dest.open('wb') as out:
                cursor=0; rc=0
                for offset,length in regions:
                    copy_bytes(inp,out,offset-cursor)
                    out.write(replacement[rc:rc+length]); rc+=length
                    cursor=offset+length
                copy_bytes(inp,out,header['source_size']-cursor)
        if dest.stat().st_size!=header['source_size'] or sha256_file(dest)!=header['source_sha256']:
            raise AssertionError('Restored GGUF SHA256 mismatch')
    except Exception:
        if dest.exists(): dest.unlink()
        raise


def decode_raw_selection(raw: bytes, dtype: int, rows: int, cols: int) -> np.ndarray:
    """Independent packed-GGUF reference: decode original bytes, not our DAG."""
    _,bs,block_bytes,_,groups=Q_TYPES[dtype]
    count=cols//bs
    if len(raw)!=rows*count*block_bytes:
        raise ValueError('Wrong original selection payload size')
    dense=np.empty((rows,cols),dtype=np.float64)
    for r in range(rows):
        for b in range(count):
            pos=(r*count+b)*block_bytes
            meta,codes=unpack_block(raw[pos:pos+block_bytes],dtype)
            for g in range(groups):
                sc,ofs=scale_offset(meta,g,dtype)
                dense[r,b*bs+g*32:b*bs+(g+1)*32]=codes[g].astype(np.float64)*sc+ofs
    return dense


def benchmark(result: CompiledSlice, seed: int, repeat: int, raw_selection: bytes) -> dict:
    rng=np.random.default_rng(seed)
    x=rng.integers(-16,17,size=result.info['cols'],dtype=np.int64)
    original_matrix=decode_raw_selection(raw_selection, result.info['dtype'],
                                         result.info['rows'], result.info['cols'])
    reference=original_matrix@x.astype(np.float64)
    optimized=result.eval_grouped(x)
    max_error=float(np.max(np.abs(reference-optimized)))
    if not np.allclose(reference,optimized,rtol=1e-10,atol=1e-8):
        raise AssertionError('DAG vs original quantized GGUF matrix mismatch')
    def test(fn):
        fn(x)
        samples=[]
        for _ in range(repeat):
            start=time.perf_counter_ns(); fn(x)
            samples.append((time.perf_counter_ns()-start)/1e6)
        return float(np.median(samples))
    def dense_fn(inp):
        return original_matrix @ inp.astype(np.float64)
    def packed_fn(inp):
        matrix=decode_raw_selection(raw_selection,result.info['dtype'],
                                    result.info['rows'],result.info['cols'])
        return matrix@inp.astype(np.float64)
    return {'predecoded_dense_numpy_matvec_ms':test(dense_fn),
            'packed_gguf_decode_plus_numpy_matvec_ms':test(packed_fn),
            'dag_numpy_matvec_ms':test(result.eval_grouped),
            'max_absolute_error_vs_original_gguf':max_error,
            'numeric_semantics':'real-number equivalence with float64 accumulation; not bit-exact llama.cpp runtime',
            'measurement':'Python/NumPy microbenchmark of selected slice; NOT full-model tokens/s',
            'baseline_fairness':'Dense baseline assumes weights predecoded in RAM; packed baseline includes Python decoding; neither uses optimized llama.cpp kernels.'}


def ollama_residency(base,model):
    """Ollama-reported loaded model VRAM allocation, NOT actual peak device usage."""
    try:
        with urllib.request.urlopen(base.rstrip('/')+'/api/ps',timeout=15) as r:
            models=json.load(r).get('models',[])
    except (urllib.error.URLError,ValueError) as exc:
        return {'available':False,'reason':str(exc)}
    for m in models:
        if model in (m.get('name'),m.get('model')):
            return {'available':True,'loaded_model_bytes':m.get('size'),
                    'ollama_reported_vram_bytes':m.get('size_vram'),
                    'source':'Ollama /api/ps; not process peak VRAM'}
    return {'available':False,'reason':'Model not loaded in Ollama /api/ps'}

def benchmark_ollama(base: str, model: str, prompt: str, n_predict: int) -> dict:
    payload={'model':model,'prompt':prompt,'stream':False,
             'keep_alive':'5m','options':{'num_predict':n_predict,'temperature':0}}
    post_json(base,'/api/generate',payload,timeout=600)  # warmup
    result=post_json(base,'/api/generate',payload,timeout=600)
    count=result.get('eval_count',0); duration=result.get('eval_duration',0)
    return {'ollama_baseline_tokens_per_second':count*1e9/duration if duration else None,
            'eval_count':count,'eval_duration_ns':duration,
            'load_duration_ns':result.get('load_duration'),
            'prompt_eval_duration_ns':result.get('prompt_eval_duration'),
            'note':'Unmodified Ollama runtime only. Compiled DAG is NOT installed into Ollama.'}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    sub=p.add_subparsers(dest='command',required=True)
    for verb in ('inspect','run'):
        subp=sub.add_parser(verb)
        inp=subp.add_mutually_exclusive_group(required=True)
        inp.add_argument('--model',help='Local Ollama model, e.g. qwen2.5-coder:3b')
        inp.add_argument('--gguf',help='Direct GGUF path, no Ollama installation needed')
        subp.add_argument('--pull','--spool',dest='pull',action='store_true',help='Download --model using local Ollama API first')
        subp.add_argument('--ollama-url',default='http://localhost:11434')
        subp.add_argument('--models-dir',help='Ollama model root (or OLLAMA_MODELS)')
        subp.add_argument('--limit',type=int,default=25,help='Inspect: max tensors displayed')
        if verb=='run':
            subp.add_argument('--tensor',help='Substring of GGUF tensor name; default first FFN gate')
            subp.add_argument('--rows',type=int,default=16)
            subp.add_argument('--cols',type=int,default=256)
            subp.add_argument('--row-start',type=int,default=0)
            subp.add_argument('--col-start',type=int,default=0)
            subp.add_argument('--tile-rows',type=int,default=16)
            subp.add_argument('--max-nodes',type=int,default=16)
            subp.add_argument('--pair-window',type=int,default=4)
            subp.add_argument('--bench-repeat',type=int,default=5)
            subp.add_argument('--out-dir',default='dag_results')
            subp.add_argument('--pack-full',action='store_true',help='Write complete reversible .dagpack (size ~ whole GGUF)')
            subp.add_argument('--bench-ollama',action='store_true',help='Benchmark original Ollama inference separately')
            subp.add_argument('--profile-memory',action=argparse.BooleanOptionalAction,default=True)
            subp.add_argument('--profile-cuda',action=argparse.BooleanOptionalAction,default=True)
            subp.add_argument('--memory-repeat',type=int,default=8)
            subp.add_argument('--memory-interval-ms',type=int,default=20)
            subp.add_argument('--prompt',default='Explain the Java memory model briefly.')
            subp.add_argument('--num-predict',type=int,default=32)
    restore=sub.add_parser('restore',help='Restore byte-identical GGUF from complete .dagpack')
    restore.add_argument('--pack',required=True)
    restore.add_argument('--out',required=True)
    args=p.parse_args(argv)
    if args.command=='restore':
        restore_full(Path(args.pack),Path(args.out)); print('PASS: restored GGUF SHA256 identical')
        return
    if args.pull and not args.model: p.error('--pull requires --model')
    if args.pull: post_json(args.ollama_url,'/api/pull',{'model':args.model,'stream':True})
    source=Path(args.gguf) if args.gguf else resolve_ollama_blob(args.model,args.models_dir)
    with GGUFReader(source) as gguf:
        if args.command=='inspect':
            jprint({'model':args.model,'gguf':str(source),'gguf_bytes':source.stat().st_size,
                    'tensors_total':len(gguf.tensors),
                    'tensors':[{'name':t.name,'dims':list(t.dims),'ggml_type':t.dtype,
                                'quant_type':Q_TYPES[t.dtype][0] if t.dtype in Q_TYPES else 'unsupported'}
                               for t in gguf.tensors[:args.limit]]})
            return
        tensor=gguf.select(args.tensor)
        rows=min(args.rows,tensor.dims[1]-args.row_start)
        _,block,_,_,_=Q_TYPES[tensor.dtype]
        cols=min(args.cols,tensor.dims[0]-args.col_start)
        cols=cols//block*block
        if rows<=0 or cols<=0: raise ValueError('Selection contains no complete quantized blocks')
        compiled,regions,elapsed=compile_slice(gguf,tensor,args.row_start,rows,args.col_start,cols,
                                                args.tile_rows,args.max_nodes,args.pair_window)
        payload=artifact_bytes(compiled)
        recovered=load_artifact(payload).source_raw()
        if len(recovered)!=sum(n for _,n in regions): raise AssertionError('Reconstruction size mismatch')
        result_dir=Path(args.out_dir); result_dir.mkdir(parents=True,exist_ok=True)
        artifact=result_dir/'selected.adag'; artifact.write_bytes(payload)
        raw_selected=b''.join(gguf.mm[o:o+n] for o,n in regions)
        bench=benchmark(compiled,seed=17,repeat=max(1,args.bench_repeat),raw_selection=raw_selected)
        source_bytes=source.stat().st_size
        selected_bytes=compiled.info['original_region_bytes']
        report={'model':args.model,'source_gguf':str(source),'tensor':tensor.name,
                'quant_type':compiled.info['dtype_name'],'selection':[rows,cols],
                'original_full_gguf_bytes':source_bytes,
                'original_selected_region_bytes':selected_bytes,
                'compiled_selected_region_bytes':len(payload),
                'selected_region_compression_ratio':selected_bytes/len(payload),
                'selected_region_saved_bytes':selected_bytes-len(payload),
                'full_gguf_with_selection_replaced_estimate_bytes':source_bytes-selected_bytes+len(payload),
                'full_gguf_estimate_is_not_a_real_generated_file':True,
                'selected_region_bit_exact_reconstruction':True,
                'compiled_dag_groups':len(compiled.dags),'shared_nodes':compiled.info['total_shared_nodes'],
                'compile_seconds':elapsed,'benchmarks':bench,
                'dag_artifact':str(artifact),'limitations':['Single tensor slice only, not whole-model optimization',
                       'No Ollama runtime integration / no generated token speedup claim',
                       'Original packed quantization metadata retained byte-for-byte',
                       'Numerical matvec equivalence is not floating-point bitwise identity']}
        if args.pack_full:
            pack=result_dir/'model.dagpack'
            pack_full(source,regions,payload,pack)
            report['actual_full_dagpack_bytes']=pack.stat().st_size
            report['actual_full_snapshot_saved_bytes']=source_bytes-pack.stat().st_size
            report['full_dagpack_path']=str(pack)
            report['full_dagpack_restore_command']=f'python ollama_dag_pipeline.py restore --pack "{pack}" --out restored.gguf'
        if args.profile_memory:
            from memory_bench import profile_slice
            report['memory_profile']=profile_slice(str(source),str(artifact),tensor.name,
                args.row_start,rows,args.col_start,cols,args.tile_rows,args.max_nodes,
                args.pair_window,repeat=max(1,args.memory_repeat),interval_ms=args.memory_interval_ms,
                profile_cuda=args.profile_cuda)
        if args.bench_ollama:
            if not args.model: raise ValueError('--bench-ollama requires --model')
            report['ollama_inference_baseline']=benchmark_ollama(args.ollama_url,args.model,args.prompt,args.num_predict)
            report['ollama_model_residency']=ollama_residency(args.ollama_url,args.model)
        report_path=result_dir/'report.json'
        report_path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
        jprint(report)

if __name__=='__main__':
    main()
