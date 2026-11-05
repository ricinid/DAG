#!/usr/bin/env python3
"""Stream all supported GGUF weight matrices into a reversible arithmetic DAG stream.

IMPORTANT: This is a *compiler and research artifact*, not an Ollama runtime.
Every supported rank-2 Q4_0/Q4_K tensor is visited. Unsupported tensors stay
referenced from the immutable source GGUF; the output is NOT standalone or a
model that can be loaded by Ollama/llama.cpp. No speed/VRAM reduction is claimed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import struct
import time
from datetime import datetime, timezone
from pathlib import Path

from ollama_dag_pipeline import GGUFReader, Q_TYPES, artifact_bytes, compile_slice, load_artifact, sha256_file

MAGIC = b'ADAGSTREAM\x01'


def atomic_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def log_event(fp, event: str, **fields) -> None:
    data = {'timestamp_utc': datetime.now(timezone.utc).isoformat(), 'event': event, **fields}
    fp.write(json.dumps(data, ensure_ascii=False) + '\n')
    fp.flush()


def eligible(t):
    return len(t.dims) == 2 and t.dtype in Q_TYPES and t.dims[0] % Q_TYPES[t.dtype][1] == 0


def work_units(reader: GGUFReader, rows: int, cols: int):
    for t in reader.tensors:
        if not eligible(t):
            continue
        block = Q_TYPES[t.dtype][1]
        width = (max(block, cols) // block) * block
        for row0 in range(0, t.dims[1], rows):
            rr = min(rows, t.dims[1] - row0)
            for col0 in range(0, t.dims[0], width):
                cc = min(width, t.dims[0] - col0)
                yield t, row0, rr, col0, cc


def compile_full(source: Path, out: Path, *, rows: int = 32, cols: int = 256,
                 max_nodes: int = 16, pair_window: int = 4,
                 resume: bool = True, verify: bool = True,
                 max_work_units: int | None = None) -> dict:
    """Incremental compile with a checkpointed binary stream and JSONL index.

    Running a model with billions of parameters can be extremely expensive.
    Resume safely from a committed work-unit boundary; all matrices are included
    by default (no implicit row/column/tensor sampling). The compiler never
    silently re-quantizes weights or holds the entire model in RAM.
    """
    if rows < 1 or cols < 32 or max_nodes < 0 or pair_window < 1:
        raise ValueError('Invalid compiler arguments')
    if max_work_units is not None and max_work_units < 1:
        raise ValueError('max_work_units must be >= 1')
    source = Path(source).resolve()
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    stream_path = out / 'weights.adagstream'
    index_path = out / 'weights.index.jsonl'
    checkpoint = out / 'compiler_state.json'
    summary_path = out / 'full_model_report.json'
    events_path = out / 'compile_events.jsonl'
    source_sha = sha256_file(source)
    config = {'source_size': source.stat().st_size, 'source_sha256': source_sha,
              'rows_per_unit': rows, 'cols_per_unit': cols, 'max_nodes': max_nodes,
              'pair_window': pair_window, 'format': 'adagstream-v1'}
    with GGUFReader(source) as reader:
        eligible_tensors = [t for t in reader.tensors if eligible(t)]
        unsupported = [t for t in reader.tensors if not eligible(t)]
        total_units = sum(((t.dims[1]+rows-1)//rows)*((t.dims[0]+max(Q_TYPES[t.dtype][1], cols)-1)//max(Q_TYPES[t.dtype][1],cols)) for t in eligible_tensors)
        # The index can be large; do not load it in memory. Verify it sequentially.
        if checkpoint.exists():
            if not resume:
                raise FileExistsError('An existing compiler checkpoint is present; use --resume or a new output folder')
            if json.loads(checkpoint.read_text()) != config:
                raise ValueError('Source/model/compiler settings changed since checkpoint; cannot resume')
        else:
            if stream_path.exists() or index_path.exists():
                raise RuntimeError('Incomplete output without compiler_state.json; select a fresh output directory')
            atomic_json(checkpoint, config)
            stream_path.write_bytes(MAGIC)
            index_path.write_text('', encoding='utf-8')
        completed = 0
        committed_end = len(MAGIC)
        if not stream_path.exists() or not index_path.exists():
            raise RuntimeError('Checkpoint exists but stream/index is missing')
        with stream_path.open('r+b') as fstream, index_path.open('r', encoding='utf-8') as findex:
            if fstream.read(len(MAGIC)) != MAGIC:
                raise ValueError('Invalid stream format')
            for line in findex:
                entry = json.loads(line)
                if entry['sequence'] != completed or entry['offset'] != committed_end:
                    raise ValueError('Corrupt index sequence or stream offsets')
                if entry['size'] < 0:
                    raise ValueError('Negative record size')
                fstream.seek(entry['offset'])
                record_len_raw = fstream.read(8)
                if len(record_len_raw) != 8 or struct.unpack('<Q',record_len_raw)[0] != entry['size']:
                    raise ValueError('Index/stream size mismatch')
                if verify:
                    h=hashlib.sha256(); remaining=entry['size']
                    while remaining:
                        data = fstream.read(min(1 << 20, remaining))
                        if not data: raise ValueError('Truncated stream')
                        h.update(data); remaining-=len(data)
                    if h.hexdigest()!=entry['artifact_sha256']:
                        raise ValueError('Index/stream artifact checksum mismatch')
                committed_end=entry['offset']+8+entry['size']
                completed+=1
            fstream.seek(0,os.SEEK_END)
            if fstream.tell() < committed_end:
                raise ValueError('Truncated stream')
            # Crash between a binary write and committing the JSONL index: discard orphan bytes.
            fstream.truncate(committed_end)
        elapsed_start=time.monotonic()
        new_units=0
        original_bytes_compiled=0
        dag_bytes_compiled=0
        shared_nodes=0
        with events_path.open('a', encoding='utf-8') as events, index_path.open('a', encoding='utf-8') as idx, stream_path.open('ab') as output:
            log_event(events, 'compile_started', already_completed=completed, total_units=total_units,
                      supported_tensors=len(eligible_tensors), unsupported_tensors=len(unsupported))
            for sequence, (tensor,r0,rr,c0,cc) in enumerate(work_units(reader,rows,cols)):
                if sequence < completed:
                    continue
                if max_work_units is not None and new_units >= max_work_units:
                    break
                compiled, regions, compile_seconds = compile_slice(reader,tensor,r0,rr,c0,cc,
                      tile_rows=rr, max_nodes=max_nodes, pair_window=pair_window)
                blob=artifact_bytes(compiled)
                raw_selected=b''.join(reader.mm[start:start+n] for start,n in regions)
                if verify and load_artifact(blob).source_raw()!=raw_selected:
                    raise AssertionError(f'Lossless reconstruction failed: {tensor.name} [{r0},{c0}]')
                offset=output.tell()
                output.write(struct.pack('<Q',len(blob)))
                output.write(blob)
                output.flush()
                entry={'sequence':sequence,'tensor':tensor.name,'row_start':r0,'rows':rr,
                       'col_start':c0,'cols':cc,'offset':offset,'size':len(blob),
                       'original_region_bytes':sum(size for _,size in regions),
                       'artifact_sha256':hashlib.sha256(blob).hexdigest(),
                       'shared_nodes':compiled.info['total_shared_nodes'],
                       'compile_seconds':round(compile_seconds,6)}
                idx.write(json.dumps(entry,separators=(',',':'))+'\n')
                idx.flush()
                new_units+=1
                original_bytes_compiled+=entry['original_region_bytes']
                dag_bytes_compiled+=len(blob)
                shared_nodes+=entry['shared_nodes']
                if new_units%10==0 or sequence+1==total_units:
                    log_event(events,'compile_progress',completed=sequence+1,total=total_units,
                              elapsed_seconds=round(time.monotonic()-elapsed_start,2))
            finished=completed+new_units==total_units
            log_event(events,'compile_finished' if finished else 'compile_checkpoint',
                      completed=completed+new_units,total=total_units)
        report={'schema_version':1,'source_path':str(source),'source_sha256':source_sha,
                'original_gguf_bytes':source.stat().st_size,'total_tensors':len(reader.tensors),
                'eligible_q4_matrix_tensors':len(eligible_tensors),
                'passthrough_tensors_count':len(unsupported),
                'passthrough_tensor_names':[t.name for t in unsupported],
                'work_units_total':total_units,'work_units_completed':completed+new_units,
                'complete':finished,'new_work_units_this_invocation':new_units,
                'new_compiled_original_region_bytes':original_bytes_compiled,
                'new_dag_payload_bytes':dag_bytes_compiled,
                'new_shared_nodes':shared_nodes,'total_stream_file_bytes':stream_path.stat().st_size,
                'compile_wall_seconds_this_invocation':round(time.monotonic()-elapsed_start,3),
                'weight_values_preserved':bool(verify),
                'representation':'DAG stream + index + original GGUF for unsupported tensors',
                'runtime_executable_by_ollama':False,
                'inference_ab_possible_without_dag_runtime':False,
                'warning':'This is NOT an executable model. No RAM/VRAM/tokens-per-second inference reduction can be inferred.'}
        atomic_json(summary_path,report)
        return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--gguf',required=True)
    p.add_argument('--out-dir',required=True)
    p.add_argument('--rows',type=int,default=32)
    p.add_argument('--cols',type=int,default=256)
    p.add_argument('--max-nodes',type=int,default=16)
    p.add_argument('--pair-window',type=int,default=4)
    p.add_argument('--no-resume',action='store_true')
    p.add_argument('--no-verify',action='store_true')
    p.add_argument('--max-work-units',type=int,help='Explicit partial smoke test (default: entire model)')
    a=p.parse_args(argv)
    result=compile_full(Path(a.gguf),Path(a.out_dir),rows=a.rows,cols=a.cols,
                        max_nodes=a.max_nodes,pair_window=a.pair_window,
                        resume=not a.no_resume,verify=not a.no_verify,
                        max_work_units=a.max_work_units)
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__=='__main__':
    main()
