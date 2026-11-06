#!/usr/bin/env python3
"""Read existing A/B JSON reports and locate first token or text divergence.

Legacy reports only have token hashes, not streamed token IDs; only the new
harness version can report a token index. This tool never infers token index
from a character diff. Runs offline; no server or compiled model needed.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def compare(a: dict, b: dict) -> dict:
    a_ids, b_ids = a.get('token_ids'), b.get('token_ids')
    a_text, b_text = a.get('text',''), b.get('text','')
    char_index = next((i for i, (x,y) in enumerate(zip(a_text,b_text)) if x != y),
                      min(len(a_text), len(b_text)))
    out = {
        'reference_text_length':len(a_text),
        'candidate_text_length':len(b_text),
        'matching_prefix_characters':char_index,
        'same_text':a_text == b_text,
        'same_token_hash':a.get('token_ids_sha256') == b.get('token_ids_sha256')
                           if a.get('token_ids_sha256') and b.get('token_ids_sha256') else None,
    }
    if (isinstance(a_ids,list) and isinstance(b_ids,list) and
            a.get('token_ids_complete') and b.get('token_ids_complete')):
        i = next((i for i,(x,y) in enumerate(zip(a_ids,b_ids)) if x != y),
                 min(len(a_ids),len(b_ids)))
        out.update({
            'token_comparison':'equal' if i==len(a_ids)==len(b_ids) else 'different',
            'baseline_ids':len(a_ids),'candidate_ids':len(b_ids),
            'matching_prefix_tokens':i,
            'first_mismatch_token_index':None if i==len(a_ids)==len(b_ids) else i,
            'baseline_token_id':a_ids[i] if i<len(a_ids) else None,
            'candidate_token_id':b_ids[i] if i<len(b_ids) else None,
        })
    else:
        out['token_comparison']='unavailable: legacy report or incomplete SSE token capture'
        out['captured_token_counts']=[
            len(a_ids) if isinstance(a_ids,list) else None,
            len(b_ids) if isinstance(b_ids,list) else None
        ]
    start=max(0,char_index-40)
    out['reference_text_at_divergence']=a_text[start:char_index+80]
    out['candidate_text_at_divergence']=b_text[start:char_index+80]
    return out


def gather(path: Path) -> dict:
    report=json.loads(path.read_text(encoding='utf-8'))
    result={}
    trials={mode:details['trials'][0] for mode,details in report['phases'].items()
            if details.get('trials')}
    if 'baseline' in trials:
        for mode,trial in trials.items():
            if mode!='baseline':
                result[f'baseline vs {mode}']=compare(trials['baseline'],trial)
    return {'source':str(path),'comparisons':result,'trials':trials}


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('reports',type=Path,nargs='+',help='One or more full_inference_ab.json reports')
    args=p.parse_args(argv)
    sources=[gather(p) for p in args.reports]
    output={'within_reports':[{k:v for k,v in src.items() if k!='trials'} for src in sources]}
    if len(sources)>1:
        candidates=[]
        for src in sources:
            modes=[m for m in src['trials'] if m!='baseline']
            if len(modes)==1:
                candidates.append((src['source'],src['trials'][modes[0]]))
        if len(candidates)==2:
            output['cross_report_dag']={
                'reference':candidates[0][0],'candidate':candidates[1][0],
                'comparison':compare(candidates[0][1],candidates[1][1])}
    print(json.dumps(output,ensure_ascii=False,indent=2))
    return 0


if __name__=='__main__':
    raise SystemExit(main())
