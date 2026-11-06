"""Offline regression for token-level parity diagnostics."""
import json

from compare_generation_parity import compare, gather, main


def trial(ids, text):
    return {'token_ids':ids,'token_ids_complete':True,'token_ids_sha256':'ignored','text':text}


def test_first_mismatch_at_token_and_character_level():
    d=compare(trial([2,4,6,8], 'once and now'),
              trial([2,4,7,8], 'once and then'))
    assert d['matching_prefix_tokens']==2
    assert d['first_mismatch_token_index']==2
    assert d['baseline_token_id']==6
    assert d['candidate_token_id']==7
    assert d['same_text'] is False
    assert d['matching_prefix_characters']==9


def test_legacy_hash_only_reports_are_honest():
    a={'text':'same','token_ids_sha256':'abc'}
    b={'text':'same','token_ids_sha256':'xyz'}
    d=compare(a,b)
    assert d['same_token_hash'] is False
    assert d['token_comparison'].startswith('unavailable')
    assert d['matching_prefix_characters']==4


def test_offline_reader(tmp_path,capsys):
    src=tmp_path/'full_inference_ab.json'
    report={'phases':{
        'baseline':{'trials':[trial([1,2,3], 'abc')]},
        'stream50':{'trials':[trial([1,2,4], 'abd')]}
    }}
    src.write_text(json.dumps(report),encoding='utf-8')
    assert main([str(src)])==0
    out=json.loads(capsys.readouterr().out)
    detail=out['within_reports'][0]['comparisons']['baseline vs stream50']
    assert detail['first_mismatch_token_index']==2
    assert detail['matching_prefix_characters']==2
