"""Portable inference parity and actual release corruption/path handling."""
import hashlib
import io
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys
import tarfile

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
verifier=runpy.run_path(str(ROOT/'scripts/deployment/verify_release.py'))


@pytest.mark.parametrize('name',['lstm128','matched-mlp','p10-mlp-67m'])
def test_export_replays_without_training_imports(name):
    exports=os.environ.get('PIPER_POLICY_EXPORTS')
    if not exports:pytest.skip('Set PIPER_POLICY_EXPORTS to exported checkpoint artifacts')
    env=dict(os.environ,PYTHONPATH=os.environ.get('PYTHONPATH',''))
    code='''import json,sys
from piper_policy_runtime.check import check
r=check(sys.argv[1]);assert not any(n in sys.modules for n in ['jax','mujoco','piper_sdk']);print(json.dumps(r))'''
    p=subprocess.run([sys.executable,'-c',code,str(Path(exports)/name)],env=env,capture_output=True,text=True,check=True)
    d=json.loads(p.stdout);assert d['max_action_abs_error']<1e-4
    assert d['robot_connection']=='none'


def test_invalid_inputs_do_not_change_policy_memory():
    exports=os.environ.get('PIPER_POLICY_EXPORTS')
    if not exports:pytest.skip('Needs exported LSTM artifact')
    from piper_policy_runtime import Policy
    policy=Policy(Path(exports)/'lstm128');before=(policy.c.copy(),policy.h.copy())
    with pytest.raises(ValueError):policy.step(np.full(48,np.nan))
    with pytest.raises(ValueError):policy.step(np.zeros(47))
    np.testing.assert_array_equal(policy.c,before[0]);np.testing.assert_array_equal(policy.h,before[1])


def test_modified_weights_are_rejected(tmp_path):
    exports=os.environ.get('PIPER_POLICY_EXPORTS')
    if not exports:pytest.skip('Needs exported LSTM artifact')
    from piper_policy_runtime import Policy
    for name in ['weights.npz','policy.json']:shutil.copy2(Path(exports)/'lstm128'/name,tmp_path/name)
    with (tmp_path/'weights.npz').open('ab') as f:f.write(b'tampered')
    with pytest.raises(ValueError,match='checksum'):Policy(tmp_path)


def test_release_file_corruption_is_detected(tmp_path):
    f=tmp_path/'model';f.write_bytes(b'original')
    manifest={'schema_version':1,'files':{'model':hashlib.sha256(f.read_bytes()).hexdigest()}}
    (tmp_path/'release.json').write_text(json.dumps(manifest))
    verifier['verify'](tmp_path)
    f.write_bytes(b'changed')
    with pytest.raises(ValueError,match='Checksum'):verifier['verify'](tmp_path)


def test_release_cannot_escape_extraction_directory(tmp_path):
    archive=tmp_path/'bad.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        info=tarfile.TarInfo('../escaped');info.size=4;tar.addfile(info,io.BytesIO(b'nope'))
    Path(str(archive)+'.sha256').write_text(hashlib.sha256(archive.read_bytes()).hexdigest())
    with pytest.raises(ValueError,match='Unsafe'):verifier['stage'](archive,tmp_path/'release')
    assert not (tmp_path/'escaped').exists()
    assert not (tmp_path/'release').exists()
