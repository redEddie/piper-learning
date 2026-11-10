#!/usr/bin/env python3
"""Hash existing large weights and write a resumable SFTP batch, without copying/downloading."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import re


def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()


def quote(s):
    s=str(s)
    if '\n' in s or '\r' in s:raise ValueError('Newline in transfer path')
    return '"'+s.replace('\\','\\\\').replace('"','\\"')+'"'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source',type=Path,default=Path('/data/models'))
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--remote-root',default='robot-models/20261021')
    args=ap.parse_args();root=PurePosixPath(args.remote_root)
    if root.is_absolute() or '..' in root.parts or not re.fullmatch(r'[A-Za-z0-9_./-]+',str(root)):
        ap.error('remote-root must be a safe path relative to the SFTP home')
    args.output.mkdir(parents=True,exist_ok=False)
    files={};sources={}
    for p in sorted(args.source.rglob('*')):
        rel=p.relative_to(args.source)
        if any(x in ['.cache','.git','__pycache__'] for x in rel.parts) or not p.is_file():continue
        files[str(rel)]={'sha256':sha(p),'bytes':p.stat().st_size};sources[str(rel)]=p.resolve()
    manifest=args.output/'models.json'
    manifest.write_text(json.dumps({'schema_version':1,'files':files,'source':str(args.source),
        'downloads_performed':False},indent=2)+'\n')
    dirs={root}
    for rel in files:
        p=root/PurePosixPath(rel)
        dirs.update(d for d in p.parents if str(d)!='.')
    cmds=['-mkdir '+quote(p) for p in sorted(dirs,key=lambda x:(len(x.parts),str(x)))]
    cmds += ['put -a '+quote(sources[rel])+' '+quote(root/PurePosixPath(rel)) for rel in files]
    # Metadata is small and uploaded fresh rather than resumed.
    cmds += ['put '+quote(manifest.resolve())+' '+quote(root/'models.json'),
             'put '+quote(Path(__file__).with_name('verify_models.py'))+' '+quote(root/'verify_models.py')]
    (args.output/'models.sftp').write_text('\n'.join(cmds)+'\n')
    print(json.dumps({'files':len(files),'bytes':sum(x['bytes'] for x in files.values()),
                      'batch':str(args.output/'models.sftp'),'remote_root':str(root)}),flush=True)


if __name__=='__main__':main()
