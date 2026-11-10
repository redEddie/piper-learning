#!/usr/bin/env python3
"""Check SFTP model transfers on the target using only the Python standard library."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('root',type=Path);args=ap.parse_args()
    m=json.loads((args.root/'models.json').read_text())
    if m['schema_version']!=1:raise ValueError('Unsupported manifest')
    failures=[]
    for name,expected in m['files'].items():
        path=PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts:raise ValueError('Invalid manifest path')
        p=args.root/name
        if not p.is_file() or p.stat().st_size!=expected['bytes']:
            failures.append(name);continue
        h=hashlib.sha256()
        with p.open('rb') as f:
            for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
        if h.hexdigest()!=expected['sha256']:failures.append(name)
    print(json.dumps({'status':'failed' if failures else 'verified','files':len(m['files']),'failures':failures}))
    raise SystemExit(1 if failures else 0)


if __name__=='__main__':main()
