#!/usr/bin/env python3
"""Verify release checksums and stage into a new directory; never activates/runs it."""
import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import tempfile


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def verify(directory):
    m=json.loads((directory/'release.json').read_text())
    if m['schema_version']!=1:raise ValueError('Unknown release schema')
    actual={str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file()}-{'release.json'}
    if actual!=set(m['files']):raise ValueError('Release file set mismatch')
    for name,expected in m['files'].items():
        p=PurePosixPath(name)
        if p.is_absolute() or '..' in p.parts:raise ValueError('Invalid manifest path')
        if digest(directory/name)!=expected:raise ValueError(f'Checksum mismatch: {name}')
    return m


def stage(archive,destination):
    expected=Path(str(archive)+'.sha256').read_text().split()[0]
    if digest(archive)!=expected:raise ValueError('Archive checksum mismatch')
    if destination.exists():raise FileExistsError('Destination already exists; choose a new release directory')
    destination.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.incoming-',dir=destination.parent) as tmp:
        root=Path(tmp)
        with tarfile.open(archive,'r:gz') as tar:
            seen=set()
            for item in tar:
                p=PurePosixPath(item.name)
                if p.is_absolute() or '..' in p.parts or not p.parts or item.name in seen:
                    raise ValueError('Unsafe or duplicate archive path')
                seen.add(item.name)
                if not (item.isdir() or item.isfile()):raise ValueError('Archive links/devices are unsupported')
                target=root/item.name
                if item.isdir():target.mkdir(parents=True,exist_ok=True)
                else:
                    target.parent.mkdir(parents=True,exist_ok=True)
                    with tar.extractfile(item) as src,target.open('xb') as dst:shutil.copyfileobj(src,dst)
        m=verify(root)
        root.rename(destination)
    return m


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('path',type=Path)
    ap.add_argument('--extract-to',type=Path);args=ap.parse_args()
    m=stage(args.path,args.extract_to) if args.extract_to else verify(args.path)
    print(json.dumps({'status':'verified','release_id':m['release_id'],'files':len(m['files']),
                      'activated':False,'repos':m['repos']}))


if __name__=='__main__':main()
