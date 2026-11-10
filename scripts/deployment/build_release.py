#!/usr/bin/env python3
"""Transport-independent release: Git bundles, actor artifacts, checksums; no venvs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import tarfile
import tempfile

ROOT=Path(__file__).resolve().parents[2]


def named(value):
    name,sep,path=value.partition('=')
    if not sep or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',name):
        raise argparse.ArgumentTypeError('Expected safe-name=/absolute/path')
    p=Path(path).resolve()
    if not p.is_dir():raise argparse.ArgumentTypeError(f'Not a directory: {p}')
    return name,p


def git(repo,*args):
    return subprocess.check_output(['git','-C',str(repo),*args],text=True).strip()


def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def copytree(src,dst):
    if any(p.is_symlink() for p in src.rglob('*')):raise ValueError(f'Symlink in artifact directory: {src}')
    shutil.copytree(src,dst,ignore=shutil.ignore_patterns('__pycache__','*.pyc','*.egg-info','.venv','.cache'))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--release-id',required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repo',type=named,action='append',required=True)
    ap.add_argument('--policy',type=named,action='append',default=[])
    ap.add_argument('--artifact',type=named,action='append',default=[])
    args=ap.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*',args.release_id):ap.error('Unsafe release-id')
    for group in [args.repo,args.policy,args.artifact]:
        if len({n for n,_ in group})!=len(group):ap.error('Duplicate names')
    if git(ROOT,'status','--porcelain'):raise RuntimeError('Commit deployment source before building release')
    for _,repo in args.repo:
        if git(repo,'status','--porcelain'):raise RuntimeError(f'Dirty repository: {repo}')
    args.output.mkdir(parents=True,exist_ok=True)
    archive=args.output/f'{args.release_id}.tar.gz'
    if archive.exists():raise FileExistsError(archive)
    with tempfile.TemporaryDirectory(prefix='piper-release-') as tmp:
        root=Path(tmp);(root/'repos').mkdir();repos={}
        for name,repo in args.repo:
            branch=git(repo,'symbolic-ref','--short','HEAD');commit=git(repo,'rev-parse','HEAD')
            git(repo,'bundle','create',str(root/f'repos/{name}.bundle'),f'refs/heads/{branch}')
            repos[name]={'branch':branch,'commit':commit}
        import piper_runtime,piper_policy_runtime
        from piper_runtime.identity import identity
        for package in (piper_runtime,piper_policy_runtime):
            copytree(Path(package.__file__).parent,root/'runtime/src'/package.__name__)
        (root/'runtime/runtime-identity.json').write_text(json.dumps(identity(),indent=2)+'\n')
        (root/'runtime/pyproject.toml').write_text('[build-system]\nrequires=["setuptools>=77"]\nbuild-backend="setuptools.build_meta"\n[project]\nname="piper-runtime"\nversion="0.4.0"\nrequires-python=">=3.11,<3.13"\ndependencies=["numpy>=2,<3"]\n[tool.setuptools.packages.find]\nwhere=["src"]\n[tool.setuptools.package-data]\npiper_policy_runtime=["*.cpp"]\npiper_runtime=["data/*","data/notices/*"]\n')
        (root/'tools').mkdir()
        for name in ['thor_preflight.py','verify_release.py']:
            shutil.copy2(ROOT/'scripts/deployment'/name,root/'tools'/name)
        for name,source in args.policy:
            dest=root/'policies'/name;dest.mkdir(parents=True)
            for file in ['policy.json','weights.npz','golden.npz','export-validation.json']:
                shutil.copy2(source/file,dest/file)
        for name,source in args.artifact:copytree(source,root/'artifacts'/name)
        # Models are explicit bulk transfers, not downloaded or silently added.
        inventory=[]
        modelroot=Path('/data/models')
        for source in sorted(modelroot.glob('*')):
            if source.is_dir():
                files=[p for p in source.rglob('*') if p.is_file()]
                inventory.append({'source':str(source),'files':len(files),'bytes':sum(p.stat().st_size for p in files),
                                  'included_in_release':False,'checksums_computed':False})
        (root/'large-model-inventory.json').write_text(json.dumps(inventory,indent=2)+'\n')
        sources=root/'third-party-sources';sources.mkdir()
        for source in list(modelroot.glob('*.SOURCE.txt')) + list(Path('/data/projects/one2many/third_party').glob('*.SOURCE.txt')):
            label=('model-' if source.parent==modelroot else 'code-')+source.name
            shutil.copy2(source,sources/label)
        pins={}
        for source in Path('/data/projects/one2many/third_party').glob('*'):
            if source.is_dir() and (source/'.git').exists():
                pins[source.name]={'commit':git(source,'rev-parse','HEAD'),
                    'submodules':git(source,'submodule','status'), 'working_directory':str(source),
                    'included_as_source_bundle':False}
        (sources/'pins.json').write_text(json.dumps(pins,indent=2)+'\n')
        files={str(p.relative_to(root)):digest(p) for p in sorted(root.rglob('*')) if p.is_file()}
        manifest={'schema_version':1,'release_id':args.release_id,'repos':repos,'files':files,
            'policies':[n for n,_ in args.policy], 'target':'Jetson Thor aarch64; exact JetPack/CUDA pending preflight',
            'transport':'SFTP initially; same immutable format for GitHub Releases/object storage later',
            'not_included':['x86 venv/conda','CUDA/JIT caches','secrets','large perception model weights'],
            'activation':'manual only after target verification; no hardware commands'}
        (root/'release.json').write_text(json.dumps(manifest,indent=2)+'\n')
        with tarfile.open(archive,'w:gz',compresslevel=3) as tar:
            for p in sorted(root.rglob('*')):
                if p.is_file():tar.add(p,arcname=str(p.relative_to(root)),recursive=False)
    Path(str(archive)+'.sha256').write_text(f'{digest(archive)}  {archive.name}\n')
    shutil.copy2(ROOT/'scripts/deployment/verify_release.py',args.output/'verify_release.py')
    shutil.copy2(ROOT/'scripts/deployment/thor_preflight.py',args.output/'thor_preflight.py')
    def q(path):
        return '"'+str(path.resolve()).replace('\\','\\\\').replace('"','\\"')+'"'
    batch=['-mkdir "robot-incoming"']
    for path in [archive,Path(str(archive)+'.sha256'),args.output/'verify_release.py',args.output/'thor_preflight.py']:
        command='put -a' if path==archive else 'put'
        batch.append(command+' '+q(path)+' "robot-incoming/'+path.name+'"')
    (args.output/'upload-release.sftp').write_text('\n'.join(batch)+'\n')
    print(json.dumps({'archive':str(archive),'bytes':archive.stat().st_size,'repos':repos,
                      'policies':manifest['policies'],'large_weights_included':False}))


if __name__=='__main__':main()
