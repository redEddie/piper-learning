#!/usr/bin/env python3
"""Read-only, standard-library probe. Does not install packages or command hardware."""
import argparse
import glob
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess


def command(argv):
    if not shutil.which(argv[0]):return {'available':False}
    try:
        p=subprocess.run(argv,capture_output=True,text=True,timeout=8)
        return {'returncode':p.returncode,'stdout':p.stdout[:5000].strip(),'stderr':p.stderr[:1000].strip()}
    except (OSError,subprocess.TimeoutExpired) as e:return {'error':type(e).__name__}


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--output',type=Path);args=ap.parse_args()
    packages={}
    for name in ['numpy','torch','torchvision','tensorrt','jax','jaxlib','piper-sdk']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    disk=shutil.disk_usage(Path.home())
    r={'schema_version':1,'machine':platform.machine(),'kernel':platform.release(),
       'python':platform.python_version(),'os_release':Path('/etc/os-release').read_text() if Path('/etc/os-release').exists() else None,
       'nv_tegra_release':Path('/etc/nv_tegra_release').read_text() if Path('/etc/nv_tegra_release').exists() else None,
       'jetpack_packages':command(['dpkg-query','-W','-f=${Package} ${Version}\n','nvidia-jetpack','nvidia-l4t-core','libnvinfer*']),
       'cuda':command(['nvcc','--version']),
       'gpu':command(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader']),
       'packages':packages,'home_disk_free_gib':disk.free/1024**3,
       'camera_nodes':glob.glob('/dev/video*'),
       'can_interfaces':[x.name for x in Path('/sys/class/net').glob('*') if x.name.startswith('can')],
       'checks_pending':['JetPack-compatible perception runtime','camera calibration/base frame','policy-only parity/latency on target','robot command adapter'],
       'hardware_commands_sent':False}
    text=json.dumps(r,indent=2,ensure_ascii=False)+'\n'
    if args.output:args.output.write_text(text)
    print(text,end='')


if __name__=='__main__':main()
