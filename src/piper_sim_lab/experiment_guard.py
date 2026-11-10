"""Read-only GPU occupancy guard: never signals or changes another process."""
import os
import subprocess


def compute_pids():
    raw = subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader,nounits"], text=True)
    return sorted({int(x.strip()) for x in raw.splitlines() if x.strip().isdigit()} - {os.getpid()})


def require_idle_gpu():
    pids = compute_pids()
    if pids:
        raise RuntimeError(f"GPU compute processes still active: {pids}; no training started")
