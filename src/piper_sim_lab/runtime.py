"""Lightweight setup: call before importing JAX or MuJoCo."""
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
import subprocess
import threading
import time


def configure(platform="gpu"):
    # Keep a CPU device next to CUDA: the RuntimeSafety host callback (wrappers.py)
    # places its inputs on the CPU, and without it the real physics error is
    # masked by "failed to find a local CPU device".
    os.environ["JAX_PLATFORMS"] = "cuda,cpu" if platform == "gpu" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path(".cache/jax").resolve()))
    os.environ.setdefault("WARP_CACHE_PATH", str(Path(".cache/warp").resolve()))


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")


def provenance():
    import jax
    from piper_sim_lab.assets import verify_assets
    repo = Path(__file__).resolve().parents[2]
    try:
        commit = subprocess.check_output(['git', '-C', str(repo), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(['git', '-C', str(repo), 'status', '--porcelain'], text=True))
        git = dict(commit=commit, dirty=dirty)
    except (OSError, subprocess.CalledProcessError):
        git = None
    lock = repo / 'uv.lock'
    packages = {}
    for name in ("piper-sim-lab", "playground", "mujoco", "mujoco-mjx", "mujoco-warp", "warp-lang", "jax", "jaxlib", "brax", "flax", "optax"):
        packages[name] = importlib.metadata.version(name)
    sources = {}
    for name in ("playground", "brax"):
        raw = importlib.metadata.distribution(name).read_text("direct_url.json")
        sources[name] = json.loads(raw) if raw else None
    try:
        gpu_hardware = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader"], text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        gpu_hardware = None
    return {"git": git, "uv_lock_sha256": hashlib.sha256(lock.read_bytes()).hexdigest() if lock.is_file() else None,
            "python": platform.python_version(), "system": platform.platform(), "gpu_hardware": gpu_hardware,
            "packages": packages, "sources": sources, "devices": [str(d) for d in jax.devices()],
            "code_sha256": {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(Path(__file__).parent.glob("*.py"))},
            "assets": verify_assets(), "platform": jax.default_backend()}


class GpuSampler:
    """Sample board memory; includes display/other processes, not allocator peak."""
    def __init__(self):
        self.samples = []
        self._stop = threading.Event()

    def _sample(self):
        while not self._stop.is_set():
            try:
                raw = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL, timeout=3)
                for row in raw.splitlines():
                    index, memory, utilization = [int(x.strip()) for x in row.split(",")]
                    self.samples.append({"time": time.time(), "gpu_index": index,
                                         "board_memory_mib": memory, "utilization_percent": utilization})
            except (OSError, ValueError, subprocess.SubprocessError):
                pass
            self._stop.wait(0.5)

    def __enter__(self):
        self._thread = threading.Thread(target=self._sample, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *_):
        self._stop.set()
        self._thread.join(timeout=4)

    def report(self):
        return {"sample_interval_seconds": 0.5, "scope": "whole board, sampled, includes other processes",
                "peak_board_memory_mib": max((s["board_memory_mib"] for s in self.samples), default=None),
                "samples": self.samples}

