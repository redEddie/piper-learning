"""Read bundled, revision-pinned Menagerie assets without network access."""
import hashlib
import json
from pathlib import Path

ASSET_ROOT = Path(__file__).with_name("assets")


# "menagerie": the v0 model (Menagerie, old URDF geometry, 70 mm gripper).
# "piper_v100": built from the vendor URDF with the 100 mm gripper by
# scripts/build_piper_v100.py (corrected geometry, SHOW-76/84).
MODEL_DIRS = {"menagerie": "agilex_piper", "piper_v100": "agilex_piper_v100"}


def get_model_path(robot_model: str = "menagerie") -> Path:
    if robot_model not in MODEL_DIRS:
        raise ValueError(f"unknown robot_model {robot_model!r}; expected one of {sorted(MODEL_DIRS)}")
    return ASSET_ROOT / MODEL_DIRS[robot_model] / "scene.xml"


def verify_assets() -> dict:
    manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
    failures = []
    for name, expected in manifest["files"].items():
        path = ASSET_ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected["sha256"]:
            failures.append(name)
    if failures:
        raise ValueError(f"Missing or modified upstream assets: {failures}")
    return {"repository": manifest["repository"], "revision": manifest["revision"],
            "license": manifest["license"], "verified_files": len(manifest["files"])}


def main():
    print(json.dumps(verify_assets(), indent=2))

