"""Build the v1 Piper MJCF directly from the vendor URDF with the 100 mm gripper.

Source: agilexrobotics/piper_ros, branch noetic, commit ac41fcb,
piper_description/urdf/piper_description_v100.urdf. Its arm (link1-link5)
matches the corrected piper_description.urdf; only the flange/gripper differ
(100 mm gripper, fingertips at link6 z = 135.0 mm). Nothing is taken from the
Menagerie model.

What MuJoCo needs on top of the URDF is added here and recorded as simulation
choices: primitive collision geoms fitted to each link mesh, position
actuators, the finger coupling, joint armature/friction, gravity compensation,
joint limits (the narrower of the vendor URDFs) and a home keyframe (SHOW-84
pose A).

Usage:
    python scripts/build_piper_v100.py --source /data/projects/one2many/assets/robots/piper_description
"""
import argparse
import hashlib
import json
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import trimesh

OUT = Path(__file__).resolve().parents[1] / "src" / "piper_sim_lab" / "assets" / "agilex_piper_v100"
URDF = "urdf/piper_description_v100.urdf"
GRIPPER_DAE = "meshes/dae/gripper_base_100_v2.dae"
LINK_STLS = ["base_link", "link1", "link2", "link3", "link4", "link5", "link7", "link8"]

# Narrower of piper_description.urdf and piper_description_v100.urdf (same commit).
JOINT_RANGES = {
    "joint1": (-2.618, 2.618), "joint2": (0.0, 3.14), "joint3": (-2.697, 0.0),
    "joint4": (-1.745, 1.745), "joint5": (-1.22, 1.22), "joint6": (-2.094, 2.094),
    "joint7": (0.0, 0.05), "joint8": (-0.05, 0.0),
}
# Simulation choices kept from the v0 environment so the controller feels the same.
ARM_GAINS = {"joint1": (80, 5), "joint2": (80, 5), "joint3": (80, 5),
             "joint4": (40, 5), "joint5": (10, 1.5), "joint6": (10, 1.5)}
HOME_QPOS = [0.0, 0.523, -0.450, 0.0, 0.0, 0.0]            # SHOW-84 pose A
# Collision primitive per link mesh (MuJoCo fits it to the mesh at compile time).
COLLISION = {"base_link": "capsule", "link1": "capsule", "link2": "capsule", "link3": "capsule",
             "link4": "capsule", "link5": "capsule", "link6": "box"}
# Fingers taper from a wide base to a thin pad whose inner face meets the centre
# line when closed. One fitted box would overhang the pad, so each finger gets
# two link6-aligned boxes cut from its own mesh: the pad (link6 z >= 106 mm,
# fingertips at 135 mm) and the base (62-100 mm).
FINGER_PARTS = {"pad": (0.106, 0.140), "base": (0.062, 0.100)}


def dae_to_stl(dae: Path, stl: Path) -> None:
    """Convert a Blender COLLADA export (triangles + node matrices) to one STL in metres."""
    ns = {"c": "http://www.collada.org/2005/11/COLLADASchema"}
    root = ET.parse(dae).getroot()
    geoms = {}
    for g in root.iterfind(".//c:library_geometries/c:geometry", ns):
        mesh = g.find("c:mesh", ns)
        verts_src = mesh.find("c:vertices/c:input[@semantic='POSITION']", ns).get("source")[1:]
        pos = np.array(mesh.find(f"c:source[@id='{verts_src}']/c:float_array", ns).text.split(), float).reshape(-1, 3)
        faces = []
        for tri in mesh.iterfind("c:triangles", ns):
            inputs = tri.findall("c:input", ns)
            stride = max(int(i.get("offset")) for i in inputs) + 1
            voff = next(int(i.get("offset")) for i in inputs if i.get("semantic") == "VERTEX")
            p = np.array(tri.find("c:p", ns).text.split(), int).reshape(-1, 3, stride)
            faces.append(p[:, :, voff])
        geoms[g.get("id")] = (pos, np.vstack(faces))
    parts = []
    for node in root.iterfind(".//c:visual_scene//c:node", ns):
        inst = node.find("c:instance_geometry", ns)
        if inst is None:
            continue
        mat = np.eye(4)
        m = node.find("c:matrix", ns)
        if m is not None:
            mat = np.array(m.text.split(), float).reshape(4, 4)
        pos, faces = geoms[inst.get("url")[1:]]
        world = pos @ mat[:3, :3].T + mat[:3, 3]
        parts.append(trimesh.Trimesh(world, faces, process=False))
    trimesh.util.concatenate(parts).export(stl)


def add_finger_boxes(spec: mujoco.MjSpec) -> None:
    """Pad and base boxes per finger, axis-aligned with link6, from the finger mesh at q = 0 (closed)."""
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_kinematics(model, data)
    b6 = model.body("link6").id
    r6, p6 = data.xmat[b6].reshape(3, 3), data.xpos[b6]
    for finger in ("link7", "link8"):
        bid = model.body(finger).id
        rb, pb = data.xmat[bid].reshape(3, 3), data.xpos[bid]
        g = next(g for g in range(model.ngeom)
                 if model.geom_bodyid[g] == bid and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH)
        mid = model.geom_dataid[g]
        verts = model.mesh_vert[model.mesh_vertadr[mid]:model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
        in6 = (verts @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g] - p6) @ r6
        rel = rb.T @ r6                                   # link6 axes expressed in the finger frame
        quat = np.zeros(4)
        mujoco.mju_mat2Quat(quat, rel.flatten())
        for part, (z0, z1) in FINGER_PARTS.items():
            pts = in6[(in6[:, 2] >= z0) & (in6[:, 2] < z1)]
            lo, hi = pts.min(0), pts.max(0)
            centre6 = (lo + hi) / 2
            box = spec.body(finger).add_geom(name=f"{finger}_{part}", type=mujoco.mjtGeom.mjGEOM_BOX)
            box.size = (hi - lo) / 2
            box.pos = rb.T @ (r6 @ centre6 + p6 - pb)
            box.quat = quat
            box.group, box.density = 3, 0
            box.rgba = [0.2, 0.4, 1.0, 0.4]


def prepare_urdf(source: Path, mesh_dir: Path) -> Path:
    text = (source / URDF).read_text()
    text = re.sub(r'<mesh filename="package://piper_description/meshes/dae/gripper_base_100_v2\.dae"\s*/>',
                  '<mesh filename="gripper_base_100.stl"/>', text)
    # Visuals are discarded on import; point them at the STL twins so the file still parses.
    text = re.sub(r"package://piper_description/meshes/dae/(\w+)\.dae", r"\1.STL", text)
    text = text.replace("package://piper_description/meshes/", "")
    text = text.replace('<link name="arm_base">', '<link name="base_link">')
    text = text.replace('<parent link="arm_base" />', '<parent link="base_link" />')
    # Fixed mount link so base_link becomes a body that the environment can lift by the plate height.
    text = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler meshdir="%s" discardvisual="true" fusestatic="false" '
                  r'balanceinertia="true" strippath="true"/></mujoco><link name="mount"/>'
                  r'<joint name="mount_to_base" type="fixed"><parent link="mount"/><child link="base_link"/>'
                  r'<origin xyz="0 0 0" rpy="0 0 0"/></joint>' % mesh_dir, text, count=1)
    out = mesh_dir.parent / "piper_v100_prepared.urdf"
    out.write_text(text)
    return out


def build(source: Path) -> dict:
    if OUT.exists():
        shutil.rmtree(OUT)
    mesh_dir = OUT / "meshes"
    mesh_dir.mkdir(parents=True)
    for name in LINK_STLS:
        shutil.copy(source / "meshes" / f"{name}.STL", mesh_dir / f"{name}.STL")
    dae_to_stl(source / GRIPPER_DAE, mesh_dir / "gripper_base_100.stl")
    urdf = prepare_urdf(source, mesh_dir)

    spec = mujoco.MjSpec.from_file(str(urdf))
    urdf.unlink()
    spec.modelname = "piper_v100"
    spec.compiler.meshdir = "meshes"
    spec.compiler.discardvisual = False          # keep the visual-only mesh geoms added below
    spec.compiler.fitaabb = True                 # fit collision primitives to the mesh bounding box, not its inertia
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
    spec.option.impratio = 10

    for body in spec.bodies:
        if body.name not in ("world", "mount"):
            body.gravcomp = 1.0
    for joint in spec.joints:
        lo, hi = JOINT_RANGES[joint.name]
        joint.range = [lo, hi]
        joint.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        joint.armature = 0.005
        joint.damping = [0.0, 0.0, 0.0]
        joint.frictionloss = 0.0 if joint.name in ("joint7", "joint8") else 0.3

    # Imported mesh geoms become visual only; a fitted primitive does the colliding.
    for geom in list(spec.geoms):
        if geom.type != mujoco.mjtGeom.mjGEOM_MESH:
            continue
        body = geom.parent
        geom.contype, geom.conaffinity, geom.group, geom.density = 0, 0, 2, 0
        geom.rgba = [0.80, 0.80, 0.82, 1]
        if body.name not in COLLISION:
            continue
        col = body.add_geom()
        col.type = mujoco.mjtGeom.mjGEOM_CAPSULE if COLLISION[body.name] == "capsule" else mujoco.mjtGeom.mjGEOM_BOX
        col.meshname = geom.meshname
        col.pos, col.quat = geom.pos, geom.quat
        col.group, col.density = 3, 0
        col.rgba = [0.9, 0.3, 0.3, 0.3]

    add_finger_boxes(spec)
    spec.add_exclude(bodyname1="base_link", bodyname2="link1")
    spec.add_exclude(bodyname1="base_link", bodyname2="link2")   # fitted capsules overlap at every pose; link1 sits between
    spec.add_exclude(bodyname1="link7", bodyname2="link8")
    spec.add_equality(type=mujoco.mjtEq.mjEQ_JOINT, name1="joint8", name2="joint7",
                      data=[0, -1, 0, 0, 0] + [0] * 6)
    for name, (kp, kv) in ARM_GAINS.items():
        act = spec.add_actuator(name=name, target=name, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        act.set_to_position(kp=kp, kv=kv)
        act.ctrlrange = list(JOINT_RANGES[name]); act.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
        act.forcerange = [-100, 100]; act.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
    grip = spec.add_actuator(name="gripper", target="joint7", trntype=mujoco.mjtTrn.mjTRN_JOINT)
    grip.set_to_position(kp=40, kv=5)
    grip.ctrlrange = list(JOINT_RANGES["joint7"]); grip.ctrllimited = mujoco.mjtLimited.mjLIMITED_TRUE
    grip.forcerange = [-10, 10]; grip.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE

    model = spec.compile()
    key = spec.add_key(name="home")
    key.qpos = HOME_QPOS + [0.0, 0.0]
    key.ctrl = HOME_QPOS + [0.0]
    spec.compile()
    (OUT / "piper.xml").write_text(spec.to_xml())
    (OUT / "scene.xml").write_text(SCENE)
    shutil.copy(source / "LICENSE.piper_ros", OUT / "LICENSE.piper_ros")
    manifest = {
        "source": "agilexrobotics/piper_ros noetic ac41fcbcdda598f01b51cf6175ed9a24d0dacadc",
        "urdf": URDF, "license": "MIT (LICENSE.piper_ros)",
        "urdf_sha256": hashlib.sha256((source / URDF).read_bytes()).hexdigest(),
        "files": {p.relative_to(OUT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in sorted(OUT.rglob("*")) if p.is_file()},
        "joint_ranges": JOINT_RANGES, "home_qpos": HOME_QPOS, "nbody": model.nbody, "ngeom": model.ngeom,
    }
    (OUT / "SOURCE.json").write_text(json.dumps(manifest, indent=2))
    return manifest


SCENE = """<mujoco model="piper_v100_scene">
  <include file="piper.xml"/>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="120" elevation="-20"/>
  </visual>
  <asset>
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
      markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
  </asset>
  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
  </worldbody>
</mujoco>
"""


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--source", type=Path, required=True, help="piper_description folder from piper_ros")
    args = parser.parse_args()
    info = build(args.source)
    print(json.dumps({k: info[k] for k in ("source", "urdf_sha256", "nbody", "ngeom")}, indent=2))


if __name__ == "__main__":
    main()
