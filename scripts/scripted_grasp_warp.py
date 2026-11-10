"""Scripted grasp on the GPU Warp backend (the training physics), 8 envs.

Usage: scripted_grasp_warp.py [configs/pick_place.yaml]  (environment overrides are taken from the YAML)
Env vars: IMPL=jax N=1 JAX_PLATFORMS=cpu runs the same script on the CPU MJX path (slow; physics-path comparison).
"""
import os, sys, yaml, numpy as np, jax, jax.numpy as jp
from piper_sim_lab.pick_place import PiperPickPlace
N = int(os.environ.get("N", 8))
IMPL = os.environ.get("IMPL", "warp")
overrides = yaml.safe_load(open(sys.argv[1]))["environment"] if len(sys.argv) > 1 else {}
overrides.pop("task", None)
if IMPL == "jax":
    overrides.setdefault("bowl_segments", 8)
env = PiperPickPlace(config_overrides={**overrides, "impl": IMPL, "batch_capacity": N})
print("block_half", env._config.block_half, "block_mass", env._config.block_mass, flush=True)
reset = jax.jit(jax.vmap(env.reset)); step = jax.jit(jax.vmap(env.step))
s = reset(jax.random.split(jax.random.key(3), N))
blk = np.asarray(s.data.xpos[:, env._block_id]); bowl = np.asarray(s.data.mocap_pos[:, env._bowl_mocap_id])
def goto(s, target, steps, grip):
    for _ in range(steps):
        tcp = s.data.site_xpos[:, env._tcp_id]; v = jp.clip((jp.array(target) - tcp)/0.01, -1, 1)
        s = step(s, jp.concatenate((v, jp.zeros((N,3)), jp.full((N,1), grip)), axis=1))
    return s
s = goto(s, blk + [0,0,0.08], 25, 1.0); s = goto(s, blk + [0,0,0.006], 20, 1.0)
at = np.asarray(s.metrics["tcp_block"])
for _ in range(8): s = step(s, jp.tile(jp.array([0,0,0,0,0,0,-1.0]), (N,1)))
w = np.asarray(s.data.qpos[:, env._finger_qpos_ids[0]] - s.data.qpos[:, env._finger_qpos_ids[1]])
s = goto(s, np.asarray(s.data.site_xpos[:, env._tcp_id]) + [0,0,0.10], 20, -1.0); h = np.asarray(s.metrics["block_height"]); g = np.asarray(s.metrics["grasped"])
s = goto(s, bowl + [0,0,0.11], 40, -1.0); xy = np.asarray(s.metrics["block_bowl_xy"])
for _ in range(18): s = step(s, jp.tile(jp.array([0,0,0,0,0,0,1.0]), (N,1)))
inside = np.asarray(s.metrics["success"]); b = np.asarray(s.data.xpos[:, env._block_id])
for i in range(N):
    print(f"env {i}: at d={at[i]:.3f} | w={w[i]:.4f} | lifted h={h[i]:.3f} grasped={int(g[i])} | above xy={xy[i]:.3f} | inside={int(inside[i])} block={np.round(b[i],3).tolist()}", flush=True)
print(IMPL, "scripted success", int(inside.sum()), "/", N, "overflow", float(jp.max(s.metrics["capacity_overflow"])), flush=True)
