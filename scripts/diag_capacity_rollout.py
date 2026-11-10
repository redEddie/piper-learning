"""Roll out a checkpoint on Warp and report which physics flag ends episodes.

Usage: diag_capacity_rollout.py <run dir> <checkpoint dir> ['{"ccd_pairs_per_env": 16, ...}']
Prints the Warp overflow bitmask the first time it is nonzero (solver-limit bits
1024/1536 are benign), then per reset stage how many envs hit done / overflow /
nonfinite, and the state just before the first failure for a few envs.
"""
import json, sys
from pathlib import Path
from piper_sim_lab.runtime import configure
configure("gpu")
import jax, jax.numpy as jp, numpy as np
from piper_sim_lab.checkpoint import load_policy
from piper_sim_lab.tasks import make_env
run = Path(sys.argv[1]); ckpt = Path(sys.argv[2]); N = 128
config = json.loads((run / "config.json").read_text())
eff = dict(config["resolved_environment"]); eff["batch_capacity"] = N
if len(sys.argv) > 3: eff.update(json.loads(sys.argv[3]))
print("contacts_per_env", eff["contacts_per_env"])
env = make_env(eff)
policy = load_policy(ckpt, observation_size=env.observation_size, action_size=env.action_size, deterministic=True)
state = jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(2026), N))
step = jax.jit(jax.vmap(env.step))
key = jax.random.key(1)
rows = []
for t in range(80):
    key, k = jax.random.split(key)
    action, _ = policy(state.obs, k)
    state = step(state, action)
    m = state.metrics
    bits = np.asarray(state.data._impl.overflow)
    if (bits != 0).any() and not globals().get("printed"):
        globals()["printed"] = True; print("t", t, "overflow bits (unique):", np.unique(bits), "nacon", np.asarray(getattr(state.data._impl, "nacon", -1)), "naccd", np.asarray(getattr(state.data._impl, "naccd", -1)))
    rows.append({n: np.asarray(m[n]) for n in ("capacity_overflow", "nonfinite", "stage", "tcp_block", "block_height", "block_bowl_xy", "grasped")} | {"done": np.asarray(state.done), "reset_stage": np.asarray(state.info["reset_stage"])})
done = np.stack([r["done"] for r in rows]); ov = np.stack([r["capacity_overflow"] for r in rows]); nf = np.stack([r["nonfinite"] for r in rows])
print("envs ever done", int((done > 0).any(0).sum()), "overflow", int((ov > 0).any(0).sum()), "nonfinite", int((nf > 0).any(0).sum()))
first = np.array([np.argmax(done[:, i] > 0) if (done[:, i] > 0).any() else -1 for i in range(N)])
print("first done step: min", first[first >= 0].min() if (first >= 0).any() else None, "median", np.median(first[first >= 0]) if (first >= 0).any() else None)
rs = rows[0]["reset_stage"]
for s in range(3):
    idx = rs == s
    print(f"reset_stage {s}: n={idx.sum()} done={int((done[:, idx] > 0).any(0).sum())} overflow={int((ov[:, idx] > 0).any(0).sum())}")
# state just before the first failure for a few envs
for i in np.where(first >= 0)[0][:6]:
    t = max(first[i] - 1, 0); r = rows[t]
    print(f"env {i} reset_stage {rs[i]} fails at t={first[i]} | before: stage {r['stage'][i]:.0f} tcp_block {r['tcp_block'][i]:.3f} height {r['block_height'][i]:.3f} xy {r['block_bowl_xy'][i]:.3f} overflow {ov[first[i], i]:.0f} nonfinite {nf[first[i], i]:.0f}")
