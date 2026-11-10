# Matched MLP / 1-layer LSTM-128 experiment (SHOW-69)

This checkout is isolated from `/home/jeonchanwook/work/piper-sim-lab`, including
its running PPO jobs. `source-snapshot.json` records the source bytes copied
from that checkout. Environment changes already present there are prerequisites,
not changes invented by this experiment. No shared Python package is reinstalled.
Use `PYTHONPATH=src` with `/data/envs/venv/piper-sim-lab/bin/python`.

## Model and learner

- MLP: upstream Brax actor `[256,256]`, swish, tanh-normal actions.
- LSTM: one Flax OptimizedLSTMCell, hidden/cell size 128, output LayerNorm,
  followed by the same `[256,256]` swish head and tanh-normal distribution.
- Both: identical privileged, memoryless Brax `[256,256]` critic.
- Actor memory reads only deployment observations (including previous action).
- `recurrent_train.py` reuses upstream **Brax compute_ppo_loss**, GAE and
  distribution, Optax Adam, and existing simulator/reset wrappers. It adds
  stateful collection, full current-episode prefix replay, and sequence SGD.
- Minibatches shuffle **environments/sequences**, never individual time steps.
  Prefixes are replayed using the current parameters and normalization before
  each minibatch. The prefix carry is detached; BPTT spans the 16 collected steps.
  Each slot's memory/history clears on done, including time-limit autoreset.
- Observation statistics are frozen during collection+SGD and updated afterward
  for **both** models. This differs from the stock Brax adapter's pre-SGD update;
  therefore the newly trained MLP in this same loop is the primary control.
- The same 16,777,216-transition DR task/config is used for both actors.
  This is one training seed (7) per actor, not a training-seed stability study.

## Resource protection

`experiment_guard.py` rejects GPU startup if any other compute PID exists.
The queue waits for a quiet minute between jobs. During training, if an external
compute process appears, this trainer yields before its next iteration without
signalling that process. It does not change priorities or power limits.
`queue.jsonl`, `occupancy.jsonl` and each run's `gpu-yields.jsonl` record occupancy.
The first collect+update includes compilation and is excluded from steady-state
throughput; collection and optimizer time are separately synchronized with JAX.
End-to-end time includes checks, logs, checkpoint saves and any waiting.

## Commands

```bash
env -u LD_LIBRARY_PATH JAX_PLATFORMS=cpu PYTHONPATH=src \
 /data/envs/venv/piper-sim-lab/bin/python -m pytest tests/test_recurrent.py -q

# Waits for existing GPU jobs, then runs train/eval for both actors serially.
/data/envs/venv/piper-sim-lab/bin/python -u experiments/lstm128/run_comparison.py

# Individual job; startup fails if another GPU job is present.
env -u LD_LIBRARY_PATH PYTHONPATH=src \
 /data/envs/venv/piper-sim-lab/bin/python -m piper_sim_lab.recurrent_train \
 --config configs/pick_place_lstm_comparison.yaml \
 --output runs/matched-lstm-s7 --actor lstm
```

## Evaluation contract

Each actor is evaluated deterministically from **empty-hand/stage-0** starts,
with held-out seeds 2026/2027/2028, 128 episodes each. The same seeds generate
initial scenes and physical randomization for both actors. Report DR-on and
DR-off conditions separately; the observed size channel and bowl collision
remain enabled in both conditions. `final_success_rate` requires the block
inside the bowl and released at step 80, with no physical failure anywhere
in the episode. Ever-success is a separate metric.
Episode-level Wilson intervals do not measure variability across training seeds.
Trajectory NPZ files contain actual simulator states for images/videos.

Checkpoints contain model/normalizer/optimizer/RNG and the transition count.
They support evaluation; exact training resume is not exposed (the simulator
state and live episode histories are not saved). They are a distinct format,
not interchangeable with the existing Brax Orbax checkpoints.

## Measured result (2026-10-21, SHOW-69)

Both models completed 16,777,216 transitions. Each evaluation condition uses
384 matched empty-hand episodes, across held-out seeds 2026/2027/2028.

| Actor in the matched learner | DR on final success | DR off final success | Steady-state transitions/s |
| --- | --- | --- | --- |
| MLP [256,256] | 301/384 (78.4%) | 332/384 (86.5%) | 25,352 |
| 1-layer LSTM-128 + [256,256] | 340/384 (88.5%) | 345/384 (89.8%) | 23,080 |

LSTM improves DR-on success by 10.16 percentage points in this pilot, with
8.96% lower throughput. Both have zero physical failures in these evaluations.
This is one training seed per model. The pre-existing stock Brax MLP at the same
transition budget is a separate reference: DR on 86.2%, off 91.4%. Its environment
and task source match, but its learner/normalization schedule differs. The LSTM
result does not establish uniform improvement over that reference.

Code is on branch `codex/lstm128-comparison`; the main checkout was untouched.
The two **final** checkpoints, configs, logs, evaluation trajectories and figures
are also copied into this new shared artifact directory, with SHA-256 manifest:

`/data/projects/piper-sim-lab/artifacts/show69-lstm128-20261021/`

The `mlp/` and `lstm/` subdirectories each work as `--run` inputs to
`piper_sim_lab.recurrent_evaluate`. The Git repository stores lightweight evidence
under `experiments/lstm128/results/`; binary checkpoints remain separate artifacts.
