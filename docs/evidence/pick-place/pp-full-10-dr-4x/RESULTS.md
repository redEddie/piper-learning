# pp-full-10-dr-4x (2026-10-21)

`configs/pick_place_dr_long.yaml` = `configs/pick_place_dr.yaml` with 4x the budget (67.1M steps, 2583 s) and a
checkpoint every 8.4M steps, so the 1x/2x/4x points come from one run (Brax PPO uses a constant learning rate).
Evaluation: 128 env, empty-hand starts, final-step success, eval seeds 2026/2027/2028.

| budget | steps | DR on | mean | DR off | mean |
| --- | --- | --- | --- | --- | --- |
| 1x | 16.8M | 87.5 / 80.5 / 90.6 | 86.2 | 93.0 / 92.2 / 89.1 | 91.4 |
| 2x | 33.6M | 95.3 / 90.6 / 95.3 | 93.8 | 96.1 / 97.7 / 96.1 | 96.6 |
| 4x | 67.1M | 96.1 / 92.2 / 97.7 | 95.3 | 85.9 / 87.5 / 96.9 | 90.1 |

Mixed starts, DR on, seed 2026: 2x 95.3, 4x 96.9 (random 13.3, zero 0).

Same config and training seed as pp-full-09-dr, whose 16.8M policy scored 78.9 (DR on). Only the eval schedule
differs (num_evals 9 vs 5), which changes the PPO random stream, so the 1x gap (86.2 vs 78.9) is run-to-run variance.
A single training seed cannot separate the budget effect from that variance; training seeds are still to do.

Arm-bowl contact (replay, DR on, seed 2026): 0 episodes at 1x, 2x and 4x. pp-full-09-dr had 66 % (wrist resting on
the rim after release), so that habit belongs to that run, not to the setup.

Training-time eval (sum of per-step success, 64 env): 8.3, 33.1, 52.6, 57.2, 58.2, 57.2, 62.0, 59.8, 60.6 at 0-67.1M
in 8.4M steps: the curve flattens after about 2x.
