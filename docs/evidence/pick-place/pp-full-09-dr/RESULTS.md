# pp-full-09-dr (2026-10-21)

`configs/pick_place_dr.yaml`: stage-free policy (`stage_obs: false`), bowl-arm collision, Dex-One2Many Table 10
randomization (actor-observation errors and delays, action delay, physics per env slot). 1024 env, 16.7M steps, 764 s.
Evaluation: 128 env, empty-hand starts (`reset_stage_probs [1,0,0,0]`), final-step success (ever in brackets).

| policy | env | seed 2026 | 2027 | 2028 | mean |
| --- | --- | --- | --- | --- | --- |
| pp-full-09-dr | DR on | 82.0 (83.6) | 78.9 (79.7) | 75.8 (82.8) | 78.9 |
| pp-full-09-dr | DR off (bowl-arm collision on) | 87.5 (90.6) | 79.7 (85.2) | 82.0 (85.2) | 83.1 |
| pp-full-08-nostage-preinside | DR on (no size obs) | 0.0 | | | |
| pp-full-08-nostage-preinside | no DR, no bowl-arm collision | 93.0 (96.9) | | | |

pp-full-09-dr mixed starts, DR on, seed 2026: 87.5 (89.8); random 13.3, zero 0.

pp-full-08 ablation (seed 2026, bowl-arm collision on): collision only 94.5, + physics 93.0, + delays only 93.0,
+ pose/size noise only 0.0, all 0.0. The 1 cm per-step block-position noise is what breaks a policy trained without it.

Arm(link1-6)-bowl contact, replayed in MuJoCo C from trajectory.npz (seed 2026):

| policy / env | episodes with contact | step fraction | max depth | links |
| --- | --- | --- | --- | --- |
| pp-full-08, collision off (would-be penetration) | 3.9 % | 0.5 % | 28.0 mm | |
| pp-full-08, collision on | 3.1 % | 0.6 % | 1.3 mm | |
| pp-full-09, DR on | 66.4 % | 14.5 % | 2.0 mm | link6 1484 frames, link5 5 |
| pp-full-09, DR off | 79.7 % | 19.0 % | 1.4 mm | |

pp-full-09 contacts: 96 % of contact frames have the block already over the bowl floor and the hand released
(TCP > 2 cm from the block); median step 70 of 80. The policy rests its wrist on the rim after releasing. The bowl is
kinematic (mocap), so leaning on it costs nothing in simulation.

Learning curve (eval sum of per-step success over 64 env, 80 steps): 8.3, 12.5, 29.3, 38.1, 45.9 at 0/4/8/12.6/16.8M —
still rising at the end of the budget.
