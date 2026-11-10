"""Measure face alignment from saved actual poses; no reward or action changes."""
from pathlib import Path
from unittest import mock
import json
import yaml
import numpy as np
import mujoco
from piper_sim_lab.tasks import make_env
from piper_sim_lab.pick_place import PiperPickPlace

ROOT=Path(__file__).resolve().parents[2]
config=yaml.safe_load((ROOT/'configs/show101/refine-v3-episode-8m.yaml').read_text())['environment']
tables={k:np.zeros((1,6)) for k in ('pregrasp','grasp','preinside')}
with mock.patch('mujoco_playground._src.mjx_env.put_model',return_value=None), \
     mock.patch.object(PiperPickPlace,'_build_reset_table',return_value=tables):
    env=make_env(config)
model=env._mj_model;data=mujoco.MjData(model);results={}
for name,path in [('show100_source50m','runs/show100/final-evaluation/empty_hand-s2026-trace.npz'),
                  ('old_refinement_intermediate','runs/show101/interim-continuation-6291456-evaluation/empty_hand-s2026-trace.npz')]:
    trace=np.load(ROOT/path);episodes=[]
    for e in range(trace['qpos'].shape[1]):
        angles=[]
        for t in range(80):
            data.qpos[:]=trace['qpos'][t,e];data.qvel[:]=trace['qvel'][t,e]
            data.ctrl[:]=trace['ctrl'][t,e];data.mocap_pos[:]=trace['mocap'][t,e]
            mujoco.mj_forward(model,data)
            if not any(data.sensordata[env._sensor_adr[n]]>0 for n in ('left','right')):continue
            rotation=data.xmat[env._block_id].reshape(3,3)
            bottom=data.xpos[env._block_id,2]-.025*np.abs(rotation[2,:]).sum()
            if bottom<=.005:continue
            closing=data.site_xmat[env._tcp_id].reshape(3,3)[:,env._closing_axis]
            u=rotation.T@closing
            angles.append(float(np.degrees(np.arccos(np.clip(np.abs(u).max(),0,1)))))
        if len(angles)>=3:
            episodes.append(dict(episode=e,contact_samples=len(angles),median_face_normal_error_degrees=float(np.median(angles))))
    errors=np.array([e['median_face_normal_error_degrees'] for e in episodes])
    results[name]=dict(evaluated_empty_hand_episodes=trace['qpos'].shape[1],episodes_with_at_least_300ms_lifted_finger_contacts=len(episodes),
        face_aligned_under10deg=int((errors<10).sum()),oblique_over25deg=int((errors>25).sum()),
        median_episode_angle_degrees=float(np.median(errors)),records=episodes)
output=dict(definition='Angle between gripper closing axis and nearest cube face normal; 0 degrees is aligned. Median over samples with actual finger contact and cube bottom >5mm above table; at least 3 sampled contacts (not necessarily consecutive).',
    limitation='Recorded 100ms native-contact/pose audit; not a robustness or hardware test, and not proof of why the optimizer chose a pose.',results=results,real_robot=False)
(ROOT/'docs/show101/grasp-axis-audit.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps({k:{n:v for n,v in r.items() if n!='records'} for k,r in results.items()}),flush=True)
