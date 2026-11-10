"""Audit saved positive labels using native contacts at recorded 100 ms poses.

This replay is not a fresh physics rollout or a reconstruction of missing 5 ms
contacts. Substep timing is tested separately and frozen GPU reruns use 5 ms.
"""
from pathlib import Path
from unittest import mock
import json
import functools
import yaml
import mujoco
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.tasks import make_env
from piper_sim_lab.pick_place import PiperPickPlace
from piper_sim_lab.placement_refinement import placement_memory,placement_update,placement_terms

ROOT=Path(__file__).resolve().parents[2]
config=yaml.safe_load((ROOT/'configs/show101/refine-v3-episode-8m.yaml').read_text())['environment']
tables={k:np.zeros((1,6)) for k in ('pregrasp','grasp','preinside')}
with mock.patch('mujoco_playground._src.mjx_env.put_model',return_value=None), \
     mock.patch.object(PiperPickPlace,'_build_reset_table',return_value=tables):
    env=make_env(config)
model=env._mj_model;data=mujoco.MjData(model);c=env._config


def features(trace,t,e):
    data.qpos[:]=trace['qpos'][t,e];data.qvel[:]=trace['qvel'][t,e]
    data.mocap_pos[:]=trace['mocap'][t,e];data.ctrl[:]=trace['ctrl'][t,e]
    mujoco.mj_forward(model,data)
    obj=data.xpos[env._block_id];bowl=data.xpos[env._bowl_id]
    ro=data.xmat[env._block_id].reshape(3,3);rb=data.xmat[env._bowl_id].reshape(3,3)
    rt=data.site_xmat[env._tcp_id].reshape(3,3)
    p=rb.T@(obj-bowl);h=p[2]-c.block_half*np.abs(rb[:,2]@ro).sum()-c.bowl_floor
    xy=np.linalg.norm(p[:2]);width=data.qpos[env._finger_qpos_ids[0]]-data.qpos[env._finger_qpos_ids[1]]
    contact={n:data.sensordata[a]>0 for n,a in env._sensor_adr.items()}
    v=data.qvel[env._block_qvel_adr:env._block_qvel_adr+3]-data.qvel[env._bowl_qvel_adr:env._bowl_qvel_adr+3]
    near=np.linalg.norm(obj-data.site_xpos[env._tcp_id])<c.grasp_distance
    grasp=near and width<2*c.block_half+.004 and obj[2]-c.block_half>c.lift_height
    above=(grasp and xy<c.bowl_r_rim-c.block_half and obj[2]>bowl[2]+env._bowl_rim_z)
    above=above or ((contact['left'] or contact['right']) and xy<c.bowl_r_rim-c.block_half)
    elevated=obj[2]-c.block_half*np.abs(ro[2,:]).sum()>c.place_carry_clearance
    result=dict(contact,above=above,elevated=elevated,distance=np.hypot(xy,max(h,0.)),
        height=h,xy=xy,speed=np.linalg.norm(v),downward_speed=max(-(rb[:,2]@v),0.),
        width=width,span=2*c.block_half*np.abs(rt[:,env._closing_axis]@ro).sum(),
        in_region=bool(xy<=c.bowl_r_bottom-c.block_half and -.001<=h<=c.place_height_tolerance))
    u=ro.T@rt[:,env._closing_axis]
    angle=float(np.degrees(np.arccos(np.clip(np.abs(u).max(),0,1))))
    return result,angle


@jax.jit
def replay(f):
    initial={k:v[0] for k,v in f.items()}
    memory=placement_memory(initial,True)
    before={k:v[:-1] for k,v in f.items()};after={k:v[1:] for k,v in f.items()}
    def step(t,ab):
        t=placement_update(t,*ab,.1,c);return t,t
    return jax.lax.scan(step,memory,(before,after))


audit=json.loads((ROOT/'docs/show101/continuation-success-label-audit.json').read_text())
results=[];angle_examples=[]
for row in audit['records']:
    path=ROOT/'runs/show101'/row['evaluation']/f"{row['recording']}-trace.npz"
    trace=np.load(path);episode=row['episode']
    stream=[];angles=[]
    for t in range(80):
        f,a=features(trace,t,episode);stream.append(f);angles.append(a)
    f={k:jp.asarray(np.array([s[k] for s in stream])) for k in stream[0]}
    final,_=jax.block_until_ready(replay(f))
    result=dict(evaluation=row['evaluation'],recording=row['recording'],episode=episode,
        old_raw_gentle=row['raw_gentle_final'],corrected_qualified=bool(final['qualified_release']),
        corrected_premature=bool(final['premature']),carried_witness=bool(final['carried']),
        corrected_opening_credit=float(final['opening_best']))
    results.append(result)
    assert not result['corrected_qualified'],result
    assert result['corrected_premature'],result
    assert result['corrected_opening_credit']==0,result
    if row['evaluation']=='interim-continuation-6291456-evaluation' and row['recording']=='empty_hand-s2026':
        carried=[i for i,s in enumerate(stream) if s['elevated'] and (s['left'] or s['right'])]
        angle_examples.append(dict(episode=episode,definition='angle between gripper closing axis and nearest cube face normal; 0 degrees is face alignment',
            first_lift_contact_seconds=(carried[0]+1)*.1 if carried else None,
            first_lift_contact_angle_deg=angles[carried[0]] if carried else None,
            median_lift_contact_angle_deg=float(np.median([angles[i] for i in carried])) if carried else None,
            median_projected_cube_span_mm=float(np.median([stream[i]['span']*1000 for i in carried])) if carried else None,
            median_gripper_width_mm=float(np.median([stream[i]['width']*1000 for i in carried])) if carried else None))
output=dict(scope=__doc__,episodes=len(results),old_raw_gentle_records=sum(r['old_raw_gentle'] for r in results),
    corrected_qualified_records=sum(r['corrected_qualified'] for r in results),
    records=results,grasp_angles=angle_examples,real_robot=False)
(ROOT/'docs/show101/episode-replay-regressions.json').write_text(json.dumps(output,indent=2)+'\n')
print(json.dumps(dict(episodes=len(results),corrected_qualified=output['corrected_qualified_records'],grasp_angles=angle_examples)),flush=True)
