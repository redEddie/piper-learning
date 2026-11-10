"""Independent scripted positive control; never used as training policy/rules."""
from pathlib import Path
import functools,json
import yaml
from piper_sim_lab.runtime import configure
configure('gpu')
import jax
import jax.numpy as jp
import numpy as np
from mujoco_playground import wrapper
from piper_sim_lab.tasks import make_env

ROOT=Path(__file__).resolve().parents[2]
c=yaml.safe_load((ROOT/'configs/show103/release-8m.yaml').read_text())['environment']
n=16;env=make_env(dict(c,batch_capacity=n,reset_stage_probs=[0.,0.,0.,1.],gentle_near_fraction=1.))
dr=wrapper.BraxDomainRandomizationVmapWrapper(env,functools.partial(env.physics_randomization_fn,
    rng=jax.random.split(jax.random.key(10318),n)))
state=jax.jit(dr.reset)(jax.random.split(jax.random.key(10317),n))
@jax.jit
def run(state):
    def step(carry,i):
        st,phase,timer=carry
        z=jp.where(phase==1,-.4,jp.where(phase==4,1.,0.))
        grip=jp.where(phase>=3,1.,0.)
        action=jp.zeros((n,7)).at[:,2].set(z).at[:,6].set(grip)
        st=dr.step(st,action)
        f=jax.vmap(env._place_features)(st.data);m=st.info['placement']
        nxt=jp.where((phase==0)&(i>=19)&(st.info['coord_phase']==0),1,phase)
        nxt=jp.where((phase==1)&(f['tcp_height']<=.056),2,nxt)
        nxt=jp.where((phase==2)&(timer>=10),3,nxt)
        nxt=jp.where((phase==3)&m['released'],4,nxt)
        nxt=jp.where((phase==4)&(f['clearance']>=.085),5,nxt)
        timer=jp.where(nxt==phase,timer+1,0)
        sample=dict(phase=nxt,height=f['height'],width=f['width'],support=f['support'],
            qualified=m['qualified_release'],premature=m['premature'],recontact=m['recontact_seen'],
            gentle=st.metrics['gentle_success'],impact=m['impact_peak'],failure=st.info['controller_failed'],
            reason=st.info['controller_failure_reason'],clearance=f['clearance'],trace=st.info['gentle_trace'],
            tcp_height=f['tcp_height'],release_tcp=m['release_tcp_height'],
            locked_speed=st.metrics['grip_arm_locked_reference_speed'],
            opening_unlocked=st.info['grip_motion_unlocked'],
            grip_speed=st.metrics['grip_arm_reference_speed'],
            qpos=st.data.qpos)
        return (st,nxt,timer),sample
    return jax.lax.scan(step,(state,jp.zeros(n,jp.int32),jp.zeros(n,jp.int32)),jp.arange(400))
_,trace=jax.block_until_ready(run(state));t=jax.tree.map(np.asarray,trace)
out=dict(scope='Scripted test only: stop, lower toward TCP55mm above bowl inner floor, settle, open100mm, withdraw after release. Not a training policy/rule.',
    worlds=n,seconds=40.,gentle_final_successes=int(t['gentle'][-1].sum()),
    target_height_release_worlds=int(t['qualified'][-1].sum()),premature_worlds=int(t['premature'][-1].sum()),
    recontact_worlds=int(t['recontact'][-1].sum()),controller_fault_worlds=int(t['failure'].any(0).sum()),
    final_phases=t['phase'][-1].tolist(),peak_contact_speed_m_s=t['impact'][-1].tolist(),
    release_tcp_height_mm=(t['release_tcp'][-1]*1000).tolist(),
    max_locked_grip_reference_speed=float(t['locked_speed'].max()),
    max_unlocked_opening_reference_speed=float(np.where(t['opening_unlocked'],t['grip_speed'],0.).max()),
    real_robot=False,learning=False)
(ROOT/'docs/show103/positive-release.json').write_text(json.dumps(out,indent=2)+'\n')
np.savez_compressed(ROOT/'runs/show103/positive-release.npz',**t)
print(json.dumps(out),flush=True)
assert out['controller_fault_worlds']==0
assert out['max_locked_grip_reference_speed']<1e-4
assert out['gentle_final_successes']>0,'No physically executed positive control met the criterion; inspect before adoption'
