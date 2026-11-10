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
c=yaml.safe_load((ROOT/'configs/show103/refine-8m.yaml').read_text())['environment']
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
        nxt=jp.where((phase==1)&f['support'],2,nxt)
        nxt=jp.where((phase==2)&(timer>=10)&m['supported'],3,nxt)
        nxt=jp.where((phase==3)&m['released']&(st.info['coord_phase']==0),4,nxt)
        nxt=jp.where((phase==4)&(f['clearance']>=.085),5,nxt)
        timer=jp.where(nxt==phase,timer+1,0)
        sample=dict(phase=nxt,height=f['height'],width=f['width'],support=f['support'],
            qualified=m['qualified_release'],premature=m['premature'],recontact=m['recontact_seen'],
            gentle=st.metrics['gentle_success'],impact=m['impact_peak'],failure=st.info['controller_failed'],
            reason=st.info['controller_failure_reason'],clearance=f['clearance'],trace=st.info['gentle_trace'],
            qpos=st.data.qpos)
        return (st,nxt,timer),sample
    return jax.lax.scan(step,(state,jp.zeros(n,jp.int32),jp.zeros(n,jp.int32)),jp.arange(400))
_,trace=jax.block_until_ready(run(state));t=jax.tree.map(np.asarray,trace)
out=dict(scope='Scripted test only: stop, lower at20mm/s until support, settle, open, withdraw. This controller is not used by training or policy evaluation.',
    worlds=n,seconds=40.,gentle_final_successes=int(t['gentle'][-1].sum()),
    supported_release_worlds=int(t['qualified'][-1].sum()),premature_worlds=int(t['premature'][-1].sum()),
    recontact_worlds=int(t['recontact'][-1].sum()),controller_fault_worlds=int(t['failure'].any(0).sum()),
    final_phases=t['phase'][-1].tolist(),peak_contact_speed_m_s=t['impact'][-1].tolist(),
    real_robot=False,learning=False)
(ROOT/'docs/show103/positive-motion.json').write_text(json.dumps(out,indent=2)+'\n')
np.savez_compressed(ROOT/'runs/show103/positive-motion.npz',**t)
print(json.dumps(out),flush=True)
assert out['controller_fault_worlds']==0
assert out['gentle_final_successes']>0,'No physically executed positive control met the criterion; inspect before adoption'
