"""Controller-only GPU timings, followed by a fixed-action integration probe.

No PPO/gradient updates or robot I/O. GPU timings include dispatch and device
synchronization, exclude compilation, DLS and physics. Existing GPU work is
recorded and never interrupted.
"""
from pathlib import Path
from dataclasses import replace
import json,time,subprocess
import numpy as np
from piper_sim_lab.runtime import configure
configure('gpu')
import jax
import jax.numpy as jp
from piper_sim_lab.control_lab.tracking_qp import TrackingSettings,tracking_problem,project_tracking
from piper_sim_lab.control_lab.controllers import Settings,floor_distance
from piper_sim_lab.control_lab.hierarchy import floor_minimum
from piper_sim_lab.control_lab.hierarchy_env import fk_floor
from piper_sim_lab.assets import get_model_path
from piper_policy_runtime.kinematics import PiperKinematics

ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'runs/qp-jerk/gpu';OUT.mkdir(parents=True,exist_ok=False)
occupancy=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,used_gpu_memory','--format=csv,noheader'],text=True)
kin=PiperKinematics(get_model_path('piper_v100').parent/'piper.xml');kin.tcp_local_position=(0.,0.,.1385)
p=TrackingSettings();s=Settings();lower=jp.asarray(kin.lower,dtype=jp.float32);upper=jp.asarray(kin.upper,dtype=jp.float32)
variants=[0.,100.,300.,1000.];rows=[];rng=np.random.default_rng(7782)
for n in [1,128,1024]:
    q=jp.asarray(np.array([0.,.523,-.45,0.,0.,0.])+rng.uniform(-.02,.02,(n,6)),dtype=jp.float32)
    target=jp.asarray(rng.uniform(-.3,.3,(n,6)),dtype=jp.float32)
    functions={};timings={j:[] for j in variants}
    for jerk in variants:
        settings=replace(s,qp_jerk=jerk>0,jerk=jerk or 100.)
        def make_one(settings):
            def one(q,target):
                z0=jp.zeros(6,dtype=q.dtype)
                preferred,lo,hi,slo,shi,*_=tracking_problem(q,z0,q,z0,target,lower,upper,settings,p,jp.zeros(6,bool),z0,xp=jp,previous_acceleration=z0)
                z,normal=fk_floor(kin,q,settings.floor_normal_base)
                v,ok=project_tracking(preferred,lo,hi,slo,shi,p.weight,normal,floor_minimum(z,settings,xp=jp))
                ref=q+settings.dt*v;next_z,_=fk_floor(kin,ref,settings.floor_normal_base)
                return ref,v,ok&(floor_distance(next_z,settings)>=-settings.tolerance)
            return one
        f=jax.jit(jax.vmap(make_one(settings)))
        started=time.perf_counter();out=jax.block_until_ready(f(q,target));cold=time.perf_counter()-started
        assert np.asarray(out[2]).all()
        for _ in range(5):jax.block_until_ready(f(q,target))
        functions[jerk]=(f,cold)
    for _ in range(120):
        for jerk in rng.permutation(variants):
            start=time.perf_counter_ns();jax.block_until_ready(functions[jerk][0](q,target));timings[jerk].append((time.perf_counter_ns()-start)/1e6)
        time.sleep(.01)
    for jerk in variants:
        x=np.array(timings[jerk]);r=dict(batch_size=n,jerk_limit=jerk or None,p50_ms=float(np.median(x)),p95_ms=float(np.percentile(x,95)),p99_ms=float(np.percentile(x,99)),mean_ms=float(x.mean()),compile_seconds=functions[jerk][1],samples=len(x));rows.append(r);print(json.dumps(r),flush=True)
    del functions
    jax.clear_caches()
result=dict(scope=__doc__,gpu_processes_present_before_start=occupancy,timings=rows,training=False,real_robot=False)
(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')

# Exercise the actual opt-in environment path, including native 5ms state carry.
import yaml
from piper_sim_lab.tasks import make_env
c=yaml.safe_load((ROOT/'configs/show100/fresh-50m.yaml').read_text())['environment']
c.update(control_jerk=100.,batch_capacity=64,reset_stage_probs=[1.,0.,0.,0.],obs_randomization=False,physics_randomization=False)
env=make_env(c)
state=jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(7783),64))
@jax.jit
def roll(state):
    def step(state,_):
        state=jax.vmap(env.step)(state,jp.tile(jp.array([.02,0,0,0,0,0,1.]),(64,1)))
        return state,dict(v=state.info['tracking_trace'][:,:,6:12],failed=state.info['controller_failed'],nonfinite=state.metrics['nonfinite'],overflow=state.metrics['capacity_overflow'])
    return jax.lax.scan(step,state,None,length=8)
_,trace=jax.block_until_ready(roll(state));t=jax.tree.map(np.asarray,trace)
v=t['v'].transpose(1,0,2,3).reshape(64,-1,6)
a=np.diff(np.concatenate([np.zeros((64,1,6)),v],axis=1),axis=1)/.005
j=np.diff(np.concatenate([np.zeros((64,1,6)),a],axis=1),axis=1)/.005
probe=dict(environments=64,policy_steps=8,control_steps_per_environment=160,max_reference_jerk=float(np.abs(j).max()),max_reference_acceleration=float(np.abs(a).max()),controller_failures=int(t['failed'].any(0).sum()),nonfinite=float(t['nonfinite'].max()),overflow=float(t['overflow'].max()),training=False,real_robot=False)
assert probe['controller_failures']==probe['nonfinite']==probe['overflow']==0,probe
assert probe['max_reference_jerk']<=100.1,probe
assert probe['max_reference_acceleration']<=8.001,probe
(OUT/'integration.json').write_text(json.dumps(probe,indent=2)+'\n');print(json.dumps(probe),flush=True)
