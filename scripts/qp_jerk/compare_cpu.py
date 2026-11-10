"""Offline matched-input DLS+tracking-QP comparison. No policy training/SDK.

Only the jerk velocity bounds differ. The greedy QP objective, plant, 10 Hz
inputs and all previous hard limits are identical. Infeasible runs stop;
failure holds are never included as smooth motion or a successful response.
"""
import os
os.environ.setdefault('JAX_PLATFORMS','cpu')
import argparse,json,time,platform,hashlib
from pathlib import Path
from dataclasses import replace
from unittest import mock
import numpy as np
import mujoco,yaml
from piper_sim_lab.tasks import make_env
from piper_sim_lab.pick_place import PiperPickPlace
from piper_sim_lab.control_lab.tracking_qp import tracking_problem,TrackingSettings
from piper_sim_lab.control_lab.controllers import floor_kinematics,floor_distance
from piper_sim_lab.control_lab.hierarchy import floor_minimum,dls_velocity

ROOT=Path(__file__).resolve().parents[2]
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--output',type=Path,default=ROOT/'runs/qp-jerk/cpu')
ap.add_argument('--replay-count',type=int,default=12)
args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
affinity=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{affinity[min(2,len(affinity)-1)]})
config=yaml.safe_load((ROOT/'configs/show100/fresh-50m.yaml').read_text())['environment']
tables={k:np.zeros((1,6)) for k in ('pregrasp','grasp','preinside')}
with mock.patch('mujoco_playground._src.mjx_env.put_model',return_value=None), \
     mock.patch.object(PiperPickPlace,'_build_reset_table',return_value=tables):
    env=make_env(config)
model=env._mj_model;data=mujoco.MjData(model);s=env.settings;p=env.tracking_settings;kin=env.kin


def project(target,lo,hi,slo,shi,weight,normal,minimum):
    if np.any(lo>hi):raise ValueError('joint_bounds_infeasible')
    if normal@np.where(normal>=0,hi,lo)<minimum-1e-6:raise ValueError('floor_and_joint_bounds_infeasible')
    def value(multiplier):
        z=target+multiplier*normal
        v=np.where(z<slo,(z+weight*slo)/(1+weight),np.where(z>shi,(z+weight*shi)/(1+weight),z))
        return np.clip(v,lo,hi)
    base=value(0.)
    if normal@base>=minimum:return base
    right=1.
    for _ in range(40):
        if normal@value(right)>=minimum:break
        right*=2
    left=0.
    for _ in range(32):
        mid=(left+right)*.5
        if normal@value(mid)<minimum:left=mid
        else:right=mid
    return value(right)


def trajectory(t,q0,kind):
    vector=np.zeros(6)
    if kind.startswith('sine'):
        frequency=.5 if kind=='sine0.5' else 1.
        vector[:2]=[.035,.012]
        return q0+vector*np.sin(2*np.pi*frequency*t),vector*(2*np.pi*frequency)*np.cos(2*np.pi*frequency*t)
    if kind=='step':
        vector[0]=.03
        return q0+vector*(t>=1.),vector*(10. if 1<=t<1.1 else 0.)
    vector[:2]=[.04,.015]
    x=np.clip(t-1,0,3)-2*np.clip(t-4,0,3)+np.clip(t-7,0,3)
    dx=0 if t<1 or t>=10 else (1 if t<4 or t>=7 else -1)
    return q0+vector*x,vector*dx


def distribution(values):
    x=np.asarray(values)
    if not len(x):return None
    return dict(mean=float(x.mean()),p50=float(np.percentile(x,50)),p95=float(np.percentile(x,95)),p99=float(np.percentile(x,99)),maximum=float(x.max()))


def run(name,jerk,pose,kind,source=None,episode=None):
    settings=replace(s,qp_jerk=jerk>0,jerk=jerk or 100.)
    mujoco.mj_resetData(model,data);data.qpos[:]=np.asarray(env._initial_qpos)
    data.qpos[env._qpos_ids]=pose;data.ctrl[:]=np.asarray(env._initial_ctrl)
    data.ctrl[env._arm_ctrl_ids]=pose
    if source is not None:
        data.qpos[:]=source['qpos'][0,episode];data.qvel[:]=0
        data.mocap_pos[:]=source['mocap'][0,episode]
        data.ctrl[env._arm_ctrl_ids]=data.qpos[env._qpos_ids]
    mujoco.mj_forward(model,data)
    q0=data.qpos[env._qpos_ids].copy();ref=q0.copy();v=np.zeros(6);acc=np.zeros(6)
    active=np.zeros(6,bool);timer=np.zeros(6);target=np.zeros(6);rows=[];failure=None
    seconds=12 if source is None else min(8.,len(source['action'])*.1)
    for i in range(round(seconds/s.dt)):
        q=data.qpos[env._qpos_ids].copy();qd=data.qvel[env._qvel_ids].copy();t=i*s.dt
        desired,_=trajectory(t+s.dt,q0,kind) if source is None else (q0,None)
        if source is None:
            qr,vr=trajectory(t,q0,kind)
            action=np.r_[kin.jacobian(qr)@vr/np.array([s.linear_scale]*3+[s.angular_scale]*3),1.]
        else:action=source['action'][min(i//20,len(source['action'])-1),episode].copy()
        started=time.perf_counter_ns();update=i%20==0
        if update:target=dls_velocity(kin,q,np.clip(action,-1.,1.),settings)
        try:
            if np.max(np.abs(ref-q))>s.lag+p.extra+s.tolerance:raise ValueError('actual_tracking_cap')
            preferred,lo,hi,slo,shi,next_active,next_timer=tracking_problem(q,qd,ref,v,target,np.asarray(env._lower),np.asarray(env._upper),
                settings,p,active,timer,previous_acceleration=acc)
            z,normal=floor_kinematics(kin,ref,settings);minimum=floor_minimum(z,settings)
            vn=project(preferred,lo,hi,slo,shi,p.weight,normal,minimum)
            candidate=ref+s.dt*vn
            if floor_distance(np.asarray(s.floor_normal_base)@kin.forward(candidate)['tcp_position'],settings)<-s.tolerance:raise ValueError('nonlinear_floor')
            if np.any(next_timer>p.timeout):raise ValueError('recovery_timeout')
        except ValueError as ex:
            failure=dict(time_s=t,reason=str(ex));break
        elapsed=(time.perf_counter_ns()-started)/1e6
        an=(vn-v)/s.dt;jn=(an-acc)/s.dt
        ref,v,acc,active,timer=candidate,vn,an,next_active,next_timer
        data.ctrl[env._arm_ctrl_ids]=ref
        mujoco.mj_step(model,data)
        rows.append(dict(t=t+s.dt,desired=desired,reference=ref.copy(),actual=data.qpos[env._qpos_ids].copy(),
            v=v.copy(),a=acc.copy(),j=jn.copy(),controller_ms=elapsed,policy_tick=update))
    tr={k:np.asarray([r[k] for r in rows]) for k in rows[0]} if rows else {}
    result=dict(case=name,jerk_limit=jerk or None,completed=failure is None,failure=failure,ticks=len(rows),
        controller_ms=distribution(tr['controller_ms'][20:]) if rows else None,
        reference_jerk_max=float(np.max(np.abs(tr['j']))) if rows else None,
        reference_acceleration_max=float(np.max(np.abs(tr['a']))) if rows else None,
        reference_velocity_max=float(np.max(np.abs(tr['v']))) if rows else None)
    if rows:
        result['reference_jerk_rms']=float(np.sqrt(np.mean(tr['j']**2)))
        for k in ['reference','actual']:
            result[k+'_position_rms_error_rad']=float(np.sqrt(np.mean((tr[k]-tr['desired'])**2))) if source is None else None
        if failure is None and kind.startswith('sine') and source is None:
            freq=.5 if kind=='sine0.5' else 1.;mask=tr['t']>=2.;times=tr['t'][mask]
            X=np.c_[np.sin(2*np.pi*freq*times),np.cos(2*np.pi*freq*times),np.ones(len(times))]
            fit={k:np.linalg.lstsq(X,tr[k][mask,0],rcond=None)[0] for k in ['desired','reference','actual']}
            for k in ['reference','actual']:
                phase=np.angle(np.exp(1j*(np.arctan2(fit[k][1],fit[k][0])-np.arctan2(fit['desired'][1],fit['desired'][0]))))
                result[k+'_phase_lag_ms']=float(-phase/(2*np.pi*freq)*1000)
                result[k+'_gain']=float(np.linalg.norm(fit[k][:2])/np.linalg.norm(fit['desired'][:2]))
        if failure is None and kind=='step' and source is None:
            for k in ['reference','actual']:
                y=tr[k][:,0]-q0[0];final=float(np.mean(y[-100:]));mask=tr['t']>=1.
                crossings=[np.flatnonzero(mask&(y>=level*final)) for level in [.1,.9]]
                result[k+'_rise_10_90_ms']=float((tr['t'][crossings[1][0]]-tr['t'][crossings[0][0]])*1000) if final>0 and all(len(v) for v in crossings) else None
                result[k+'_final_gain']=final/.03
                result[k+'_overshoot_percent']=float(max(0.,y[mask].max()/.03-1)*100)
    if jerk and rows:assert result['reference_jerk_max']<=jerk+1e-6,result
    np.savez_compressed(args.output/f'{name}-j{jerk}.npz',**tr)
    return result


results=[]
variants=[0.,100.,300.,1000.]
poses={'A':np.array(config['nominal_qpos']),'topdown':np.array([0.,1.743,-1.087,0.,1.002,0.])}
rng=np.random.default_rng(7781)
for pose_name,pose in poses.items():
    for kind in ['sine0.5','sine1','step','reversal']:
        for jerk in rng.permutation(variants):
            r=run(pose_name+'-'+kind,float(jerk),pose,kind);results.append(r)
            print(json.dumps(r),flush=True)
source_path=ROOT.parent/'show100-ternary-dls-qp/runs/show100/final-evaluation/empty_hand-s2026-trace.npz'
if args.replay_count:
    source=np.load(source_path)
    for episode in range(args.replay_count):
        for jerk in rng.permutation(variants):
            r=run(f'replay-{episode}',float(jerk),poses['A'],'replay',source,episode);results.append(r);print(json.dumps(r),flush=True)
manifest=dict(scope=__doc__,settings=dict(dt=s.dt,policy_dt=s.policy_dt,velocity=s.velocity,acceleration=s.acceleration,
    jerk_limits=variants,tracking_cap=s.lag+p.extra,tcp_floor=0.,tcp_local_z=.1385),
    computation='controller includes DLS on each20th tick, bounds, floor FK/Jacobian, QP projection, nonlinear floor check; excludes MuJoCo step and logging; first20 ticks excluded',
    phase='0.5/1Hz least-squares fit after2s, q1; includes 10Hz sampling and native plant response',
    replay='Fixed recorded actions starting at first recorded pose with controller and plant velocity reset to rest; not a policy success evaluation',
    affinity=sorted(os.sched_getaffinity(0)),platform=platform.platform(),source_actions=str(source_path),training=False,real_robot=False)
(args.output/'results.json').write_text(json.dumps(dict(manifest=manifest,results=results),indent=2)+'\n')
print('CPU comparison complete',flush=True)
