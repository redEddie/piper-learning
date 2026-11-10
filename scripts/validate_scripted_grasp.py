"""Scripted approach/close/lift/transport/release using the actual patched task.

C uses the same DLS equations with MuJoCo's analytic Jacobian; JAX/Warp use
the task's env.step. This is a closed-loop regression, not a policy eval.
"""
import argparse,os,sys,json,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--backend',choices=['c','jax','warp'],required=True)
p.add_argument('--ls',type=int,default=20);p.add_argument('--num-envs',type=int,default=8)
p.add_argument('--block',type=int,choices=[30,50],default=50)
p.add_argument('--output-root',type=Path,default=Path(__file__).resolve().parents[1]/'runs/contact-solver-validation')
a=p.parse_args();out=a.output_root/f'grasp-{a.backend}-block{a.block}-ls{a.ls}'
out.mkdir(parents=True,exist_ok=True)
os.environ['JAX_PLATFORMS']='cuda,cpu' if a.backend=='warp' else 'cpu'
os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false';os.environ['MUJOCO_GL']='egl'
os.environ['JAX_DEFAULT_MATMUL_PRECISION']='highest'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import jax,jax.numpy as jp,numpy as np,mujoco
from mujoco import mjx
from piper_sim_lab.pick_place import PiperPickPlace
N=a.num_envs
env=PiperPickPlace(config_overrides={'impl':'warp' if a.backend=='warp' else 'jax','batch_capacity':N,
 'solver_ls_iterations':a.ls,'reset_stage_probs':[1.,0.,0.],
 'block_half':a.block/2000,'block_mass':.02 if a.block==30 else .08})
c=env._config;m=env.mj_model
mujoco.mj_saveModel(m,str(out/'model.mjb'))
s=jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(3),N));jax.block_until_ready(s)
if a.backend=='c':
 ds=mjx.get_data(m,s.data)
 jacp=np.zeros((3,m.nv));jacr=np.zeros((3,m.nv))
 def tcp():return np.stack([d.site_xpos[env._tcp_id].copy() for d in ds])
 def block():return np.stack([d.xpos[env._block_id].copy() for d in ds])
 def bowl():return np.stack([d.mocap_pos[env._bowl_mocap_id].copy() for d in ds])
 def poses():return np.stack([d.qpos.copy() for d in ds]),np.stack([d.mocap_pos.copy() for d in ds])
 def widths():return np.array([d.qpos[env._finger_qpos_ids[0]]-d.qpos[env._finger_qpos_ids[1]] for d in ds])
 def step(actions):
  for d,action in zip(ds,actions):
   mujoco.mj_comPos(m,d);mujoco.mj_jacSite(m,d,jacp,jacr,env._tcp_id)
   J=np.concatenate((jacp,jacr))[:,env._qvel_ids]
   twist=np.r_[action[:3]*c.max_lin_vel,action[3:6]*c.max_ang_vel]
   qv=J.T@np.linalg.solve(J@J.T+c.ik_damping**2*np.eye(6),twist)
   qv=np.clip(qv,-c.max_joint_vel,c.max_joint_vel)
   desired=d.ctrl[env._arm_ctrl_ids]+qv*c.ctrl_dt
   d.ctrl[env._arm_ctrl_ids]=np.clip(np.clip(desired,d.qpos[env._qpos_ids]-c.max_target_lag,d.qpos[env._qpos_ids]+c.max_target_lag),np.asarray(env._lower),np.asarray(env._upper))
   previous=d.ctrl[env._gripper_ctrl_id]
   d.ctrl[env._gripper_ctrl_id]=np.clip(c.gripper_max*(action[6]+1)/2,previous-c.gripper_rate,previous+c.gripper_rate)
   for _ in range(env.n_substeps):mujoco.mj_step(m,d)
   mujoco.mj_kinematics(m,d)
else:
 step_fn=jax.jit(jax.vmap(env.step))
 def tcp():return np.asarray(s.data.site_xpos[:,env._tcp_id])
 def block():return np.asarray(s.data.xpos[:,env._block_id])
 def bowl():return np.asarray(s.data.mocap_pos[:,env._bowl_mocap_id])
 def poses():return np.asarray(s.data.qpos),np.asarray(s.data.mocap_pos)
 def widths():return np.asarray(s.data.qpos[:,env._finger_qpos_ids[0]]-s.data.qpos[:,env._finger_qpos_ids[1]])
 def step(actions):
  global s
  s=step_fn(s,jp.array(actions));jax.block_until_ready(s)
trajectory=[];phase=[]
def record(label):
 q,mo=poses();trajectory.append((q.copy(),mo.copy()));phase.append(label)
def goto(target,count,grip,label):
 for _ in range(count):
  velocity=np.clip((target-tcp())/.01,-1,1)
  step(np.c_[velocity,np.zeros((N,3)),np.full(N,grip)]);record(label)
def grip(value,count,label):
 for _ in range(count):step(np.tile([0,0,0,0,0,0,value],(N,1)));record(label)
start=time.monotonic();initial_block=block().copy();initial_bowl=bowl().copy();record('start')
goto(initial_block+[0,0,.08],25,1,'approach')
goto(initial_block+[0,0,.006],20,1,'descend')
approach_distance=np.linalg.norm(block()-tcp(),axis=1)
grip(-1,8,'close');grip_width=widths()
goto(tcp()+[0,0,.10],20,-1,'lift');lift_height=block()[:,2]-c.block_half
goto(initial_bowl+[0,0,.11],40,-1,'transport');xy=np.linalg.norm((block()-bowl())[:,:2],axis=1)
grip(1,18,'release')
b=block();t=tcp();bo=bowl();near=np.linalg.norm(b-t,axis=1)<c.grasp_distance
inside=(np.linalg.norm((b-bo)[:,:2],axis=1)<c.bowl_r_bottom-c.block_half)&(b[:,2]<bo[:,2]+env._bowl_rim_z)&(b[:,2]>bo[:,2])&~near
q,mocap=map(np.stack,zip(*trajectory));np.savez_compressed(out/'trajectory.npz',qpos=q,mocap_pos=mocap,phase=np.array(phase))
result={'backend':a.backend,'ls_iterations':a.ls,'num_envs':N,'block_mm':a.block,'seed':3,
 'reset_stage_probs':[1,0,0],'scripted_success':int(inside.sum()),'approach_distance_m':approach_distance.tolist(),
 'grip_width_m':grip_width.tolist(),'lift_height_m':lift_height.tolist(),'above_bowl_xy_m':xy.tolist(),
 'final_block_position':b.tolist(),'finite':bool(np.isfinite(q).all()),'wall_seconds_including_compilation':time.monotonic()-start,
 'scope':'scripted closed-loop grasp, not learned policy; C uses equivalent NumPy DLS controller',
 'capacity_overflow':float(jp.max(s.metrics['capacity_overflow'])) if a.backend!='c' else None}
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
