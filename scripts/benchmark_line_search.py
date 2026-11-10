"""Steady-state Warp task-step benchmark; compile/warmup excluded.

256 parallel held-block environments, identical initial seeds and commands.
Both model variants are compiled before alternating timed runs.
"""
import os,sys,time,json,hashlib,argparse
from pathlib import Path
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--num-envs',type=int,default=256)
parser.add_argument('--output',type=Path,default=Path(__file__).resolve().parents[1]/'runs/contact-solver-validation/benchmark.json')
args=parser.parse_args()
if args.num_envs<=0:parser.error('--num-envs must be positive')
args.output.parent.mkdir(parents=True,exist_ok=True)
os.environ['JAX_PLATFORMS']='cuda,cpu';os.environ['XLA_PYTHON_CLIENT_PREALLOCATE']='false'
os.environ['JAX_DEFAULT_MATMUL_PRECISION']='highest'
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import jax,jax.numpy as jp,numpy as np
from piper_sim_lab.pick_place import PiperPickPlace
root=args.output.parent;N=args.num_envs;control_steps=16
variants={};samples={4:[],20:[]};warmup={}
for limit in [4,20]:
 env=PiperPickPlace(config_overrides={'impl':'warp','batch_capacity':N,'solver_ls_iterations':limit,
  'reset_stage_probs':[0.,0.,1.],'block_half':.025,'block_mass':.08})
 initial=jax.jit(jax.vmap(env.reset))(jax.random.split(jax.random.key(2026),N));jax.block_until_ready(initial)
 action=jp.tile(jp.array([0.,0.,0.,0.,0.,0.,-1.]),(N,1))
 def make_rollout(env,action):
  @jax.jit
  def rollout(state):
   def step(state,_):
    state=jax.vmap(env.step)(state,action)
    return state,None
   return jax.lax.scan(step,state,None,length=control_steps)[0]
  return rollout
 rollout=make_rollout(env,action)
 start=time.perf_counter()
 for _ in range(3):jax.block_until_ready(rollout(initial))
 warmup[limit]=time.perf_counter()-start
 variants[limit]=(initial,rollout)
 print('compiled and warmed',limit,round(warmup[limit],3),flush=True)
# Alternate order each round to reduce drift/thermal/order bias.
for repeat in range(9):
 for limit in ([4,20] if repeat%2==0 else [20,4]):
  initial,rollout=variants[limit]
  start=time.perf_counter();final=rollout(initial);jax.block_until_ready(final)
  elapsed=time.perf_counter()-start;samples[limit].append(elapsed)
  assert np.isfinite(np.asarray(final.data.qpos)).all()
  assert float(jp.max(final.metrics['capacity_overflow']))==0
  print('repeat',repeat,'limit',limit,'seconds',round(elapsed,6),flush=True)
med={k:float(np.median(v)) for k,v in samples.items()}
paired=np.array(samples[20])/np.array(samples[4])
result={'backend':'MJX-Warp','num_envs':N,'control_steps_per_call':control_steps,
 'physics_steps_per_control':env.n_substeps,'policy_dt':float(env.dt),'physics_dt':float(env.sim_dt),
 'repeats':9,'warmup_calls_per_variant':3,'warmup_seconds_excluded':warmup,
 'seconds_per_call':samples,'median_seconds':med,'increase_percent':100*(med[20]/med[4]-1),
 'paired_median_increase_percent':float(100*(np.median(paired)-1)),
 'mean_seconds':{k:float(np.mean(v)) for k,v in samples.items()},
 'mean_increase_percent':float(100*(np.mean(samples[20])/np.mean(samples[4])-1)),
 'paired_ratio_q25_q75':np.quantile(paired,[.25,.75]).tolist(),
 'env_control_steps_per_second':{k:N*control_steps/val for k,val in med.items()},
 'source_sha256':hashlib.sha256((Path(__file__).resolve().parents[1]/'src/piper_sim_lab/pick_place.py').read_bytes()).hexdigest(),
 'scope':'task.step including IK, physics and reward; held-block state; no PPO optimization, perception or rendering; actual full training slowdown is not measured'}
args.output.write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
