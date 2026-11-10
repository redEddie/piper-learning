"""Copy SHOW-100 weights, insert an initially ignored observed phase channel."""
from pathlib import Path
import copy,hashlib,json
import jax
import jax.numpy as jp
import numpy as np
from ml_collections import ConfigDict
from brax.training import checkpoint
from piper_sim_lab.checkpoint import load_policy

ROOT=Path(__file__).resolve().parents[2]
SOURCE=ROOT.parent/'show100-ternary-dls-qp/runs/show100/mlp-fresh-50m/checkpoints/000050331648'
OUT=ROOT/'runs/smooth-grip/bootstrap'
if OUT.exists():raise FileExistsError(OUT)
params=checkpoint.load(str(SOURCE));old=copy.deepcopy(params)
stats=params[0]
def insert(x,value):return jp.concatenate([jp.asarray(x[:42]),jp.array([value],dtype=jp.asarray(x).dtype),jp.asarray(x[42:])])
count=float(np.asarray(stats.count.hi))*2**32+float(np.asarray(stats.count.lo))
params[0]=stats.replace(mean={k:insert(v,0.) for k,v in stats.mean.items()},
    std={k:insert(v,1.) for k,v in stats.std.items()},
    summed_variance={k:insert(v,count) for k,v in stats.summed_variance.items()})
for network in params[1:]:
    kernel=network['params']['hidden_0']['kernel']
    network['params']['hidden_0']['kernel']=np.insert(kernel,42,0.,axis=0)
raw=json.loads((SOURCE/'ppo_network_config.json').read_text())
raw['observation_size']['state']['shape']=[50];raw['observation_size']['privileged']['shape']=[67]
# Brax's saver accepts initializer callables, not the saved string registry keys.
for field in ['policy_network_kernel_init_fn','value_network_kernel_init_fn','mean_kernel_init_fn']:
    raw['network_factory_kwargs'].pop(field,None)
raw['network_factory_kwargs'].pop('activation',None)
# Write parameters with raw metadata separately, preserving exact saved network semantics.
checkpoint.save(str(OUT),50331648,params,ConfigDict(raw),'ppo_network_config.json')
target=OUT/'000050331648'
source_meta=json.loads((SOURCE/'ppo_network_config.json').read_text())
source_meta['observation_size']['state']['shape']=[50];source_meta['observation_size']['privileged']['shape']=[67]
(target/'ppo_network_config.json').write_text(json.dumps(source_meta)+'\n')
original=load_policy(SOURCE,observation_size={'state':(49,),'privileged':(66,)},action_size=7)
expanded=load_policy(target,observation_size={'state':(50,),'privileged':(67,)},action_size=7)
rng=np.random.default_rng(10024)
obs={k:jp.asarray(rng.normal(size=(64,n)),jp.float32) for k,n in [('state',49),('privileged',66)]}
newobs={k:jp.concatenate([v[:,:42],jp.asarray(rng.uniform(0,1,(64,1)),jp.float32),v[:,42:]],axis=1) for k,v in obs.items()}
a=np.asarray(original(obs,jax.random.key(0))[0]);b=np.asarray(expanded(newobs,jax.random.key(0))[0]);error=float(np.max(np.abs(a-b)))
assert error<1e-5,error
report=dict(source=str(SOURCE),target=str(target),added_phase_at_index=42,actor_dimensions=[49,50],critic_dimensions=[66,67],
    initial_policy_max_action_difference=error,source_sha256={str(p.relative_to(SOURCE)):hashlib.sha256(p.read_bytes()).hexdigest() for p in SOURCE.rglob('*') if p.is_file()},
    policy_value_normalizer_restored=True,optimizer_and_rng_restart=True,real_robot=False)
(ROOT/'docs/smooth-grip/bootstrap.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
