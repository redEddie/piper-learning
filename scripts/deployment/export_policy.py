"""Export existing actor weights; never trains, rewrites checkpoints or connects CAN."""
import argparse
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
os.environ['JAX_PLATFORMS']='cpu'


from piper_runtime.contracts import observation_layout, tcp_local_position


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--run',type=Path,required=True);ap.add_argument('--checkpoint',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--cumulative-training-steps',type=int,help='Include parent warm-start steps in the lineage total')
    ap.add_argument('--runtime-profile', choices=['show108-mlp-lstm-v1'],
                    help='Validate the export against the installed common piper-runtime contract')
    args=ap.parse_args()
    import jax
    import jax.numpy as jp
    import numpy as np
    from flax import serialization
    from brax.training import checkpoint as brax_checkpoint
    from brax.training.acme import running_statistics as stats
    from piper_sim_lab.recurrent import ActorAdapter,zero_carry
    cfg=json.loads((args.run/'config.json').read_text())
    checkpoint=args.checkpoint or sorted((args.run/'checkpoints').iterdir())[-1]
    if (checkpoint/'learner.msgpack').exists():
        raw=serialization.msgpack_restore((checkpoint/'learner.msgpack').read_bytes())
        params=raw['params']['policy'];normalizer=raw['normalizer'];step=int(raw['step'])
        kind=cfg['actor'];format_name='matched-recurrent-msgpack'
    else:
        norm,params,*_=brax_checkpoint.load(str(checkpoint.resolve()))
        normalizer=serialization.to_state_dict(norm);step=int(checkpoint.name)
        kind='mlp';format_name='brax-orbax'
        raw=json.loads((checkpoint/'ppo_network_config.json').read_text())
        kwargs=raw['network_factory_kwargs']
        if kwargs.get('activation','swish') not in ('swish','silu') or kwargs.get('distribution_type','tanh_normal')!='tanh_normal':
            raise ValueError('Exporter supports the existing swish / tanh_normal policy only')
    if kind not in ('mlp', 'lstm', 'mlp_lstm'):
        raise ValueError('Portable export supports mlp, lstm and mlp_lstm actors')
    env=cfg.get('resolved_environment',cfg['environment']);key=cfg['network']['policy_obs_key']
    sizes={k:tuple(np.shape(v)) for k,v in normalizer['mean'].items()}
    actor=ActorAdapter(sizes,7,kind,cfg['network'],cfg.get('recurrent_width',128),
                       cfg.get('actor_architecture'),cfg.get('exploration'))
    norm=serialization.from_state_dict(stats.init_state({k:jp.zeros(v) for k,v in sizes.items()}),normalizer)
    weights={}
    def flatten(node,prefix=''):
        for k,v in node.items():
            path=f'{prefix}/{k}' if prefix else k
            if isinstance(v,Mapping):flatten(v,path)
            else:weights[path]=np.asarray(v,dtype=np.float32)
    flatten(params)
    weights['normalizer/mean']=np.asarray(normalizer['mean'][key],dtype=np.float32)
    weights['normalizer/std']=np.asarray(normalizer['std'][key],dtype=np.float32)
    if kind in ('lstm', 'mlp_lstm'):
        weights['runtime/lstm_input_kernel']=np.concatenate([weights[f'params/lstm/i{k}/kernel'] for k in 'ifgo'],axis=-1)
        weights['runtime/lstm_hidden_kernel']=np.concatenate([weights[f'params/lstm/h{k}/kernel'] for k in 'ifgo'],axis=-1)
        weights['runtime/lstm_bias']=np.concatenate([weights[f'params/lstm/h{k}/bias'] for k in 'ifgo'])
    layout=observation_layout(env);offset=layout[-1]['stop']
    if offset!=sizes[key][0]:raise ValueError('Unknown observation contract; refusing to label export')
    metadata={'schema_version':1,'actor':kind,'recurrent_width':actor.width if kind!='mlp' else 0,
        'head_sizes':cfg['network']['policy_hidden_layer_sizes'],'observation_size':offset,'action_size':7,
        'observation_layout':layout,'action_contract':'normalized [-1,1]: base-frame linear/angular twist commands (6) and gripper action (1); NOT direct joint/CAN commands',
        'environment':env,'training_step':step,'checkpoint_format':format_name,
        'source_checkpoint':str(checkpoint.resolve()),
        'code_commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip()}
    if kind == 'mlp_lstm':
        metadata.update(encoder_sizes=list((cfg.get('actor_architecture') or {}).get('hidden_sizes',[256,64])),
                        head_sizes=[], checkpoint_sha256=hashlib.sha256((checkpoint/'learner.msgpack').read_bytes()).hexdigest())
    if (args.run/'provenance.json').exists():
        metadata['training_source_commit'] = json.loads((args.run/'provenance.json').read_text()).get('git',{}).get('commit')
    cumulative = args.cumulative_training_steps if args.cumulative_training_steps is not None else step
    if cumulative < step:
        raise ValueError('Cumulative steps cannot be smaller than checkpoint steps')
    from piper_sim_lab.assets import get_model_path
    from piper_sim_lab.pick_place import ROBOT_MODELS
    robot_model = env.get('robot_model', 'menagerie')
    model_xml = get_model_path(robot_model).with_name('piper.xml')
    metadata.update(cumulative_training_steps=cumulative,checkpoint_step=step,parent_training_steps=cumulative-step,
        warm_start=(json.loads((args.run/'invocation.json').read_text()).get('warm_start')
                    if (args.run/'invocation.json').exists() else None),
        training_config_sha256=hashlib.sha256((args.run/'config.json').read_bytes()).hexdigest(),
        kinematics={'robot_model':robot_model,'tcp_local_position_m':tcp_local_position(ROBOT_MODELS[robot_model]['tcp'],env),
                    'piper_xml_sha256':hashlib.sha256(model_xml.read_bytes()).hexdigest(),
                    'observation_frame':'base_link','base_height_m':env.get('base_height',0.)})
    if args.runtime_profile:
        from piper_runtime.contracts import contract
        from piper_runtime.identity import identity
        metadata.update(runtime_profile_id=args.runtime_profile, runtime_identity=identity())
        metadata['runtime_contract'] = contract(metadata)
    args.output.mkdir(parents=True,exist_ok=False)
    np.savez_compressed(args.output/'weights.npz',**weights)
    metadata['weights_sha256']=hashlib.sha256((args.output/'weights.npz').read_bytes()).hexdigest()
    (args.output/'policy.json').write_text(json.dumps(metadata,indent=2)+'\n')
    shutil.copy2(model_xml,args.output/'piper.xml')
    shutil.copy2(args.run/'config.json',args.output/'training-config.json')
    # Synthetic vectors check transfer arithmetic, not physical success.
    rng=np.random.default_rng(901);obs=(weights['normalizer/mean']+weights['normalizer/std']*
        rng.normal(0,.7,(80,4,offset))).astype(np.float32)
    reset=np.zeros((80,4),bool);reset[0]=True;reset[25,1]=True;reset[50,2]=True;reset[60]=True
    def f(c,item):
        x,mask=item;c,logits=actor.step(norm,params,c,{key:x},mask)
        return c,(actor.dist.mode(logits),logits,c[0],c[1])
    _,values=jax.jit(lambda:jax.lax.scan(f,zero_carry(4,actor.width),(jp.asarray(obs),jp.asarray(reset))))()
    actions,logits,memory_c,memory_h=values
    if format_name == 'brax-orbax':
        from piper_sim_lab.checkpoint import load_policy
        original = load_policy(checkpoint, observation_size=sizes, action_size=7, deterministic=True)
        original_obs = {name:jp.zeros((80,4)+shape) for name,shape in sizes.items()}
        original_obs[key] = jp.asarray(obs)
        actions,_ = jax.jit(original)(original_obs,jax.random.key(901))
    vectors=dict(observations=obs,resets=reset,actions=np.asarray(actions))
    if kind != 'mlp':
        vectors.update(logits=np.asarray(logits),memory_c=np.asarray(memory_c),memory_h=np.asarray(memory_h))
    np.savez_compressed(args.output/'golden.npz',**vectors)
    from piper_policy_runtime.check import check
    report=check(args.output);(args.output/'export-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'output':str(args.output),'validation':report}))


if __name__=='__main__':main()
