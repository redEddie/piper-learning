from dataclasses import replace
import jax
import jax.numpy as jp
import numpy as np
import pytest
from piper_sim_lab.control_lab.controllers import Settings
from piper_sim_lab.control_lab.smooth_plan import velocity_step,stop_sequence,speed_bounds

SETTINGS=[Settings(velocity=.75,acceleration=4.,jerk=300.,qp_jerk=True),
          Settings(velocity=1.,acceleration=5.,jerk=100.,qp_jerk=True)]


@pytest.mark.parametrize('S',SETTINGS)
def test_velocity_step_does_not_enter_previously_infeasible_speed_corner(S):
    @jax.jit
    def run():
        def f(carry,i):
            v,a=carry;target=jp.full(6,jp.where(i<500,S.velocity,-S.velocity))
            vn,an=velocity_step(v,a,target,S)
            return (vn,an),jp.concatenate([vn,an,(an-a)/S.dt])
        return jax.lax.scan(f,(jp.zeros(6),jp.zeros(6)),jp.arange(1000))[1]
    x=np.asarray(run())
    assert np.max(np.abs(x[:,:6]))<=S.velocity+1e-5
    assert np.max(np.abs(x[:,6:12]))<=S.acceleration+.001
    assert np.max(np.abs(x[:,12:]))<=S.jerk+.1
    np.testing.assert_allclose(x[499,:6],S.velocity,atol=1e-5)
    np.testing.assert_allclose(x[-1,:6],-S.velocity,atol=1e-5)


@pytest.mark.parametrize('S',SETTINGS)
def test_every_reachable_velocity_state_has_a_bounded_full_stop(S):
    @jax.jit
    def generate():
        def f(carry,i):
            ref,v,a=jp.split(carry,3)
            target=jp.sin(jp.arange(6)+i*.13)*S.velocity
            vn,an=velocity_step(v,a,target,S)
            st=jp.concatenate([ref+S.dt*vn,vn,an])
            return st,st
        return jax.lax.scan(f,jp.zeros(18),jp.arange(600))[1]
    states=generate()[::7]
    paths=jax.jit(jax.vmap(lambda st:stop_sequence(st,S,64)))(states)
    x=np.asarray(paths);initial=np.asarray(states)
    assert np.max(np.abs(x[:,-1,6:]))<1e-3
    previous=np.concatenate([initial[:,None,:],x[:,:-1]],axis=1)
    assert np.max(np.abs(x[:,:,6:12]))<=S.velocity+1e-5
    assert np.max(np.abs(x[:,:,12:]))<=S.acceleration+.001
    assert np.max(np.abs((x[:,:,12:]-previous[:,:,12:])/S.dt))<=S.jerk+.1


@pytest.mark.parametrize('S',SETTINGS)
def test_velocity_step_reaches_zero_without_pulse_area_limit_cycle(S):
    @jax.jit
    def run():
        def f(carry,i):
            v,a=carry;target=jp.full(6,jp.where(i<20,.3,0.))
            vn,an=velocity_step(v,a,target,S)
            return (vn,an),jp.concatenate([vn,an])
        return jax.lax.scan(f,(jp.zeros(6),jp.zeros(6)),jp.arange(200))[1]
    x=np.asarray(run());assert np.max(np.abs(x[-20:]))==0
