import numpy as np
import jax
import jax.numpy as jp
import osqp
from scipy import sparse
from piper_sim_lab.control_lab.tracking_qp import project_tracking
from piper_sim_lab.control_lab.tracking_qp import tracking_problem,TrackingSettings
from piper_sim_lab.control_lab.controllers import Settings
from dataclasses import replace
import pytest


def independent_solve(target,lo,hi,slo,shi,weight,normal,minimum):
    eye=np.eye(6);zero=np.zeros((6,6))
    matrix=np.vstack([np.c_[eye,zero,zero],np.r_[normal,np.zeros(12)][None],
        np.c_[eye,eye,zero],np.c_[eye,zero,-eye],np.c_[np.zeros((12,6)),np.eye(12)]])
    solver=osqp.OSQP()
    solver.setup(P=sparse.diags(np.r_[np.ones(6),np.full(12,weight)],format='csc'),
        q=np.r_[-target,np.zeros(12)],A=sparse.csc_matrix(matrix),
        l=np.r_[lo,minimum,slo,np.full(6,-np.inf),np.zeros(12)],
        u=np.r_[hi,np.inf,np.full(6,np.inf),shi,np.full(12,np.inf)],
        verbose=False,eps_abs=1e-8,eps_rel=1e-8,max_iter=100000,polishing=True)
    r=solver.solve(raise_error=False)
    assert r.info.status_val==1
    return r.x[:6]


def test_piecewise_qp_matches_independent_slack_qp():
    rng=np.random.default_rng(9901);solve=jax.jit(project_tracking)
    for _ in range(100):
        lo=rng.uniform(-1,0,6);hi=rng.uniform(0,1,6)
        slo=rng.uniform(-2,.5,6);shi=slo+rng.uniform(.1,2,6)
        target,normal=rng.normal(size=(2,6));weight=float(rng.choice([1.,10.,100.]))
        minimum=normal@np.where(normal>=0,hi,lo)-rng.uniform(.001,2)
        expected=independent_solve(target,lo,hi,slo,shi,weight,normal,minimum)
        got,ok=solve(*map(jp.asarray,(target,lo,hi,slo,shi,weight,normal,minimum)))
        assert ok
        np.testing.assert_allclose(got,expected,atol=3e-5)


def test_hard_infeasibility_is_not_hidden_by_soft_tracking():
    v,ok=project_tracking(jp.zeros(6),jp.ones(6),-jp.ones(6),
        -jp.ones(6),jp.ones(6),100.,jp.ones(6),0.)
    assert not ok
    v,ok=project_tracking(jp.zeros(6),-jp.ones(6),jp.ones(6),
        -jp.ones(6),jp.ones(6),100.,jp.ones(6),7.)
    assert not ok


def problem(s,velocity,acceleration,**kwargs):
    q=np.zeros(6)
    return tracking_problem(q,q,q,velocity,np.ones(6),np.full(6,-3.),np.full(6,3.),
        s,TrackingSettings(),np.zeros(6,bool),q,previous_acceleration=acceleration,**kwargs)


def test_jerk_window_is_acceleration_change_not_velocity_change():
    s=Settings(qp_jerk=True,jerk=100.)
    velocity=np.full(6,.2);acceleration=np.full(6,2.)
    _,lo,hi,*_=problem(s,velocity,acceleration)
    np.testing.assert_allclose(lo,.2075,atol=1e-12)
    np.testing.assert_allclose(hi,.2125,atol=1e-12)
    for v in (lo,hi):
        assert np.max(np.abs(((v-velocity)/s.dt-acceleration)/s.dt))<=100.+1e-9


def test_disabled_jerk_does_not_change_existing_bounds():
    s=Settings()
    v=np.full(6,.1)
    a=problem(s,v,None)
    b=problem(s,v,np.full(6,1e6))
    for x,y in zip(a,b):np.testing.assert_array_equal(x,y)


def test_enabled_jerk_requires_history():
    with pytest.raises(ValueError,match='previous reference acceleration'):
        problem(Settings(qp_jerk=True),np.zeros(6),None)


def test_jerk_does_not_silently_relax_conflicting_velocity_limit():
    s=Settings(qp_jerk=True,jerk=100.)
    target,lo,hi,slo,shi,*_=problem(s,np.full(6,1.49),np.full(6,8.))
    _,ok=project_tracking(*map(jp.asarray,(target,lo,hi,slo,shi,100.,np.zeros(6),-1.)))
    assert np.any(lo>hi) and not ok


def test_jerk_projection_matches_independent_qp():
    s=Settings(qp_jerk=True,jerk=300.)
    rng=np.random.default_rng(7761)
    solve=jax.jit(project_tracking)
    for _ in range(20):
        v=rng.uniform(-.3,.3,6);a=rng.uniform(-3.,3.,6)
        target,lo,hi,slo,shi,*_=problem(s,v,a)
        normal=rng.normal(size=6);minimum=normal@np.clip(v+s.dt*a,lo,hi)-.001
        expected=independent_solve(target,lo,hi,slo,shi,100.,normal,minimum)
        got,ok=solve(*map(jp.asarray,(target,lo,hi,slo,shi,100.,normal,minimum)))
        assert ok
        np.testing.assert_allclose(got,expected,atol=3e-5)
