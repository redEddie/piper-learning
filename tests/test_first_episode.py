"""Successful termination must remain successful; no post-terminal metrics leak."""
from typing import NamedTuple
import jax
import jax.numpy as jp
import numpy as np
from piper_sim_lab.first_episode import latch


class State(NamedTuple):
    obs: object
    done: object
    metrics: object


def test_success_and_failure_are_frozen_independently():
    before=State(jp.zeros((2,3)),jp.zeros(2),dict(success=jp.zeros(2)))
    ending=State(jp.ones((2,3)),jp.ones(2),dict(success=jp.array([1.,0.])))
    step=jax.jit(latch)
    ended,terminal,failed,active=step(before,ending,jp.zeros(2,bool),jp.zeros(2,bool))
    np.testing.assert_array_equal(terminal,[True,True]);np.testing.assert_array_equal(failed,[False,True])
    np.testing.assert_array_equal(active,[True,True])
    later=State(jp.full((2,3),100.),jp.zeros(2),dict(success=jp.array([0.,1.])))
    final,terminal,failed,active=step(ended,later,terminal,failed)
    np.testing.assert_array_equal(final.obs,ending.obs)
    np.testing.assert_array_equal(final.metrics['success'],[1.,0.])
    np.testing.assert_array_equal(active,[False,False]);np.testing.assert_array_equal(failed,[False,True])


def test_shared_empty_warp_buffers_have_no_world_axis_and_stay_fresh():
    before=State(dict(per_world=jp.zeros((2,3)),empty=jp.zeros((0,)),shared=jp.zeros((6,))),jp.ones(2),dict(success=jp.ones(2)))
    later=State(dict(per_world=jp.ones((2,3)),empty=jp.zeros((0,)),shared=jp.ones((6,))),jp.zeros(2),dict(success=jp.zeros(2)))
    result,_,_,_=jax.jit(latch)(before,later,jp.ones(2,bool),jp.zeros(2,bool))
    np.testing.assert_array_equal(result.obs['per_world'],before.obs['per_world'])
    np.testing.assert_array_equal(result.obs['shared'],later.obs['shared'])
    assert result.obs['empty'].shape==(0,)
