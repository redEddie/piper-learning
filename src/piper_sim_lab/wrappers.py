"""Fresh Playground resets that retain Brax's terminal transition metadata.

At the pinned Playground revision, full_reset=True resets every info field
except its documented preserve-info payload, steps, and reset counters. Brax
PPO needs truncation and completed-episode statistics from the transition that
just ended. Capture them through that hook and restore them after autoreset.
The upstream physics, vectorization, episode limits, and resets are reused.
"""

from typing import Any, Callable

from brax.envs.wrappers import training as brax_training
import jax
import jax.numpy as jp
from mujoco_playground import wrapper
from mujoco_playground._src import mjx_env


_PRESERVE = "AutoResetWrapper_preserve_info"
_BOUNDARY = "piper_sim_lab_episode_boundary"


def _raise_physics_error(nonfinite, capacity_overflow):
    # vmap can turn cond into select and execute both branch callbacks. Check
    # the supplied flags again so a healthy mapped slot never raises an error.
    failures = []
    if bool(nonfinite):
        failures.append("nonfinite")
    if bool(capacity_overflow):
        failures.append("capacity_overflow")
    if failures:
        raise RuntimeError("Invalid Piper physics: " + ", ".join(failures))


class RuntimeSafety(wrapper.Wrapper):
    """Reject invalid batched physics before autoreset can hide the failure.

    Place outside VmapWrapper and inside Episode/AutoReset wrappers. Healthy
    steps only reduce flags and branch on device; the host callback runs in the
    failing branch. Environments without either metric remain unchanged.
    """

    @staticmethod
    def _check(state: mjx_env.State) -> mjx_env.State:
        names = ("nonfinite", "capacity_overflow")
        if not any(name in state.metrics for name in names):
            return state

        def failed(name):
            value = jp.asarray(state.metrics.get(name, 0.0))
            return jp.any((value > 0) | ~jp.isfinite(value))

        nonfinite, capacity = (failed(name) for name in names)

        def reject(_):
            jax.debug.callback(_raise_physics_error, nonfinite, capacity)

        jax.lax.cond(nonfinite | capacity, reject, lambda _: None, operand=None)
        return state

    def reset(self, rng):
        return self._check(self.env.reset(rng))

    def step(self, state, action):
        return self._check(self.env.step(state, action))


class _CaptureEpisodeBoundary(wrapper.Wrapper):
    """Runs inside autoreset, immediately after Brax's EpisodeWrapper."""

    @staticmethod
    def _capture(state: mjx_env.State) -> mjx_env.State:
        info = dict(state.info)
        preserved = dict(info.get(_PRESERVE, {}))
        preserved[_BOUNDARY] = {
            "truncation": info["truncation"],
            "episode_done": info["episode_done"],
            "episode_metrics": dict(info["episode_metrics"]),
        }
        info[_PRESERVE] = preserved
        return state.replace(info=info)

    def reset(self, rng):
        return self._capture(self.env.reset(rng))

    def step(self, state, action):
        return self._capture(self.env.step(state, action))


class _RestoreEpisodeBoundary(wrapper.Wrapper):
    """Runs outside autoreset, leaving fresh data, observations and task info."""

    def step(self, state, action):
        state = self.env.step(state, action)
        info = dict(state.info)
        boundary = info[_PRESERVE][_BOUNDARY]
        info["truncation"] = boundary["truncation"]
        info["episode_done"] = boundary["episode_done"]
        info["episode_metrics"] = dict(boundary["episode_metrics"])
        return state.replace(info=info)


def wrap_for_training(
    env: mjx_env.MjxEnv,
    episode_length: int = 1000,
    action_repeat: int = 1,
    randomization_fn: Callable[..., Any] | None = None,
    strict_physics: bool = True,
) -> wrapper.Wrapper:
    """Batch a state-observation environment with fresh per-episode resets.

    ``done`` and episode metadata describe the completed transition, while
    ``data``, ``obs`` and ordinary task ``info`` already describe the new episode
    for terminated slots. Other slots continue without being reset. Existing
    custom values in the official preserve-info payload are retained.

    ``strict_physics=False`` is for tests that deliberately inject failures to
    inspect autoreset behavior; production training and benchmarks keep it on.
    """
    if episode_length <= 0 or action_repeat <= 0:
        raise ValueError("episode_length and action_repeat must be positive")
    task_horizon = getattr(env, "finite_horizon_steps", episode_length)
    if task_horizon != episode_length:
        raise ValueError("finite task horizon must equal episode_length; early wrapper timeout would discard failure costs")
    if hasattr(env,"failure_discount") and action_repeat != 1:
        raise ValueError("failure-cost settlement requires action_repeat=1 to match its discount time unit")
    if randomization_fn is None:
        env = brax_training.VmapWrapper(env)
    else:
        env = wrapper.BraxDomainRandomizationVmapWrapper(env, randomization_fn)
    if strict_physics:
        env = RuntimeSafety(env)
    env = brax_training.EpisodeWrapper(env, episode_length, action_repeat)
    env = _CaptureEpisodeBoundary(env)
    env = wrapper.BraxAutoResetWrapper(env, full_reset=True)
    return _RestoreEpisodeBoundary(env)
