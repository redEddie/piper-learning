"""Actual Brax checkpoint round trips, including its nullable initializer."""

import functools
import json
import os

os.environ["JAX_PLATFORMS"] = "cpu"

from brax.training.acme import running_statistics, specs
from brax.training.agents.ppo import checkpoint, networks
import jax
import jax.numpy as jp
import numpy as np
import pytest

from piper_sim_lab.checkpoint import load_policy


@pytest.fixture(scope="module")
def saved_policy(tmp_path_factory):
    root = tmp_path_factory.mktemp("brax-checkpoint")
    factory = functools.partial(
        networks.make_ppo_networks,
        policy_hidden_layer_sizes=(8, 8), value_hidden_layer_sizes=(8, 8),
        policy_obs_key="state", value_obs_key="state",
        mean_kernel_init_fn=None,
    )
    observation_size = {"state": specs.Array((24,), jp.float32)}
    network = factory(
        {"state": (24,)}, 6, preprocess_observations_fn=running_statistics.normalize,
    )
    normalizer = running_statistics.init_state({"state": jp.zeros(24)}).replace(
        mean={"state": jp.linspace(-0.2, 0.2, 24)},
        std={"state": jp.linspace(0.7, 1.3, 24)},
    )
    params = (
        normalizer,
        network.policy_network.init(jax.random.PRNGKey(0)),
        network.value_network.init(jax.random.PRNGKey(1)),
    )
    config = checkpoint.network_config(observation_size, 6, True, factory)
    checkpoint.save(root, 10, params, config)
    path = root / "000000000010"
    policy = networks.make_inference_fn(network)(params, deterministic=True)
    return path, policy


def test_saved_policy_restores_identical_actions_and_keeps_files_unchanged(saved_policy):
    path, original = saved_policy
    before = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    config = json.loads((path / "ppo_network_config.json").read_text())
    assert config["network_factory_kwargs"]["mean_kernel_init_fn"] is None
    restored = load_policy(path, observation_size={"state": (24,)}, action_size=6)
    obs = {"state": jp.linspace(-1.0, 1.0, 72).reshape(3, 24)}
    key = jax.random.PRNGKey(2)
    expected, _ = original(obs, key)
    actual, _ = restored(obs, key)
    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)
    assert actual.shape == (3, 6)
    after = {p.relative_to(path): p.read_bytes() for p in path.rglob("*") if p.is_file()}
    assert before == after


@pytest.mark.parametrize(
    "observation_size, action_size, message",
    [({"state": (30,)}, 6, "observation shape mismatch"),
     ({"state": (24,)}, 7, "action size mismatch")],
)
def test_checkpoint_rejects_incompatible_environment(saved_policy, observation_size, action_size, message):
    path, _ = saved_policy
    with pytest.raises(ValueError, match=message):
        load_policy(path, observation_size=observation_size, action_size=action_size)
