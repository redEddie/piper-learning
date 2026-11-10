"""Load upstream Brax policies while preserving optional null initializers.

The pinned Brax saver writes ``mean_kernel_init_fn: null`` but its config
loader looks up None in the initializer registry. This adapter only decodes
that configuration; network construction and parameter loading stay upstream.
"""

from collections.abc import Mapping
import json
from numbers import Integral
from pathlib import Path

from brax.training import checkpoint as brax_checkpoint
from brax.training import networks as brax_networks
from brax.training.agents.ppo import networks as ppo_networks
from ml_collections import config_dict


_INITIALIZER_FIELDS = (
    "policy_network_kernel_init_fn", "value_network_kernel_init_fn",
    "q_network_kernel_init_fn", "mean_kernel_init_fn",
)


def _observation_shapes(spec):
    """Normalize saved specs and MjxEnv shape tuples for an exact comparison."""
    if isinstance(spec, Mapping):
        if "shape" in spec:
            return _observation_shapes(spec["shape"])
        return {key: _observation_shapes(value) for key, value in spec.items()}
    if isinstance(spec, Integral) and not isinstance(spec, bool):
        spec = (int(spec),)
    if isinstance(spec, (tuple, list)) and spec and all(
        isinstance(size, Integral) and not isinstance(size, bool) and size > 0
        for size in spec
    ):
        return tuple(int(size) for size in spec)
    raise ValueError(f"Unsupported checkpoint observation shape: {spec!r}")


def load_policy(path, *, observation_size, action_size, deterministic=True):
    """Restore a saved policy only when its observation/action shapes match.

    No checkpoint files are rewritten. Shape compatibility does not certify
    matching observation semantics; use the checkpoint's environment config.
    """
    path = Path(path).resolve()
    raw = json.loads((path / "ppo_network_config.json").read_text())
    saved_shapes = _observation_shapes(raw["observation_size"])
    expected_shapes = _observation_shapes(observation_size)
    if saved_shapes != expected_shapes:
        raise ValueError(
            f"Checkpoint observation shape mismatch: saved={saved_shapes}, "
            f"environment={expected_shapes}"
        )
    if raw["action_size"] != action_size:
        raise ValueError(
            f"Checkpoint action size mismatch: saved={raw['action_size']}, "
            f"environment={action_size}"
        )

    kwargs = raw["network_factory_kwargs"]
    if "activation" in kwargs:
        name = kwargs["activation"]
        if name not in brax_networks.ACTIVATION:
            raise ValueError(f"Unknown checkpoint activation: {name!r}")
        kwargs["activation"] = brax_networks.ACTIVATION[name]
    for field in _INITIALIZER_FIELDS:
        name = kwargs.get(field)
        if name is None:
            continue
        if name not in brax_networks.KERNEL_INITIALIZER:
            raise ValueError(f"Unknown checkpoint initializer for {field}: {name!r}")
        kwargs[field] = brax_networks.KERNEL_INITIALIZER[name]

    network = brax_checkpoint.get_network(
        config_dict.create(**raw), ppo_networks.make_ppo_networks,
    )
    params = brax_checkpoint.load(str(path))
    return ppo_networks.make_inference_fn(network)(params, deterministic=deterministic)
