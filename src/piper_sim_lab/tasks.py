"""Task registry: ``environment.task`` in a config selects the environment class."""

from __future__ import annotations

from typing import Any

from mujoco_playground._src import mjx_env


def make_env(environment: dict[str, Any], **overrides: Any) -> mjx_env.MjxEnv:
    """Build the task named by ``environment['task']`` (default ``reach``).

    The ``task`` key is removed before the rest is passed as config overrides,
    so each environment only sees its own settings.
    """
    config = dict(environment, **overrides)
    task = config.pop("task", "reach")
    if task == 'stage_reward_pick_place_five_starts_v2':
        from piper_sim_lab.five_start_stage import FiveStartStageRewardPickPlace
        return FiveStartStageRewardPickPlace(config_overrides=config)
    if task == "stage_reward_pick_place_random_positions":
        from piper_sim_lab.position_random_stage import PositionRandomStageRewardPickPlace
        return PositionRandomStageRewardPickPlace(config_overrides=config)
    if task == "stage_reward_pick_place":
        from piper_sim_lab.stage_reward_env import StageRewardPickPlace
        return StageRewardPickPlace(config_overrides=config)
    if task == "reach":
        from piper_sim_lab.env import PiperReach
        return PiperReach(config_overrides=config)
    if task == "pick_place":
        from piper_sim_lab.pick_place import PiperPickPlace
        return PiperPickPlace(config_overrides=config)
    if task == "show76_ternary_qp":
        from piper_sim_lab.ternary_control import TernaryControlledPickPlace
        return TernaryControlledPickPlace(config_overrides=config)
    if task == "show100_smooth_grip":
        from piper_sim_lab.smooth_grip import SmoothGripPickPlace
        return SmoothGripPickPlace(config_overrides=config)
    if task == "gentle_placement":
        from piper_sim_lab.gentle_placement import GentlePlacement
        return GentlePlacement(config_overrides=config)
    if task == "controlled_release":
        from piper_sim_lab.controlled_release import ControlledRelease
        return ControlledRelease(config_overrides=config)
    if task == "quadratic_release":
        from piper_sim_lab.quadratic_release import QuadraticRelease
        return QuadraticRelease(config_overrides=config)
    if task == "direct_gripper_release":
        from piper_sim_lab.direct_gripper import DirectGripperRelease
        return DirectGripperRelease(config_overrides=config)
    if task == "show76_place_refine":
        from piper_sim_lab.placement_refinement import PlacementControlledPickPlace
        return PlacementControlledPickPlace(config_overrides=config)
    raise ValueError(f"unknown task {task!r}; expected 'reach' or 'pick_place'")
