"""Downstream extension: reuse robot/reset/observations, change only the reward.

Illustrates the inheritance boundary; this sparse variant is not the trained
baseline and may need a curriculum. Pass its instance to the same Brax trainer.
"""
from piper_sim_lab.env import PiperReach


class SparsePiperReach(PiperReach):
    def step(self, state, action):
        next_state = super().step(state, action)
        reward = (
            next_state.metrics["success"]
            - next_state.metrics["action_penalty"]
            - next_state.metrics["nonfinite"]
        )
        return next_state.replace(reward=reward)
