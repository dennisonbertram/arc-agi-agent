"""Reward shaping for ARC-AGI-3."""
import numpy as np


class RewardShaper:
    def __init__(self, win_reward=10.0, game_over_penalty=-5.0, level_bonus=2.0,
                 step_cost=-0.01, undo_penalty=-0.05, reset_penalty=-0.1,
                 grid_change_bonus=0.2, no_change_penalty=-0.005):
        self.win_reward = win_reward
        self.game_over_penalty = game_over_penalty
        self.level_bonus = level_bonus
        self.step_cost = step_cost
        self.undo_penalty = undo_penalty
        self.reset_penalty = reset_penalty
        self.grid_change_bonus = grid_change_bonus
        self.no_change_penalty = no_change_penalty

    def compute_reward(self, prev_grid, curr_grid, curr_state, action_type, levels_before, levels_after):
        r = self.step_cost
        s = str(curr_state)
        if "WIN" in s:
            r += self.win_reward
        elif "GAME_OVER" in s:
            r += self.game_over_penalty
        if levels_after > levels_before:
            r += self.level_bonus * (1.0 + 0.5 * levels_after)
        if prev_grid is not None and curr_grid is not None:
            r += self.grid_change_bonus if self._differ(prev_grid, curr_grid) else self.no_change_penalty
        if action_type == 7:
            r += self.undo_penalty
        if action_type == 0 and "GAME_OVER" not in s:
            r += self.reset_penalty
        return r

    def _differ(self, a, b):
        try:
            return not np.array_equal(np.asarray(a), np.asarray(b))
        except Exception:
            return True
