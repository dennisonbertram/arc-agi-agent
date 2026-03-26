"""Tests for RewardShaper."""
import pytest
from src.training.reward_shaper import RewardShaper


@pytest.fixture
def shaper():
    return RewardShaper(
        win_reward=10.0,
        game_over_penalty=-5.0,
        level_bonus=2.0,
        step_cost=-0.01,
        undo_penalty=-0.05,
        reset_penalty=-0.1,
        grid_change_bonus=0.02,
        no_change_penalty=-0.005,
        novelty_bonus=0.1,
    )


def same_grid():
    return [[0] * 4 for _ in range(4)]


def different_grid_b():
    return [[1] * 4 for _ in range(4)]


class TestWinReward:
    def test_win_state_positive(self, shaper):
        r = shaper.compute_reward(None, None, "WIN", action_type=1,
                                  levels_before=0, levels_after=0)
        assert r > 0.0

    def test_win_reward_includes_win_bonus(self, shaper):
        r = shaper.compute_reward(None, None, "WIN", action_type=1,
                                  levels_before=0, levels_after=0)
        # step_cost + win_reward = -0.01 + 10.0 = 9.99
        assert pytest.approx(r, abs=1e-4) == shaper.step_cost + shaper.win_reward

    def test_not_win_state_no_win_reward(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0)
        assert r < shaper.win_reward


class TestGameOverPenalty:
    def test_game_over_negative(self, shaper):
        r = shaper.compute_reward(None, None, "GAME_OVER", action_type=1,
                                  levels_before=0, levels_after=0)
        assert r < 0.0

    def test_game_over_includes_penalty(self, shaper):
        r = shaper.compute_reward(None, None, "GAME_OVER", action_type=1,
                                  levels_before=0, levels_after=0)
        assert pytest.approx(r, abs=1e-4) == shaper.step_cost + shaper.game_over_penalty

    def test_normal_step_no_game_over_penalty(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0)
        assert r > shaper.game_over_penalty


class TestLevelBonus:
    def test_level_increase_gives_bonus(self, shaper):
        r_no_level = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                           levels_before=0, levels_after=0)
        r_level = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                        levels_before=0, levels_after=1)
        assert r_level > r_no_level

    def test_no_level_increase_no_bonus(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=2, levels_after=2)
        # Only step_cost applies (no grid given)
        assert pytest.approx(r, abs=1e-4) == shaper.step_cost

    def test_level_bonus_scaling_with_level_count(self, shaper):
        r1 = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                   levels_before=0, levels_after=1)
        r2 = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                   levels_before=1, levels_after=2)
        # level_bonus * (1.0 + 0.5 * levels_after): higher levels_after -> bigger bonus
        assert r2 > r1

    def test_level_bonus_formula(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=2)
        expected = shaper.step_cost + shaper.level_bonus * (1.0 + 0.5 * 2)
        assert pytest.approx(r, abs=1e-4) == expected


class TestStepCost:
    def test_step_cost_always_applied(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0)
        assert pytest.approx(r, abs=1e-4) == shaper.step_cost

    def test_step_cost_is_negative(self, shaper):
        assert shaper.step_cost < 0.0


class TestUndoPenalty:
    def test_undo_action_incurs_penalty(self, shaper):
        r_normal = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                         levels_before=0, levels_after=0)
        r_undo = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=7,
                                       levels_before=0, levels_after=0)
        assert r_undo < r_normal

    def test_undo_penalty_value(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=7,
                                  levels_before=0, levels_after=0)
        expected = shaper.step_cost + shaper.undo_penalty
        assert pytest.approx(r, abs=1e-4) == expected


class TestResetPenalty:
    def test_reset_action_incurs_penalty(self, shaper):
        r_normal = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                         levels_before=0, levels_after=0)
        r_reset = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=0,
                                        levels_before=0, levels_after=0)
        assert r_reset < r_normal

    def test_reset_no_penalty_on_game_over(self, shaper):
        r_game_over_reset = shaper.compute_reward(
            None, None, "GAME_OVER", action_type=0, levels_before=0, levels_after=0)
        # GAME_OVER state: reset_penalty not applied, only step + game_over_penalty
        expected = shaper.step_cost + shaper.game_over_penalty
        assert pytest.approx(r_game_over_reset, abs=1e-4) == expected


class TestGridChangeBonus:
    def test_grid_change_gives_bonus(self, shaper):
        g1 = same_grid()
        g2 = different_grid_b()
        r_change = shaper.compute_reward(g1, g2, "NOT_FINISHED", action_type=1,
                                         levels_before=0, levels_after=0)
        r_no_change = shaper.compute_reward(g1, g1, "NOT_FINISHED", action_type=1,
                                             levels_before=0, levels_after=0)
        assert r_change > r_no_change

    def test_grid_change_bonus_value(self, shaper):
        g1 = same_grid()
        g2 = different_grid_b()
        r = shaper.compute_reward(g1, g2, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0)
        expected = shaper.step_cost + shaper.grid_change_bonus
        assert pytest.approx(r, abs=1e-4) == expected

    def test_no_change_penalty_value(self, shaper):
        g = same_grid()
        r = shaper.compute_reward(g, g, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0)
        expected = shaper.step_cost + shaper.no_change_penalty
        assert pytest.approx(r, abs=1e-4) == expected

    def test_none_grids_no_grid_reward(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0)
        assert pytest.approx(r, abs=1e-4) == shaper.step_cost


class TestNoveltyBonus:
    def test_novelty_bonus_applied_when_novel(self, shaper):
        r_novel = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                        levels_before=0, levels_after=0,
                                        info={"is_novel_state": True})
        r_no_novel = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                           levels_before=0, levels_after=0,
                                           info={"is_novel_state": False})
        assert r_novel > r_no_novel

    def test_novelty_bonus_value(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0,
                                  info={"is_novel_state": True})
        expected = shaper.step_cost + shaper.novelty_bonus
        assert pytest.approx(r, abs=1e-4) == expected

    def test_no_novelty_when_info_missing(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0)
        assert pytest.approx(r, abs=1e-4) == shaper.step_cost

    def test_no_novelty_when_info_none(self, shaper):
        r = shaper.compute_reward(None, None, "NOT_FINISHED", action_type=1,
                                  levels_before=0, levels_after=0, info=None)
        assert pytest.approx(r, abs=1e-4) == shaper.step_cost

    def test_shape_reward_applies_novelty(self, shaper):
        base = 1.0
        r = shaper.shape_reward(base, {"is_novel_state": True})
        assert pytest.approx(r, abs=1e-4) == base + shaper.novelty_bonus

    def test_shape_reward_no_novelty(self, shaper):
        base = 1.0
        r = shaper.shape_reward(base, {"is_novel_state": False})
        assert pytest.approx(r, abs=1e-4) == base

    def test_default_novelty_bonus_value(self):
        shaper_default = RewardShaper()
        assert shaper_default.novelty_bonus == 0.1

    def test_default_win_reward(self):
        shaper_default = RewardShaper()
        assert shaper_default.win_reward == 100.0

    def test_default_level_bonus(self):
        shaper_default = RewardShaper()
        assert shaper_default.level_bonus == 50.0

    def test_default_step_cost(self):
        shaper_default = RewardShaper()
        assert shaper_default.step_cost == -0.05

    def test_default_grid_change_bonus(self):
        shaper_default = RewardShaper()
        assert shaper_default.grid_change_bonus == 0.02
