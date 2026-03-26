"""Tests for ReplayBuffer."""
import pytest
import torch
import numpy as np
from src.training.replay_buffer import ReplayBuffer, Transition


def make_transition(reward=0.0, done=False, value=0.5, log_prob=-1.0,
                    action_type=1, action_x=0, action_y=0):
    return Transition(
        grid=torch.zeros(16, 64, 64),
        aux=torch.zeros(15),
        available_actions=torch.ones(8, dtype=torch.bool),
        action_type=action_type,
        action_x=action_x,
        action_y=action_y,
        reward=reward,
        done=done,
        log_prob=log_prob,
        value=value,
    )


class TestAddAndLen:
    def test_empty_buffer_len(self):
        buf = ReplayBuffer(max_size=100)
        assert len(buf) == 0

    def test_add_increments_len(self):
        buf = ReplayBuffer(max_size=100)
        buf.add(make_transition())
        assert len(buf) == 1

    def test_add_multiple(self):
        buf = ReplayBuffer(max_size=100)
        for _ in range(5):
            buf.add(make_transition())
        assert len(buf) == 5


class TestMaxSizeCap:
    def test_exceeds_max_size_caps_at_max(self):
        buf = ReplayBuffer(max_size=3)
        for i in range(5):
            buf.add(make_transition(reward=float(i)))
        assert len(buf) == 3

    def test_oldest_removed_when_full(self):
        buf = ReplayBuffer(max_size=3)
        for i in range(5):
            buf.add(make_transition(reward=float(i)))
        # After 5 adds with max_size=3, buffer holds rewards 2.0, 3.0, 4.0
        rewards = [t.reward for t in buf.transitions]
        assert rewards == [2.0, 3.0, 4.0]


class TestClear:
    def test_clear_empties_buffer(self):
        buf = ReplayBuffer(max_size=100)
        for _ in range(10):
            buf.add(make_transition())
        buf.clear()
        assert len(buf) == 0

    def test_clear_resets_episode_boundaries(self):
        buf = ReplayBuffer(max_size=100)
        buf.add(make_transition())
        buf.mark_episode_end()
        buf.clear()
        assert buf.episode_boundaries == [0]

    def test_clear_resets_gae_cache(self):
        buf = ReplayBuffer(max_size=100)
        buf.add(make_transition())
        buf.compute_gae()
        buf.clear()
        assert buf._adv is None
        assert buf._ret is None


class TestGAE:
    def test_gae_empty_buffer(self):
        buf = ReplayBuffer(max_size=100)
        adv, ret = buf.compute_gae()
        assert len(adv) == 0
        assert len(ret) == 0

    def test_gae_single_transition_done(self):
        buf = ReplayBuffer(max_size=100)
        buf.add(make_transition(reward=1.0, done=True, value=0.0))
        adv, ret = buf.compute_gae(gamma=0.99, lam=0.95)
        assert adv.shape == (1,)
        # delta = reward + gamma * 0 * 0 - value = 1.0 - 0.0 = 1.0
        # gae = delta = 1.0
        assert pytest.approx(adv[0].item(), abs=1e-4) == 1.0
        assert pytest.approx(ret[0].item(), abs=1e-4) == 1.0

    def test_gae_single_episode_multi_step(self):
        buf = ReplayBuffer(max_size=100)
        # 3-step episode: rewards=[0,0,1], values=[0.5,0.5,0], last done
        buf.add(make_transition(reward=0.0, done=False, value=0.5))
        buf.add(make_transition(reward=0.0, done=False, value=0.5))
        buf.add(make_transition(reward=1.0, done=True, value=0.0))
        buf.mark_episode_end()
        adv, ret = buf.compute_gae(gamma=0.99, lam=0.95)
        assert adv.shape == (3,)
        assert ret.shape == (3,)
        # Returns should be positive (bootstrapped from rewards)
        assert ret[0].item() > 0.0

    def test_gae_advantages_sum_reasonable(self):
        buf = ReplayBuffer(max_size=100)
        for _ in range(10):
            buf.add(make_transition(reward=0.1, done=False, value=0.5))
        buf.add(make_transition(reward=1.0, done=True, value=0.0))
        buf.mark_episode_end()
        adv, ret = buf.compute_gae()
        # Advantages should not all be zero
        assert adv.abs().sum().item() > 0.0

    def test_compute_gae_called_by_sample_if_needed(self):
        buf = ReplayBuffer(max_size=100)
        buf.add(make_transition(reward=1.0, done=True, value=0.0))
        assert buf._adv is None
        buf.sample_minibatch(1)
        assert buf._adv is not None


class TestSampleMinibatch:
    def test_minibatch_keys(self):
        buf = ReplayBuffer(max_size=100)
        for _ in range(5):
            buf.add(make_transition())
        batch = buf.sample_minibatch(3)
        expected_keys = {
            'grids', 'aux', 'available_actions', 'action_types',
            'action_x', 'action_y', 'old_log_probs', 'old_values',
            'advantages', 'returns',
        }
        assert set(batch.keys()) == expected_keys

    def test_minibatch_shapes(self):
        buf = ReplayBuffer(max_size=100)
        for _ in range(10):
            buf.add(make_transition())
        batch = buf.sample_minibatch(4)
        assert batch['grids'].shape == (4, 16, 64, 64)
        assert batch['aux'].shape == (4, 15)
        assert batch['available_actions'].shape == (4, 8)
        assert batch['action_types'].shape == (4,)
        assert batch['advantages'].shape == (4,)
        assert batch['returns'].shape == (4,)

    def test_minibatch_capped_at_buffer_size(self):
        buf = ReplayBuffer(max_size=100)
        for _ in range(3):
            buf.add(make_transition())
        batch = buf.sample_minibatch(100)
        assert batch['grids'].shape[0] == 3

    def test_minibatch_raises_on_empty(self):
        buf = ReplayBuffer(max_size=100)
        with pytest.raises((AssertionError, Exception)):
            buf.sample_minibatch(4)


class TestEpisodeStats:
    def test_empty_buffer(self):
        buf = ReplayBuffer(max_size=100)
        stats = buf.episode_stats()
        assert stats["num_episodes"] == 0

    def test_single_episode_stats(self):
        buf = ReplayBuffer(max_size=100)
        for i in range(5):
            buf.add(make_transition(reward=1.0, done=(i == 4)))
        buf.mark_episode_end()
        stats = buf.episode_stats()
        assert stats["num_episodes"] == 1
        assert stats["mean_length"] == 5
        assert pytest.approx(stats["mean_total_reward"], abs=1e-4) == 5.0

    def test_two_episodes_stats(self):
        buf = ReplayBuffer(max_size=100)
        # episode 1: 3 steps, reward=1 each
        for _ in range(3):
            buf.add(make_transition(reward=1.0))
        buf.mark_episode_end()
        # episode 2: 2 steps, reward=2 each
        for _ in range(2):
            buf.add(make_transition(reward=2.0))
        buf.mark_episode_end()
        stats = buf.episode_stats()
        assert stats["num_episodes"] == 2
        assert pytest.approx(stats["mean_length"], abs=1e-4) == 2.5
        assert pytest.approx(stats["mean_total_reward"], abs=1e-4) == (3.0 + 4.0) / 2
