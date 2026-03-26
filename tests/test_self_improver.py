"""Tests for SelfImprover mode parameter support."""
import pytest
from unittest.mock import MagicMock, patch
from src.training.self_improver import SelfImprover


def test_self_improver_default_mode_is_offline():
    """SelfImprover should default to OFFLINE mode."""
    trainer = MagicMock()
    improver = SelfImprover(trainer=trainer)
    assert improver.mode == "OFFLINE"


def test_self_improver_accepts_online_mode():
    """SelfImprover should accept ONLINE mode."""
    trainer = MagicMock()
    improver = SelfImprover(trainer=trainer, mode="ONLINE")
    assert improver.mode == "ONLINE"


def test_self_improver_accepts_normal_mode():
    """SelfImprover should accept NORMAL mode."""
    trainer = MagicMock()
    improver = SelfImprover(trainer=trainer, mode="NORMAL")
    assert improver.mode == "NORMAL"


def test_self_improver_stores_api_key():
    """SelfImprover should store the api_key parameter."""
    trainer = MagicMock()
    improver = SelfImprover(trainer=trainer, api_key="test-key-123")
    assert improver.api_key == "test-key-123"


def test_self_improver_default_api_key_is_empty():
    """SelfImprover should default api_key to empty string."""
    trainer = MagicMock()
    improver = SelfImprover(trainer=trainer)
    assert improver.api_key == ""


def test_train_iteration_passes_mode_to_env():
    """_train_iteration should pass self.mode to ArcEnvWrapper."""
    trainer = MagicMock()
    trainer.collect_rollout.return_value = {"mean_reward": 0.0, "episodes": 1}
    trainer.update.return_value = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.5}

    improver = SelfImprover(
        trainer=trainer,
        game_ids=["ls20"],
        train_steps_per_iter=1,
        mode="ONLINE",
    )

    with patch("src.training.self_improver.ArcEnvWrapper") as mock_env_cls:
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        improver._train_iteration()
        # Verify ArcEnvWrapper was instantiated with mode="ONLINE"
        call_kwargs = mock_env_cls.call_args
        assert call_kwargs is not None
        # mode may be positional or keyword
        args, kwargs = call_kwargs
        mode_passed = kwargs.get("mode") or (args[1] if len(args) > 1 else None)
        assert mode_passed == "ONLINE", f"Expected mode='ONLINE', got {call_kwargs}"


def test_train_iteration_passes_api_key_to_env():
    """_train_iteration should pass self.api_key to ArcEnvWrapper."""
    trainer = MagicMock()
    trainer.collect_rollout.return_value = {"mean_reward": 0.0, "episodes": 1}
    trainer.update.return_value = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.5}

    improver = SelfImprover(
        trainer=trainer,
        game_ids=["ls20"],
        train_steps_per_iter=1,
        api_key="my-api-key",
    )

    with patch("src.training.self_improver.ArcEnvWrapper") as mock_env_cls:
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        improver._train_iteration()
        call_kwargs = mock_env_cls.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get("api_key") == "my-api-key", f"Expected api_key='my-api-key', got {call_kwargs}"


def test_evaluate_passes_mode_to_env():
    """_evaluate should pass self.mode to ArcEnvWrapper."""
    import torch
    trainer = MagicMock()
    # policy.sample returns tensors
    trainer.policy.sample.return_value = (
        torch.tensor(1),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0.5),
        torch.tensor(0.0),
    )

    improver = SelfImprover(
        trainer=trainer,
        eval_game_ids=["ls20"],
        mode="ONLINE",
    )

    with patch("src.training.self_improver.ArcEnvWrapper") as mock_env_cls:
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        mock_env.reset.return_value = {
            "grid": torch.zeros(1, 64, 64),
            "aux": torch.zeros(8),
            "available_actions": torch.ones(8),
        }
        mock_env.step.return_value = (
            {
                "grid": torch.zeros(1, 64, 64),
                "aux": torch.zeros(8),
                "available_actions": torch.ones(8),
            },
            0.0,
            True,  # done immediately
            {"state": "NOT_FINISHED", "levels_completed": 0},
        )

        improver._evaluate()

        call_kwargs = mock_env_cls.call_args
        assert call_kwargs is not None
        args, kwargs = call_kwargs
        mode_passed = kwargs.get("mode") or (args[1] if len(args) > 1 else None)
        assert mode_passed == "ONLINE", f"Expected mode='ONLINE', got {call_kwargs}"


def test_evaluate_passes_api_key_to_env():
    """_evaluate should pass self.api_key to ArcEnvWrapper."""
    import torch
    trainer = MagicMock()
    trainer.policy.sample.return_value = (
        torch.tensor(1),
        torch.tensor(0),
        torch.tensor(0),
        torch.tensor(0.5),
        torch.tensor(0.0),
    )

    improver = SelfImprover(
        trainer=trainer,
        eval_game_ids=["ls20"],
        api_key="eval-key-xyz",
    )

    with patch("src.training.self_improver.ArcEnvWrapper") as mock_env_cls:
        mock_env = MagicMock()
        mock_env_cls.return_value = mock_env
        mock_env.reset.return_value = {
            "grid": torch.zeros(1, 64, 64),
            "aux": torch.zeros(8),
            "available_actions": torch.ones(8),
        }
        mock_env.step.return_value = (
            {
                "grid": torch.zeros(1, 64, 64),
                "aux": torch.zeros(8),
                "available_actions": torch.ones(8),
            },
            0.0,
            True,
            {"state": "NOT_FINISHED", "levels_completed": 0},
        )

        improver._evaluate()

        call_kwargs = mock_env_cls.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs.get("api_key") == "eval-key-xyz", f"Expected api_key='eval-key-xyz', got {call_kwargs}"
