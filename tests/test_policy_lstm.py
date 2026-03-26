"""Tests for CNNLSTMPolicy."""
import torch
import pytest
from src.models.policy_lstm import CNNLSTMPolicy


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def model():
    return CNNLSTMPolicy(
        num_colors=16,
        embedding_dim=128,
        aux_dim=15,
        aux_hidden=32,
        lstm_hidden=128,
        num_actions=8,
        grid_size=64,
    )


def _make_inputs(batch_size: int, device: torch.device = torch.device("cpu")):
    grid = torch.zeros(batch_size, 16, 64, 64, device=device)
    # Mark one color active per cell to create a valid one-hot grid
    grid[:, 0, :, :] = 1.0
    aux = torch.randn(batch_size, 15, device=device)
    mask = torch.ones(batch_size, 8, dtype=torch.bool, device=device)
    return grid, aux, mask


# ---------------------------------------------------------------------------
# forward() shape tests
# ---------------------------------------------------------------------------

class TestForwardShapes:
    def test_forward_batch1(self, model):
        grid, aux, mask = _make_inputs(1)
        action_logits, x_logits, y_logits, value, lstm_state = model.forward(grid, aux, mask)

        assert action_logits.shape == (1, 8)
        assert x_logits.shape == (1, 64)
        assert y_logits.shape == (1, 64)
        assert value.shape == (1,)

    def test_forward_batch4(self, model):
        grid, aux, mask = _make_inputs(4)
        action_logits, x_logits, y_logits, value, lstm_state = model.forward(grid, aux, mask)

        assert action_logits.shape == (4, 8)
        assert x_logits.shape == (4, 64)
        assert y_logits.shape == (4, 64)
        assert value.shape == (4,)

    def test_forward_no_mask(self, model):
        grid, aux, _ = _make_inputs(2)
        action_logits, x_logits, y_logits, value, lstm_state = model.forward(grid, aux)
        assert action_logits.shape == (2, 8)
        assert value.shape == (2,)

    def test_forward_returns_lstm_state_tuple(self, model):
        grid, aux, mask = _make_inputs(1)
        _, _, _, _, lstm_state = model.forward(grid, aux, mask)
        h, c = lstm_state
        assert h.shape == (1, 1, 128)
        assert c.shape == (1, 1, 128)


# ---------------------------------------------------------------------------
# LSTM state tests
# ---------------------------------------------------------------------------

class TestLSTMState:
    def test_state_is_maintained_across_steps(self, model):
        """Hidden state from step N should differ from the initial zeros."""
        grid, aux, mask = _make_inputs(1)
        h0 = model.init_hidden(1, torch.device("cpu"))

        _, _, _, _, state1 = model.forward(grid, aux, mask, lstm_state=h0)
        _, _, _, _, state2 = model.forward(grid, aux, mask, lstm_state=state1)

        # The hidden state tensors should change across steps
        assert not torch.allclose(state1[0], state2[0])

    def test_state_reset_gives_zeros(self, model):
        """init_hidden should return zero tensors."""
        h, c = model.init_hidden(3, torch.device("cpu"))
        assert h.shape == (1, 3, 128)
        assert c.shape == (1, 3, 128)
        assert torch.all(h == 0)
        assert torch.all(c == 0)

    def test_fresh_state_differs_from_carried_state(self, model):
        """Running forward with fresh vs. carried state should produce different logits."""
        grid, aux, mask = _make_inputs(1)

        # Take two steps with carried state
        _, _, _, _, state1 = model.forward(grid, aux, mask)
        logits_with_state, _, _, _, _ = model.forward(grid, aux, mask, lstm_state=state1)

        # Run the same second step but with fresh state
        logits_fresh, _, _, _, _ = model.forward(grid, aux, mask, lstm_state=None)

        assert not torch.allclose(logits_with_state, logits_fresh)

    def test_lstm_state_batch_size_matches_input(self, model):
        """lstm_state batch dimension must match the input batch size."""
        for B in [1, 2, 8]:
            grid, aux, mask = _make_inputs(B)
            _, _, _, _, (h, c) = model.forward(grid, aux, mask)
            assert h.shape[1] == B
            assert c.shape[1] == B


# ---------------------------------------------------------------------------
# sample() tests
# ---------------------------------------------------------------------------

class TestSample:
    def test_sample_returns_correct_types(self, model):
        grid, aux, mask = _make_inputs(1)
        action, x, y, log_prob, entropy, value, lstm_state = model.sample(grid, aux, mask)

        assert action.dtype == torch.long
        assert x.dtype == torch.long
        assert y.dtype == torch.long
        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)

    def test_sample_action_in_range(self, model):
        grid, aux, mask = _make_inputs(1)
        for _ in range(20):
            action, x, y, _, _, _, _ = model.sample(grid, aux, mask)
            assert 0 <= action.item() < 8
            assert 0 <= x.item() < 64
            assert 0 <= y.item() < 64

    def test_sample_respects_mask(self, model):
        """With only action 3 available, sample should always return action 3."""
        grid, aux, _ = _make_inputs(1)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, 3] = True  # Only action 3 is available

        for _ in range(20):
            action, _, _, _, _, _, _ = model.sample(grid, aux, mask)
            assert action.item() == 3

    def test_sample_carries_lstm_state(self, model):
        grid, aux, mask = _make_inputs(1)
        _, _, _, _, _, _, state1 = model.sample(grid, aux, mask)
        h, c = state1
        assert h.shape == (1, 1, 128)

    def test_sample_batch4(self, model):
        grid, aux, mask = _make_inputs(4)
        action, x, y, log_prob, entropy, value, lstm_state = model.sample(grid, aux, mask)

        assert action.shape == (4,)
        assert log_prob.shape == (4,)
        assert value.shape == (4,)


# ---------------------------------------------------------------------------
# evaluate() tests
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_evaluate_shapes_batch1(self, model):
        grid, aux, mask = _make_inputs(1)
        actions = torch.zeros(1, dtype=torch.long)
        act_x = torch.zeros(1, dtype=torch.long)
        act_y = torch.zeros(1, dtype=torch.long)

        log_prob, entropy, value, lstm_state = model.evaluate(
            grid, aux, actions, act_x, act_y, mask
        )

        assert log_prob.shape == (1,)
        assert entropy.shape == (1,)
        assert value.shape == (1,)

    def test_evaluate_shapes_batch4(self, model):
        grid, aux, mask = _make_inputs(4)
        actions = torch.randint(0, 8, (4,))
        act_x = torch.randint(0, 64, (4,))
        act_y = torch.randint(0, 64, (4,))

        log_prob, entropy, value, lstm_state = model.evaluate(
            grid, aux, actions, act_x, act_y, mask
        )

        assert log_prob.shape == (4,)
        assert entropy.shape == (4,)
        assert value.shape == (4,)

    def test_evaluate_with_lstm_state(self, model):
        grid, aux, mask = _make_inputs(2)
        actions = torch.zeros(2, dtype=torch.long)
        act_x = torch.zeros(2, dtype=torch.long)
        act_y = torch.zeros(2, dtype=torch.long)
        lstm_state = model.init_hidden(2, torch.device("cpu"))

        log_prob, entropy, value, out_state = model.evaluate(
            grid, aux, actions, act_x, act_y, mask, lstm_state=lstm_state
        )

        assert log_prob.shape == (2,)
        h, c = out_state
        assert h.shape == (1, 2, 128)

    def test_evaluate_log_prob_is_finite(self, model):
        grid, aux, mask = _make_inputs(4)
        actions = torch.randint(0, 8, (4,))
        act_x = torch.randint(0, 64, (4,))
        act_y = torch.randint(0, 64, (4,))

        log_prob, entropy, value, _ = model.evaluate(
            grid, aux, actions, act_x, act_y, mask
        )

        assert torch.all(torch.isfinite(log_prob))
        assert torch.all(torch.isfinite(entropy))
        assert torch.all(torch.isfinite(value))


# ---------------------------------------------------------------------------
# Action masking tests
# ---------------------------------------------------------------------------

class TestActionMasking:
    def test_masked_actions_get_neg_inf_logits(self, model):
        grid, aux, _ = _make_inputs(1)
        mask = torch.zeros(1, 8, dtype=torch.bool)
        mask[0, 0] = True  # Only action 0 is valid

        action_logits, _, _, _, _ = model.forward(grid, aux, mask)

        # Actions 1-7 should be -inf
        assert torch.all(action_logits[0, 1:] == float('-inf'))
        # Action 0 should be finite
        assert torch.isfinite(action_logits[0, 0])

    def test_all_actions_available_gives_finite_logits(self, model):
        grid, aux, mask = _make_inputs(1)
        action_logits, _, _, _, _ = model.forward(grid, aux, mask)
        assert torch.all(torch.isfinite(action_logits))


# ---------------------------------------------------------------------------
# init_hidden tests
# ---------------------------------------------------------------------------

class TestInitHidden:
    def test_init_hidden_device_cpu(self, model):
        h, c = model.init_hidden(4, torch.device("cpu"))
        assert h.device.type == "cpu"
        assert c.device.type == "cpu"

    def test_init_hidden_shapes(self, model):
        for B in [1, 5, 16]:
            h, c = model.init_hidden(B, torch.device("cpu"))
            assert h.shape == (1, B, 128)
            assert c.shape == (1, B, 128)
