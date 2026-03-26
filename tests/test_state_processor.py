"""Tests for StateProcessor."""
import pytest
import torch
from src.environment.state_processor import StateProcessor, MockFrame


@pytest.fixture
def processor():
    return StateProcessor()


def make_frame(grid_2d=None, state="NOT_FINISHED", levels_completed=0,
               win_levels=5, available_actions=None):
    if grid_2d is None:
        grid_2d = [[0] * 64 for _ in range(64)]
    if available_actions is None:
        available_actions = [1, 2, 3, 4, 5]
    # Wrap as [grid_2d] so frame_to_tensor sees a 3-level list
    return MockFrame(
        frame=[grid_2d],
        state=state,
        levels_completed=levels_completed,
        win_levels=win_levels,
        available_actions=available_actions,
    )


class TestFrameToTensor:
    def test_output_shape(self, processor):
        frame = make_frame()
        t = processor.frame_to_tensor(frame)
        assert t.shape == (16, 64, 64)

    def test_one_hot_sum_per_pixel(self, processor):
        """Each spatial location should have exactly one hot channel."""
        frame = make_frame()
        t = processor.frame_to_tensor(frame)
        sums = t.sum(dim=0)
        assert torch.all(sums == 1.0), "Each pixel must be one-hot across color channels"

    def test_correct_color_encoding(self, processor):
        """A grid filled with color 3 should set channel 3 everywhere."""
        grid = [[3] * 64 for _ in range(64)]
        frame = make_frame(grid_2d=grid)
        t = processor.frame_to_tensor(frame)
        assert t[3].sum().item() == 64 * 64
        assert t[0].sum().item() == 0.0

    def test_color_modulo(self, processor):
        """Color values >= NUM_COLORS should be taken mod 16."""
        grid = [[16] * 64 for _ in range(64)]  # 16 % 16 == 0
        frame = make_frame(grid_2d=grid)
        t = processor.frame_to_tensor(frame)
        assert t[0].sum().item() == 64 * 64

    def test_padding_small_grid(self, processor):
        """Grids smaller than 64x64 should be zero-padded to 64x64."""
        small = [[1] * 4 for _ in range(4)]
        frame = make_frame(grid_2d=small)
        t = processor.frame_to_tensor(frame)
        assert t.shape == (16, 64, 64)
        # top-left 4x4 should be color 1
        assert t[1, :4, :4].sum().item() == 16.0
        # rest should be color 0 (padding)
        assert t[0, 4:, 4:].sum().item() == (64 - 4) * (64 - 4)

    def test_flat_frame_format(self, processor):
        """frame.frame as a flat 2D list (not nested) should also work."""
        grid = [[5] * 64 for _ in range(64)]
        frame = MockFrame(frame=grid)  # flat 2D, no outer list
        t = processor.frame_to_tensor(frame)
        assert t.shape == (16, 64, 64)
        assert t[5].sum().item() == 64 * 64


class TestExtractAuxFeatures:
    def test_output_shape(self, processor):
        frame = make_frame()
        aux = processor.extract_aux_features(frame)
        assert aux.shape == (15,)

    def test_levels_progress_ratio(self, processor):
        frame = make_frame(levels_completed=2, win_levels=4)
        aux = processor.extract_aux_features(frame)
        assert pytest.approx(aux[0].item(), abs=1e-4) == 0.5

    def test_win_levels_normalized(self, processor):
        frame = make_frame(win_levels=255)
        aux = processor.extract_aux_features(frame)
        assert pytest.approx(aux[1].item(), abs=1e-4) == 1.0

    def test_action_count_normalized(self, processor):
        frame = make_frame()
        aux = processor.extract_aux_features(frame, action_count=100, max_actions=200)
        assert pytest.approx(aux[2].item(), abs=1e-4) == 0.5

    def test_prev_action_one_hot(self, processor):
        """prev_action=3 should set features[6] (3+3) to 1.0."""
        frame = make_frame()
        aux = processor.extract_aux_features(frame, prev_action=3)
        assert aux[3 + 3].item() == 1.0
        # other action slots should be 0
        for i in range(8):
            if i != 3:
                assert aux[3 + i].item() == 0.0

    def test_prev_action_out_of_range_ignored(self, processor):
        """prev_action values outside [0,7] should not set any slot."""
        frame = make_frame()
        aux = processor.extract_aux_features(frame, prev_action=10)
        assert aux[3:11].sum().item() == 0.0

    def test_state_not_finished_encoding(self, processor):
        frame = make_frame(state="NOT_FINISHED")
        aux = processor.extract_aux_features(frame)
        # NOT_FINISHED -> STATE_MAP value 1 -> features[12]
        assert aux[12].item() == 1.0

    def test_state_win_encoding(self, processor):
        frame = make_frame(state="WIN")
        aux = processor.extract_aux_features(frame)
        # WIN -> STATE_MAP value 2 -> features[13]
        assert aux[13].item() == 1.0

    def test_state_game_over_encoding(self, processor):
        frame = make_frame(state="GAME_OVER")
        aux = processor.extract_aux_features(frame)
        # GAME_OVER -> STATE_MAP value 3 -> features[14]
        assert aux[14].item() == 1.0


class TestAvailableActionsMask:
    def test_output_shape(self, processor):
        frame = make_frame()
        mask = processor.get_available_actions_mask(frame)
        assert mask.shape == (8,)
        assert mask.dtype == torch.bool

    def test_action_zero_always_true(self, processor):
        """Action 0 (RESET) is always available."""
        frame = make_frame(available_actions=[])
        mask = processor.get_available_actions_mask(frame)
        assert mask[0].item() is True

    def test_listed_actions_enabled(self, processor):
        frame = make_frame(available_actions=[1, 3, 5])
        mask = processor.get_available_actions_mask(frame)
        assert mask[1].item() is True
        assert mask[3].item() is True
        assert mask[5].item() is True
        assert mask[2].item() is False
        assert mask[4].item() is False

    def test_out_of_range_actions_ignored(self, processor):
        frame = make_frame(available_actions=[8, 99, -1])
        mask = processor.get_available_actions_mask(frame)
        # only action 0 should be set
        assert mask[0].item() is True
        assert mask[1:].sum().item() == 0

    # --- valid_actions parameter (per-game tag filtering) ---

    def test_valid_actions_click_only(self, processor):
        """click games: RESET(0), INTERACT(5), COORDINATE(6), UNDO(7) only."""
        click_actions = [0, 5, 6, 7]
        mask = processor.get_available_actions_mask(valid_actions=click_actions)
        expected = [True, False, False, False, False, True, True, True]
        for i, exp in enumerate(expected):
            assert mask[i].item() is exp, f"action {i}: expected {exp}, got {mask[i].item()}"

    def test_valid_actions_keyboard_only(self, processor):
        """keyboard games: RESET(0), UP(1), DOWN(2), LEFT(3), RIGHT(4), UNDO(7) only."""
        keyboard_actions = [0, 1, 2, 3, 4, 7]
        mask = processor.get_available_actions_mask(valid_actions=keyboard_actions)
        expected = [True, True, True, True, True, False, False, True]
        for i, exp in enumerate(expected):
            assert mask[i].item() is exp, f"action {i}: expected {exp}, got {mask[i].item()}"

    def test_valid_actions_keyboard_click_all(self, processor):
        """keyboard_click games: all 8 actions available."""
        all_actions = [0, 1, 2, 3, 4, 5, 6, 7]
        mask = processor.get_available_actions_mask(valid_actions=all_actions)
        assert mask.all().item() is True, "All actions should be enabled for keyboard_click"

    def test_valid_actions_overrides_frame(self, processor):
        """When valid_actions is given, the frame's available_actions field is ignored."""
        frame = make_frame(available_actions=[1, 2, 3])
        click_actions = [0, 5, 6, 7]
        mask = processor.get_available_actions_mask(frame, valid_actions=click_actions)
        # keyboard actions 1-4 should be False despite frame saying [1,2,3]
        assert mask[1].item() is False
        assert mask[2].item() is False
        assert mask[5].item() is True
        assert mask[6].item() is True

    def test_valid_actions_none_falls_back_to_frame(self, processor):
        """Passing valid_actions=None falls back to frame-level available_actions."""
        frame = make_frame(available_actions=[1, 3])
        mask = processor.get_available_actions_mask(frame, valid_actions=None)
        assert mask[0].item() is True   # always on via frame fallback
        assert mask[1].item() is True
        assert mask[3].item() is True
        assert mask[2].item() is False
        assert mask[4].item() is False
