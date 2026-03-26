"""Converts ARC-AGI-3 frame data to PyTorch tensors."""
import torch
import numpy as np
from dataclasses import dataclass, field


@dataclass
class MockFrame:
    """Mock frame for testing when arc-agi is not installed."""
    frame: list
    state: str = "NOT_FINISHED"
    levels_completed: int = 0
    win_levels: int = 5
    available_actions: list = field(default_factory=lambda: [1, 2, 3, 4, 5])


class StateProcessor:
    """Converts ARC-AGI-3 frame data to tensors for neural networks."""

    GRID_SIZE = 64
    NUM_COLORS = 16
    AUX_DIM = 15
    STATE_MAP = {"NOT_PLAYED": 0, "NOT_STARTED": 0, "NOT_FINISHED": 1, "WIN": 2, "GAME_OVER": 3}

    def frame_to_tensor(self, frame) -> torch.Tensor:
        """Convert frame grid to one-hot [16, 64, 64]."""
        raw = frame.frame
        if isinstance(raw, list) and len(raw) > 0:
            if isinstance(raw[0], list) and len(raw[0]) > 0 and isinstance(raw[0][0], list):
                grid_2d = raw[0]
            else:
                grid_2d = raw
        else:
            grid_2d = [[0] * self.GRID_SIZE for _ in range(self.GRID_SIZE)]

        h, w = len(grid_2d), len(grid_2d[0]) if grid_2d else 0
        arr = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.int64)
        for y in range(min(h, self.GRID_SIZE)):
            for x in range(min(w, self.GRID_SIZE)):
                arr[y, x] = int(grid_2d[y][x]) % self.NUM_COLORS

        tensor = torch.zeros(self.NUM_COLORS, self.GRID_SIZE, self.GRID_SIZE)
        indices = torch.from_numpy(arr).long().unsqueeze(0)
        tensor.scatter_(0, indices, 1.0)
        return tensor

    def extract_aux_features(self, frame, action_count=0, prev_action=0, max_actions=200) -> torch.Tensor:
        features = torch.zeros(self.AUX_DIM)
        levels = getattr(frame, 'levels_completed', 0)
        win_levels = getattr(frame, 'win_levels', 5)
        features[0] = levels / max(win_levels, 1)
        features[1] = win_levels / 255.0
        features[2] = action_count / max_actions
        if 0 <= prev_action < 8:
            features[3 + prev_action] = 1.0
        state = str(getattr(frame, 'state', 'NOT_FINISHED'))
        if hasattr(frame.state, 'name'):
            state = frame.state.name
        features[11 + self.STATE_MAP.get(state, 1)] = 1.0
        return features

    def get_available_actions_mask(self, frame) -> torch.Tensor:
        mask = torch.zeros(8, dtype=torch.bool)
        mask[0] = True
        for a in getattr(frame, 'available_actions', [1, 2, 3, 4, 5]):
            if 0 <= a < 8:
                mask[a] = True
        return mask
