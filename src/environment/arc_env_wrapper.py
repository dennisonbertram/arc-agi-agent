"""RL-friendly wrapper around arc-agi SDK."""
import numpy as np
from typing import Optional
from src.environment.state_processor import StateProcessor, MockFrame
from src.training.reward_shaper import RewardShaper

try:
    from arcengine import GameAction, GameState
    import arc_agi
    HAS_ARC = True
except ImportError:
    HAS_ARC = False


class ArcEnvWrapper:
    """Gym-like interface: reset() -> obs, step(action) -> (obs, reward, done, info)"""

    def __init__(self, game_id: str, mode: str = "OFFLINE", api_key: str = "",
                 max_actions: int = 200, reward_shaper: Optional[RewardShaper] = None):
        self.game_id = game_id
        self.max_actions = max_actions
        self.processor = StateProcessor()
        self.reward_shaper = reward_shaper or RewardShaper()
        self.action_count = 0
        self.prev_action = 0
        self.prev_grid = None
        self.prev_levels = 0
        self.current_frame = None
        self._env = None

        if HAS_ARC:
            from arc_agi import Arcade, OperationMode
            modes = {"OFFLINE": OperationMode.OFFLINE, "ONLINE": OperationMode.ONLINE, "NORMAL": OperationMode.NORMAL}
            arc = Arcade(operation_mode=modes.get(mode, OperationMode.OFFLINE), arc_api_key=api_key or None)
            self._env = arc.make(game_id)

    def reset(self) -> dict:
        self.action_count = 0
        self.prev_action = 0
        self.prev_grid = None
        self.prev_levels = 0
        if self._env:
            self.current_frame = self._env.reset()
        else:
            self.current_frame = MockFrame(frame=[[[0]*64 for _ in range(64)]])
        return self._obs()

    def step(self, action_type: int, x: int = 0, y: int = 0):
        self.prev_grid = self._grid()
        self.prev_levels = getattr(self.current_frame, 'levels_completed', 0)

        if self._env and HAS_ARC:
            action = self._to_game_action(action_type, x, y)
            data = {"x": x, "y": y} if action_type == 6 else None
            self.current_frame = self._env.step(action, data=data)
        else:
            grid = [[np.random.randint(0, 16) for _ in range(64)] for _ in range(64)]
            self.current_frame = MockFrame(frame=[grid])

        self.action_count += 1
        curr_levels = getattr(self.current_frame, 'levels_completed', 0)
        state = str(getattr(self.current_frame, 'state', 'NOT_FINISHED'))

        reward = self.reward_shaper.compute_reward(
            self.prev_grid, self._grid(), state, action_type, self.prev_levels, curr_levels)
        done = "WIN" in state or self.action_count >= self.max_actions
        info = {"action_count": self.action_count, "levels_completed": curr_levels, "state": state}
        self.prev_action = action_type
        return self._obs(), reward, done, info

    def _obs(self):
        return {
            "grid": self.processor.frame_to_tensor(self.current_frame),
            "aux": self.processor.extract_aux_features(self.current_frame, self.action_count, self.prev_action),
            "available_actions": self.processor.get_available_actions_mask(self.current_frame),
        }

    def _grid(self):
        if not self.current_frame:
            return None
        raw = self.current_frame.frame
        if isinstance(raw, list) and raw:
            return raw[0] if isinstance(raw[0], list) and isinstance(raw[0][0], list) else raw
        return raw

    def _to_game_action(self, at, x=0, y=0):
        if not HAS_ARC:
            return at
        m = {0: GameAction.RESET, 1: GameAction.ACTION1, 2: GameAction.ACTION2,
             3: GameAction.ACTION3, 4: GameAction.ACTION4, 5: GameAction.ACTION5,
             6: GameAction.ACTION6, 7: GameAction.ACTION7}
        a = m.get(at, GameAction.ACTION1)
        if at == 6:
            a.set_data({"x": x, "y": y})
        return a
