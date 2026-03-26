"""RL-friendly wrapper around arc-agi SDK."""
import time
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
        self.valid_actions = None  # None means use frame's available_actions

        if HAS_ARC:
            from arc_agi import Arcade, OperationMode
            modes = {"OFFLINE": OperationMode.OFFLINE, "ONLINE": OperationMode.ONLINE, "NORMAL": OperationMode.NORMAL}
            arc = Arcade(operation_mode=modes.get(mode, OperationMode.OFFLINE), arc_api_key=api_key if api_key is not None else "")
            self._env = arc.make(game_id)
            # Detect game tag to restrict valid actions
            game_tags = []
            for env_info in arc.available_environments:
                if env_info.game_id == game_id:
                    game_tags = getattr(env_info, 'tags', [])
                    break
            self.valid_actions = self._actions_for_tags(game_tags)

    def reset(self) -> dict:
        self.action_count = 0
        self.prev_action = 0
        self.prev_grid = None
        self.prev_levels = 0
        if self._env:
            time.sleep(0.05)  # base delay to stay under 600 RPM
            self.current_frame = self._env.reset()
        else:
            self.current_frame = MockFrame(frame=[[[0]*64 for _ in range(64)]])
        return self._obs()

    def step(self, action_type: int, x: int = 0, y: int = 0):
        self.prev_grid = self._grid()
        self.prev_levels = getattr(self.current_frame, 'levels_completed', 0)
        step_error_done = False

        if self._env and HAS_ARC:
            import logging
            action = self._to_game_action(action_type, x, y)
            data = {"x": x, "y": y} if action_type == 6 else None
            max_retries = 3
            backoff = 1.0
            time.sleep(0.05)  # base delay to stay under 600 RPM
            for attempt in range(max_retries + 1):
                try:
                    self.current_frame = self._env.step(action, data=data)
                    break  # success
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str and attempt < max_retries:
                        sleep_time = min(backoff * (2 ** attempt), 10.0)
                        logging.warning(
                            f"Step 429 rate-limited for action {action_type} ({action}), "
                            f"attempt {attempt + 1}/{max_retries}. Sleeping {sleep_time:.1f}s."
                        )
                        time.sleep(sleep_time)
                        continue
                    # Non-retryable error or retries exhausted
                    if "400" in err_str:
                        logging.warning(
                            f"Step 400 error for action {action_type} ({action}): game session ended. "
                            f"Marking episode done."
                        )
                        step_error_done = True
                    elif "429" in err_str:
                        logging.warning(
                            f"Step 429 max retries exhausted for action {action_type} ({action}). "
                            f"Marking episode done."
                        )
                        step_error_done = True
                    else:
                        logging.warning(
                            f"Step error for action {action_type} ({action}): {e}. Keeping previous frame."
                        )
                    break
            # current_frame is unchanged on error; count action and continue
        else:
            grid = [[np.random.randint(0, 16) for _ in range(64)] for _ in range(64)]
            self.current_frame = MockFrame(frame=[grid])

        self.action_count += 1
        curr_levels = getattr(self.current_frame, 'levels_completed', 0)
        state = str(getattr(self.current_frame, 'state', 'NOT_FINISHED'))

        reward = self.reward_shaper.compute_reward(
            self.prev_grid, self._grid(), state, action_type, self.prev_levels, curr_levels)
        done = step_error_done or "WIN" in state or self.action_count >= self.max_actions
        info = {"action_count": self.action_count, "levels_completed": curr_levels, "state": state,
                "step_error": step_error_done}
        self.prev_action = action_type
        return self._obs(), reward, done, info

    def _obs(self):
        frame = self.current_frame
        if frame is None:
            frame = MockFrame(frame=[[[0] * self.processor.GRID_SIZE for _ in range(self.processor.GRID_SIZE)]])
        return {
            "grid": self.processor.frame_to_tensor(frame),
            "aux": self.processor.extract_aux_features(frame, self.action_count, self.prev_action),
            "available_actions": self.processor.get_available_actions_mask(frame, valid_actions=self.valid_actions),
        }

    @staticmethod
    def _actions_for_tags(tags: list) -> list[int] | None:
        """Return the valid action indices for a game based on its tags.

        Returns None when tags are unknown (falls back to frame-level mask).
        """
        tag_set = {str(t).lower() for t in tags}
        if "keyboard_click" in tag_set:
            return [0, 1, 2, 3, 4, 5, 6, 7]
        if "click" in tag_set:
            return [0, 5, 6, 7]
        if "keyboard" in tag_set:
            return [0, 1, 2, 3, 4, 7]
        return None

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
        return m.get(at, GameAction.ACTION1)
