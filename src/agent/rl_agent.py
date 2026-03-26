"""RL Agent that uses trained policy/value networks."""
import torch
from pathlib import Path
from typing import Optional

from src.agent.base_agent import BaseAgent
from src.models.policy_net import PolicyNetwork
from src.models.value_net import ValueNetwork
from src.environment.state_processor import StateProcessor


class RLAgent(BaseAgent):
    """Agent that uses trained neural networks for action selection.

    Can be used with the ARC-AGI-3 agent framework.
    """

    MAX_ACTIONS = 200

    def __init__(self, game_id: str, checkpoint_path: "Optional[str | Path]" = None,
                 device: str = "cpu", **kwargs):
        super().__init__()
        self.game_id = game_id
        self.device = torch.device(device)
        self.processor = StateProcessor()
        self.policy = PolicyNetwork().to(self.device)
        self.policy.eval()
        self.action_count = 0
        self._prev_action = 0

        if checkpoint_path:
            self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, path: "str | Path"):
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        if "policy_state_dict" in ckpt:
            self.policy.load_state_dict(ckpt["policy_state_dict"])
        else:
            self.policy.load_state_dict(ckpt)
        self.policy.eval()

    def is_done(self, frames=None, latest_frame=None) -> bool:
        # Support both call signatures: is_done() from BaseAgent and
        # is_done(frames, latest_frame) used by ArcEnvWrapper/scripts.
        if latest_frame is not None:
            state = getattr(latest_frame, 'state', None)
            if state is None:
                return self.action_count >= self.MAX_ACTIONS
            state_str = str(state)
            if hasattr(state, 'name'):
                state_str = state.name
            return "WIN" in state_str or self.action_count >= self.MAX_ACTIONS
        return self._done or self.action_count >= self.MAX_ACTIONS

    def choose_action(self, frame):
        """Select an action using the current policy.

        Accepts either a raw frame object (for BaseAgent compatibility) or
        pre-processed tensors via keyword arguments.
        """
        grid = self.processor.frame_to_tensor(frame).unsqueeze(0).to(self.device)
        aux = self.processor.extract_aux_features(
            frame, self.action_count, self._prev_action
        ).unsqueeze(0).to(self.device)
        mask = self.processor.get_available_actions_mask(frame).unsqueeze(0).to(self.device)

        with torch.no_grad():
            action, x, y, _, _ = self.policy.sample(grid, aux, mask)

        self.action_count += 1
        action_int = action.item()
        self._prev_action = action_int

        # Convert to GameAction format
        try:
            from arcengine import GameAction
            action_map = {
                0: GameAction.RESET, 1: GameAction.ACTION1, 2: GameAction.ACTION2,
                3: GameAction.ACTION3, 4: GameAction.ACTION4, 5: GameAction.ACTION5,
                6: GameAction.ACTION6, 7: GameAction.ACTION7,
            }
            ga = action_map.get(action_int, GameAction.ACTION1)
            if action_int == 6:
                ga.set_data({"x": x.item(), "y": y.item()})
            ga.reasoning = f"RL policy action {action_int}"
            return ga
        except ImportError:
            return action_int

    def on_episode_end(self, reward: float, info=None) -> None:
        super().on_episode_end(reward, info)
        self.action_count = 0
        self._prev_action = 0

    @property
    def prev_action(self):
        return self._prev_action

    @prev_action.setter
    def prev_action(self, value):
        self._prev_action = value
