"""LLM-based agent for ARC-AGI-3 using Claude API."""
import os
import re
import json
import random
import numpy as np
from typing import Optional
from anthropic import Anthropic


class LLMAgent:
    """Uses Claude to reason about ARC-AGI-3 grid puzzles and choose actions.

    The agent converts the 64x64 colored grid to a compact text representation,
    maintains a history of recent actions and their effects, and asks Claude to
    reason step-by-step about the pattern and what action to take next.
    """

    # Color abbreviations sent to the model (single character per color value)
    COLOR_MAP = {
        0: '.', 1: 'B', 2: 'R', 3: 'G', 4: 'Y',
        5: 'X', 6: 'M', 7: 'O', 8: 'C', 9: 'W',
        10: 'w', 11: 'P', 12: 'g', 13: 'o', 14: 'T', 15: 'r',
    }

    COLOR_LEGEND = (
        ". = black(0), B = blue(1), R = red(2), G = green(3), Y = yellow(4), "
        "X = gray(5), M = magenta(6), O = orange(7), C = cyan(8), W = maroon(9), "
        "w = maroon-light(10), P = purple(11), g = dk-green(12), o = dk-orange(13), "
        "T = teal(14), r = rose(15)"
    )

    SYSTEM_PROMPT = """You are an expert at solving ARC-AGI-3 visual puzzles.

The game shows a 64x64 grid of colored cells. Your goal is to figure out the
transformation rule and apply it to complete each level. There are multiple levels
per game; completing all levels wins the game.

Color legend: {color_legend}

Available actions depend on the game type:
  Keyboard games  → ACTION:UP / ACTION:DOWN / ACTION:LEFT / ACTION:RIGHT / ACTION:UNDO
  Click games     → ACTION:CLICK(x,y) where x=column, y=row (both 0-63) / ACTION:UNDO
  Hybrid games    → Any of the above plus ACTION:INTERACT
  Any game        → ACTION:RESET (restart the current level)

Reasoning strategy:
1. Describe what you see in the grid (patterns, shapes, colors).
2. Recall what happened on the last action — what changed?
3. Form or update a hypothesis about the rule or goal.
4. Decide which action advances toward that goal.

Respond with a short reasoning paragraph, then end your response with exactly one
action line in the format:
  ACTION:<command>
Examples: ACTION:UP  ACTION:CLICK(12,34)  ACTION:UNDO  ACTION:RESET
""".format(color_legend=COLOR_LEGEND)

    def __init__(self, model: str = "claude-sonnet-4-20250514", max_history: int = 10):
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        self.client = Anthropic(api_key=api_key)
        self.model = model
        self.max_history = max_history
        self.history: list[dict] = []
        self.prev_grid: Optional[list] = None
        self.game_tags: list = []
        self.levels_completed: int = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def choose_action(
        self,
        grid,
        game_tags=None,
        levels_completed: int = 0,
        available_actions=None,
    ) -> tuple[int, int, int]:
        """Use Claude to choose the next action.

        Parameters
        ----------
        grid:
            The current 64x64 grid (list-of-lists or numpy array).
        game_tags:
            List of tag strings for the current game (e.g. ['keyboard'], ['click']).
        levels_completed:
            How many levels have been finished so far.
        available_actions:
            Optional list of integer action codes that are currently valid.

        Returns
        -------
        (action_type, x, y) where action_type is 0-7, x/y are 0-63.
        """
        if game_tags is not None:
            self.game_tags = game_tags
        self.levels_completed = levels_completed

        grid_list = grid.tolist() if isinstance(grid, np.ndarray) else grid
        grid_text = self.grid_to_text(grid_list)
        diff_text = self.diff_grids(self.prev_grid, grid_list)

        user_msg = self._build_user_message(grid_text, diff_text, available_actions)

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=600,
                system=self.SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_msg}],
            )
            reply = response.content[0].text
            print(f"  LLM: {reply[:300]}{'...' if len(reply) > 300 else ''}")

            action_type, x, y = self._parse_action(reply)

        except Exception as exc:
            print(f"  LLM error: {exc}")
            action_type, x, y = self._fallback_action()

        # Record history entry before updating prev_grid
        self.history.append({
            "action": self._action_label(action_type, x, y),
            "diff": diff_text[:120],
        })
        # Trim history
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        self.prev_grid = grid_list
        return action_type, x, y

    def reset(self) -> None:
        """Reset agent state for a new episode."""
        self.history = []
        self.prev_grid = None
        self.levels_completed = 0

    # ------------------------------------------------------------------
    # Grid representation helpers
    # ------------------------------------------------------------------

    def grid_to_text(self, grid: list) -> str:
        """Convert a 64x64 grid to a compact text representation.

        Finds the bounding box of non-zero (non-black) cells and renders
        only that region, capped at 40x40 characters to keep prompts short.
        """
        non_zero = [
            (y, x, grid[y][x])
            for y in range(len(grid))
            for x in range(len(grid[y]))
            if grid[y][x] != 0
        ]

        if not non_zero:
            return "Grid is entirely black (all zeros)."

        min_y = min(p[0] for p in non_zero)
        max_y = max(p[0] for p in non_zero)
        min_x = min(p[1] for p in non_zero)
        max_x = max(p[1] for p in non_zero)

        cap = 40  # maximum rows / columns to show
        lines = [f"Grid region rows {min_y}-{min(max_y, min_y + cap - 1)}, "
                 f"cols {min_x}-{min(max_x, min_x + cap - 1)}:"]

        for y in range(min_y, min(max_y + 1, min_y + cap)):
            row = ""
            for x in range(min_x, min(max_x + 1, min_x + cap)):
                row += self.COLOR_MAP.get(grid[y][x], "?")
            lines.append(row)

        return "\n".join(lines)

    def diff_grids(self, old_grid, new_grid) -> str:
        """Describe what changed between two grid states."""
        if old_grid is None:
            return "First observation — no previous state."

        changes = []
        rows = min(len(old_grid), len(new_grid))
        cols = min(len(old_grid[0]), len(new_grid[0])) if rows > 0 else 0
        for y in range(rows):
            for x in range(cols):
                if old_grid[y][x] != new_grid[y][x]:
                    changes.append(f"({y},{x}): {old_grid[y][x]}→{new_grid[y][x]}")

        if not changes:
            return "No visible grid changes."
        if len(changes) > 20:
            return f"{len(changes)} cells changed. First 20: " + ", ".join(changes[:20])
        return f"{len(changes)} cells changed: " + ", ".join(changes)

    # ------------------------------------------------------------------
    # Prompt construction
    # ------------------------------------------------------------------

    def _build_user_message(self, grid_text: str, diff_text: str, available_actions) -> str:
        tag_set = {str(t).lower() for t in self.game_tags}
        if "keyboard_click" in tag_set:
            action_hint = "Keyboard + click game. Can use UP/DOWN/LEFT/RIGHT, CLICK(x,y), INTERACT, UNDO, RESET."
        elif "click" in tag_set:
            action_hint = "Click-only game. Can use CLICK(x,y), INTERACT (ACTION5), UNDO, RESET."
        elif "keyboard" in tag_set:
            action_hint = "Keyboard game. Can use UP/DOWN/LEFT/RIGHT, UNDO, RESET."
        else:
            action_hint = "Unknown game type. Try UP/DOWN/LEFT/RIGHT or CLICK(x,y)."

        if available_actions:
            action_hint += f" Available action codes: {available_actions}"

        history_text = ""
        if self.history:
            history_text = "\nRecent actions and effects:\n"
            for entry in self.history:
                history_text += f"  {entry['action']} → {entry['diff']}\n"

        return (
            f"Game tags: {self.game_tags}\n"
            f"Levels completed: {self.levels_completed}\n"
            f"Action type: {action_hint}\n"
            f"{history_text}\n"
            f"Current grid:\n{grid_text}\n\n"
            f"Last action effect: {diff_text}\n\n"
            "What is the pattern? What should I do next? End with ACTION:<command>."
        )

    # ------------------------------------------------------------------
    # Action parsing
    # ------------------------------------------------------------------

    def _parse_action(self, text: str) -> tuple[int, int, int]:
        """Parse LLM response text into (action_type, x, y).

        Searches from the end of the response for the last ACTION: directive.
        """
        upper = text.upper()
        for line in reversed(upper.split("\n")):
            line = line.strip()
            if "ACTION:" not in line:
                continue
            action_str = line.split("ACTION:")[-1].strip()

            if action_str.startswith("CLICK"):
                m = re.search(r"CLICK\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)", action_str)
                if m:
                    return 6, int(m.group(1)), int(m.group(2))
                return 6, 32, 32  # default center click

            if "UP" in action_str:
                return 1, 0, 0
            if "DOWN" in action_str:
                return 2, 0, 0
            if "LEFT" in action_str:
                return 3, 0, 0
            if "RIGHT" in action_str:
                return 4, 0, 0
            if "INTERACT" in action_str:
                return 5, 0, 0
            if "UNDO" in action_str:
                return 7, 0, 0
            if "RESET" in action_str:
                return 0, 0, 0

        # Fallback if no recognizable action found
        return self._fallback_action()

    def _fallback_action(self) -> tuple[int, int, int]:
        """Return a random valid action based on game tags."""
        tag_set = {str(t).lower() for t in self.game_tags}
        if "click" in tag_set and "keyboard" not in tag_set:
            return 6, random.randint(0, 63), random.randint(0, 63)
        return random.choice([1, 2, 3, 4]), 0, 0

    @staticmethod
    def _action_label(action_type: int, x: int, y: int) -> str:
        labels = {0: "RESET", 1: "UP", 2: "DOWN", 3: "LEFT", 4: "RIGHT",
                  5: "INTERACT", 7: "UNDO"}
        if action_type == 6:
            return f"CLICK({x},{y})"
        return labels.get(action_type, f"ACTION{action_type}")
