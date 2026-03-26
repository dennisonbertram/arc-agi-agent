"""Tests for LLMAgent (src/agent/llm_agent.py).

These tests are fully offline — no API calls are made.
"""
import importlib
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Stub the anthropic package so tests work without a real API key / network
# ---------------------------------------------------------------------------
def _make_anthropic_stub():
    stub = types.ModuleType("anthropic")

    class FakeMessage:
        class content:
            pass

    class FakeMessages:
        def create(self, *args, **kwargs):
            msg = MagicMock()
            msg.content = [MagicMock(text="Let me think...\nACTION:UP")]
            return msg

    class FakeAnthropic:
        def __init__(self, api_key=""):
            self.messages = FakeMessages()

    stub.Anthropic = FakeAnthropic
    return stub


# Patch before importing the module under test
sys.modules.setdefault("anthropic", _make_anthropic_stub())

from src.agent.llm_agent import LLMAgent  # noqa: E402  (import after stub)


# ---------------------------------------------------------------------------
# Helper: build a simple 64x64 grid
# ---------------------------------------------------------------------------
def _empty_grid(val=0):
    return [[val] * 64 for _ in range(64)]


def _grid_with_cell(y, x, color):
    g = _empty_grid()
    g[y][x] = color
    return g


# ===========================================================================
# Tests
# ===========================================================================

class TestGridToText(unittest.TestCase):
    def setUp(self):
        self.agent = LLMAgent()

    def test_all_black_grid(self):
        grid = _empty_grid(0)
        result = self.agent.grid_to_text(grid)
        self.assertIn("entirely black", result.lower())

    def test_single_colored_cell(self):
        grid = _grid_with_cell(10, 5, 3)  # color 3 -> 'G'
        result = self.agent.grid_to_text(grid)
        self.assertIn("G", result)

    def test_bounding_box_reported(self):
        grid = _grid_with_cell(2, 4, 1)  # one cell at row=2, col=4
        result = self.agent.grid_to_text(grid)
        # Should mention the row range and col range
        self.assertIn("2", result)
        self.assertIn("4", result)

    def test_numpy_array_accepted(self):
        arr = np.zeros((64, 64), dtype=int)
        arr[5, 10] = 7  # color 7 -> 'O'
        result = self.agent.grid_to_text(arr.tolist())
        self.assertIn("O", result)

    def test_large_region_capped(self):
        # Fill 50x50 non-zero region; output should be capped at 40 rows
        grid = [[1] * 64 for _ in range(64)]
        result = self.agent.grid_to_text(grid)
        lines = [l for l in result.split("\n") if l and not l.startswith("Grid")]
        self.assertLessEqual(len(lines), 40)

    def test_color_map_coverage(self):
        """Every color value 0-15 should produce a single character."""
        for color, char in LLMAgent.COLOR_MAP.items():
            self.assertEqual(len(char), 1, f"Color {color} maps to multi-char: {char!r}")


class TestDiffGrids(unittest.TestCase):
    def setUp(self):
        self.agent = LLMAgent()

    def test_first_observation(self):
        result = self.agent.diff_grids(None, _empty_grid())
        self.assertIn("First", result)

    def test_no_change(self):
        grid = _empty_grid()
        result = self.agent.diff_grids(grid, grid)
        self.assertIn("No visible", result)

    def test_single_cell_change(self):
        old = _empty_grid()
        new = _grid_with_cell(3, 7, 2)
        result = self.agent.diff_grids(old, new)
        self.assertIn("(3,7)", result)
        self.assertIn("0→2", result)

    def test_many_changes_truncated(self):
        old = _empty_grid()
        new = [[1] * 64 for _ in range(64)]  # all cells changed
        result = self.agent.diff_grids(old, new)
        # Should mention count > 20 and truncate list
        self.assertIn("First 20", result)


class TestParseAction(unittest.TestCase):
    def setUp(self):
        self.agent = LLMAgent()

    def test_up(self):
        self.assertEqual(self.agent._parse_action("I should go up.\nACTION:UP"), (1, 0, 0))

    def test_down(self):
        self.assertEqual(self.agent._parse_action("ACTION:DOWN"), (2, 0, 0))

    def test_left(self):
        self.assertEqual(self.agent._parse_action("ACTION:LEFT"), (3, 0, 0))

    def test_right(self):
        self.assertEqual(self.agent._parse_action("ACTION:RIGHT"), (4, 0, 0))

    def test_interact(self):
        self.assertEqual(self.agent._parse_action("ACTION:INTERACT"), (5, 0, 0))

    def test_undo(self):
        self.assertEqual(self.agent._parse_action("ACTION:UNDO"), (7, 0, 0))

    def test_reset(self):
        self.assertEqual(self.agent._parse_action("ACTION:RESET"), (0, 0, 0))

    def test_click_with_coords(self):
        result = self.agent._parse_action("ACTION:CLICK(15,30)")
        self.assertEqual(result, (6, 15, 30))

    def test_click_with_spaces(self):
        result = self.agent._parse_action("ACTION:CLICK( 5 , 10 )")
        self.assertEqual(result, (6, 5, 10))

    def test_click_no_coords_defaults_center(self):
        result = self.agent._parse_action("ACTION:CLICK()")
        # No valid coords — should default to center (32,32)
        self.assertEqual(result[0], 6)
        self.assertEqual(result[1], 32)
        self.assertEqual(result[2], 32)

    def test_action_found_in_last_line(self):
        text = "Line 1\nLine 2\nACTION:RIGHT"
        self.assertEqual(self.agent._parse_action(text), (4, 0, 0))

    def test_last_action_wins_when_multiple(self):
        # If the model accidentally writes two ACTION: lines, last one wins
        text = "ACTION:UP\nBut actually ACTION:DOWN"
        result = self.agent._parse_action(text)
        self.assertEqual(result, (2, 0, 0))

    def test_no_action_returns_fallback(self):
        # No recognizable action → fallback (random directional or click)
        result = self.agent._parse_action("I have no idea what to do here.")
        self.assertIn(result[0], [1, 2, 3, 4, 6])

    def test_case_insensitive(self):
        self.assertEqual(self.agent._parse_action("action:up"), (1, 0, 0))


class TestReset(unittest.TestCase):
    def setUp(self):
        self.agent = LLMAgent()

    def test_reset_clears_history(self):
        self.agent.history = [{"action": "UP", "diff": "changed"}]
        self.agent.prev_grid = _empty_grid()
        self.agent.levels_completed = 3
        self.agent.reset()
        self.assertEqual(self.agent.history, [])
        self.assertIsNone(self.agent.prev_grid)
        self.assertEqual(self.agent.levels_completed, 0)


class TestChooseAction(unittest.TestCase):
    """Integration-level tests for choose_action with a stubbed LLM response."""

    def setUp(self):
        self.agent = LLMAgent()

    def test_returns_valid_action_tuple(self):
        grid = _grid_with_cell(10, 10, 1)
        result = self.agent.choose_action(grid, game_tags=["keyboard"], levels_completed=0)
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        action_type, x, y = result
        self.assertIn(action_type, range(8))
        self.assertIn(x, range(64))
        self.assertIn(y, range(64))

    def test_history_grows(self):
        grid = _grid_with_cell(5, 5, 2)
        self.agent.choose_action(grid)
        self.assertEqual(len(self.agent.history), 1)
        self.agent.choose_action(grid)
        self.assertEqual(len(self.agent.history), 2)

    def test_history_capped_at_max(self):
        self.agent.max_history = 3
        grid = _grid_with_cell(1, 1, 1)
        for _ in range(10):
            self.agent.choose_action(grid)
        self.assertLessEqual(len(self.agent.history), 3)

    def test_numpy_grid_accepted(self):
        arr = np.zeros((64, 64), dtype=np.int64)
        arr[20, 20] = 5
        result = self.agent.choose_action(arr, game_tags=["keyboard"])
        self.assertEqual(len(result), 3)

    def test_game_tags_stored(self):
        grid = _empty_grid()
        self.agent.choose_action(grid, game_tags=["click"])
        self.assertEqual(self.agent.game_tags, ["click"])


class TestActionLabel(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(LLMAgent._action_label(0, 0, 0), "RESET")
        self.assertEqual(LLMAgent._action_label(1, 0, 0), "UP")
        self.assertEqual(LLMAgent._action_label(2, 0, 0), "DOWN")
        self.assertEqual(LLMAgent._action_label(3, 0, 0), "LEFT")
        self.assertEqual(LLMAgent._action_label(4, 0, 0), "RIGHT")
        self.assertEqual(LLMAgent._action_label(5, 0, 0), "INTERACT")
        self.assertEqual(LLMAgent._action_label(6, 12, 34), "CLICK(12,34)")
        self.assertEqual(LLMAgent._action_label(7, 0, 0), "UNDO")


if __name__ == "__main__":
    unittest.main()
