#!/usr/bin/env python3
"""Play ARC-AGI-3 games with the LLM agent (Claude-based reasoning)."""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
load_dotenv()

import numpy as np

from src.agent.llm_agent import LLMAgent


def _get_grid_from_frame(frame) -> list:
    """Extract the last grid from a frame's animation sequence."""
    raw = frame.frame
    if isinstance(raw, np.ndarray):
        grid = raw if raw.ndim == 2 else raw[-1]
        return grid.tolist()
    if isinstance(raw, list) and raw:
        last = raw[-1]
        if isinstance(last, np.ndarray):
            return last.tolist()
        # Nested Python lists
        if isinstance(last, list) and last and isinstance(last[0], list):
            return last
        return raw
    return []


def main():
    parser = argparse.ArgumentParser(description="Play ARC-AGI-3 with the LLM agent.")
    parser.add_argument("--game", default="tr87", help="Partial game ID to match")
    parser.add_argument("--max-actions", type=int, default=50)
    parser.add_argument("--model", default="claude-sonnet-4-20250514")
    parser.add_argument("--mode", default="ONLINE", choices=["ONLINE", "OFFLINE", "NORMAL"])
    args = parser.parse_args()

    arc_api_key = os.getenv("ARC_API_KEY", "")
    anthropic_key = os.getenv("ANTHROPIC_API_KEY", "")

    if not anthropic_key:
        print("ERROR: ANTHROPIC_API_KEY not set.", file=sys.stderr)
        sys.exit(1)

    # ------------------------------------------------------------------ #
    # Import arc_agi
    # ------------------------------------------------------------------ #
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction
    except ImportError as exc:
        print(f"ERROR: arc_agi SDK not installed — {exc}", file=sys.stderr)
        sys.exit(1)

    mode_map = {
        "ONLINE": OperationMode.ONLINE,
        "OFFLINE": OperationMode.OFFLINE,
        "NORMAL": OperationMode.NORMAL,
    }
    arcade = Arcade(arc_api_key=arc_api_key, operation_mode=mode_map[args.mode])

    # ------------------------------------------------------------------ #
    # Find matching game
    # ------------------------------------------------------------------ #
    game_info = None
    for env_info in arcade.available_environments:
        if args.game in env_info.game_id:
            game_info = env_info
            break

    if game_info is None:
        print(f"ERROR: Game matching '{args.game}' not found.", file=sys.stderr)
        available = [e.game_id for e in arcade.available_environments]
        print(f"Available games: {available}", file=sys.stderr)
        sys.exit(1)

    print(f"Game  : {game_info.game_id}")
    print(f"Tags  : {getattr(game_info, 'tags', [])}")
    print(f"Model : {args.model}")
    print(f"Max   : {args.max_actions} actions")
    print("-" * 60)

    env = arcade.make(game_info.game_id, include_frame_data=True)
    agent = LLMAgent(model=args.model)
    agent.game_tags = getattr(game_info, "tags", [])

    # ------------------------------------------------------------------ #
    # Episode loop
    # ------------------------------------------------------------------ #
    frame = env.reset()

    action_map = {
        0: GameAction.RESET,
        1: GameAction.ACTION1,
        2: GameAction.ACTION2,
        3: GameAction.ACTION3,
        4: GameAction.ACTION4,
        5: GameAction.ACTION5,
        6: GameAction.ACTION6,
        7: GameAction.ACTION7,
    }

    for step in range(1, args.max_actions + 1):
        state = str(getattr(frame, "state", "NOT_FINISHED"))
        levels = getattr(frame, "levels_completed", 0)
        win_levels = getattr(frame, "win_levels", "?")

        print(f"\nStep {step}/{args.max_actions} | state={state} | "
              f"levels={levels}/{win_levels}")

        grid = _get_grid_from_frame(frame)
        available = getattr(frame, "available_actions", None)

        action_type, x, y = agent.choose_action(
            grid,
            game_tags=agent.game_tags,
            levels_completed=levels,
            available_actions=available,
        )

        print(f"  -> action_type={action_type}, x={x}, y={y} "
              f"({agent._action_label(action_type, x, y)})")

        # Execute action
        game_action = action_map.get(action_type, GameAction.ACTION1)
        data = {"x": x, "y": y} if action_type == 6 else None

        try:
            time.sleep(0.05)  # respect 600 RPM rate limit
            frame = env.step(game_action, data=data)
        except Exception as exc:
            print(f"  Step error: {exc}")
            break

        if frame is None:
            print("  Null frame returned — stopping.")
            break

        new_state = str(getattr(frame, "state", "NOT_FINISHED"))
        new_levels = getattr(frame, "levels_completed", 0)

        if new_levels > levels:
            print(f"\n  *** LEVEL COMPLETED! Now at {new_levels}/{win_levels} ***")

        if "WIN" in new_state:
            print(f"\n  *** WIN at step {step}! Levels: {new_levels}/{win_levels} ***")
            break

        if "GAME_OVER" in new_state:
            print(f"\n  GAME_OVER at step {step}.")
            break

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print("\n" + "=" * 60)
    final_state = str(getattr(frame, "state", "?")) if frame else "?"
    final_levels = getattr(frame, "levels_completed", 0) if frame else 0
    print(f"Final state  : {final_state}")
    print(f"Final levels : {final_levels}")
    print("=" * 60)


if __name__ == "__main__":
    main()
