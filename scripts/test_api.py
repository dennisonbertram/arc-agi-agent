"""Test ARC-AGI-3 API connection and list available games."""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

api_key = os.getenv("ARC_API_KEY", "")
print(f"API Key: {api_key[:8]}...{api_key[-4:]}")

# Test 1: Direct API call
import urllib.request
import json

req = urllib.request.Request(
    "https://three.arcprize.org/api/games",
    headers={"X-API-Key": api_key}
)
try:
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
        print(f"\nAPI Response Status: {resp.status}")
        print(f"Response type: {type(data)}")
        if isinstance(data, list):
            print(f"Number of games: {len(data)}")
            for g in data[:5]:
                print(f"  - {g}")
        elif isinstance(data, dict):
            print(f"Keys: {list(data.keys())}")
            for k, v in list(data.items())[:5]:
                print(f"  {k}: {v}")
except Exception as e:
    print(f"API Error: {e}")

# Test 2: Try arc-agi SDK
try:
    from arc_agi import Arcade, OperationMode
    print("\n--- arc-agi SDK Test ---")
    # NOTE: correct parameter is arc_api_key (not api_key)
    arcade = Arcade(arc_api_key=api_key, operation_mode=OperationMode.ONLINE)
    games = [env.game_id for env in arcade.available_environments]
    print(f"SDK found {len(games)} games")
    for g in games[:10]:
        print(f"  - {g}")

    # Try playing one game
    if games:
        game_id = games[0]
        print(f"\nPlaying game: {game_id}")
        env = arcade.make(game_id)
        obs = env.reset()
        print(f"  Obs type: {type(obs)}")
        print(f"  State: {obs.state}")
        print(f"  Levels completed: {obs.levels_completed}")
        print(f"  Win levels: {obs.win_levels}")
        print(f"  Available actions: {obs.available_actions}")

        # Take a few steps with click data
        import arcengine.enums
        action = arcengine.enums.GameAction.ACTION6
        for i in range(3):
            try:
                result = env.step(action, data={"x": 0, "y": 0})
                if result:
                    print(f"  Step {i+1}: state={result.state}, levels={result.levels_completed}")
                else:
                    print(f"  Step {i+1}: returned None")
            except Exception as e:
                print(f"  Step {i+1} error: {e}")
                break

except ImportError as e:
    print(f"\narc-agi SDK not available: {e}")
except Exception as e:
    print(f"\nSDK Error: {e}")
    import traceback
    traceback.print_exc()
