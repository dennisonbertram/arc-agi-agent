"""Diagnose what frame data the ARC-AGI-3 SDK returns in ONLINE mode."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

from arc_agi import Arcade, OperationMode

arcade = Arcade(arc_api_key=os.getenv("ARC_API_KEY", ""), operation_mode=OperationMode.ONLINE)
games = arcade.available_environments
print(f"Available games: {len(games)}")

# Pick first game
game = games[0]
print(f"\nGame: {game.game_id}")
print(f"Game attrs: {[a for a in dir(game) if not a.startswith('_')]}")

env = arcade.make(game.game_id, include_frame_data=True)
print(f"\nEnv type: {type(env)}")
print(f"Env attrs: {[a for a in dir(env) if not a.startswith('_')]}")

# Reset and examine the observation
obs = env.reset()
print(f"\nObs type: {type(obs)}")
print(f"Obs attrs: {[a for a in dir(obs) if not a.startswith('_')]}")

# Print every attribute value
for attr in sorted(dir(obs)):
    if attr.startswith('_'):
        continue
    try:
        val = getattr(obs, attr)
        if callable(val):
            continue
        if isinstance(val, (list, tuple)) and len(val) > 10:
            print(f"  obs.{attr}: type={type(val).__name__}, len={len(val)}, first_few={val[:3]}...")
        elif isinstance(val, bytes) and len(val) > 100:
            print(f"  obs.{attr}: bytes, len={len(val)}")
        else:
            print(f"  obs.{attr} = {repr(val)}")
    except Exception as e:
        print(f"  obs.{attr}: ERROR {e}")

# Specifically check frame/grid data
print("\n--- Frame Data Analysis ---")
if hasattr(obs, 'frame'):
    f = obs.frame
    print(f"frame type: {type(f)}, value: {repr(f)[:200] if f else 'None/Empty'}")
    if f and isinstance(f, (list, tuple)):
        print(f"frame shape: {len(f)}x{len(f[0]) if f else 0}")

if hasattr(obs, 'grid'):
    g = obs.grid
    print(f"grid type: {type(g)}, value: {repr(g)[:200] if g else 'None/Empty'}")
    if g and isinstance(g, (list, tuple)):
        print(f"grid shape: {len(g)}x{len(g[0]) if g else 0}")

if hasattr(obs, 'pixels'):
    p = obs.pixels
    print(f"pixels type: {type(p)}, len: {len(p) if p else 'None'}")

if hasattr(obs, 'image'):
    i = obs.image
    print(f"image type: {type(i)}, len: {len(i) if i else 'None'}")

# Check the raw response data
if hasattr(obs, 'model_dump'):
    print(f"\nmodel_dump: {repr(obs.model_dump())[:500]}")
elif hasattr(obs, '__dict__'):
    print(f"\n__dict__: {repr(obs.__dict__)[:500]}")

# Take a step and examine
print("\n--- After Step ---")
try:
    step_result = env.step(5, data={"x": 0, "y": 0})  # INTERACT at (0,0)
    print(f"Step result type: {type(step_result)}")
    for attr in sorted(dir(step_result)):
        if attr.startswith('_') or callable(getattr(step_result, attr, None)):
            continue
        try:
            val = getattr(step_result, attr)
            if isinstance(val, (list, tuple)) and len(val) > 10:
                print(f"  step.{attr}: type={type(val).__name__}, len={len(val)}")
            elif isinstance(val, bytes) and len(val) > 100:
                print(f"  step.{attr}: bytes, len={len(val)}")
            else:
                print(f"  step.{attr} = {repr(val)}")
        except:
            pass
except Exception as e:
    print(f"Step error: {e}")
    import traceback
    traceback.print_exc()

# Also check the arc_agi module for FrameDataRaw definition
print("\n--- SDK Classes ---")
import arc_agi
for name in dir(arc_agi):
    obj = getattr(arc_agi, name)
    if isinstance(obj, type):
        print(f"  {name}: {[a for a in dir(obj) if not a.startswith('_') and not callable(getattr(obj, a, None))][:10]}")
