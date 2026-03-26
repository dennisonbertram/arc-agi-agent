# ARC-AGI-3 API Test Results

**Date:** 2026-03-26

## Summary

API connection is confirmed working. The arc-agi SDK successfully connects to `https://three.arcprize.org`, lists 25 games, creates game sessions, and accepts step actions. Training against real games using `train.py --mode ONLINE` completes successfully.

---

## Step 1: Direct curl Test

```
curl -s -H "X-API-Key: REDACTED-..." https://three.arcprize.org/api/games
```

**Result:** HTTP 200 OK

Returns a JSON array of 25 game objects. Example structure:
```json
{
  "game_id": "tn36-ab4f63cc",
  "title": "TN36",
  "tags": ["click"],
  "baseline_actions": [23, 22, 26, 37, 25, 56, 61]
}
```

---

## Step 2: arc-agi SDK Test

### SDK Interface Discovery

The `Arcade` constructor signature:
```python
Arcade(
    arc_api_key: str = '',          # NOT 'api_key' — must use 'arc_api_key'
    arc_base_url: str = 'https://three.arcprize.org',
    operation_mode: OperationMode = OperationMode.NORMAL,
    environments_dir: str = 'environment_files',
    recordings_dir: str = 'recordings',
    logger: Logger | None = None
)
```

**Critical fix:** The parameter is `arc_api_key`, not `api_key`. The original test script template was incorrect.

### Available Games (25 total)

| game_id | title | tags |
|---------|-------|------|
| tn36-ab4f63cc | TN36 | click |
| sb26-7fbdac44 | SB26 | keyboard_click |
| ft09-0d8bbf25 | FT09 | (none) |
| re86-4e57566e | RE86 | keyboard_click |
| vc33-9851e02b | VC33 | click |
| ls20-9607627b | LS20 | keyboard |
| wa30-ee6fef47 | WA30 | keyboard |
| sp80-0ee2d095 | SP80 | keyboard_click |
| tr87-cd924810 | TR87 | keyboard |
| g50t-5849a774 | G50T | keyboard |
| cd82-fb555c5d | CD82 | keyboard_click |
| lp85-305b61c3 | LP85 | click |
| sc25-f9b21a2f | SC25 | keyboard_click |
| r11l-aa269680 | R11L | click |
| lf52-271a04aa | LF52 | click |
| dc22-4c9bff3e | DC22 | keyboard_click |
| sk48-41055498 | SK48 | keyboard_click |
| bp35-0a0ad940 | BP35 | keyboard_click |
| su15-4c352900 | SU15 | click |
| s5i5-a48e4b1d | S5I5 | click |
| ar25-e3c63847 | AR25 | keyboard_click |
| ka59-9f096b4a | KA59 | keyboard_click |
| m0r0-dadda488 | M0R0 | keyboard_click |
| tu93-2b534c15 | TU93 | keyboard_click |
| cn04-65d47d14 | CN04 | keyboard_click |

### SDK Observation Structure

`arcade.make(game_id)` returns a `RemoteEnvironmentWrapper`. Calling `env.reset()` returns `arcengine.enums.FrameDataRaw` with:
- `game_id` — string
- `state` — `GameState.NOT_FINISHED` | `GameState.WIN` | `GameState.LOSE`
- `levels_completed` — int
- `win_levels` — int (total levels to complete)
- `available_actions` — list of int action IDs
- `guid` — session UUID
- `action_input` — the last action taken

### Action API

`env.step(action, data=None)` where:
- `action` is a `GameAction` enum (ACTION1–ACTION7, RESET)
- For click-based games, `data={"x": int, "y": int}` is required
- Without click data on click games, server returns HTTP 500

**Action mapping used in `arc_env_wrapper.py`:**
```
0 -> RESET
1 -> ACTION1
2 -> ACTION2
3 -> ACTION3
4 -> ACTION4
5 -> ACTION5
6 -> ACTION6  (click, requires x/y data)
7 -> ACTION7
```

### `arcade.list_games()` — Does Not Exist

The method `list_games()` does not exist on `Arcade`. The correct way to enumerate games:
```python
games = arcade.available_environments  # returns list[EnvironmentInfo]
game_ids = [e.game_id for e in games]
```

---

## Step 3: Training Run

Command:
```bash
uv run python scripts/train.py --mode ONLINE --steps 5 --rollout 50 --game tn36-ab4f63cc
```

### Training Output

```
Training on game: tn36-ab4f63cc (mode: ONLINE)
Steps: 5, Rollout: 50, LR: 0.0003
Successfully fetched 25 environment(s) from API
Created new scorecard: 41ea0482-a3a7-4063-ac92-c035ff318c03
Successfully reset game tn36-ab4f63cc
Step 1/5 | Reward: -2.200 | Episodes: 1 | P-Loss: 0.4378 | V-Loss: 0.1579 | Entropy: 4.9867
[...game resets between rollouts...]
Step 5/5 | Reward: -2.450 | Episodes: 1 | P-Loss: 0.1887 | V-Loss: 0.2100 | Entropy: 4.4926
Saved checkpoint to checkpoints/latest.pt
```

**Result:** PASS — training ran 5 steps with 50-step rollouts against a live ARC-AGI-3 game.

---

## Bugs Found and Fixed

### Bug 1: Incorrect `Arcade` constructor parameter in test_api.py

The provided template used `api_key=api_key` but the SDK requires `arc_api_key=api_key`. Fixed in `scripts/test_api.py`.

### Bug 2: `arcade.list_games()` does not exist

The template called `arcade.list_games()` which doesn't exist. Correct usage is `arcade.available_environments`. Fixed in `scripts/test_api.py`.

### Bug 3: ACTION6 without click data returns HTTP 500

Calling `env.step(GameAction.ACTION6)` on a click-based game with no `data` argument causes a server 500 error. Must pass `data={"x": int, "y": int}`. The existing `arc_env_wrapper.py` handles this correctly (line 54).

---

## Known Issues / Notes

- The `arc_env_wrapper.py` passes `api_key=""` to `Arcade` when called from `train.py` with no explicit key argument. However, the `Arcade` constructor apparently also reads from the environment (`ARC_API_KEY` var in `.env`), which is why training still works.
- The `FrameDataRaw` object does not expose a pixel grid directly — `frame` attribute appears to be None/empty for remote games in initial testing. State info (levels_completed, state, available_actions) is available.
- Each `arcade.make()` call creates a new scorecard on the server. Running many training episodes will generate many scorecards.

---

## Files Changed

- `/Users/dennisonbertram/Develop/arc-agi-agent/scripts/test_api.py` — created and corrected
- `/Users/dennisonbertram/Develop/arc-agi-agent/docs/investigations/api-test-results.md` — this file
- `/Users/dennisonbertram/Develop/arc-agi-agent/checkpoints/latest.pt` — created by training run
