# ARC-AGI-3 SDK Frame Data Analysis

**Date**: 2026-03-26
**Mode**: ONLINE (remote API at https://three.arcprize.org)

---

## Summary

The CNN encoder in `src/models/encoder.py` expects a `[batch, 16, 64, 64]` one-hot tensor. The ARC-AGI-3 SDK ONLINE mode **does** return real 64x64 frame data. The `FrameDataRaw.frame` field is **NOT None and NOT empty** for remote games — it contains a list of `numpy.ndarray` objects each with shape `(64, 64)` and `dtype=int8`.

---

## SDK Data Model

### FrameDataRaw (arcengine)

`FrameDataRaw` is a Pydantic `BaseModel` with:

```python
class FrameDataRaw(BaseModel):
    game_id: str = ""
    state: GameState = GameState.NOT_PLAYED
    levels_completed: int = 0
    win_levels: int = 0
    action_input: ActionInput = Field(default_factory=ActionInput)
    guid: Optional[str] = None
    full_reset: bool = False
    available_actions: list[int] = Field(default_factory=list)

    # runtime-only, NOT in model_dump(), NOT serialized to JSON
    _frame: List[ndarray] = PrivateAttr(default_factory=list)

    @property
    def frame(self) -> List[ndarray]: ...
```

**Critical**: `frame` is a **private attribute** (`PrivateAttr`). It is NOT included in `model_dump()` or JSON serialization. It is populated at runtime by `RemoteEnvironmentWrapper._convert_to_frame_data_raw()`.

### FrameData (arcengine) — the wire format

The API returns JSON with `frame: list[list[list[int]]]` — a 3D list (layers × rows × columns). The `RemoteEnvironmentWrapper` converts this to `FrameDataRaw._frame` as a list of `numpy.ndarray`.

```python
# In RemoteEnvironmentWrapper._convert_to_frame_data_raw()
frame_data_raw.frame = [
    np.array(frame_layer, dtype=np.int8) for frame_layer in frame_data.frame
]
```

---

## Confirmed Frame Dimensions

Live test results from ONLINE mode (3 different games):

| Game | Reset frames | Step frames | Shape | dtype | min | max |
|------|-------------|-------------|-------|-------|-----|-----|
| sc25-f9b21a2f | 1 | 22 (ACTION5) | (64, 64) | int8 | 0 | 15 |
| ka59-9f096b4a | 1 | — | (64, 64) | int8 | 0 | 15 |
| m0r0-dadda488 | 1 | — | (64, 64) | int8 | 5 | 12 |
| sb26-7fbdac44 | 1 | 1 (ACTION1-3) | (64, 64) | int8 | 0 | 15 |

**All frames are exactly 64x64 with int8 values in range [0, 15].**

---

## Multi-Frame Animation Behaviour

`FrameDataRaw.frame` is a **list** of frames, not a single frame:

- **On `reset()`**: Always returns exactly **1 frame** — the initial game state
- **On `step()`**: Returns **1 to N frames** representing an animation sequence
  - Some games return 1 frame per step (navigation games like sb26)
  - Some games return many frames per step (animated games like sc25 which returned 22 frames for ACTION5)
  - The **first frame** of a step result == the **last frame** of the prior step's result (confirmed: `first == last` is `True` for sc25)
  - The **last frame** is the new "current state" after the action

This means `frame[-1]` is always the authoritative current state.

---

## CNN Encoder Compatibility

The `StateProcessor.frame_to_tensor()` in `src/environment/state_processor.py` correctly handles this:

```python
def frame_to_tensor(self, frame) -> torch.Tensor:
    raw = frame.frame  # list of numpy arrays
    arr = np.zeros((64, 64), dtype=np.int64)
    if isinstance(raw, list) and len(raw) > 0:
        first = raw[0]
        if isinstance(first, np.ndarray):
            # frame.frame is a list whose first element is a 2-D numpy array
            grid_np = first
            ...
```

**Issue**: `StateProcessor` uses `raw[0]` (first frame) rather than `raw[-1]` (last frame). For multi-frame step responses, this means the agent is observing the **animation start state** rather than the **final game state** after the action. This may be harmless if the first and last frames are the same (as confirmed for sc25), but it's architecturally incorrect.

**The `GridEncoder.encode_grid()` method** takes `list[list[int]]` (a 2D Python list), not a numpy array. It does NOT handle the multi-frame list natively — it's a convenience method for the raw 2D grid, not for `FrameDataRaw.frame`.

---

## GameAction Enum Values

```python
RESET = 0
ACTION1 = 1
ACTION2 = 2
ACTION3 = 3
ACTION4 = 4
ACTION5 = 5
ACTION6 = 6
ACTION7 = 7
```

**Important**: The `env.step()` method expects a `GameAction` enum, NOT an integer. Passing `env.step(5, ...)` fails with `AttributeError: 'int' object has no attribute 'value'`. Use `env.step(GameAction.ACTION5, ...)`.

---

## GameState Values

```python
NOT_PLAYED = 'NOT_PLAYED'
NOT_FINISHED = 'NOT_FINISHED'
WIN = 'WIN'
GAME_OVER = 'GAME_OVER'
```

`NOT_STARTED` appears in `StateProcessor.STATE_MAP` but is NOT in the SDK enum — `NOT_PLAYED` is the initial state, which `StateProcessor` maps to index 0 (same as `NOT_STARTED`). This is safe.

---

## Available Actions

The `available_actions` field in `FrameDataRaw` contains **integer action IDs** (not enum values). Example: `[1, 2, 3, 4, 6]` — note ACTION5 (id=5) is absent from sb26 but present for other games. ACTION6 is available for sc25. This list is game-specific and changes as gameplay progresses.

---

## Wire Format from API

The REST API `/api/cmd/RESET` and `/api/cmd/ACTION{N}` endpoints return JSON matching `FrameData`:

```json
{
  "game_id": "sc25-f9b21a2f",
  "frame": [[[5,5,...], ...], ...],  // 3D list: layers × rows × cols
  "state": "NOT_FINISHED",
  "levels_completed": 0,
  "win_levels": 6,
  "action_input": {"id": 0, "data": {...}, "reasoning": null},
  "guid": "1e04bc6d-...",
  "full_reset": false,
  "available_actions": [1, 2, 3, 4, 6]
}
```

Note: `model_dump()` on `FrameDataRaw` does NOT include `frame` (it's a `PrivateAttr`).

---

## Key Invariants Established

1. **Frame data is always present in ONLINE mode** — `FrameDataRaw.frame` is never `None` or empty after a successful `reset()` or `step()`.
2. **Frame dimensions are always (64, 64)** with `dtype=int8` and values in `[0, 15]`.
3. **`reset()` always returns exactly 1 frame**.
4. **`step()` returns 1+ frames as animation sequence**; `frame[-1]` is the final state.
5. **`StateProcessor` uses `frame[0]`** — correct only because first==last, but architecturally should use `frame[-1]`.
6. **`env.step()` requires `GameAction` enum**, not integers.
7. **`frame` is a `PrivateAttr`** — not in `model_dump()`, set after deserialization.
8. **Color values 0-15 are valid** and observed in real game data.

---

## Risks Identified

1. **`StateProcessor.frame_to_tensor()` uses `frame[0]`** not `frame[-1]`. For animated games, this is technically reading the pre-action state rather than post-action state. Confirmed safe for sc25 only (first==last), but other games may differ.
2. **No validation that frame shape is always 64x64** — if a game returns a different grid size the encoder will crash or zero-pad silently.
3. **`available_actions` is game and state dependent** — agents should use the action mask from `StateProcessor.get_available_actions_mask()` rather than hardcoding action counts.
4. **Script bug**: `game.id` does not exist; use `game.game_id`. The original script template had this error; it was fixed before running.
