# ARC-AGI-3 Win Conditions Research

**Date:** 2026-03-26
**Status:** Comprehensive research completed

---

## Executive Summary

ARC-AGI-3 games are **won when `frame.state == GameState.WIN`**. Each game has multiple **levels** that must be completed sequentially. A complete WIN requires reaching the `win_levels` threshold on a single game. The framework provides clear feedback via frame data, reward signals, and scorecard tracking.

---

## 1. Core WIN Condition

### What Triggers a WIN?

The **only** way to win a game is when the frame data returned from the API indicates:

```
frame.state == GameState.WIN
```

In Python SDK terms (from `arcengine`):
```python
from arcengine import GameState

# After calling env.step(action)
if frame.state == GameState.WIN:
    print("Game WON!")
    # Episode terminates
```

In our environment wrapper (`src/environment/arc_env_wrapper.py`):
```python
state = str(getattr(self.current_frame, 'state', 'NOT_FINISHED'))
done = step_error_done or "WIN" in state or self.action_count >= self.max_actions
```

The condition checks if the string representation of state contains `"WIN"`.

### Frame State Values

The SDK defines four possible game states:

| State | Enum Value | Meaning |
|-------|-----------|---------|
| `NOT_PLAYED` | `'NOT_PLAYED'` | Initial state before game starts |
| `NOT_FINISHED` | `'NOT_FINISHED'` | Game in progress, keep playing |
| `WIN` | `'WIN'` | Player completed the game successfully |
| `GAME_OVER` | `'GAME_OVER'` | Player lost or game ended unsuccessfully |

**Critical:** `NOT_STARTED` appears in our code but is NOT in the SDK enum. Our `StateProcessor` maps it to the same index as `NOT_PLAYED` for compatibility.

---

## 2. Game Structure: Levels

### How Levels Work

Games are not single-puzzle tasks. They are **multi-level sequences** where:

- Each game has **multiple levels** (typically 1-6 based on observations)
- Levels are played **sequentially** - you cannot skip ahead
- Each level completion advances `levels_completed` counter
- A game is **won only when `levels_completed >= win_levels`**

### Frame Data Fields Related to Progress

```json
{
  "game_id": "ls20-016295f7601e",
  "state": "NOT_FINISHED|WIN|GAME_OVER",
  "levels_completed": 0,           // How many levels you've beaten
  "win_levels": 6,                 // How many you need to beat to WIN
  "available_actions": [1, 2, 3, 4, 5, 6]
}
```

### Example: `sc25-f9b21a2f`
- Initial frame shows: `levels_completed: 0, win_levels: 6`
- This game requires **winning all 6 levels** to achieve overall WIN
- Each action progresses the game; some actions complete a level
- When a level is completed, the next level loads automatically (or you see `full_reset: true`)

### Level Transition Detection

From `src/environment/arc_env_wrapper.py`:
```python
self.prev_levels = getattr(self.current_frame, 'levels_completed', 0)
# ... step ...
curr_levels = getattr(self.current_frame, 'levels_completed', 0)

# Reward shaper tracks this:
if levels_after > levels_before:
    r += self.level_bonus * (1.0 + 0.5 * levels_after)
```

When `levels_completed` increases, it signals a level was completed, and the reward jumps.

---

## 3. Detailed WIN Condition Logic

### In the API Response

The server returns `state: "WIN"` **only after** the agent's action completes the final required level. At that moment:

1. `state` field becomes `"WIN"`
2. `levels_completed` reaches `== win_levels`
3. Episode should terminate

### In Our Environment Wrapper

```python
# arc_env_wrapper.py, line 113
done = step_error_done or "WIN" in state or self.action_count >= self.max_actions
```

**Three ways an episode ends:**
1. `"WIN"` appears in state string → victory
2. `step_error_done = True` → API session expired (400/429 error)
3. `action_count >= max_actions` → timeout (default 200 actions)

### In Reward Shaping

```python
# reward_shaper.py, lines 21-22
if "WIN" in s:
    r += self.win_reward  # +10.0 by default
```

When WIN is detected, agent receives a **+10.0 reward spike** (plus any level bonus already earned).

---

## 4. Frame Data Anatomy

### Complete Frame Response Structure

```json
{
  "game_id": "sc25-f9b21a2f",
  "guid": "1e04bc6d-...",
  "frame": [[[5,5,...], ...], ...],  // 3D list: layers × 64 rows × 64 cols
  "state": "NOT_FINISHED|WIN|GAME_OVER",
  "levels_completed": 0,
  "win_levels": 6,
  "action_input": {
    "id": 0,
    "data": {},
    "reasoning": null
  },
  "full_reset": false,
  "available_actions": [1, 2, 3, 4, 6]
}
```

### Multi-Frame Animation Behavior

**Important discovery:** `frame` is **not** a single grid — it's a **list of grids** for animation:

- **On `reset()`:** Always returns exactly **1 frame** (the initial state)
- **On `step(action)`:** Returns **1 to N frames** (animation sequence)
  - Example: `sc25` returned **22 frames** for one ACTION5 call
  - The first frame of each step roughly matches the last frame of the previous step
  - The **final frame** (`frame[-1]`) is the new "current state" after the action

**Note:** Our `StateProcessor.frame_to_tensor()` uses `frame[0]` (first frame) rather than `frame[-1]` (last frame). This is technically incorrect for animated games, but our testing showed first == last for `sc25`, so it works in practice. This should be fixed architecturally.

### Frame Dimensions

All games return frames of exactly **64x64** with:
- `dtype=int8` (numpy)
- Values in range `[0, 15]` (16 colors)

---

## 5. How Games Differ: Tag-Based Action Restrictions

Different games support different action sets. Our code restricts valid actions based on game tags:

```python
# arc_env_wrapper.py, lines 130-142
def _actions_for_tags(tags: list) -> list[int] | None:
    tag_set = {str(t).lower() for t in tags}
    if "keyboard_click" in tag_set:
        return [0, 1, 2, 3, 4, 5, 6, 7]  # All actions
    if "click" in tag_set:
        return [0, 5, 6, 7]              # No keyboard movement
    if "keyboard" in tag_set:
        return [0, 1, 2, 3, 4, 7]        # No click actions
    return None  # Unknown: use frame's available_actions
```

### Game Tag Distribution (from M3 training results)

| Tag | Count | Example | Valid Actions |
|-----|-------|---------|----------------|
| `keyboard` | 4 | `tr87-cd924810` | 0,1,2,3,4,7 (directional + undo) |
| `click` | 7 | `tn36-ab4f63cc` | 0,5,6,7 (interact + click + undo) |
| `keyboard_click` | 13 | `m0r0-dadda488` | 0,1,2,3,4,5,6,7 (all) |
| `[]` (no tag) | 1 | `ft09-0d8bbf25` | Use frame's mask |

### Action Meanings by Game Type

**Keyboard games (e.g., `tr87-cd924810`):**
- ACTION1-4: Move cursor up/down/left/right
- ACTION5: Unavailable or unused
- ACTION6: Unavailable or unused
- ACTION7: Undo

**Click games (e.g., `tn36-ab4f63cc`):**
- ACTION1-4: Unavailable
- ACTION5: Interact/Enter/Select
- ACTION6: Click at (x, y) coordinate
- ACTION7: Undo

**Hybrid games (e.g., `m0r0-dadda488`):**
- ACTION1-4: Movement
- ACTION5: Interact
- ACTION6: Click
- ACTION7: Undo

---

## 6. Training Performance & Reward Signals

### Reward Shaper Parameters

From `src/training/reward_shaper.py`:

```python
RewardShaper(
    win_reward=10.0,              # Bonus for WIN
    game_over_penalty=-5.0,       # Penalty for GAME_OVER
    level_bonus=2.0,              # Bonus per level completed
    step_cost=-0.01,              # Cost per action (discourage wasting moves)
    undo_penalty=-0.05,           # Extra penalty for undo
    reset_penalty=-0.1,           # Extra penalty for unnecessary resets
    grid_change_bonus=0.02,       # Bonus for changing the grid state
    no_change_penalty=-0.005      # Small penalty for no-ops
)
```

### Learning Trajectories (from M3 training results)

Ran training for 15 steps on three game types with `rollout=50` actions per episode:

**Keyboard game (`tr87-cd924810`):**
| Step | Reward | Trend |
|------|--------|-------|
| 1    | -0.850 | Struggling |
| 5    | +0.225 | Improving |
| 10   | +0.425 | Learning |
| 15   | +0.500 | Strong positive learning |

**Click game (`tn36-ab4f63cc`, large action space 64×64):**
| Step | Reward | Trend |
|------|--------|-------|
| 1    | -2.350 | Difficult (high action space) |
| 5    | -0.950 | Still struggling |
| 10   | -0.400 | Improving |
| 15   | -0.075 | Nearly zero |

**Hybrid game (`m0r0-dadda488`):**
| Step | Reward | Trend |
|------|--------|-------|
| 1    | -0.950 | Mixed difficulty |
| 5    | -0.400 | Learning |
| 10   | +0.100 | Positive crossing |
| 15   | +0.200 | Strong learning |

**Key insight:** All three game types showed **positive learning trajectory** with **zero 400-error crashes** after fixing action masking and terminal state detection.

---

## 7. Detection of Game States

### In Frame Data

You can detect progress by monitoring:

1. **`frame.state`** - The definitive state indicator
   - `"WIN"` → episode won
   - `"GAME_OVER"` → episode lost
   - `"NOT_FINISHED"` → keep playing

2. **`frame.levels_completed`** - Level progress
   - Increments when a level is completed
   - Check if `levels_completed >= win_levels` for final status

3. **`frame.available_actions`** - What you're allowed to do
   - Dynamically changes based on game state
   - Always respect this list (never issue invalid actions)

4. **`frame.full_reset`** flag - Level transition marker
   - `true` indicates the previous level ended and a new one began
   - Useful for resetting internal agent memory between levels

### In API Error Codes

- **HTTP 400:** Game session expired (invalid `guid`, or game ended but agent continued sending actions)
  - **Action:** Treat as episode termination
  - **Fix:** Check for "WIN" or "GAME_OVER" and stop taking actions

- **HTTP 429:** Rate limit exceeded (more than 600 requests/minute)
  - **Action:** Exponential backoff retry
  - **Fix:** Add delays between steps (current code uses 0.05s base delay + backoff)

---

## 8. API Rate Limits & Session Management

### Rate Limiting

- **Limit:** 600 requests per minute (RPM)
- **Error:** HTTP 429 with `{"error":"RATE_LIMIT_EXCEEDED"}`
- **Handling:** Exponential backoff (implemented in `arc_env_wrapper.py`)

```python
max_retries = 3
backoff = 1.0
for attempt in range(max_retries + 1):
    try:
        self.current_frame = self._env.step(action, data=data)
        break
    except Exception as e:
        if "429" in str(e) and attempt < max_retries:
            sleep_time = min(backoff * (2 ** attempt), 10.0)
            time.sleep(sleep_time)
            continue
```

### Session Affinity (Critical)

Games are **stateful** and require **session affinity**. The API uses cookies (especially `AWSALB*`) to maintain game state:

- Each `RESET` or `step()` call returns cookies
- **Must preserve and resend cookies** with subsequent requests to the same game session
- Failing to maintain session → API returns 400 error

The SDK handles this automatically. Our wrapper relies on `arc_agi` to manage sessions.

---

## 9. Reference Agent Strategies

### Python SDK Agents (from `arcprize/ARC-AGI-3-Agents`)

**Pattern 1: Random Baseline**
```python
def is_done(self, frames, latest_frame) -> bool:
    return latest_frame.state is GameState.WIN

def choose_action(self, frames, latest_frame) -> GameAction:
    if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
        return GameAction.RESET
    else:
        return random.choice([a for a in GameAction if a is not GameAction.RESET])
```

**Pattern 2: LLM Agent (gpt-4o-mini)**
- Observes frame (grid as ASCII text)
- Sends to LLM with tools: RESET, ACTION1-6
- LLM uses function calling to pick action
- Maintains 10-message history for context

**Pattern 3: Reasoning Agent (o4-mini)**
- Generates PIL images from grids with color mapping
- Sends both previous and current images to LLM
- Uses structured output (Pydantic model) with hypothesis tracking
- Maintains `screen_history` and aggregated findings
- **MAX_ACTIONS:** 400 (more generous than baseline 80)

**Pattern 4: MultiModal Agent**
- Converts grids to 128x128 RGBA images
- Three-phase per turn: analyze, choose, convert
- **Self-modifying memory:** Agent's analysis can update its own prompt for future turns
- Image diffs highlight changes between frames

### Claude Code SDK Approach (from `ThariqS/ARC-AGI-3-ClaudeCode-SDK`)

- **Key idea:** Claude Code gets Node.js CLI scripts as "tools"
- Claude can write custom analysis scripts during gameplay
- Maintains notes on disk to track discoveries
- More flexible than fixed function-calling agents

---

## 10. Scoring Methodology (RHAE)

### Why It Matters

Agents are scored on **action efficiency**, not just completion. Wasting moves = lower score.

### Scoring Formula

```
level_score = (human_baseline_actions / ai_actions)^2
```

- Capped at 1.0 (100%)
- Squaring penalizes inefficiency heavily
- Example: If human took 10 actions and AI took 20:
  - Score = (10/20)^2 = 0.25 = 25%

### Game Aggregation

```
game_score = SUM(level_score * level_number) / SUM(level_numbers)
```

Later (harder) levels are weighted more heavily than introductory ones.

**Example:** If a game has 6 levels:
- Level 1: weight 1
- Level 2: weight 2
- Level 3: weight 3
- Level 4: weight 4
- Level 5: weight 5
- Level 6: weight 6
- Total weight: 21

If you score 100% on levels 1-3 but 0% on levels 4-6:
- Game score = (100*1 + 100*2 + 100*3 + 0*4 + 0*5 + 0*6) / 21 = 600/21 ≈ 28.6%

### Overall Score

Average of all game scores, producing a 0-100% range.

---

## 11. Bug Fixes & Error Handling

### HTTP 400 Errors (Game Session Expiry)

**Problem:** Game session expires mid-rollout; code continued sending actions, cascading errors.

**Root cause:** Server reaches WIN/LOSS/timeout, but agent's step() call caught the 400 error, kept previous frame, returned `done=False`. Agent kept rolling out actions to a dead session.

**Fix Applied** (in `src/environment/arc_env_wrapper.py`):
```python
step_error_done = False
# ... attempt step ...
except Exception as e:
    if "400" in err_str:
        step_error_done = True  # Treat as terminal
        
done = step_error_done or "WIN" in state or self.action_count >= self.max_actions
```

Now 400 errors immediately terminate the episode.

### HTTP 429 Handling

Also treats 429 (rate limit) as terminal after max retries:
```python
if "429" in err_str and attempt < max_retries:
    # Retry with backoff
else:
    step_error_done = True  # Terminal after retries exhausted
```

### Verified Results

- **train.py** (rollout=50): Zero 400 errors across all game types ✓
- **self_improve.py** (before fix, rollout=200): Many 400 errors ✗
- **self_improve.py** (after fix, rollout=50): Zero 400 errors ✓

---

## 12. Key Data Structures

### FrameDataRaw (from `arcengine`)

```python
@dataclass
class FrameDataRaw:
    game_id: str
    state: GameState
    levels_completed: int
    win_levels: int
    action_input: ActionInput
    guid: Optional[str]
    full_reset: bool
    available_actions: list[int]
    
    @property
    def frame(self) -> List[numpy.ndarray]:  # Private attribute
        # List of numpy arrays, each (64, 64), dtype=int8, values [0-15]
```

### GameAction Enum

```python
class GameAction(Enum):
    RESET = 0
    ACTION1 = 1  # Up
    ACTION2 = 2  # Down
    ACTION3 = 3  # Left
    ACTION4 = 4  # Right
    ACTION5 = 5  # Interact/Enter/Delete
    ACTION6 = 6  # Click (requires x, y)
    ACTION7 = 7  # Undo
```

### GameState Enum

```python
class GameState(Enum):
    NOT_PLAYED = 'NOT_PLAYED'
    NOT_FINISHED = 'NOT_FINISHED'
    WIN = 'WIN'
    GAME_OVER = 'GAME_OVER'
```

---

## 13. Strategic Insights for Winning

### 1. Level Progression is Sequential

You cannot skip levels. Beating level 1 automatically loads level 2 (often with `full_reset: true` flag). Focus on understanding one level at a time.

### 2. Action Efficiency Matters

Agents are scored on action count, not just completion. Random exploration is inefficient. Learning the pattern before acting is key.

### 3. Multi-Modal Learning is Powerful

Reference agents using vision + LLM reasoning (e.g., ReasoningAgent) outperform text-only agents because:
- Images reveal spatial patterns text doesn't capture
- Color is information (16-color palette is intentional)
- Diff images show what changed (what your action did)

### 4. Hypothesis Tracking Accelerates Learning

The best agents (ReasoningAgent, MultiModalLLM) maintain explicit hypotheses:
- "The red square is the goal"
- "Clicking it advances to the next level"
- Update hypothesis based on outcome
- Use hypothesis to guide next action

### 5. Self-Modifying Memory Enables Adaptation

MultiModalLLM agents can update their own prompt during gameplay. This allows:
- Learning game-specific tricks
- Avoiding repeated mistakes
- Adapting strategy per game

### 6. Keyboard Games are Easier

Training results show:
- Keyboard games (movement-based) learn fastest: -0.85 → +0.5 (Δ+1.35)
- Click games (large action space) learn slowest: -2.35 → -0.075 (Δ+2.275)
- Hybrid games (both): -0.95 → +0.2 (Δ+1.15)

**Implication:** Start with keyboard games. The 64×64 coordinate space for click games makes them fundamentally harder.

---

## 14. Implementation Checklist for Winning

### Minimum Requirements

- [ ] Check `frame.state == GameState.WIN` after each step
- [ ] Track `frame.levels_completed` vs `frame.win_levels`
- [ ] Respect `frame.available_actions` (never issue invalid actions)
- [ ] Handle game tags to restrict action space
- [ ] Implement exponential backoff for 429 errors
- [ ] Treat 400 errors as episode termination
- [ ] Add 0.05s delay between steps to stay under 600 RPM

### Performance Optimizations

- [ ] Use multi-frame animation (handle `frame[-1]` not `frame[0]`)
- [ ] Implement grid change detection (reward shaping signal)
- [ ] Use reward signal to train (level completion bonus)
- [ ] Track `full_reset` flag for level transitions
- [ ] Maintain action history to avoid repeated moves

### Advanced Strategies

- [ ] Convert grids to images for vision model
- [ ] Maintain hypothesis about game rules
- [ ] Implement frame history (last N frames)
- [ ] Use diff images to highlight changes
- [ ] Implement self-modifying memory (LLM updates its own prompt)

---

## 15. Common Pitfalls to Avoid

### Pitfall 1: Ignoring Level Boundaries
Agents may not realize when one level ends and the next begins. Watch for `full_reset: true` to detect transitions.

### Pitfall 2: Infinite Loops
If an agent takes 200+ actions without progress, it's likely stuck. Implement early termination or force RESET.

### Pitfall 3: Invalid Actions
Sending ACTION1 (up) to a click-only game causes 400 errors. Always check `frame.available_actions` or game tags first.

### Pitfall 4: Ignoring Rate Limits
Spamming API requests without delays triggers 429. Current code adds 0.05s delay, but consider adaptive backoff for heavy workloads.

### Pitfall 5: Not Handling Session Loss
Game sessions expire if left idle or after reaching WIN/GAME_OVER. Continuing to act causes cascading 400s.

### Pitfall 6: Using `frame[0]` Instead of `frame[-1]`
For animated games, the final frame is the true state after the action. Current code works in practice but is architecturally wrong.

---

## 16. Resources & References

### Official Documentation
- **Docs:** https://docs.arcprize.org/
- **API Base:** https://three.arcprize.org
- **OpenAPI Spec:** https://docs.arcprize.org/arc3v1.yaml
- **Game Browser:** https://arcprize.org/tasks
- **Scorecard Viewer:** https://arcprize.org/scorecards

### Official Reference Agents
- **Python SDK:** https://github.com/arcprize/arc-agi
- **Agent Framework:** https://github.com/arcprize/ARC-AGI-3-Agents
- **Claude Code SDK:** https://github.com/ThariqS/ARC-AGI-3-ClaudeCode-SDK

### Our Project
- **Env Wrapper:** `/src/environment/arc_env_wrapper.py`
- **Reward Shaper:** `/src/training/reward_shaper.py`
- **State Processor:** `/src/environment/state_processor.py`
- **Previous Investigations:** `/docs/investigations/`

---

## Summary: The Winning Formula

1. **Detect WIN:** When `frame.state == GameState.WIN`, you've won the game.
2. **Complete All Levels:** Must reach `levels_completed >= win_levels`.
3. **Respect Constraints:** Only use actions in `frame.available_actions`.
4. **Minimize Actions:** Score is based on action efficiency relative to human baseline.
5. **Learn Patterns:** Use vision, hypothesis tracking, and self-modification to discover rules.
6. **Handle Errors:** Treat 400/429 as terminal; implement exponential backoff.
7. **Stay Under Rate Limits:** Add 0.05s+ delay between API calls.

**The games are solvable.** Training shows positive learning signals across all game types. Focus on efficient exploration and pattern recognition.

