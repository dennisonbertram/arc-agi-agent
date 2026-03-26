# Real Game Training Results — ARC-AGI-3

**Date:** 2026-03-26

## Available Games (25 total)

| Game ID | Tags |
|---------|------|
| sp80-0ee2d095 | keyboard_click |
| tn36-ab4f63cc | click |
| ft09-0d8bbf25 | (none) |
| ar25-e3c63847 | keyboard_click |
| tu93-2b534c15 | keyboard_click |
| re86-4e57566e | keyboard_click |
| vc33-9851e02b | click |
| sb26-7fbdac44 | keyboard_click |
| dc22-4c9bff3e | keyboard_click |
| su15-4c352900 | click |
| sc25-f9b21a2f | keyboard_click |
| cd82-fb555c5d | keyboard_click |
| tr87-cd924810 | keyboard |
| g50t-5849a774 | keyboard |
| bp35-0a0ad940 | keyboard_click |
| ls20-9607627b | keyboard |
| wa30-ee6fef47 | keyboard |
| cn04-65d47d14 | keyboard_click |
| lf52-271a04aa | click |
| sk48-41055498 | keyboard_click |
| lp85-305b61c3 | click |
| m0r0-dadda488 | keyboard_click |
| ka59-9f096b4a | keyboard_click |
| r11l-aa269680 | click |
| s5i5-a48e4b1d | click |

## Training Run: tn36-ab4f63cc (10 Steps, 100 Rollout)

Command: `uv run python scripts/train.py --mode ONLINE --steps 10 --rollout 100 --game tn36-ab4f63cc`

| Step | Reward | P-Loss | V-Loss | Entropy |
|------|--------|--------|--------|---------|
| 1/10 | -4.850 | 0.6565 | 0.1881 | 4.4691 |
| 5/10 | -1.950 | -0.0918 | 0.1047 | 6.7256 |
| 10/10 | 0.000 | -0.0465 | 0.0030 | 7.3132 |

**Observations:**
- Reward improved from -4.85 → 0.0 over 10 steps (monotonic improvement)
- Policy loss went negative (policy collapsed toward random/uniform), entropy increased from 4.47 → 7.31 (approaching log(8)=2.08 nats for 8-action space)
- Value loss decreased significantly (0.1881 → 0.0030), model better calibrated its value estimates
- Checkpoint saved to `checkpoints/latest.pt`

## Self-Improvement Loop (3 Iterations)

Command: `uv run python scripts/self_improve.py --iterations 3 --train-steps 5 --games tn36-ab4f63cc vc33-9851e02b su15-4c352900`

**Note:** `self_improve.py` does not accept `--mode` argument; uses OFFLINE mode internally for game scanning (hence "Game X not found in scanned environments" errors). However training and evaluation still executed via API using the game IDs directly.

| Iteration | Score | New Best? | Duration |
|-----------|-------|-----------|----------|
| 1 | 1.5000 | Yes | 109.7s |
| 2 | 1.9000 | Yes | 115.3s |
| 3 | 1.7000 | No (1 consecutive no-improve) | 110.7s |

**Trajectory:** [1.5, 1.9, 1.7]

**Best score: 1.9000 at iteration 2**

Saved checkpoints: `iter_0001.pt`, `iter_0002.pt`, `iter_0003.pt`, `best.pt`, `final.pt`

## Evaluation Results

### latest.pt on tn36-ab4f63cc

Command: `uv run python scripts/evaluate.py --mode ONLINE --checkpoint checkpoints/latest.pt --games tn36-ab4f63cc`

| Game | Reward | Actions | Final State | Wins |
|------|--------|---------|-------------|------|
| tn36-ab4f63cc | 1.50 | 200 | NOT_FINISHED | 0 |

**Summary:** 0/1 wins, mean reward: 1.500

### best.pt on 3 games

Command: `uv run python scripts/evaluate.py --mode ONLINE --checkpoint checkpoints/best.pt --games tn36-ab4f63cc vc33-9851e02b su15-4c352900`

| Game | Reward | Actions | Final State | Wins |
|------|--------|---------|-------------|------|
| tn36-ab4f63cc | 1.50 | 200 | NOT_FINISHED | 0 |
| vc33-9851e02b | -6.00 | 200 | NOT_FINISHED | 0 |
| su15-4c352900 | -12.20 | 200 | NOT_FINISHED | 0 |

**Summary:** 0/3 wins, mean reward: -7.713

## Issues Encountered

### 1. Evaluation Crash: NoneType AttributeError
- **Error:** `AttributeError: 'NoneType' object has no attribute 'frame'`
- **Root cause:** When the API returns 400 for an invalid action (e.g., `ACTION6` with invalid coordinates), the step raised an exception, leaving `current_frame = None`. The next call to `_obs()` then tried to access `None.frame`.
- **Fix applied:** Added `try/except` around `self._env.step()` in `arc_env_wrapper.py` to log warnings on API errors and keep the previous frame. Also added a guard in `_obs()` to substitute a blank `MockFrame` if `current_frame` is still `None`.

### 2. ACTION6/ACTION4 400 Errors
- The agent frequently attempts `ACTION6` (click) and `ACTION4` with coordinates outside the valid range for the game. The API returns 400 "Bad Request".
- After the fix, these are logged as warnings and the episode continues with a penalty (negative reward from unchanged grid).
- **Underlying issue:** The policy hasn't learned valid action/coordinate constraints for each game. The 400s for `vc33` and `su15` are more frequent, leading to large negative rewards.

### 3. self_improve.py Lacks --mode Argument
- The script accepts only `--games`, `--iterations`, `--train-steps`, `--checkpoint`, `--device`.
- It does NOT accept `--mode ONLINE`. Game scanning uses OFFLINE mode (no API), hence the "Game X not found" warning messages.
- Training still ran correctly against the API because `ArcEnvWrapper` creates its own `Arcade` connection internally.

## Key Invariants Discovered

1. **Game tags matter:** `click`-tag games require ACTION6 with valid x/y. `keyboard`-tag games use keyboard actions. `keyboard_click` games use both. The current policy does not gate actions by game type.
2. **400 errors are recoverable:** The API returns 400 for out-of-bounds or illegal actions but does not terminate the session. The agent can continue.
3. **self_improve.py uses OFFLINE game scanning:** The "Game not found" errors are cosmetic — training executes fine because game IDs are passed directly to the `Arcade.make()` call.
4. **Entropy inflation:** High entropy (7.3) after 10 steps suggests the policy is near-uniform, not converging to meaningful behavior. More steps and a properly shaped reward are needed.
5. **Checkpoint naming:** `best.pt` = best self-improvement iteration, `latest.pt` = last train.py checkpoint, `iter_N.pt` = per-iteration snapshots from self_improve.

## Recommendations for Next Training Run

1. **Filter actions by game tags** — only allow ACTION6 (click) for `click`/`keyboard_click` games, and only with valid coordinate ranges from the game metadata.
2. **Increase training steps** — 10 steps is insufficient; 100-500 steps needed to see meaningful policy convergence.
3. **Add `--mode` support to self_improve.py** — currently cannot force ONLINE mode for game scanning.
4. **Reward shaping for invalid actions** — penalize 400-error actions explicitly to teach the policy to avoid them.
5. **Use keyboard-only games first** — games like `tr87`, `g50t`, `ls20`, `wa30` (tags=['keyboard']) are simpler (no x/y coordinates required) and may produce cleaner learning signals.
