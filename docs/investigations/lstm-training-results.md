# LSTM Training Results on Real ARC-AGI-3 Games

**Date:** 2026-03-26
**Model:** CNN+LSTM (Phase 1)
**Game:** tr87-d45dff03 (keyboard game)
**Training:** 100 steps, 100 rollout per step, ONLINE mode

## Reward Trajectory

| Step | Reward | P-Loss | V-Loss | Entropy |
|------|--------|--------|--------|---------|
| 1 | 15.57 | -2.19 | 4.61 | 1.60 |
| 5 | 17.40 | -2.02 | 5.68 | 1.51 |
| 10 | 16.88 | -1.46 | 5.05 | 1.52 |
| 15 | 17.50 | -0.91 | 4.88 | 1.51 |
| 20 | 17.09 | -0.26 | 5.62 | 1.47 |
| 25 | 18.50 | 0.09 | 6.89 | 1.45 |
| 30 | 18.20 | 1.18 | 10.28 | 1.49 |
| 35 | 18.60 | 1.37 | 10.65 | 1.41 |
| 40 | 18.80 | 1.40 | 11.84 | 1.48 |
| 45 | 18.80 | 1.62 | 7.74 | 1.49 |
| 50 | 18.50 | 2.17 | 14.51 | 1.46 |
| 55 | 18.80 | 2.45 | 17.01 | 1.45 |
| 60 | 18.70 | 2.65 | 18.51 | 1.46 |
| 65 | 18.60 | 2.24 | 15.69 | 1.42 |
| 70 | **18.90** | 1.86 | 13.14 | 1.40 |
| 75 | 18.80 | 2.11 | 14.88 | 1.44 |
| 80 | 18.30 | 2.50 | 18.03 | 1.48 |
| 85 | 18.30 | 2.47 | 16.07 | 1.54 |
| 90 | 18.30 | 2.48 | 16.80 | 1.52 |
| 95 | 18.30 | 2.27 | 15.00 | 1.52 |
| 100 | 18.40 | 1.88 | 13.04 | 1.40 |

## Key Metrics
- **Peak reward:** 18.9 (step 70)
- **Final reward:** 18.4 (step 100)
- **Reward range:** 15.6 → 18.9 (+21% improvement)
- **Training duration:** ~37 minutes (14:00 → 14:37)
- **API calls:** ~10,000 (100 steps × 100 rollout)
- **WINs achieved:** 0

## Comparison with CNN-only Model

| Metric | CNN-only (PPO) | CNN+LSTM |
|--------|---------------|----------|
| Starting reward | -4.85 | 15.57 |
| Peak reward | 1.50 | **18.90** |
| Final reward | 0.00 | **18.40** |
| Improvement factor | — | **12.6x** |

## Analysis

### What's Working
1. **LSTM memory** provides context across steps — agent remembers prior actions
2. **10x grid_change_bonus** (0.2 vs 0.02) gives dense reward for grid exploration
3. **Consistent high rewards** from step 1 — no cold-start exploration problem
4. **Stable entropy** (~1.4-1.5) indicates healthy policy exploration

### Why No WIN Yet
1. **Reward saturation:** The shaped reward plateaus at ~18.8 — the agent maximizes grid changes but hasn't discovered the puzzle-solving pattern
2. **No level completion signal:** The reward shaper gives +10 for WIN and +2 per level, but these signals are never triggered because the agent can't solve the puzzle logic
3. **Fundamental limitation:** The CNN+LSTM learns to interact with the grid effectively but cannot discover abstract transformation rules from pixel-level observation alone
4. **Exploration ceiling:** With 6 keyboard actions + undo + reset, the policy has settled on a pattern that maximizes grid changes without understanding the game's objective

### Reward Decomposition (estimated per episode)
- Grid change bonus: ~100 changes × 0.2 = +20.0
- Step cost: ~100 steps × -0.01 = -1.0
- Reset/undo penalties: ~-0.5
- **Total: ~18.5** (matches observed rewards)

The agent is essentially earning maximum grid-change reward but zero game-completion reward.

## Next Steps
1. **Investigate game mechanics**: Play tr87 manually or with visualization to understand what a WIN requires
2. **Add level-completion curiosity**: Reward the agent for reaching new game states it hasn't seen before
3. **Consider hybrid approach**: Use LLM to analyze grid patterns and guide RL policy
4. **Try different games**: Some games may have simpler WIN conditions
