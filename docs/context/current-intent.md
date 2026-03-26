# Current Intent

## What We're Building
An RL agent that plays ARC-AGI-3 interactive grid games and recursively improves via PPO-based reinforcement learning.

## Architecture
- **CNN Encoder**: 4-block conv net, 16ch one-hot input → 256-dim embedding
- **Policy Network**: Encoder + aux features → 3-head action decoder (type, x, y)
- **Value Network**: Encoder + aux features → scalar value estimate
- **PPO Trainer**: Clipped objective + GAE + entropy bonus
- **Self-Improvement Loop**: Train → Evaluate → Analyze → Adapt hyperparameters

## Current Status
- **M1 Foundation**: ✅ Complete — all components implemented, 73 tests passing
- **M2 Real Games**: ✅ Complete — API verified, training on real games works
- **M3 Training Quality**: 🔜 Next — need action masking, extended training, first WIN
- API Key: configured in .env
- 25 real games available via ARC-AGI-3 API
- Best self-improvement score: 1.9 (but no actual level wins yet)

## Key Files
- `src/config.py` — All hyperparameters
- `src/environment/arc_env_wrapper.py` — SDK wrapper
- `src/training/trainer.py` — PPO implementation
- `src/training/self_improver.py` — Recursive improvement loop
- `scripts/` — Entry points (train, evaluate, play_game, self_improve)
