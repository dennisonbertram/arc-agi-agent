# ARC-AGI-3 RL Agent — Active Plan

## Milestone 1: Foundation ✅ COMPLETE
- [x] Project scaffolding, config, models
- [x] CNN encoder, policy/value networks, action heads
- [x] PPO trainer with GAE
- [x] Replay buffer, reward shaper
- [x] Environment wrapper (arc-agi SDK)
- [x] State processor, grid visualization
- [x] Entry scripts (train, evaluate, play, self-improve)
- [x] 73 unit tests passing
- [x] Mock training verified (3 iterations)

## Milestone 2: Real Game Integration ✅ COMPLETE
- [x] API key configured and verified
- [x] SDK interface fixes (frame indexing, GameAction enum, singleton mutation)
- [x] 25 games discovered (7 click, 5 keyboard, 13 keyboard_click)
- [x] Integration test passing against real API
- [x] Training on real games verified (10 steps on tn36-ab4f63cc)
- [x] Self-improvement loop on real games (3 iterations, best score 1.9)
- [x] Evaluation pipeline working

## Milestone 3: Training Quality Improvements 🔜 NEXT
- [ ] Per-game action filtering (mask invalid actions by game tag)
- [ ] Add --mode argument to self_improve.py
- [ ] Extended training runs (100+ steps)
- [ ] Multi-game training curriculum
- [ ] Hyperparameter tuning based on real game feedback
- [ ] Achieve first actual level WIN

## Milestone 4: Advanced Features 🔮 FUTURE
- [ ] Attention-based encoder (replace/augment CNN)
- [ ] Curriculum learning (easy→hard game progression)
- [ ] Experience replay across games
- [ ] Model distillation / architecture search
- [ ] Kaggle submission preparation
