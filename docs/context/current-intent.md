# Current Intent

## Project Goal

Build a reinforcement learning agent that can solve ARC-AGI-3 tasks using PPO with a recursive self-improvement loop.

ARC-AGI-3 (Abstraction and Reasoning Corpus, version 3) is a benchmark of visual puzzle tasks that test general fluid intelligence. Each task presents a small set of input/output grid examples; the agent must infer the transformation rule and apply it to a novel test input.

## Why RL?

Traditional supervised learning on ARC requires explicit access to the correct output at training time. RL allows the agent to explore action sequences and receive reward signals based on how close its output is to the target — enabling learning without step-by-step supervision.

## Core Components (in build order)

1. **Environment wrapper** (`src/environment/`) — wraps the `arc-agi` SDK so the PPO trainer can call `reset()` / `step()` / `render()`.
2. **State processor** (`src/environment/state_processor.py`) — converts raw ARC frames (grids) into tensors the neural network can consume.
3. **Encoder** (`src/models/encoder.py`) — CNN that maps a one-hot encoded grid to a dense embedding.
4. **Policy + Value networks** (`src/models/`) — standard actor-critic architecture on top of the encoder.
5. **Replay buffer** (`src/training/replay_buffer.py`) — stores transitions for PPO updates.
6. **Reward shaper** (`src/training/reward_shaper.py`) — adds dense intermediate rewards (pixel accuracy delta) to sparse terminal signals.
7. **PPO Trainer** (`src/training/trainer.py`) — the main training loop: collect rollouts → compute GAE → PPO update → checkpoint.
8. **RL Agent** (`src/agent/rl_agent.py`) — integrates encoder + policy net into the `BaseAgent` interface.
9. **Evaluator** (`src/evaluation/`) — periodic unbiased performance measurement on held-out tasks.
10. **Self-Improver** (`src/training/self_improver.py`) — outer loop that fine-tunes on the agent's own high-reward trajectories.

## Current Status (as of scaffolding)

- All modules are scaffolded with class signatures, docstrings, and `NotImplementedError` stubs.
- The `RandomAgent` and `BaseAgent` are fully implemented — useful for smoke-testing the environment integration before any ML code exists.
- The `Config` class is fully implemented via `pydantic-settings`.
- All other modules need implementation in the order listed above.

## Next Immediate Steps

1. Implement `StateProcessor.grid_to_tensor` and `StateProcessor.process`.
2. Implement `ArcEnvWrapper.reset`, `step`, `close`.
3. Implement `GridEncoder.build` and `forward`.
4. Implement `PolicyNetwork` and `ValueNetwork`.
5. Wire up `RLAgent.choose_action` using the encoder + policy net.
6. Implement `ReplayBuffer` CRUD methods.
7. Implement `PPOTrainer.collect_rollout`, `compute_advantages`, `update`, `train`.
8. Implement `Evaluator.evaluate`.
9. Implement `SelfImprover.run`.

## Design Decisions

- **Offline-first**: All development and testing uses local ARC task data (no API key required). Online mode is added later.
- **Modular**: Each component has a clean interface so it can be swapped or tested independently.
- **Config-driven**: All hyperparameters live in `src/config.py` and can be overridden via `.env` or environment variables.
- **TDD-friendly**: Stubs raise `NotImplementedError` so tests can verify the interface before implementation.
