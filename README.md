# ARC-AGI-3 RL Agent

A reinforcement learning agent designed to solve ARC-AGI-3 tasks using Proximal Policy Optimization (PPO) with recursive self-improvement loops.

## Overview

ARC-AGI-3 (Abstraction and Reasoning Corpus) presents a benchmark of tasks that require general intelligence — each task is a unique visual puzzle requiring novel reasoning. This project builds a PPO-based RL agent that:

1. Plays ARC-AGI-3 tasks via the `arc-agi` SDK
2. Learns from rewards using PPO with a CNN-based grid encoder
3. Recursively self-improves by re-training on its own trajectory data
4. Tracks performance metrics with TensorBoard

## Architecture

```
arc-agi-agent/
├── src/
│   ├── config.py              # Centralized configuration via pydantic-settings
│   ├── agent/
│   │   ├── base_agent.py      # Abstract base class for all agents
│   │   ├── random_agent.py    # Baseline random agent
│   │   └── rl_agent.py        # PPO-based RL agent
│   ├── models/
│   │   ├── encoder.py         # CNN grid encoder
│   │   ├── policy_net.py      # Policy network (actor)
│   │   ├── value_net.py       # Value network (critic)
│   │   └── action_head.py     # Action output head
│   ├── training/
│   │   ├── trainer.py         # PPO training loop
│   │   ├── replay_buffer.py   # Trajectory buffer
│   │   ├── reward_shaper.py   # Reward shaping logic
│   │   └── self_improver.py   # Recursive self-improvement loop
│   ├── environment/
│   │   ├── arc_env_wrapper.py # Wraps arc-agi SDK as RL environment
│   │   └── state_processor.py # Converts game frames to tensors
│   ├── evaluation/
│   │   ├── evaluator.py       # Runs evaluation episodes
│   │   └── metrics.py         # Computes and tracks metrics
│   └── utils/
│       ├── grid_viz.py        # Grid visualization utilities
│       ├── logger.py          # Structured logging
│       └── trajectory.py      # Trajectory serialization
├── scripts/
│   ├── train.py               # Main training entry point
│   ├── evaluate.py            # Evaluation entry point
│   ├── play_game.py           # Play a single game interactively
│   └── self_improve.py        # Run self-improvement loop
└── tests/                     # Test suite
```

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

```bash
# Clone the repo
git clone <repo-url>
cd arc-agi-agent

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv sync

# Copy and configure environment variables
cp .env.example .env
# Edit .env and add your ARC API key
```

### Configuration

All configuration is managed through `src/config.py` using `pydantic-settings`. You can override any setting via environment variables or the `.env` file.

Key settings:

| Setting | Default | Description |
|---|---|---|
| `ARC_API_KEY` | `""` | Your ARC Prize API key |
| `OPERATION_MODE` | `"OFFLINE"` | `"OFFLINE"` or `"ONLINE"` |
| `GRID_SIZE` | `64` | Grid dimensions (NxN) |
| `LEARNING_RATE` | `3e-4` | PPO learning rate |
| `MAX_ACTIONS_PER_GAME` | `200` | Max steps per episode |
| `IMPROVEMENT_ITERATIONS` | `10` | Self-improvement loop iterations |

## Usage

### Train the agent

```bash
# Basic training run
python scripts/train.py

# With custom settings
python scripts/train.py --games 50 --iterations 20 --checkpoint checkpoints/run1
```

### Evaluate a checkpoint

```bash
python scripts/evaluate.py --checkpoint checkpoints/latest.pt --games 10
```

### Play a single game

```bash
python scripts/play_game.py --task-id <task-id>
```

### Run self-improvement loop

```bash
python scripts/self_improve.py --checkpoint checkpoints/base.pt --iterations 5
```

## Development

### Running tests

```bash
pytest
```

### Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

### TensorBoard

```bash
tensorboard --logdir logs/
```

## Operation Modes

- **OFFLINE**: Uses local ARC task data for training and evaluation. No API key required.
- **ONLINE**: Connects to `https://three.arcprize.org` to play live games. Requires `ARC_API_KEY`.

## Self-Improvement Loop

The recursive self-improvement loop (`SelfImprover`) works as follows:

1. Play N games with the current policy, recording trajectories
2. Identify high-reward trajectories as "positive examples"
3. Re-train the policy on these trajectories via PPO
4. Evaluate the updated policy on held-out tasks
5. If improvement exceeds threshold, keep the update; otherwise revert
6. Repeat for M iterations

This allows the agent to iteratively refine its strategy without requiring new labeled data.

## License

MIT
