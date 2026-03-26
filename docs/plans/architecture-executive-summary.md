# ARC-AGI-3 Architecture Planning: Executive Summary

**Date:** 2026-03-26
**Source:** 4-turn planning session with `gpt-4.1-2025-04-14`
**Full plan:** `docs/plans/specialized-architecture-plan.md`
**Raw JSON:** `docs/plans/gpt-architecture-plan.json`

---

## Key Finding: CNN+PPO Will Not Work

The model's central conclusion: pure RL with a CNN encoder is a poor fit for ARC-AGI-3 because:

1. ARC games require symbolic reasoning and transformation detection, not pixel statistics
2. The action space (8 types × 64×64 coordinates) is too sparse for random exploration to find wins
3. There is no memory — each step is independent, so the agent cannot track what it tried or what changed
4. Sparse reward (only +10 at WIN) gives almost no signal; winning is astronomically unlikely via exploration

**The fastest path to a first WIN is NOT the most sophisticated architecture.** It is a CNN+LSTM with aggressive reward shaping and a focused curriculum on the single easiest game.

---

## Phase 1: Quick Wins (target: first WIN in 1–3 days)

**Principle:** Minimum viable changes. Get a WIN. Everything else is secondary.

### Changes Required

**New file: `src/models/policy_lstm.py`**
```python
import torch
import torch.nn as nn
from src.models.encoder import GridEncoder

class CNNLSTMPolicy(nn.Module):
    def __init__(self, num_colors=16, aux_dim=15, embedding_dim=128, lstm_hidden=128, num_actions=8):
        super().__init__()
        self.encoder = GridEncoder(num_colors, embedding_dim)
        self.aux_fc = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU())
        self.lstm = nn.LSTM(embedding_dim + 32, lstm_hidden, batch_first=True)
        self.policy_head = nn.Linear(lstm_hidden, num_actions)
        self.value_head = nn.Linear(lstm_hidden, 1)

    def forward(self, grid, aux, lstm_state=None):
        # grid: [B, 16, 64, 64], aux: [B, 15]
        emb = self.encoder(grid)          # [B, embedding_dim]
        aux_emb = self.aux_fc(aux)        # [B, 32]
        x = torch.cat([emb, aux_emb], dim=-1).unsqueeze(1)  # [B, 1, D]
        out, lstm_state = self.lstm(x, lstm_state)           # [B, 1, H]
        h = out.squeeze(1)
        return self.policy_head(h), self.value_head(h).squeeze(-1), lstm_state
```

**Modify `src/training/trainer.py`:**
- Track and reset LSTM hidden state at episode boundaries
- Pass `lstm_state` into policy during rollout

**Modify `src/training/reward_shaper.py`:**
- Set `grid_change_bonus = 0.2` (up from 0.02 — 10x increase)
- Keep all other rewards the same

**Curriculum: train on ONE game only**
- Pick the easiest click game (e.g., `tn36-ab4f63cc`)
- Do not rotate games until first WIN is confirmed

### Phase 1 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | `3e-4` |
| LSTM hidden size | `128` |
| PPO clip epsilon | `0.2` |
| PPO epochs per update | `4` |
| Max steps per episode | `200` |
| Grid change bonus | `0.2` |

### Why This Works
LSTM gives the agent memory to recognize when the same action was tried before and failed. The 10x larger grid-change bonus provides much denser reward signal, guiding the agent toward actions that transform the grid even before wins occur. Single-game focus maximizes experience on one solvable task.

---

## Phase 2: Specialized Architecture (target: multi-game generalization, 1–2 weeks)

**Prerequisite: Phase 1 must have produced at least one WIN.**

### New Files to Create

**`src/models/transformer_grid_encoder.py`** — ViT-based grid encoder
- 8×8 patches → 64 tokens of dim 128
- 4-layer TransformerEncoder, 8 heads
- 2D learned positional embeddings
- Output: `[B, 64, 128]` patch token sequence

**`src/models/temporal_memory.py`** — Temporal transformer over step history
- Maintains history of last N=16 steps: `[B, T, 512]`
- 2-layer TransformerEncoder, 8 heads
- Output: `[B, T, 512]`; use `[:, -1]` as memory summary

**`src/models/pointer_policy.py`** — Pointer-network action head
- Replaces independent x/y heads with attention over grid patches
- Action type head: MLP(mem_summary + level_emb) → 8 logits
- Pointer head: query(mem_summary) × keys(patch_emb) → 64 patch logits
- Map winning patch index to (x, y) pixel coordinates

**`src/models/arc_agent.py`** — Full agent wrapper connecting all modules

### Phase 2 Hyperparameters

| Parameter | Value |
|-----------|-------|
| Learning rate | `1e-4` |
| PPO epochs per update | `4` |
| Memory window (T) | `16` steps |
| Aux loss coefficient | `0.05` |
| Aux loss type | World model MSE |

### Architecture Summary Table

| Module | Input Shape | Output Shape |
|--------|-------------|--------------|
| GridEncoderViT | `[B, 16, 64, 64]` | `[B, 64, 128]` |
| Grid→Mem projection | `[B, 64*128]` | `[B, 512]` |
| TemporalMemoryTF | `[B, T, 512]` | `[B, T, 512]` |
| PointerPolicyHead | mem:`[B,512]` + level:`[B,32]` | `[B,8]`, `[B,64]` |
| ValueHead | `[B, 512+32]` | `[B, 1]` |

---

## Phase 3: Advanced Strategies (target: scaling, 2+ weeks)

**Prerequisite: Phase 2 working and demonstrably generalizing.**

### New Files

- **`src/models/world_model.py`** — Predicts next patch embeddings from (grid_embed, action, aux). Use MSE loss as auxiliary signal and optionally as curiosity bonus.
- **`src/llm/llm_guidance.py`** — Offline LLM analysis of grids to extract features (symmetry, object masks, color mappings). Feed as additional aux inputs to the RL agent.
- **`src/training/model_based.py`** — Dyna-style rollouts: use world model to generate synthetic transitions between real env steps (2-5x sample efficiency boost).
- **`src/training/curriculum.py`** — Track per-game win rates; schedule games with lower win rates more frequently.

### Phase 3 Hyperparameters

| Parameter | Value |
|-----------|-------|
| World model loss coef | `0.01` |
| Real:synthetic rollout ratio | `1:1` |
| Curriculum: unsolved game weight | `2x` vs solved |

---

## Critical Pitfalls to Avoid

1. **LSTM state alignment**: Reset hidden state at episode boundaries only. Never carry state between episodes.
2. **Pointer → pixel mapping**: Pointer head outputs patch index (0–63 for 8x8 grid of 8×8 patches). Convert: `x = (patch_idx % 8) * 8 + 4`, `y = (patch_idx // 8) * 8 + 4`.
3. **Aux loss instability**: World model MSE can dominate if coef is too high. Start at `0.05` and reduce if policy loss degrades.
4. **Reward saturation**: Curiosity rewards must be normalized/clipped. True task reward must remain the dominant signal.
5. **Batch dimension errors**: Use `batch_first=True` everywhere in `nn.TransformerEncoderLayer`. Add shape assertions in `forward()` during development.

---

## Honest Assessment from GPT

> "Your best chance at a first WIN is a well-tuned CNN+LSTM baseline, with heavy reward shaping and focused curriculum. The fancy transformer/slot/world model approach is better in the long run—but unlikely to reach a WIN faster under ARC-AGI-3 constraints unless you already have a large offline dataset. Once you get a WIN, invest in more complex models and hybrid approaches for scaling to harder games."

**Comparison table from GPT (approximate API calls to first WIN):**

| Approach | API Calls to WIN | Feasibility |
|----------|------------------|-------------|
| ViT + Transformer (from scratch) | 10^6+ | Low-Med |
| CNN + LSTM (Phase 1) | 10^4–10^5 | High |
| LLM-only hybrid | 10^3–10^4 | Medium |
| RL + LLM features | <10^4 | High (if allowed) |

At 600 RPM, 10^4 calls = ~17 minutes. 10^5 calls = ~2.8 hours. Feasible.

---

## Recommended Immediate Next Step

Implement `src/models/policy_lstm.py` with the `CNNLSTMPolicy` class shown above, update `trainer.py` to track LSTM state per episode, bump `grid_change_bonus` to `0.2` in `reward_shaper.py`, and run a single-game training session on the simplest available click game with `max_actions=200` and at least 50 rollout episodes.
