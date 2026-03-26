# ARC-AGI-3 Specialized Architecture Plan

*Generated via multi-turn planning with gpt-4.1-2025-04-14*

---

## Planning Prompt 1

[Code context provided from current codebase]

## Current Problems:
1. CNN sees raw pixels but ARC games require understanding spatial relationships, symmetry, and transformation rules
2. No memory - each step is independent, agent can't track what it tried or what changed
3. Blind exploration - 8 actions x 64x64 coordinates = astronomically unlikely to find solutions randomly
4. Games require completing 1-6 sequential levels, each potentially requiring different strategies

## What We Need:
Design a specialized model architecture that can actually WIN ARC-AGI-3 games. Consider:

1. **Grid understanding**: How should we encode the 64x64 grid to capture spatial patterns, symmetry, object boundaries?
2. **Memory**: How should the agent remember past actions and their effects?
3. **World model**: Should we predict grid outcomes before acting?
4. **Action selection**: How to make the huge action space tractable?
5. **Level generalization**: How to transfer learning between levels of the same game?
6. **Architecture specifics**: Exact layer types, dimensions, attention patterns, loss functions

Please provide a DETAILED architecture design with:
- Complete model architecture (layers, dimensions, connections)
- Training procedure
- Key innovations over standard CNN+PPO
- Why each component helps with ARC specifically
- Pseudocode for the forward pass

## GPT Response 1

Absolutely! ARC-AGI-3 is fundamentally about abstract reasoning, transformation, and sequential deduction—tasks that far outstrip what a vanilla CNN+PPO policy can discover via brute-force exploration. Let’s design a **specialized, memory-augmented, transformer-based world model agent** tailored for ARC-AGI-3.

---

# 1. High-Level Architecture Overview

**Key principles:**
- **Structured Perception:** Extract object-centric, relational, and symmetric features from the grid—not just raw pixels.
- **Memory & Planning:** Maintain a memory trace (action history, grid diffs, hidden state) to support multi-step reasoning and credit assignment.
- **Action Modeling:** Decompose the huge action space via pointer networks or spatial attention, reducing random search.
- **World Modeling:** Predict effects of actions to support curiosity, planning, and reward shaping.
- **Level Generalization:** Use meta-level embeddings per level, enabling transfer and adaptation.

**Major Modules:**
1. **Grid Encoder:** Patch-wise Vision Transformer (ViT) + Relational Attention + Object Extractor.
2. **Temporal Memory:** State-Action recurrent transformer (memory over N steps).
3. **World Model Head:** Predicts grid delta given action.
4. **Policy Head:** Pointer-based action decoder (spatial attention over grid).
5. **Value Head:** Level-conditioned value estimator.
6. **Aux Features:** Level-progress, previous reward, (optionally) game-specific context.

---

# 2. Detailed Model Architecture

## 2.1. Grid Encoder

**Motivation:** ARC is about spatial relationships, symmetry, and transformation. We need to detect objects, their relations, and possible invariants.

- **Patchify:** Split grid into 8x8 non-overlapping patches, so 64 patches (each 8x8x16).
- **Patch Embedding:** Linear layer projects each patch to 128d.
- **Patch Positional Encoding:** (Learned or sinusoidal) 2D encoding per patch.
- **ViT Encoder:** 4-layer Transformer encoder, 8 heads, dim 128.
- **Relational Head:** For each patch, compute cross-patch attention to model spatial relationships.
- **Object Extractor:** (Optional) Slot Attention layer (3-8 slots) to extract object-centric representations.
- **Output:** Grid embedding [batch, grid_embed_dim=512].

## 2.2. Temporal Memory

**Motivation:** Games are sequential, require tracking changes and strategies across steps and levels.

- Maintain a buffer of last N=16 steps: [(grid_embed), (prev action), (aux), (delta), ...]
- **Temporal Transformer:** 2-layer transformer, input sequence length N, dim 512, 8 heads.
- **Key inputs for each time step:**
  - Grid embedding (ViT output)
  - Previous action (embedded/categorical)
  - Aux features (level progress, etc.)
  - Grid delta embedding (difference from previous grid)
- **Memory integration:** At each step, attend over recent history (including current step) to produce a memory summary vector [batch, mem_dim=512].

## 2.3. World Model Head

**Motivation:** Model-based RL: learn to predict grid changes from actions, supports planning, curiosity, and denoising.

- **Input:** [grid_embed] + [action_emb] + [aux]
- **MLP:** 2 layers, 512→512→patches*128
- **Output:** Predicts next grid patch embeddings (regression or contrastive loss).
- **Usage:** During RL, use prediction loss as auxiliary signal; optionally use in a planning loop.

## 2.4. Policy Head (Pointer Network for Action Selection)

**Motivation:** Action space is huge (8 action types, 64x64 for coordinate actions). Use spatial attention/pointer mechanism.

- **Input:** [memory_summary] + (optionally) [object/slot embeddings]
- **Action Type Head:** MLP, outputs logits over 8 action types.
- **Pointer Head:** If action type requires coordinates, use a pointer network:
  - Attend over all grid patches (or slots), output distribution over spatial locations (patch or pixel).
  - For patch, output (patch_x, patch_y); for pixel, refine within patch via additional attention.
- **Output:** Action type, (x, y) if required, log-probabilities.

## 2.5. Value Head

- **Input:** [memory_summary] + [level meta-embedding]
- **Output:** Scalar value estimate.

## 2.6. Level Meta-Embedding

- For each level (1–6), learn or compute a meta-embedding (could be a learned vector or derived from grid).
- Concatenate to all heads to enable level-specific adaptation.

---

# 3. Training Procedure

## 3.1. PPO with Auxiliary Losses

- **Main loss:** PPO policy gradient loss.
- **Auxiliary:** World model prediction loss (delta grid embedding), contrastive object loss (if using slot attention), curiosity bonus (e.g., prediction error).
- **Reward shaping:** As before, but can augment with world model surprise or object count changes.
- **Curriculum:** Train on easier levels first, increase complexity.
- **Data augmentation:** Rotate/flip grids; randomize color mapping (forces agent to learn structure, not color idiosyncrasies).

## 3.2. Memory Handling

- At each environment step, update memory buffer with latest [grid_embed, action, aux, delta, ...].
- For the forward pass, use the last N steps as input to the temporal transformer.

## 3.3. Exploration

- Use intrinsic reward from world model prediction error to encourage novel transformations.
- Optionally, implement "object count change" curiosity.

---

# 4. Key Innovations vs. CNN+PPO

- **Structured Perception:** Patch-based ViT + relational attention captures global symmetries, object structure, and spatial relations far better than CNNs.
- **Explicit Memory:** Temporal transformer encodes history, critical for multi-level, sequential reasoning.
- **World Modeling:** Predicting grid deltas provides richer self-supervision and supports planning.
- **Pointer-based Action Selection:** Makes coordinate selection tractable and purposeful.
- **Level Conditioning:** Meta-embedding per level enables transfer and adaptation across levels of a game.
- **Auxiliary Losses:** World model and slot attention losses speed up learning and encode richer structure.

---

# 5. Pseudocode for the Model Forward Pass

```python
def forward(obs, memory_buffer, available_actions=None):
    """
    obs: dict with keys 'grid' [batch, 16, 64, 64], 'aux' [batch, aux_dim]
    memory_buffer: list of last N steps, each with grid_embed, action, aux, delta
    available_actions: [batch, 8] mask

    Returns:
        action_type_logits: [batch, 8]
        coord_logits: [batch, grid_size, grid_size] if needed
        value: [batch, 1]
        world_model_pred: [batch, num_patches, patch_dim]
    """
    # 1. Structured grid encoding
    grid_patches = patchify(obs['grid'])             # [batch, 64, 8, 8, 16]
    patch_emb = patch_linear(grid_patches)           # [batch, 64, 128]
    patch_emb += patch_pos_emb                       # Add positional encoding

    grid_tokens = ViT_encoder(patch_emb)             # [batch, 64, 128]
    relational_emb = relational_attention(grid_tokens) # [batch, 64, 128]
    grid_embed = flatten_and_project(relational_emb)  # [batch, grid_embed_dim=512]

    # 2. Memory buffer encoding
    mem_inputs = [concat(g, a, aux, d) for g, a, aux, d in memory_buffer]
    mem_inputs = stack(mem_inputs, dim=1)            # [batch, N, mem_input_dim]
    mem_summary = temporal_transformer(mem_inputs)    # [batch, N, 512]
    mem_summary = mem_summary[:, -1]                 # Take last step summary

    # 3. World Model Prediction
    pred_grid_delta = world_model_head(concat(grid_embed, prev_action_emb, obs['aux']))

    # 4. Action selection
    action_type_logits = action_type_head(concat(mem_summary, level_meta_emb))
    action_type = CategoricalMasked(action_type_logits, available_actions).sample()

    if action_type requires coordinate:
        pointer_logits = pointer_head(mem_summary, grid_tokens)  # Attend over grid tokens [batch, 64]
        patch_idx = Categorical(pointer_logits).sample()
        # Optionally refine to pixel within patch

        # Map patch_idx to (x, y) in grid
        x, y = patch_idx_to_xy(patch_idx)
    else:
        x, y = None, None

    # 5. Value prediction
    value = value_head(concat(mem_summary, level_meta_emb))

    return {
        "action_type_logits": action_type_logits,
        "coord_logits": pointer_logits if needed else None,
        "value": value,
        "world_model_pred": pred_grid_delta,
    }
```

---

# 6. Why Each Part Helps for ARC

- **ViT + Relational Attention:** Captures symmetry, objectness, and global patterns—core to ARC reasoning.
- **Patchification:** Reduces input size, focuses on meaningful structure, and enables pointer attention.
- **Temporal Transformer:** Enables multi-step reasoning, sequential credit assignment, and memory of failed trials or past grid states.
- **World Model Head:** Allows for efficient exploration, better reward assignment, and planning.
- **Pointer Network Action Head:** Makes spatial action selection efficient, interpretable, and structured; agent can "point" at objects or regions, not guess random coordinates.
- **Auxiliary Losses:** Accelerate learning by providing dense, structured feedback beyond sparse environment rewards.

---

# 7. Implementation Notes

- **PyTorch:** Use `torch.nn.TransformerEncoder`, `nn.MultiheadAttention`, custom pointer networks.
- **Efficiency:** Patchify and process with fast batched matrix ops; keep memory buffer on GPU.
- **Integration:** All modules can be fused into a single `nn.Module`, with separate heads for PPO and auxiliary losses.
- **Compatibility:** Policy/value heads remain compatible with PPO update logic (log-probs, entropy, values).

---

# 8. Optional Enhancements

- **Contrastive Representation Learning:** Use InfoNCE or BYOL-style loss on grid embeddings.
- **Object Tracking:** Track slots/objects over time for explicit object permanence.
- **Imagination Rollouts:** Use world model for imagined rollouts to guide action selection (model-based RL).

---

# 9. References

- [Perceiver IO](https://arxiv.org/abs/2107.14795): Generalizes transformers for structured, multi-modal input.
- [Slot Attention](https://arxiv.org/abs/2006.15055): Object-centric representation from pixels.
- [Pointer Networks](https://arxiv.org/abs/1506.03134): Neural attention for spatial selection.
- [Decision Transformer](https://arxiv.org/abs/2106.01345): Transformer RL with trajectory conditioning.

---

# 10. Summary Table

| Module           | Layer/Type           | Input Shape              | Output Shape          |
|------------------|---------------------|--------------------------|----------------------|
| Patch Linear     | Linear(8*8*16→128)  | [B, 64, 1024]            | [B, 64, 128]         |
| ViT Encoder      | 4x Transf. Encoder  | [B, 64, 128]             | [B, 64, 128]         |
| Relational Attn  | MultiheadAttn       | [B, 64, 128]             | [B, 64, 128]         |
| Slot Attention   | SlotAttn (optional) | [B, 64, 128]             | [B, S, 128]          |
| Temporal Mem     | 2x Transf. Encoder  | [B, N, mem_input_dim]    | [B, N, 512]          |
| World Model      | MLP                 | [B, input_dim]           | [B, 64, 128]         |
| Policy Head      | MLP + Pointer       | [B, mem_dim+level_dim]   | [B, 8], [B, 64]      |
| Value Head       | MLP                 | [B, mem_dim+level_dim]   | [B, 1]               |

---

This architecture captures the essence of ARC: spatial reasoning, object manipulation, memory, and sequential logic. It leverages modern neural-symbolic ideas, supports model-based RL, and is fully compatible with PPO in PyTorch.

**If you’d like, I can provide a concrete PyTorch skeleton for this model.**

---

## Planning Prompt 2

Great. Now let's drill deeper:

1. **Exact PyTorch module definitions**: Give me the complete nn.Module classes with __init__ and forward methods. Use real tensor shapes and dimensions.

2. **Training loop changes**: How should PPO be modified? What auxiliary losses should we add (world model loss, contrastive loss, etc.)?

3. **Reward shaping improvements**: Our current reward shaper gives +10 for win, -0.01 per step, etc. What reward signals would help the specialized architecture learn faster?

4. **Curriculum strategy**: We have 25 games (7 click, 5 keyboard, 13 keyboard_click). How should we order training? Should we start with single-game mastery or multi-game training?

5. **Critical implementation pitfalls**: What are the top 5 things that could go wrong and how to avoid them?

Please be extremely specific with code. I want to be able to implement this directly.

## GPT Response 2

Absolutely! Here’s a **detailed, code- and implementation-focused breakdown** targeting your new architecture and training loop for ARC-AGI-3.

---

# 1. Exact PyTorch `nn.Module` Definitions

Below are the core modules: **GridEncoderViT**, **TemporalMemoryTransformer**, **WorldModelHead**, **PointerPolicyHead**, **ValueHead**, and a **full Agent** wrapper. All shapes/comments are explicit.

### 1.1. Patch-based Vision Transformer Encoder

```python
import torch
import torch.nn as nn

class GridEncoderViT(nn.Module):
    """
    Input: [B, 16, 64, 64] (one-hot grid)
    Output: [B, 64, 128] (patch embeddings)
    """
    def __init__(self, patch_size=8, embed_dim=128, num_layers=4, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (64 // patch_size) ** 2  # 8x8 = 64 patches
        self.patch_dim = patch_size * patch_size * 16  # 1024

        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)  # projects flat patch to embed_dim

        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))  # [1,64,128]

        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # x: [B, 16, 64, 64]
        B = x.size(0)
        # unfold to [B, 16*8*8, 8*8] then permute to [B, num_patches, patch_dim]
        patches = x.unfold(2, self.patch_size, self.patch_size) \
                   .unfold(3, self.patch_size, self.patch_size)  # [B, 16, 8, 8, 8, 8]
        patches = patches.permute(0,2,3,1,4,5).contiguous()  # [B, 8, 8, 16, 8, 8]
        patches = patches.view(B, 64, self.patch_dim)  # [B, 64, 1024]
        patch_emb = self.patch_embed(patches)  # [B, 64, 128]
        patch_emb = patch_emb + self.pos_embed  # [B, 64, 128]
        out = self.transformer(patch_emb)  # [B, 64, 128]
        return out  # [B, 64, 128]
```

### 1.2. Relational Attention Layer

```python
class RelationalAttention(nn.Module):
    """Extra multihead self-attention for relation modeling over patches."""
    def __init__(self, embed_dim=128, num_heads=8):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, x):  # [B, 64, 128]
        out, _ = self.attn(x, x, x)  # standard self-attn
        return out  # [B, 64, 128]
```

### 1.3. Slot Attention (optional, object-centric extraction)

```python
class SlotAttention(nn.Module):
    """
    Extracts K object-centric slots from patch embeddings.
    Input: [B, 64, 128], Output: [B, K, 128]
    """
    def __init__(self, num_slots=6, slot_dim=128, iters=3):
        super().__init__()
        self.num_slots = num_slots
        self.slot_dim = slot_dim
        self.iters = iters
        self.slots_mu = nn.Parameter(torch.zeros(1, 1, slot_dim))
        self.slots_sigma = nn.Parameter(torch.ones(1, 1, slot_dim))
        self.project_q = nn.Linear(slot_dim, slot_dim)
        self.project_k = nn.Linear(slot_dim, slot_dim)
        self.project_v = nn.Linear(slot_dim, slot_dim)
        self.gru = nn.GRUCell(slot_dim, slot_dim)
        self.mlp = nn.Sequential(
            nn.Linear(slot_dim, slot_dim),
            nn.ReLU(),
            nn.Linear(slot_dim, slot_dim)
        )
        self.norm_inputs = nn.LayerNorm(slot_dim)
        self.norm_slots = nn.LayerNorm(slot_dim)
        self.norm_mlp = nn.LayerNorm(slot_dim)

    def forward(self, x):  # [B, 64, 128]
        B, N, D = x.shape
        slots = torch.randn(B, self.num_slots, D, device=x.device) * self.slots_sigma + self.slots_mu
        x = self.norm_inputs(x)
        for _ in range(self.iters):
            slots_prev = slots
            q = self.project_q(self.norm_slots(slots))  # [B, K, D]
            k = self.project_k(x)                       # [B, N, D]
            attn_logits = torch.einsum('bkd,bnd->bkn', q, k) / D**0.5  # [B, K, N]
            attn = attn_logits.softmax(dim=1)  # [B, K, N]
            attn = attn / (attn.sum(dim=-1, keepdim=True)+1e-8)
            updates = torch.einsum('bkn,bnd->bkd', attn, self.project_v(x))  # [B, K, D]
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            ).reshape(B, self.num_slots, D)
            slots = slots + self.mlp(self.norm_mlp(slots))
        return slots  # [B, K, D]
```

### 1.4. Temporal Memory Transformer

```python
class TemporalMemoryTransformer(nn.Module):
    """
    Sequence model over previous steps.
    Input: [B, T, mem_dim]
    Output: [B, T, mem_dim]
    """
    def __init__(self, mem_dim=512, num_layers=2, num_heads=8):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(d_model=mem_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x):  # [B, T, mem_dim]
        return self.transformer(x)  # [B, T, mem_dim]
```

### 1.5. World Model Head

```python
class WorldModelHead(nn.Module):
    """
    Predicts next grid patch embeddings given current grid, action, aux.
    Input: [B, 512+action_dim+aux_dim]
    Output: [B, 64, 128] (predicted next grid patches)
    """
    def __init__(self, input_dim, num_patches=64, patch_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, num_patches * patch_dim)
        )
        self.num_patches = num_patches
        self.patch_dim = patch_dim

    def forward(self, x):  # [B, input_dim]
        out = self.fc(x)  # [B, 64*128]
        out = out.view(-1, self.num_patches, self.patch_dim)
        return out  # [B, 64, 128]
```

### 1.6. Pointer Policy Head

```python
class PointerPolicyHead(nn.Module):
    """
    - Action type logits
    - Pointer distribution over grid (for coordinate actions)
    """
    def __init__(self, mem_dim=512, level_dim=32, num_actions=8, patch_dim=128, num_patches=64):
        super().__init__()
        self.action_type_head = nn.Sequential(
            nn.Linear(mem_dim + level_dim, 256),
            nn.ReLU(),
            nn.Linear(256, num_actions)
        )
        # Pointer over patches (for coordinate actions)
        self.pointer_query_proj = nn.Linear(mem_dim + level_dim, patch_dim)
        self.pointer_key_proj = nn.Linear(patch_dim, patch_dim)
        self.num_patches = num_patches

    def forward(self, mem_summary, level_emb, patch_emb, available_actions=None):
        # mem_summary: [B, mem_dim], level_emb: [B, level_dim], patch_emb: [B, 64, 128]
        B = mem_summary.size(0)
        inp = torch.cat([mem_summary, level_emb], dim=-1)  # [B, mem_dim + level_dim]
        action_type_logits = self.action_type_head(inp)    # [B, 8]
        if available_actions is not None:
            action_type_logits = action_type_logits.masked_fill(~available_actions, float('-inf'))
        # Pointer: attend over patches
        query = self.pointer_query_proj(inp).unsqueeze(1)  # [B,1,128]
        keys = self.pointer_key_proj(patch_emb)            # [B,64,128]
        pointer_logits = torch.matmul(query, keys.transpose(1,2)).squeeze(1) / (patch_emb.size(-1)**0.5)  # [B,64]
        return action_type_logits, pointer_logits          # pointer_logits for coordinate actions only
```

### 1.7. Value Head

```python
class ValueHead(nn.Module):
    def __init__(self, mem_dim=512, level_dim=32):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(mem_dim + level_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
    def forward(self, mem_summary, level_emb):
        x = torch.cat([mem_summary, level_emb], dim=-1)
        return self.fc(x).squeeze(-1)  # [B]
```

### 1.8. Level Embedding

```python
class LevelEmbedding(nn.Module):
    def __init__(self, max_levels=6, level_dim=32):
        super().__init__()
        self.embed = nn.Embedding(max_levels+1, level_dim)
    def forward(self, level_idx):
        return self.embed(level_idx)  # [B, level_dim]
```

---

### 1.9. Full Agent Module

```python
class ARCAgent(nn.Module):
    def __init__(self, aux_dim=15, num_actions=8, num_levels=6):
        super().__init__()
        self.grid_encoder = GridEncoderViT()
        self.relation_attn = RelationalAttention()
        self.slot_attention = SlotAttention()  # optional
        self.mem_dim = 512
        self.temporal_mem = TemporalMemoryTransformer(mem_dim=self.mem_dim)
        self.level_embed = LevelEmbedding(max_levels=num_levels)
        self.policy_head = PointerPolicyHead(mem_dim=self.mem_dim)
        self.value_head = ValueHead(mem_dim=self.mem_dim)
        self.world_model = WorldModelHead(input_dim=self.mem_dim+num_actions+aux_dim)
        self.aux_dim = aux_dim

        # Project grid/slot features to mem_dim
        self.grid_to_mem = nn.Linear(128*64, self.mem_dim)
        # Action embedding
        self.action_embed = nn.Embedding(num_actions, 16)
        # (Optionally, project aux features)
        self.aux_proj = nn.Linear(aux_dim, 32)

    def forward(self, obs, history, level_idx, available_actions=None):
        """
        obs['grid']: [B, 16, 64, 64]
        obs['aux']: [B, aux_dim]
        history: [B, T, mem_dim]
        level_idx: [B]
        available_actions: [B, 8]
        """
        B = obs['grid'].size(0)
        patch_emb = self.grid_encoder(obs['grid'])        # [B, 64, 128]
        patch_emb = self.relation_attn(patch_emb)         # [B, 64, 128]
        grid_flat = patch_emb.view(B, -1)                 # [B, 64*128]
        grid_embed = self.grid_to_mem(grid_flat)          # [B, mem_dim]

        # Optionally, object slots:
        # slots = self.slot_attention(patch_emb)          # [B, K, 128]

        # Build memory input for this step (example: [grid_embed | prev_action_emb | aux])
        # history: [B, T, mem_dim]
        # For simplicity: use last step in history as summary (could use transformer memory output)
        if history is not None:
            mem_out = self.temporal_mem(history)          # [B, T, mem_dim]
            mem_summary = mem_out[:, -1]                  # [B, mem_dim]
        else:
            mem_summary = grid_embed                      # [B, mem_dim]

        level_emb = self.level_embed(level_idx)           # [B, level_dim=32]
        aux_proj = self.aux_proj(obs['aux'])              # [B, 32]

        # Policy
        action_type_logits, pointer_logits = self.policy_head(mem_summary, level_emb, patch_emb, available_actions)

        # Value
        value = self.value_head(mem_summary, level_emb)

        # World model
        # For world model, need prev_action and aux: concatenate [grid_embed, action_emb, aux_proj]
        # (For training, you provide prev_action and aux from history)
        #world_model_pred = self.world_model(torch.cat([grid_embed, prev_action_emb, aux_proj], dim=-1))

        return {
            "action_type_logits": action_type_logits,  # [B, 8]
            "pointer_logits": pointer_logits,          # [B, 64]
            "value": value,                            # [B]
            #"world_model_pred": world_model_pred,      # [B, 64, 128]
            "grid_embed": grid_embed,                  # [B, mem_dim]
            "patch_emb": patch_emb                     # [B, 64, 128]
        }
```

---

# 2. **Training Loop Changes**

## **PPO changes & Auxiliary Losses**

### 2.1. **New loss calculation**
- **Standard PPO loss:** As before.
- **Auxiliary losses:**
    - **World model loss:** MSE or contrastive loss between predicted and actual next patch embeddings.
    - **(Optional) Contrastive/slot loss:** InfoNCE between slots of current and next grid.
    - **(Optional) Curiosity reward:** Use prediction error as intrinsic reward.

### 2.2. **Sample code for loss computation**
```python
def compute_aux_losses(agent, obs, action, aux, next_obs):
    # 1. World model loss: predict next patch_emb from (grid_embed, action, aux)
    grid_embed = agent.grid_to_mem(agent.grid_encoder(obs['grid']).view(obs['grid'].size(0), -1))
    action_emb = agent.action_embed(action)
    aux_proj = agent.aux_proj(aux)
    world_model_in = torch.cat([grid_embed, action_emb, aux_proj], dim=-1)
    pred_next_patch = agent.world_model(world_model_in)  # [B, 64, 128]
    with torch.no_grad():
        next_patch_emb = agent.grid_encoder(next_obs['grid'])
        next_patch_emb = agent.relation_attn(next_patch_emb)
    world_model_loss = nn.functional.mse_loss(pred_next_patch, next_patch_emb)

    # Optionally, add slot contrastive loss, e.g. InfoNCE between slot embeddings
    # Optionally, add world model prediction error as intrinsic reward
    return world_model_loss
```

### 2.3. **Training loop changes**
- Add auxiliary losses to the total loss:
    ```python
    loss = policy_loss + value_coef * value_loss - entropy_coef * entropy + aux_loss_coef * world_model_loss
    ```
- Use intrinsic reward from prediction error when shaping `reward` during rollout.

---

# 3. **Reward Shaping Improvements**

**Key signals to help this architecture:**
- **Prediction-based curiosity:** Add reward proportional to world model prediction error (novelty bonus).
- **Object/slot change bonus:** Reward for discovering new objects or changing object count.
- **Transformation detection:** Bonus when grid structure changes (not just any change, but e.g. symmetry, color swap, etc.).
- **Intermediate goal bonus:** If agent completes a subgoal (detected via grid similarity to known intermediate states), bonus.

**Example improved reward function:**
```python
def compute_reward(prev_grid, curr_grid, curr_state, action_type, levels_before, levels_after, 
                  pred_error=None, slot_change=None):
    r = -0.01  # step cost
    if "WIN" in curr_state:
        r += 10.0
    elif "GAME_OVER" in curr_state:
        r -= 5.0
    if levels_after > levels_before:
        r += 2.0 * (1.0 + 0.5 * levels_after)
    if prev_grid is not None and curr_grid is not None:
        grid_diff = not np.array_equal(np.asarray(prev_grid), np.asarray(curr_grid))
        r += 0.02 if grid_diff else -0.005
    if action_type == 7:  # undo
        r -= 0.05
    if action_type == 0 and "GAME_OVER" not in curr_state:  # reset
        r -= 0.1
    # Add curiosity bonus
    if pred_error is not None:
        r += 0.1 * float(pred_error)
    # Bonus for object/slot count change
    if slot_change is not None and slot_change > 0:
        r += 0.05 * float(slot_change)
    return r
```

---

# 4. **Curriculum Strategy**

- **Phase 1: "Single-game mastery"**  
    - Start with **one easy click game** (lowest average solution length).
    - Train until agent wins at least 50% of episodes.
- **Phase 2: "Intra-category transfer"**
    - Add more click games, then keyboard games, then keyboard_click games.
    - Cycle through games within each category.
- **Phase 3: "Multi-game/interleaved curriculum"**
    - Sample games from all categories, with higher probability for easier/solved games, lower for harder.
    - Use "success rate" to dynamically upweight unsolved games.
- **Phase 4: "Full set, random order"**
    - Once agent can master all but hardest, train with all games equally.

**Why?**  
- ARC games are diverse; early focus on simple click games lets the model learn transformation/discovery strategies.
- Gradually introducing complexity helps the model's relational/temporal mechanisms generalize.
- Interleaving games after initial mastery prevents catastrophic forgetting and encourages transfer.

**Implementation:**
- Use a "GameSampler" class that tracks per-game win rates, and schedules games accordingly.

---

# 5. **Critical Implementation Pitfalls**

### (1) **Memory Misalignment**
- **Problem:** Memory buffer (history) must align step-by-step with env transitions.
- **Solution:** Always update memory at the same time as stepping env; test with synthetic envs.

### (2) **Pointer Action/Coordinate Mapping**
- **Problem:** Pointer head outputs patch indices, but env expects (x, y) pixel coordinates.
- **Solution:** Map patch index to grid region, optionally refine within patch for finer precision.

### (3) **Auxiliary Loss Instability**
- **Problem:** World model prediction loss may dominate or destabilize training.
- **Solution:** Tune `aux_loss_coef` carefully (start with 0.01–0.1); optionally use target networks for next-patch embedding.

### (4) **Reward Signal Saturation or Sparsity**
- **Problem:** If curiosity/auxiliary rewards dominate, agent may ignore true task reward.
- **Solution:** Normalize and clip all auxiliary rewards; always ensure true task reward is the main driver.

### (5) **Batch/Sequence Dimension Mistakes**
- **Problem:** Transformers and history modules are sensitive to shape [B,T,D] vs [T,B,D].
- **Solution:** Use `batch_first=True` everywhere; add asserts or shape prints in all forward().

---

## **Summary Table of Key Module Shapes**

| Module            | Input Shape                    | Output Shape            |
|-------------------|-------------------------------|------------------------|
| GridEncoderViT    | [B, 16, 64, 64]               | [B, 64, 128]           |
| RelationalAttn    | [B, 64, 128]                  | [B, 64, 128]           |
| SlotAttention     | [B, 64, 128]                  | [B, K, 128]            |
| Grid2Mem          | [B, 64*128]                   | [B, 512]               |
| TemporalMemoryTF  | [B, T, 512]                   | [B, T, 512]            |
| WorldModelHead    | [B, 512+16+32]                | [B, 64, 128]           |
| PointerPolicyHead | mem_summary:[B,512], ...      | [B,8], [B,64]          |
| ValueHead         | mem_summary:[B,512], ...      | [B]                    |

---

**With these precise modules, training loop changes, and reward/curriculum strategies, you have a complete blueprint to implement and train a world-model, memory-augmented ARC-AGI-3 agent in PyTorch.**  
Let me know if you want full training loop code or further details on any submodule!

---

## Planning Prompt 3

Now I want you to adversarially review your own proposal. Consider:

1. **Complexity vs. sample efficiency**: ARC games give sparse rewards. Can this architecture learn from so few positive signals? How many API calls / training steps are realistically needed?

2. **Comparison with simpler baselines**: Would a simpler approach (e.g., just adding LSTM to the existing CNN, or using a smaller grid representation) get us WINs faster? Sometimes simpler is better.

3. **The elephant in the room**: Most ARC-AGI solutions use LLMs or program synthesis, not RL. Should we consider a hybrid approach where the RL agent learns a meta-policy that invokes different strategies? Or should we use an LLM to analyze grid patterns and guide the RL policy?

4. **Practical constraints**: We have ~600 API calls/minute and 200 max actions per game. Training with real API calls is slow. How do we maximize learning per API call?

5. **Revised recommendation**: Given all the above, what is the MINIMUM VIABLE architecture that gives us the best chance of getting a first WIN in the shortest time?

Be brutally honest about what will and won't work.

## GPT Response 3

Fantastic—this is the level of critical review that separates research prototypes from real competition progress. Let’s break down the proposal **adversarially** and give you a hard-nosed, practical recommendation for ARC-AGI-3.

---

## 1. **Complexity vs. Sample Efficiency: Is this justified?**

**Strengths of the Proposed Model:**
- The architecture is state-of-the-art in theory: ViT, relational attn, memory transformer, pointer action—this would be *ideal* for a large offline dataset, or a simulated environment.

**BUT:**
- **ARC games are sparse-reward, high-dimensional, and reward signals (WIN) are *extremely* rare.**
- Even with advanced architectures, RL is notoriously sample inefficient, especially with random exploration and sparse rewards.
- Realistically, unless the agent stumbles into a WIN early, **it may require millions of steps** before ever seeing a positive signal, even with curiosity bonuses and auxiliary losses.

**Estimation:**
- With 600 API calls/minute, assume 1 game/episode per 200 steps = 3 games/minute.
- 1,000,000 steps (a small RL run) = 5,000 episodes ≈ 28 hours of wall clock time, but *still* unlikely to see a WIN unless the agent’s inductive biases are a perfect fit for a particular game’s structure.
- **Sample efficiency will be a huge bottleneck; this architecture is powerful but not data-efficient for tabula rasa RL on complex ARC-AGI games.**

---

## 2. **Comparison: Simpler Baselines**

**What if we just:**
- Add an LSTM on top of the existing CNN?
- Downsample the grid (e.g., 16x16), or use max pooling to compress input?
- Use a shallow MLP or ResNet instead of ViT?

**Potential Results:**
- **LSTM-CNN**: Immediate access to memory; can at least remember failed attempts and try not to repeat them.
- **Downsampling**: Reduces input space; makes exploration less sparse, but loses fine-grained structure.
- **Shallow ResNet**: Slightly better inductive bias than a plain CNN, but still struggles with object/relational/abstract reasoning.

**But critically:**
- *Even a simple LSTM-CNN baseline can stumble into a WIN faster*, purely by virtue of being faster to train and easier to debug. If you combine this with heavy reward shaping (see below), you could get your first WIN while still prototyping more complex architectures in parallel.

---

## 3. **The Elephant in the Room: RL vs. LLMs/Program Synthesis**

- **ARC is fundamentally about program induction and symbolic reasoning.**
- Most winning ARC-AGI entries (prior rounds) use LLMs, search, or program synthesis—**not RL**.
- RL is good for low-level control and exploration, but bad at discovering symbolic, compositional solutions from scratch.

**Hybrid Strategies:**
- **Option A:** Use LLMs to pre-analyze the grid, extract features (symmetry, object masks, color mappings), and feed those as context to your RL agent.
- **Option B:** Use RL as a meta-controller: it chooses *which strategy* (from a set of hand-crafted/LLM-generated policies) to try on each level, rather than directly emitting low-level actions.
- **Option C:** Use LLM-generated pseudo-rewards (e.g., "looks like a target pattern" or "applies a common transformation") as dense learning signals.

**Bottom line:**  
**Pure RL is at a huge disadvantage.** Hybridizing with LLMs or program synth will likely be necessary for superhuman sample efficiency, especially for multi-level games and unseen transformations.

---

## 4. **Practical Constraints: Maximizing Learning per API Call**

- **API rate limit is an absolute cap.**  
  You can't brute-force your way to a solution.

**Maximize data per call:**
- **Model-based RL:** Use a learned world model to perform "imagination rollouts" offline, between real env steps.
- **Off-policy replay:** Store and reuse all transitions, even failed ones, for auxiliary/self-supervised learning.
- **Auxiliary tasks:** Predict grid deltas, object counts, or future rewards in parallel to main RL objective.
- **Reward shaping:** Dense, informative intermediate rewards (grid change, object movement, symmetry improvement).
- **Curriculum:** Start with *the easiest possible* levels/games (single-level, click-based), and repeat until mastery.

---

## 5. **Revised Recommendation: Minimum Viable Architecture for First WIN**

### **a. Start Simple, Then Add Complexity**

**Minimum viable, WIN-oriented architecture:**
1. **CNN + LSTM baseline**
    - Use your current CNN grid encoder.
    - Add a single-layer LSTM (hidden size 128–256) after grid+aux embedding, before the policy/value heads.
    - Use PPO as before.

2. **Aggressive reward shaping**
    - +0.2 for any grid change
    - +1.0 for matching any known intermediate/target pattern (see below)
    - -0.1 for undo/reset; -0.01/step as before

3. **Downsample grid (optional)**
    - If learning is too slow, downsample input grids to 32x32/16x16 using block max or avg pooling.

4. **Curriculum**
    - Train on *one* easy click game until you get a WIN.
    - Only expand to more games after first success.

### **b. When to Add Complexity**

- If/when you get your first WIN, start adding:
    - **World model head** for auxiliary loss and curiosity
    - **Attention/transformers** on top of LSTM if you see evidence of LSTM bottlenecking
    - **Hybrid LLM features** (e.g., object masks, symmetry tags) if possible

### **c. Maximum learning per API call**
- **Replay Buffer:** Keep all transitions (including across games) for off-policy auxiliary learning.
- **Self-supervised learning:** Train world model, grid embedding, or slot/object extractor offline on stored data between env calls.
- **Model-based rollouts (if feasible):** Use your learned world model to generate synthetic rollouts for additional policy updates (Dyna-style).

---

### **Sample Minimum Viable PyTorch Policy**

```python
class CNNLSTMPolicy(nn.Module):
    def __init__(self, num_colors=16, aux_dim=15, embedding_dim=128, lstm_hidden=128, num_actions=8):
        super().__init__()
        self.encoder = GridEncoder(num_colors, embedding_dim)
        self.aux_fc = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU())
        self.lstm = nn.LSTM(embedding_dim+32, lstm_hidden, batch_first=True)
        self.policy_head = nn.Linear(lstm_hidden, num_actions)
        self.value_head = nn.Linear(lstm_hidden, 1)

    def forward(self, grid, aux, lstm_state=None):
        # grid: [B, 16, 64, 64], aux: [B, 15]
        emb = self.encoder(grid)
        aux_emb = self.aux_fc(aux)
        x = torch.cat([emb, aux_emb], dim=-1).unsqueeze(1)  # [B,1,D]
        out, lstm_state = self.lstm(x, lstm_state)          # [B,1,H]
        h = out.squeeze(1)
        return self.policy_head(h), self.value_head(h).squeeze(-1), lstm_state
```

- **During rollouts:** Maintain and pass LSTM state per episode.
- **For coordinate actions:** Add a small pointer MLP or use an additional head for (x, y).

---

## **Summary Table**

| Approach        | API Calls to WIN | Complexity | Feasibility | Comments              |
|-----------------|------------------|------------|-------------|-----------------------|
| ViT+Transformer | 10^6+            | High       | Low-Med     | Needs many calls      |
| CNN+LSTM        | 10^4–10^5        | Low        | High        | Reachable in practice |
| LLM/Hybrid      | 10^3–10^4        | High       | Med         | If LLMs allowed       |
| RL+LLM hybrid   | <10^4            | Med-High   | High        | If LLM features avail |

---

# **Brutally Honest Take**

- **Your best chance at a first WIN is a well-tuned CNN+LSTM baseline, with heavy reward shaping and focused curriculum.**
- The fancy transformer/slot/world model approach is *better* in the long run (for generalization, high-level reasoning, multi-level adaptation)—but it’s unlikely to reach a WIN faster under the ARC-AGI-3 constraints, unless you already have a large offline dataset or can simulate games.
- If you can hybridize with LLMs (even for feature extraction, not action selection), do it. Otherwise, maximize sample efficiency and learning signal per API call.
- Once you get a WIN, invest in more complex models and hybrid approaches for scaling to harder games.

---

**Recommendation:**  
1. Implement a CNN+LSTM baseline, with the best possible reward shaping and a curriculum starting at the easiest click game.
2. Only add transformers/world model/slots once you’ve proven the baseline can get a WIN or if you hit a clear bottleneck.
3. If at all possible, incorporate LLM-derived features or pseudo-rewards, even if the RL agent remains the low-level controller.

**This will maximize your chance of a first WIN given ARC-AGI-3’s sample constraints.**

---

## Planning Prompt 4

Based on our entire discussion, give me the FINAL implementation plan. Structure it as:

## Phase 1: Quick Wins (get first WIN ASAP)
- Minimum changes to current codebase
- Most likely to produce a WIN quickly
- Specific files to modify and how

## Phase 2: Specialized Architecture
- The transformer/memory model
- Detailed module definitions (PyTorch code)
- Training procedure changes

## Phase 3: Advanced Strategies
- World model / model-based RL
- Hybrid LLM+RL approach
- Multi-game curriculum

For each phase, specify:
- Exact files to create/modify
- Complete PyTorch nn.Module code
- Training hyperparameters
- Expected improvement and timeline
- Dependencies on previous phases

This will be directly handed to implementation agents, so be extremely precise and complete.

## GPT Response 4

Absolutely. Here is a **pragmatic, stepwise implementation plan** for your ARC-AGI-3 agent, structured for real-world engineering execution. Each phase is actionable, file-specific, and designed for handoff to a team.

---

# ## Phase 1: Quick Wins (First WIN ASAP)

### **Goal:**  
Get a WIN on the easiest ARC-AGI-3 game *as fast as possible* with minimal changes.

---

### **A. Modifications**

**1. Add LSTM memory to policy**
- Add LSTM on top of your existing CNN encoder.

**2. Aggressive reward shaping**
- Increase grid-change bonus, add curiosity if possible.

**3. Focused curriculum**
- Only train on the simplest click game until first WIN.

---

### **B. File Changes**

- `src/models/policy_net.py` (major change)
- `src/training/trainer.py` (minor, for LSTM state)
- `src/training/reward_shaper.py` (reward shaping)
- Optionally: `src/environment/game_sampler.py` (to restrict to one game)

---

### **C. Code: LSTM Policy Implementation**

**`src/models/policy_lstm.py`**
```python
import torch
import torch.nn as nn
from src.models.encoder import GridEncoder

class CNNLSTMPolicy(nn.Module):
    def __init__(self, num_colors=16, aux_dim=15, embedding_dim=128, lstm_hidden=128, num_actions=8):
        super().__init__()
        self.encoder = GridEncoder(num_colors, embedding_dim)
        self.aux_fc = nn.Sequential(nn.Linear(aux_dim, 32), nn.ReLU())
        self.lstm = nn.LSTM(embedding_dim+32, lstm_hidden, batch_first=True)
        self.policy_head = nn.Linear(lstm_hidden, num_actions)
        self.value_head = nn.Linear(lstm_hidden, 1)

    def forward(self, grid, aux, lstm_state=None):
        # grid: [B, 16, 64, 64], aux: [B, 15]
        emb = self.encoder(grid)         # [B, embedding_dim]
        aux_emb = self.aux_fc(aux)       # [B, 32]
        x = torch.cat([emb, aux_emb], dim=-1).unsqueeze(1)  # [B,1,D]
        out, lstm_state = self.lstm(x, lstm_state)          # [B,1,H]
        h = out.squeeze(1)
        return self.policy_head(h), self.value_head(h).squeeze(-1), lstm_state
```

**Update `src/training/trainer.py`:**
- Track and reset LSTM state at episode boundaries.
- Pass LSTM state in rollouts.

**Update `src/training/reward_shaper.py`:**
- Set `grid_change_bonus=0.2`, add optional curiosity bonus.

---

### **D. Hyperparameters**

- LR: `3e-4`
- LSTM hidden: `128`
- PPO epochs per update: `4`
- Max steps per episode: `200`
- Reward shaping: as above

---

### **E. Expected Improvement/Timeline**

- **First WIN possible in hours to a day** (if curriculum and reward shaping are tuned).
- Unlocks ability to expand to Phase 2.
- **Dependencies:** None (can run immediately).

---

# ## Phase 2: Specialized Architecture

### **Goal:**  
Introduce transformer, memory, and pointer mechanisms for better generalization and complex games.

---

### **A. File Changes**

- **Create new files:**
  - `src/models/transformer_grid_encoder.py`
  - `src/models/temporal_memory.py`
  - `src/models/pointer_policy.py`
  - `src/models/arc_agent.py` (main module wrapper)
- **Modify:**
  - `src/training/trainer.py` (add auxiliary losses, handle memory buffer/history)
  - `src/training/reward_shaper.py` (add reward for model curiosity)
  - `src/environment/game_sampler.py` (curriculum)

---

### **B. PyTorch nn.Module Code**

**`src/models/transformer_grid_encoder.py`**
```python
import torch
import torch.nn as nn
class GridEncoderViT(nn.Module):
    def __init__(self, patch_size=8, embed_dim=128, num_layers=4, num_heads=8):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_patches = (64 // patch_size) ** 2  # 8x8 patches
        self.patch_dim = patch_size * patch_size * 16
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, x):
        B = x.size(0)
        patches = x.unfold(2, self.patch_size, self.patch_size)\
                   .unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0,2,3,1,4,5).contiguous()
        patches = patches.view(B, 64, self.patch_dim)
        patch_emb = self.patch_embed(patches) + self.pos_embed
        return self.transformer(patch_emb)  # [B, 64, 128]
```

**`src/models/temporal_memory.py`**
```python
import torch.nn as nn
class TemporalMemoryTransformer(nn.Module):
    def __init__(self, mem_dim=512, num_layers=2, num_heads=8):
        super().__init__()
        enc_layer = nn.TransformerEncoderLayer(d_model=mem_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
    def forward(self, x):  # [B, T, mem_dim]
        return self.transformer(x)
```

**`src/models/pointer_policy.py`**
```python
import torch
import torch.nn as nn
class PointerPolicyHead(nn.Module):
    def __init__(self, mem_dim=512, level_dim=32, num_actions=8, patch_dim=128, num_patches=64):
        super().__init__()
        self.action_type_head = nn.Sequential(
            nn.Linear(mem_dim + level_dim, 256), nn.ReLU(), nn.Linear(256, num_actions))
        self.pointer_query_proj = nn.Linear(mem_dim + level_dim, patch_dim)
        self.pointer_key_proj = nn.Linear(patch_dim, patch_dim)
        self.num_patches = num_patches
    def forward(self, mem_summary, level_emb, patch_emb, available_actions=None):
        inp = torch.cat([mem_summary, level_emb], dim=-1)
        action_type_logits = self.action_type_head(inp)
        if available_actions is not None:
            action_type_logits = action_type_logits.masked_fill(~available_actions, float('-inf'))
        query = self.pointer_query_proj(inp).unsqueeze(1)
        keys = self.pointer_key_proj(patch_emb)
        pointer_logits = torch.matmul(query, keys.transpose(1,2)).squeeze(1) / (patch_emb.size(-1)**0.5)
        return action_type_logits, pointer_logits
```

**`src/models/arc_agent.py`**
```python
import torch.nn as nn
from .transformer_grid_encoder import GridEncoderViT
from .temporal_memory import TemporalMemoryTransformer
from .pointer_policy import PointerPolicyHead

class ARCAgent(nn.Module):
    def __init__(self, aux_dim=15, num_actions=8, num_levels=6):
        super().__init__()
        self.grid_encoder = GridEncoderViT()
        self.mem_dim = 512
        self.temporal_mem = TemporalMemoryTransformer(mem_dim=self.mem_dim)
        self.level_embed = nn.Embedding(num_levels+1, 32)
        self.policy_head = PointerPolicyHead(mem_dim=self.mem_dim)
        self.value_head = nn.Sequential(
            nn.Linear(self.mem_dim+32, 256), nn.ReLU(), nn.Linear(256, 1))
        self.grid_to_mem = nn.Linear(128*64, self.mem_dim)
        self.aux_proj = nn.Linear(aux_dim, 32)
    def forward(self, obs, history, level_idx, available_actions=None):
        B = obs['grid'].size(0)
        patch_emb = self.grid_encoder(obs['grid'])
        grid_flat = patch_emb.view(B, -1)
        grid_embed = self.grid_to_mem(grid_flat)
        if history is not None:
            mem_out = self.temporal_mem(history)
            mem_summary = mem_out[:, -1]
        else:
            mem_summary = grid_embed
        level_emb = self.level_embed(level_idx)
        action_type_logits, pointer_logits = self.policy_head(mem_summary, level_emb, patch_emb, available_actions)
        value = self.value_head(torch.cat([mem_summary, level_emb], dim=-1)).squeeze(-1)
        return {"action_type_logits": action_type_logits, "pointer_logits": pointer_logits, "value": value, "grid_embed": grid_embed, "patch_emb": patch_emb}
```

---

### **C. Training Procedure Changes**

- Add auxiliary loss for world model prediction.
- Pass memory/history to agent in rollouts.
- Store grid embeddings and actions for memory transformer.
- Optionally, use prediction error as intrinsic reward.
- Curriculum: Gradually add more games/levels after success.

---

### **D. Hyperparameters**

- LR: `1e-4`
- PPO epochs: `4`
- Memory length: `16`
- Aux loss coef: `0.05`
- World model MSE loss

---

### **E. Expected Improvement/Timeline**

- Should enable multi-level reasoning, better generalization.
- Timeline: 1-2 weeks (including debugging).
- **Dependencies:** Phase 1 (should have at least one WIN, ensures infra is working).

---

# ## Phase 3: Advanced Strategies

### **Goal:**  
Push beyond RL: model-based rollouts, LLM guidance, automated curriculum.

---

### **A. File Changes**

- **New files:**
  - `src/models/world_model.py`
  - `src/llm/llm_guidance.py`
  - `src/training/model_based.py` (for Dyna-style rollouts)
  - `src/training/curriculum.py` (automated curriculum logic)

---

### **B. Code**

**`src/models/world_model.py`**
```python
import torch.nn as nn
class WorldModelHead(nn.Module):
    def __init__(self, input_dim, num_patches=64, patch_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, num_patches * patch_dim)
        )
        self.num_patches = num_patches
        self.patch_dim = patch_dim
    def forward(self, x):
        out = self.fc(x)
        return out.view(-1, self.num_patches, self.patch_dim)
```
- Train this head to predict next patch embeddings given current grid, action, aux.

**`src/llm/llm_guidance.py`**
- Use LLM (offline or via API) to analyze grid, extract features (symmetry, object masks, pattern types).
- Feed those features to the RL agent as extra aux input or for reward shaping.

**`src/training/model_based.py`**
- Between real env steps, use world model to simulate synthetic transitions, update the RL agent off-policy.

**`src/training/curriculum.py`**
- Track per-game/level win rates and dynamically schedule games for training to maximize learning rate.

---

### **C. Hyperparameters**

- World model loss coef: `0.01`
- Model-based rollout ratio: `1:1` (real:synth)
- Curriculum: sample unsolved games 2x more often

---

### **D. Expected Improvement/Timeline**

- **Model-based RL**: 2-5x sample efficiency boost.
- **LLM guidance**: Potential order-of-magnitude improvement if LLM features are high-quality.
- **Curriculum**: Ensures generalization across all games.
- Timeline: 2+ weeks for effective integration and results.

- **Dependencies:** Phase 2 (need robust, modular architecture before adding model-based or LLM hybridization).

---

# **Summary Table**

| Phase      | Key Modules/Files                                      | Training Changes            | Timeline | Dependencies |
|------------|--------------------------------------------------------|-----------------------------|----------|--------------|
| 1: Quick   | `policy_lstm.py`, edits to `trainer.py`, `reward_shaper.py` | LSTM, reward shaping, focus | 1-3 days | None         |
| 2: Spec.   | `transformer_grid_encoder.py`, `arc_agent.py`, etc.    | Memory, transformer, pointer| 1-2 wks  | Phase 1      |
| 3: Adv.    | `world_model.py`, `llm_guidance.py`, etc.              | Model-based, LLM, curriculum| 2+ wks   | Phase 2      |

---

# **Hand-off Notes**

- **Start with Phase 1.** Only move to Phase 2 after confirming at least one WIN in curriculum.
- **Phase 2 and 3** can be prototyped in parallel after the baseline works.
- **Unit test** each module in isolation; integration tests on easiest game before scaling up.

---

**This plan will get you a WIN fast, then builds a scalable, generalizable architecture for harder games and full AGI-style reasoning.**

---

