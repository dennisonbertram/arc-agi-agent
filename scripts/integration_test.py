"""End-to-end integration test against real ARC-AGI-3 API."""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

print("=== ARC-AGI-3 Integration Test ===\n")

# Test 1: Config loads correctly
from src.config import config
print(f"[1] Config loaded: API key={config.arc_api_key[:8]}...{config.arc_api_key[-4:]}")
assert config.arc_api_key, "API key missing"
print("    PASS\n")

# Test 2: Env wrapper works with real API
from src.environment.arc_env_wrapper import ArcEnvWrapper
env = ArcEnvWrapper(game_id="tn36-ab4f63cc", api_key=config.arc_api_key, mode="ONLINE")
obs = env.reset()
print(f"[2] Env reset: grid shape={obs['grid'].shape}, aux shape={obs['aux'].shape}")
assert obs['grid'].shape[0] == 16, f"Expected 16 channels, got {obs['grid'].shape[0]}"
assert obs['grid'].shape[1] == 64 and obs['grid'].shape[2] == 64, f"Expected 64x64, got {obs['grid'].shape[1:]}"
print("    PASS\n")

# Test 3: Step works
next_obs, reward, done, info = env.step(5, 0, 0)  # INTERACT at (0,0)
print(f"[3] Step: reward={reward:.4f}, done={done}, grid shape={next_obs['grid'].shape}")
print("    PASS\n")

# Test 4: Policy network forward pass
import torch
from src.models.policy_net import PolicyNetwork
from src.models.value_net import ValueNetwork

policy = PolicyNetwork()
value = ValueNetwork()

grid = obs['grid'].unsqueeze(0)
aux = obs['aux'].unsqueeze(0)
mask = obs['available_actions'].unsqueeze(0)

with torch.no_grad():
    action, x, y, log_prob, entropy = policy.sample(grid, aux, mask)
    v = value(grid, aux)

print(f"[4] Policy: action={action.item()}, x={x.item()}, y={y.item()}, logp={log_prob.item():.4f}, value={v.item():.4f}")
print("    PASS\n")

# Test 5: Short training loop (3 steps)
from src.training.trainer import PPOTrainer
trainer = PPOTrainer(device="cpu")

print("[5] Training loop (3 rollout+update cycles)...")
for i in range(3):
    rollout_stats = trainer.collect_rollout(env, num_steps=20)
    update_stats = trainer.update()
    print(f"    Step {i+1}: episodes={rollout_stats['episodes']}, mean_reward={rollout_stats['mean_reward']:.3f}, "
          f"policy_loss={update_stats['policy_loss']:.4f}, value_loss={update_stats['value_loss']:.4f}")

print("    PASS\n")

print("=== ALL INTEGRATION TESTS PASSED ===")
