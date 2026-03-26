"""Multi-turn architecture planning session with GPT-5.4 for ARC-AGI-3."""
import os
import json
import sys

# Try openai package
try:
    from openai import OpenAI
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "openai"])
    from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY", "")
if not api_key:
    # Try reading from .zshrc
    import re
    with open(os.path.expanduser("~/.zshrc")) as f:
        for line in f:
            m = re.search(r'export OPENAI_API_KEY[="\s]+([^\s"]+)', line)
            if m:
                api_key = m.group(1)
                break

client = OpenAI(api_key=api_key)

# Load context files
context_files = {}
for path in [
    "src/models/encoder.py",
    "src/models/policy_net.py",
    "src/models/action_head.py",
    "src/training/trainer.py",
    "src/training/reward_shaper.py",
    "src/environment/arc_env_wrapper.py",
    "src/config.py",
]:
    full_path = os.path.join("/Users/dennisonbertram/Develop/arc-agi-agent", path)
    if os.path.exists(full_path):
        with open(full_path) as f:
            context_files[path] = f.read()

# Build context summary
code_context = "\n\n".join([f"### {k}\n```python\n{v}\n```" for k, v in context_files.items()])

system_prompt = """You are a world-class deep learning researcher specializing in reinforcement learning for abstract reasoning tasks. You have deep expertise in:
- Transformer architectures and attention mechanisms
- World models and model-based RL
- Program synthesis and neural-symbolic approaches
- The ARC (Abstraction and Reasoning Corpus) challenge

You are helping design a specialized neural network architecture for ARC-AGI-3, an interactive version of ARC where an agent plays turn-based grid games (64x64, 16 colors) using 8 action types. Games have multiple levels that must be completed sequentially.

Current approach (CNN + PPO) has failed to achieve any level WINs after training. We need a fundamentally better architecture.

Key constraints:
- Must work with PyTorch
- Must be compatible with PPO training
- Games are interactive (not static input->output transformation)
- 64x64 grid with 16 color channels
- 8 action types (keyboard movement, click/coordinate, undo, reset)
- Must complete 1-6 sequential levels per game
- Rate limited to ~600 API calls per minute
- No internet access during final Kaggle evaluation
"""

messages = [
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": f"""Here is our current codebase for an ARC-AGI-3 RL agent. It uses CNN + PPO but has failed to achieve any level WINs.

{code_context}

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
- Pseudocode for the forward pass"""}
]

print("=" * 80)
print("TURN 1: Initial Architecture Proposal")
print("=" * 80)

# Use the best available model - try gpt-5.4 first, fall back
model = "gpt-4.1-2025-04-14"  # Latest available model
for try_model in ["gpt-5.4", "o4-mini", "gpt-4.1-2025-04-14", "gpt-4o", "gpt-4-turbo"]:
    try:
        response = client.chat.completions.create(
            model=try_model,
            messages=messages,
            max_tokens=16000,
            temperature=0.7,
        )
        model = try_model
        print(f"[Using model: {model}]")
        break
    except Exception as e:
        if "model" in str(e).lower() or "not found" in str(e).lower():
            continue
        # If it's a different error (like rate limit), still try to use this model
        model = try_model
        response = client.chat.completions.create(
            model=try_model,
            messages=messages,
            max_tokens=16000,
            temperature=0.7,
        )
        break

reply1 = response.choices[0].message.content
print(reply1)
messages.append({"role": "assistant", "content": reply1})

# Turn 2: Drill into specifics
print("\n" + "=" * 80)
print("TURN 2: Implementation Details & Training Strategy")
print("=" * 80)

messages.append({"role": "user", "content": """Great. Now let's drill deeper:

1. **Exact PyTorch module definitions**: Give me the complete nn.Module classes with __init__ and forward methods. Use real tensor shapes and dimensions.

2. **Training loop changes**: How should PPO be modified? What auxiliary losses should we add (world model loss, contrastive loss, etc.)?

3. **Reward shaping improvements**: Our current reward shaper gives +10 for win, -0.01 per step, etc. What reward signals would help the specialized architecture learn faster?

4. **Curriculum strategy**: We have 25 games (7 click, 5 keyboard, 13 keyboard_click). How should we order training? Should we start with single-game mastery or multi-game training?

5. **Critical implementation pitfalls**: What are the top 5 things that could go wrong and how to avoid them?

Please be extremely specific with code. I want to be able to implement this directly."""})

response2 = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=16000,
    temperature=0.7,
)
reply2 = response2.choices[0].message.content
print(reply2)
messages.append({"role": "assistant", "content": reply2})

# Turn 3: Challenge and refine
print("\n" + "=" * 80)
print("TURN 3: Adversarial Review & Refinement")
print("=" * 80)

messages.append({"role": "user", "content": """Now I want you to adversarially review your own proposal. Consider:

1. **Complexity vs. sample efficiency**: ARC games give sparse rewards. Can this architecture learn from so few positive signals? How many API calls / training steps are realistically needed?

2. **Comparison with simpler baselines**: Would a simpler approach (e.g., just adding LSTM to the existing CNN, or using a smaller grid representation) get us WINs faster? Sometimes simpler is better.

3. **The elephant in the room**: Most ARC-AGI solutions use LLMs or program synthesis, not RL. Should we consider a hybrid approach where the RL agent learns a meta-policy that invokes different strategies? Or should we use an LLM to analyze grid patterns and guide the RL policy?

4. **Practical constraints**: We have ~600 API calls/minute and 200 max actions per game. Training with real API calls is slow. How do we maximize learning per API call?

5. **Revised recommendation**: Given all the above, what is the MINIMUM VIABLE architecture that gives us the best chance of getting a first WIN in the shortest time?

Be brutally honest about what will and won't work."""})

response3 = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=16000,
    temperature=0.7,
)
reply3 = response3.choices[0].message.content
print(reply3)
messages.append({"role": "assistant", "content": reply3})

# Turn 4: Final actionable plan
print("\n" + "=" * 80)
print("TURN 4: Final Implementation Plan")
print("=" * 80)

messages.append({"role": "user", "content": """Based on our entire discussion, give me the FINAL implementation plan. Structure it as:

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

This will be directly handed to implementation agents, so be extremely precise and complete."""})

response4 = client.chat.completions.create(
    model=model,
    messages=messages,
    max_tokens=16000,
    temperature=0.7,
)
reply4 = response4.choices[0].message.content
print(reply4)
messages.append({"role": "assistant", "content": reply4})

# Save the full conversation
output = {
    "model": model,
    "turns": len(messages),
    "conversation": messages,
}

output_path = "/Users/dennisonbertram/Develop/arc-agi-agent/docs/plans/gpt-architecture-plan.json"
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, "w") as f:
    json.dump(output, f, indent=2)

# Also save as readable markdown
md_path = "/Users/dennisonbertram/Develop/arc-agi-agent/docs/plans/specialized-architecture-plan.md"
with open(md_path, "w") as f:
    f.write("# ARC-AGI-3 Specialized Architecture Plan\n\n")
    f.write(f"*Generated via multi-turn planning with {model}*\n\n")
    f.write("---\n\n")
    for i, msg in enumerate(messages):
        if msg["role"] == "system":
            continue
        if msg["role"] == "user":
            f.write(f"## Planning Prompt {(i+1)//2}\n\n")
            # Truncate code context for readability
            content = msg["content"]
            if len(content) > 2000 and "```python" in content:
                f.write("[Code context provided from current codebase]\n\n")
                # Extract just the questions part
                parts = content.split("## Current Problems:")
                if len(parts) > 1:
                    f.write("## Current Problems:" + parts[1])
            else:
                f.write(content)
            f.write("\n\n")
        else:
            f.write(f"## GPT Response {i//2}\n\n")
            f.write(msg["content"])
            f.write("\n\n---\n\n")

print(f"\n\nSaved conversation to {output_path}")
print(f"Saved readable plan to {md_path}")
