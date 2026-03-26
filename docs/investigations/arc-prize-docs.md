# ARC Prize & ARC-AGI-3 Documentation Research

**Date:** 2026-03-25
**Source:** https://docs.arcprize.org/

---

## 1. What is ARC Prize

### Overview
ARC Prize is a competition offering **$2M in total prizes** aimed at accelerating progress toward AGI. The competition centers on ARC-AGI (Abstraction and Reasoning Corpus for Artificial General Intelligence), a series of benchmarks measuring AI systems' ability to acquire new skills and generalize to novel situations.

### ARC-AGI-3 (Current Version)
ARC-AGI-3 is the first **interactive reasoning benchmark** designed to measure human-like intelligence in AI agents. Unlike previous static versions (ARC-AGI-1 and ARC-AGI-2), ARC-AGI-3 requires agents to:

- **Explore** novel environments without natural language instructions
- **Acquire goals** on the fly
- **Build adaptable world models**
- **Learn continuously** through perception-to-action cycles
- **Plan** over long horizons with sparse feedback

A perfect score means AI agents can beat every game as efficiently as humans.

### Competition Tracks (2026)
1. **ARC-AGI-3** - Interactive reasoning benchmark (agent-based)
2. **ARC-AGI-2** - Static reasoning benchmark (traditional puzzle-solving)
3. **Paper Prize** - Papers advancing understanding of ARC-AGI performance

### Timeline
| Date | Event |
|------|-------|
| March 25, 2026 | Competition launch |
| June 30, 2026 | ARC-AGI-3 Milestone #1 |
| September 30, 2026 | ARC-AGI-3 Milestone #2 |
| November 2, 2026 | Submission deadline |
| November 8, 2026 | Paper submission deadline |
| December 4, 2026 | Results announcement |

### Rules
- **Open Source Mandate:** All leading participants must open source their solutions (CC0, MIT-0 licenses)
- **No Internet During Evaluation:** Solutions submitted via Kaggle; no API-based systems (GPT/Claude/etc.) during evaluation
- **Third-party code** requires permissive licenses (Apache-2.0, GPLv3)
- Prizes awarded at organizers' discretion after technical review

---

## 2. ARC-AGI-3 Tasks/Games

### Format
ARC-AGI-3 games are **turn-based environments where agents interact with 2D grids through a standardized action interface**.

### Grid Specifications
- **Maximum size:** 64x64
- **Cell values:** Integer 0-15 representing different states/colors (4-bit color indices)
- **Origin:** Top-left corner at (0,0) using (x,y) format

### Game Identification
- Games use the pattern `<game_name>-<version_hash>` (e.g., `ls20-016295f7601e`)
- Game names remain consistent but versions may evolve

### Game Loop
1. Agent receives a state frame (JSON) containing the grid and metadata
2. Agent selects an action from available actions
3. Environment returns updated state
4. Repeat until WIN, GAME_OVER, or max steps

### Action System
Seven core actions serve as the standardized input interface:

| Action | Name | Description |
|--------|------|-------------|
| RESET | Reset | Initialize or restart the game/level state |
| ACTION1 | Up | Directional movement/input mapped upward |
| ACTION2 | Down | Directional movement/input mapped downward |
| ACTION3 | Left | Directional movement/input mapped leftward |
| ACTION4 | Right | Directional movement/input mapped rightward |
| ACTION5 | Interact | Interact, select, rotate, attach/detach, execute, etc. |
| ACTION6 | Coordinate | Complex action requiring x,y coordinates (0-63 range) |
| ACTION7 | Undo | Reverses previous actions |

- Each frame's metadata indicates which actions are currently permitted
- ACTION6 requires both x and y parameters (0-63 range)
- Available actions are visually highlighted in the UI

### Game States
- `NOT_STARTED` - Game has not begun
- `NOT_FINISHED` - Game is in progress
- `WIN` - Agent has won
- `GAME_OVER` - Agent has lost

### Frame Response Structure
```json
{
  "game_id": "string",
  "guid": "string (session ID)",
  "frame": "array of 64x64 grids with 4-bit color indices (0-15)",
  "state": "NOT_FINISHED | NOT_STARTED | WIN | GAME_OVER",
  "levels_completed": "integer (0-254)",
  "win_levels": "integer (0-254, threshold to WIN)",
  "action_input": { "id": "action_id", "data": {} },
  "available_actions": [1, 2, 3, 4, 5, 6]
}
```

### Available Games
- Three games available to anonymous users
- API key required for full game list
- Example games: `ls20`, `ft09`, `vc33`
- Browse at https://arcprize.org/tasks
- Programmatic discovery via `arc.get_environments()` or `GET /api/games`

---

## 3. Scoring Methodology

### Metric: Relative Human Action Efficiency (RHAE)
Measures action efficiency per level relative to human performance, normalized across all games.

### Two Scoring Dimensions
1. **Completion** - Number of levels finished per game
2. **Efficiency** - Action count relative to human benchmarks

### Action Definition
An action is a discrete interaction that alters the environment. Internal operations (tool calls, reasoning steps, retries) are NOT counted.

### Human Baseline
- Derived from first-time players encountering each game
- The **2nd best performer** (fewest actions) per game establishes the reference point

### Level Scoring Formula
```
level_score = (human_baseline_actions / ai_actions)^2
```
- Maximum per-level score is capped at 1.0
- Squaring penalizes inefficiency heavily

### Game Aggregation (Weighted Average)
```
game_score = SUM(level_score * level_number) / SUM(level_numbers)
```
Later (harder) levels are weighted more heavily than introductory ones.

### Overall Score
Average of all game scores, producing a 0-100% range.

---

## 4. API Documentation

### Base URL
```
https://three.arcprize.org
```

### Authentication
- **Header:** `X-API-Key` (required for all endpoints)
- Obtained from https://arcprize.org/platform (sign in with Google or GitHub)
- **Environment Variable:** `ARC_API_KEY`

### Session Management (Critical)
Games are **stateful and require session affinity**. The API uses cookies (especially `AWSALB*`) to maintain game state. These cookies MUST be preserved and resent with all subsequent requests to the same game session.

### Rate Limits
- **600 requests per minute (RPM)**
- Exceeded: HTTP `429` with `{"error":"RATE_LIMIT_EXCEEDED","message":"rate limit has been exceeded"}`
- Use **exponential backoff** for retries
- No formal SLAs during research preview
- Contact team@arcprize.org for higher limits

### Endpoints

#### List Games
```
GET /api/games
Headers: X-API-Key: <key>

Response 200: Array of Game objects
  - game_id (string): e.g., "ls20-016295f7601e"
  - title (string): e.g., "LS20"

Response 401: Invalid API key
```

#### Start/Reset Game (RESET)
```
POST /api/cmd/RESET
Headers: X-API-Key: <key>
Content-Type: application/json

Body (ResetCommand):
{
  "game_id": "ls20-016295f7601e",  // required
  "card_id": "scorecard-uuid",     // required
  "guid": null                     // null = new game, string = reset existing
}

Response 200: FrameResponse (see format above)
Response 400: Invalid game_id, guid, or malformed request
Response 401: Missing/invalid API key
```

**Reset Behavior:**
- `guid` omitted or `null` -> creates new game instance
- `guid` provided + actions issued since level transition -> resets current level only
- Two consecutive RESETs guarantee complete reset

#### Execute Simple Action (ACTION1-ACTION5, ACTION7)
```
POST /api/cmd/ACTION1  (through ACTION5, ACTION7)
Headers: X-API-Key: <key>
Content-Type: application/json
Cookies: AWSALB* (from prior RESET/ACTION response)

Body:
{
  "game_id": "string",     // required
  "guid": "string",        // required (from RESET response)
  "reasoning": {}          // optional, <= 16KB JSON blob
}

Response 200: FrameResponse
Response 400: Invalid game_id, guid, or malformed reasoning
Response 401: Missing/invalid API key
```

#### Execute Complex Action (ACTION6 - Coordinate-Based)
```
POST /api/cmd/ACTION6
Headers: X-API-Key: <key>
Content-Type: application/json
Cookies: AWSALB* (from prior response)

Body:
{
  "game_id": "string",     // required
  "guid": "string",        // required
  "x": integer,            // required, 0-63
  "y": integer,            // required, 0-63
  "reasoning": {}          // optional, <= 16KB JSON blob
}

Response 200: FrameResponse
Response 400: Invalid game_id, guid, or coordinates out of range
Response 401: Missing/invalid API key
```

#### Open Scorecard
```
POST /api/scorecard/open
Headers: X-API-Key: <key>
Content-Type: application/json

Body (all fields optional):
{
  "source_url": "string (URI)",
  "tags": ["string"],
  "opaque": {}              // <= 16KB free-form JSON
}

Response 200: { "card_id": "uuid-string" }
Response 401: Invalid API key
```

#### Retrieve Scorecard
```
GET /api/scorecard/{card_id}
Headers: X-API-Key: <key>

Response 200: ScorecardSummary
{
  "card_id": "string",
  "score": integer,
  "source_url": "string",
  "tags": ["string"],
  "user_name": "string",
  "user_id": "string",
  "open_at": "datetime",
  "last_update": "datetime",
  "published_at": "datetime",
  "opaque": {},
  "total_environments": integer,
  "total_environments_completed": integer,
  "total_levels": integer,
  "total_levels_completed": integer,
  "total_actions": integer,
  "environments": [EnvironmentSummary],
  "tags_scores": [TagScore]
}

Response 401: Invalid API key
Response 404: Scorecard not found
```

#### Close Scorecard
```
POST /api/scorecard/close
Headers: X-API-Key: <key>
Content-Type: application/json

Body:
{
  "card_id": "string"
}

Response 200: ScorecardSummary (final results)
Response 401: Invalid API key
Response 404: Scorecard not found
```

### OpenAPI Spec
Available at:
- `https://docs.arcprize.org/arc3v1.yaml`
- `https://docs.arcprize.org/api-reference/openapi.json`

---

## 5. SDK / Client Libraries

### Python Toolkit (`arc-agi`)

**Installation:**
```bash
uv add arc-agi
# or
pip install arc-agi
```

**Core Class: `Arcade`**
```python
import arc_agi
from arcengine import GameAction, GameState

arc = arc_agi.Arcade(
    operation_mode=OperationMode.NORMAL,  # NORMAL | OFFLINE | ONLINE
    arc_api_key="your-key",               # or ARC_API_KEY env var
    environments_dir="environment_files",  # local game files
    recordings_dir="recordings",           # recording output
    arc_base_url="https://three.arcprize.org",  # API base URL
)
```

**Operation Modes:**

| Mode | Source | Rate Limits | Scorecards | Speed |
|------|--------|-------------|------------|-------|
| NORMAL (default) | Local + Remote | Yes (API) | Yes | Mixed |
| OFFLINE | Local only | None | No | ~2,000 FPS |
| ONLINE | Remote only | 600 RPM | Yes | API-bound |

**Key Methods:**
```python
# List available games
games = arc.get_environments()  # Returns list[EnvironmentInfo]

# Create environment
env = arc.make("ls20", render_mode="terminal")

# Game loop
obs = env.reset()
obs = env.step(GameAction.ACTION1)
obs = env.step(GameAction.ACTION6, data={"x": 10, "y": 20})

# Scorecards
card_id = arc.create_scorecard(tags=["experiment-1"])
scorecard = arc.get_scorecard(card_id)
arc.close_scorecard(card_id)
```

**EnvironmentWrapper Methods:**
- `reset()` - Reinitialize, returns FrameDataRaw
- `step(action, data=None, reasoning=None)` - Execute action, returns FrameDataRaw
- `observation_space` - Current game state (FrameDataRaw)
- `action_space` - List of available GameAction objects
- `info` - EnvironmentInfo (game_id, title, tags)

### Minimal Agent Example
```python
import random
from arcengine import GameAction, GameState
import arc_agi

arc = arc_agi.Arcade()
env = arc.make("ls20", render_mode="terminal")

for step in range(100):
    action = random.choice(env.action_space)
    action_data = {}
    if action.is_complex():
        action_data = {"x": random.randint(0, 63), "y": random.randint(0, 63)}

    obs = env.step(action, data=action_data)

    if obs and obs.state == GameState.WIN:
        print(f"Game won at step {step}!")
        break
    elif obs and obs.state == GameState.GAME_OVER:
        env.reset()

scorecard = arc.get_scorecard()
if scorecard:
    print(f"Final Score: {scorecard.score}")
```

### Agent Framework (ARC-AGI-3-Agents Repository)

**Repository:** https://github.com/arcprize/ARC-AGI-3-Agents

**Setup:**
```bash
git clone https://github.com/arcprize/ARC-AGI-3-Agents.git
cd ARC-AGI-3-Agents
uv sync
```

**Running Agents:**
```bash
uv run main.py --agent=random --game=ls20
uv run main.py --agent=llm --game=ls20
uv run main.py --agent=llm  # all games (swarm mode)
```

**Custom Agent Structure:**
Agents inherit from `Agent` base class and implement two methods:

```python
from arcengine import GameAction, GameState
from agents import Agent, FrameData

class MyAgent(Agent):
    def is_done(self, frame_history, latest_frame):
        return latest_frame.state is GameState.WIN

    def choose_action(self, frame_history, current_frame):
        action = GameAction.ACTION1
        action.reasoning = "Moving up to explore"
        return action
        # For complex actions:
        # action = GameAction.ACTION6
        # action.set_data({"x": 10, "y": 20})
        # return action
```

**Registration:** Add to `agents/__init__.py` and `AVAILABLE_AGENTS` dict.

### Built-in LLM Agent Templates
| Agent | Model | Description | Command |
|-------|-------|-------------|---------|
| `llm` | gpt-4o-mini | Standard OpenAI function-calling agent, 10-msg history | `--agent=llm` |
| `fastllm` | gpt-4o-mini | Skips observation for speed | `--agent=fastllm` |
| `reasoningllm` | o4-mini | Captures detailed reasoning tokens | `--agent=reasoningllm` |
| `guidedllm` | o3 | Game-specific rules (educational only, won't generalize) | `--agent=guidedllm` |

### Partner Integration Templates
- **Anthropic (Claude):** https://github.com/ThariqS/ARC-AGI-3-ClaudeCode-SDK (Node.js SDK, includes auto-solver `play-arc-with-claude.js`)
- **LangChain:** Template at `/partner_templates/langchain`
- **HuggingFace:** Template at `/partner_templates/huggingface`
- **AgentOps:** Template at `/partner_templates/agentops`

### Anthropic/Claude SDK (Node.js)
```bash
git clone https://github.com/ThariqS/ARC-AGI-3-ClaudeCode-SDK.git
cd ARC-AGI-3-ClaudeCode-SDK && npm install
node init.js --api-key YOUR_ARC_API_KEY
node play-arc-with-claude.js                    # auto-solver
node play-arc-with-claude.js ls20-016295f7601e 100  # specific game, 100 turn limit
```

### Benchmarking Harness (`arcagi3`)
- Compare model versions and prompt strategies
- Identify performance regressions
- Generate official scorecards
- Supports providers: OpenAI, Anthropic, Google Gemini, OpenRouter, Fireworks, Groq, DeepSeek, HuggingFace
- Results saved at https://arcprize.org/scorecards

### Swarm System (Multi-Agent)
Run agents across multiple games concurrently:
```bash
uv run main.py --agent <agent_name> [--game <filter>] [--tags <tags>]
uv run main.py --agent llm --game ls20,ft09 --tags experiment-v2
```
- Creates one agent instance per game
- Runs all agents concurrently using threads
- Automatically manages scorecards and cleanup
- Generates replay links

---

## 6. Recordings & Replays

- **API games:** Viewable at `https://arcprize.org/scorecards/<scorecard_id>`
- **Swarm games:** Saved locally as JSONL in `recordings/` directory
- **Local toolkit games:** No recordings generated
- **File format:** JSONL with timestamped entries (ISO 8601), containing frame data, game state, score, action input, and reasoning
- **Filename pattern:** `{game_id}.{agent_type}.{max_actions}.{guid}.recording.jsonl`

---

## 7. Key Vocabulary

| Term | Definition |
|------|-----------|
| **ARC-AGI** | Abstraction and Reasoning Corpus for Artificial General Intelligence |
| **ARC-AGI-3** | Latest interactive version with game environments |
| **Arcade** | Main entry point class (`arc_agi.Arcade()`) for configuration, environment discovery, and scorecard management |
| **Environment** | Interactive game instance created via `arc.make("game_id")` |
| **Swarm** | System for managing multiple agents across multiple games concurrently |
| **Toolkit** | Open-source Python SDK (`arc-agi` package) for interacting with ARC-AGI-3 |
| **RHAE** | Relative Human Action Efficiency - the scoring metric |
| **FrameDataRaw** | Observation data returned from step/reset containing grid, state, actions |

---

## 8. Key URLs

| Resource | URL |
|----------|-----|
| Documentation | https://docs.arcprize.org/ |
| Docs Index (llms.txt) | https://docs.arcprize.org/llms.txt |
| API Base URL | https://three.arcprize.org |
| OpenAPI Spec (YAML) | https://docs.arcprize.org/arc3v1.yaml |
| OpenAPI Spec (JSON) | https://docs.arcprize.org/api-reference/openapi.json |
| API Key Registration | https://arcprize.org/platform |
| Game Browser | https://arcprize.org/tasks |
| Scorecards | https://arcprize.org/scorecards |
| Competition 2026 | https://arcprize.org/competitions/2026 |
| Toolkit Repository | https://github.com/arcprize/arc-agi |
| Agent Repository | https://github.com/arcprize/ARC-AGI-3-Agents |
| Claude SDK | https://github.com/ThariqS/ARC-AGI-3-ClaudeCode-SDK |
| Rate Limit Increase | team@arcprize.org (subject: "Increase Rate Limits") |
