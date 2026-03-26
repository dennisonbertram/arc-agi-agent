# ARC-AGI-3 Reference Repository Analysis

**Date:** 2026-03-25
**Repos analyzed:**
- `arcprize/ARC-AGI-3-Agents` (Python, official agent framework)
- `ThariqS/ARC-AGI-3-ClaudeCode-SDK` (Node.js, Claude Code SDK approach)

---

## 1. ARC-AGI-3-Agents (Python)

### Project Structure

```
ARC-AGI-3-Agents/
  main.py                          # Entry point - CLI arg parsing, swarm setup
  pyproject.toml                   # Dependencies (Python 3.12+)
  agents/
    __init__.py                    # Agent registry (AVAILABLE_AGENTS dict)
    agent.py                       # Base Agent ABC + Playback class
    swarm.py                       # Multi-agent orchestration
    recorder.py                    # JSONL recording/playback
    tracing.py                     # AgentOps observability integration
    templates/
      random_agent.py              # Random action agent (simplest)
      llm_agents.py                # LLM, FastLLM, ReasoningLLM, GuidedLLM
      reasoning_agent.py           # ReasoningAgent (multimodal, hypothesis-driven)
      multimodal.py                # MultiModalLLM (image-based reasoning)
      smolagents.py                # HuggingFace smolagents integration
      langgraph_functional_agent.py
      langgraph_random_agent.py
      langgraph_thinking/          # LangGraph thinking agent
```

### Dependencies

```toml
requires-python = ">=3.12"
dependencies = [
    "arc-agi>=0.9.1",          # Official ARC environment SDK
    "dotenv>=0.9.9",
    "langchain[openai]>=0.3.27",
    "langgraph>=0.6.3",
    "numpy>=2.3.2",
    "openai==1.72.0",
    "pillow>=11.2.1",
    "pydantic>=2.11.7",
    "requests>=2.32.4",
    "smolagents>=1.20.0",
]
```

**Key dependency:** `arc-agi` (>= 0.9.1) provides `EnvironmentWrapper`, `Arcade`, `OperationMode`, `EnvironmentScorecard`. The `arcengine` package provides `FrameData`, `FrameDataRaw`, `GameAction`, `GameState`.

**Note:** `arc-agi` package is NOT currently installed in this environment.

### Base Agent Class (`agents/agent.py`)

The abstract base class defines the core game loop:

```python
class Agent(ABC):
    MAX_ACTIONS: int = 80

    def __init__(self, card_id, game_id, agent_name, ROOT_URL, record, arc_env, tags=None):
        # arc_env is an EnvironmentWrapper from arc_agi
        self.arc_env = arc_env
        self.frames = [FrameData(levels_completed=0)]

    @trace_agent_session
    def main(self) -> None:
        """The main agent loop."""
        self.timer = time.time()
        while (
            not self.is_done(self.frames, self.frames[-1])
            and self.action_counter <= self.MAX_ACTIONS
        ):
            action = self.choose_action(self.frames, self._convert_raw_frame_data(...))
            if frame := self.take_action(action):
                self.append_frame(frame)
            self.action_counter += 1
        self.cleanup()

    @abstractmethod
    def is_done(self, frames: list[FrameData], latest_frame: FrameData) -> bool:
        raise NotImplementedError

    @abstractmethod
    def choose_action(self, frames: list[FrameData], latest_frame: FrameData) -> GameAction:
        raise NotImplementedError
```

**Key abstractions:**
- `FrameData` - Contains `frame` (list of 2D grids), `state` (GameState enum), `levels_completed`, `guid`, `available_actions`
- `GameAction` - Enum with `RESET`, `ACTION1`-`ACTION7`. Has `is_simple()`, `is_complex()`, `set_data()`, `reasoning` field
- `GameState` - Enum: `NOT_PLAYED`, `NOT_FINISHED`, `WIN`, `GAME_OVER`
- Actions interact via `arc_env.step(action, data, reasoning)` which returns `FrameDataRaw`

### Swarm System (`agents/swarm.py`)

Orchestrates multiple agents playing multiple games in parallel using threads:

```python
class Swarm:
    def __init__(self, agent: str, ROOT_URL: str, games: list[str], tags=[]):
        self._arc = Arcade()  # from arc_agi

    def main(self) -> EnvironmentScorecard | None:
        # 1. Open scorecard
        self.card_id = self.open_scorecard()

        # 2. Create one agent per game
        for g in self.GAMES:
            a = self.agent_class(
                card_id=self.card_id,
                game_id=g,
                arc_env=self._arc.make(g, scorecard_id=self.card_id),
                ...
            )
            self.agents.append(a)

        # 3. Run each agent in its own thread
        for a in self.agents:
            self.threads.append(Thread(target=a.main, daemon=True))
        for t in self.threads:
            t.start()
        for t in self.threads:
            t.join()

        # 4. Close scorecard and report
        scorecard = self.close_scorecard(card_id)
```

**Pattern:** `Arcade` creates environments, `Swarm` manages threads, each `Agent` runs independently.

### Random Agent (`agents/templates/random_agent.py`)

Simplest possible agent -- good reference for the minimum interface:

```python
class Random(Agent):
    MAX_ACTIONS = 80

    def is_done(self, frames, latest_frame) -> bool:
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        if latest_frame.state in [GameState.NOT_PLAYED, GameState.GAME_OVER]:
            action = GameAction.RESET
        else:
            action = random.choice([a for a in GameAction if a is not GameAction.RESET])

        if action.is_simple():
            action.reasoning = f"RNG told me to pick {action.value}"
        elif action.is_complex():
            action.set_data({"x": random.randint(0, 63), "y": random.randint(0, 63)})
            action.reasoning = {"desired_action": f"{action.value}", "my_reason": "RNG said so!"}
        return action
```

### LLM Agent (`agents/templates/llm_agents.py`)

Uses OpenAI chat completions with function calling:

```python
class LLM(Agent):
    MAX_ACTIONS = 80
    DO_OBSERVATION = True
    MODEL = "gpt-4o-mini"
    MESSAGE_LIMIT = 10
    MODEL_REQUIRES_TOOLS = False
```

**Key patterns:**
- Exposes game actions as OpenAI functions/tools (RESET, ACTION1-ACTION6)
- ACTION6 takes x,y coordinates (0-63)
- Two-phase per turn: (1) observation step where LLM comments on frame, (2) action selection via function calling
- FIFO message buffer with `MESSAGE_LIMIT` to control context window
- Frame data sent as text grid representation via `pretty_print_3d()`
- Prompt tells LLM: "You are an agent playing a dynamic game. Your objective is to WIN and avoid GAME_OVER while minimizing actions."

**LLM Variants:**
- `FastLLM` - Skips observation step, uses gpt-4o-mini
- `ReasoningLLM` - Uses o4-mini with reasoning token tracking
- `GuidedLLM` - Uses o3 with high reasoning effort + explicit game rules in prompt (LockSmith-specific)

### ReasoningAgent (`agents/templates/reasoning_agent.py`)

Most sophisticated agent -- combines multimodal vision with structured hypothesis tracking:

```python
class ReasoningAgent(ReasoningLLM):
    MAX_ACTIONS = 400
    MODEL = "o4-mini"
    REASONING_EFFORT = "high"
    ZONE_SIZE = 16
```

**Key patterns:**
- Generates PIL images from grid data with color mapping and zone coordinates
- Sends both previous and current screen images to LLM
- Uses structured output via Pydantic model:
  ```python
  class ReasoningActionResponse(BaseModel):
      name: Literal["ACTION1", "ACTION2", "ACTION3", "ACTION4", "RESET"]
      reason: str
      short_description: str
      hypothesis: str           # Current hypothesis about game mechanics
      aggregated_findings: str  # Summary of discoveries so far
  ```
- Maintains `screen_history` (last 10 screens) and `history` of action responses
- Clears history on level transitions (`full_reset`)
- Prompt instructs: "Determine the game rules based on how the game reacted to the previous action"

### MultiModalLLM Agent (`agents/templates/multimodal.py`)

A self-programming multimodal agent:

```python
class MultiModalLLM(Agent):
    MAX_ACTIONS = 40
    MODEL = "gpt-4o-mini"
```

**Key patterns:**
- Converts grids to 128x128 RGBA images using a 16-color palette
- Three-phase per turn: (1) Analyze previous action outcome, (2) Choose human-like action, (3) Convert to game action
- **Self-modifying memory**: Analysis response contains a `---` separator; text after it replaces `_memory_prompt` for future turns
- Memory prompt tracks: Known inputs, Current goal, Game rules, Action log
- Image diff highlighting (red pixels for changes between frames)
- Human-action translation: LLM first describes action in human terms ("Click on the red square"), then a second LLM call maps it to ACTION1-6

### 16-Color Palette (used across agents)

```python
# 0: White, 1: Off-white, 2: Neutral light, 3: Neutral, 4: Off-black, 5: Black
# 6: Magenta, 7: Magenta light, 8: Red, 9: Blue, 10: Blue light
# 11: Yellow, 12: Orange, 13: Maroon, 14: Green, 15: Purple
```

### Entry Point (`main.py`)

```bash
uv run main.py --agent=random --game=locksmith
# or
uv run main.py --agent=llm --game=ls20-016295f7601e --tags=experiment,v1
```

**Flow:**
1. Parse args (agent name, game filter, tags)
2. Fetch game list from API (`GET /api/games`)
3. Initialize AgentOps tracing
4. Create `Swarm` with selected agent class and games
5. Run swarm in daemon thread with SIGINT handler for cleanup

### API Communication

The `arc_agi` package abstracts the API. Key interactions:
- `Arcade()` - Creates the environment manager
- `arcade.make(game_id, scorecard_id)` - Creates an `EnvironmentWrapper` for a game
- `env.step(action, data, reasoning)` - Submits action, returns `FrameDataRaw`
- `arcade.open_scorecard(tags)` / `arcade.close_scorecard(card_id)` - Scorecard lifecycle
- Supports both `ONLINE` and offline `OperationMode`
- API base URL: configurable, default `http://localhost:8001`
- Auth: `X-API-Key` header from `ARC_API_KEY` env var

---

## 2. ARC-AGI-3-ClaudeCode-SDK (Node.js)

### Project Structure

```
ARC-AGI-3-ClaudeCode-SDK/
  package.json                # Dependencies: @anthropic-ai/claude-code, chalk, commander
  CLAUDE.MD                   # Prompt/instructions for Claude Code
  play-arc-with-claude.js     # Claude Code SDK integration
  init.js                     # API key configuration
  serve-visualizer.js         # HTML visualizer server
  visualizer.html             # Game visualization UI
  utils.js                    # API helpers, frame saving
  actions/
    action.js                 # Execute ACTION1-6
    start-game.js             # Start a game session
    reset-game.js             # Reset current game
    list-games.js             # List available games
    status.js                 # Check game status
    open-scorecard.js         # Open scorecard
    close-scorecard.js        # Close scorecard
    get-scorecard.js          # View scorecard
  helpers/
    frame-analysis.js         # Frame comparison utilities
    grid-analysis.js          # Pattern detection, color analysis
    grid-visualization.js     # ASCII grid rendering
  notes/                      # Claude's notes during gameplay
```

### How It Works

This repo takes a fundamentally different approach: it uses the **Claude Code SDK** (`@anthropic-ai/claude-code`) to give Claude autonomous tool use for playing ARC games.

**`play-arc-with-claude.js`:**
```javascript
import { query } from "@anthropic-ai/claude-code";

async function playArcWithClaude(gameName, maxTurns = 100) {
  const initialPrompt = `Play the ARC AGI 3 game "${gameName}". Read CLAUDE.md to understand how to play. Keep playing until you win or reach ${maxTurns} turns.`;

  for await (const message of query({
    prompt: initialPrompt,
    abortController: new AbortController(),
    options: {
      maxTurns: maxTurns,
      cwd: __dirname,  // Claude can access all files in this directory
    },
  })) {
    // Stream and log messages (text, tool_use, tool_result)
  }
}
```

**Key insight:** Claude Code gets access to the Node.js CLI scripts as "tools" via the filesystem. The `CLAUDE.MD` file teaches Claude how to play by documenting the available scripts and game mechanics.

### CLAUDE.MD Prompt Structure

The prompt teaches Claude:
1. **What ARC-AGI-3 is** - Grid puzzles, 64x64 cells, values 0-15, states (NOT_PLAYED -> WIN/GAME_OVER)
2. **Available CLI commands** - `node actions/list-games.js`, `node actions/action.js --type 1`, etc.
3. **Game mechanics** - Actions 1-5 are simple (directional + enter), Action 6 is click with x,y
4. **Helper utilities** - Frame analysis, grid pattern detection, visualization tools
5. **Script writing** - Claude can write custom analysis scripts in `helpers/` or `games/[id]/scripts/`
6. **Note-taking** - Claude writes observations to `notes/` folder

### API Layer (`utils.js`)

```javascript
export async function makeRequest(endpoint, options = {}) {
  const config = await getConfig();
  const response = await fetch(`${config.baseUrl}${endpoint}`, {
    ...options,
    headers: {
      'X-API-Key': config.apiKey,
      'Content-Type': 'application/json',
    }
  });
  return response.json();
}
```

- Base URL: `https://three.arcprize.org` (production API)
- Endpoints: `/api/games`, `/api/cmd/ACTION{1-6}`, `/api/cmd/RESET`, `/api/scorecards`
- Frame data saved to `games/[game-id]/frames/frame_XXXX.json`

### Action Execution (`actions/action.js`)

```bash
node actions/action.js --type 1                           # Simple action
node actions/action.js --type 6 --x 10 --y 20             # Complex action with coords
node actions/action.js --type 1 --reasoning '{"strategy": "explore"}'  # With reasoning
```

Each action returns frame data with `state`, `score`, `win_score`, and full grid data.

---

## 3. Key Architecture Patterns

### Pattern 1: Agent Loop
Both repos follow the same core loop:
1. **RESET** to start game
2. **Observe** frame (grid data, state, score)
3. **Decide** next action (abstract reasoning)
4. **Execute** action, get new frame
5. **Repeat** until WIN, GAME_OVER, or max actions

### Pattern 2: Action Space
- `RESET` - Start/restart game
- `ACTION1-4` - Directional movement (Up, Down, Left, Right)
- `ACTION5` - Enter/Spacebar/Delete
- `ACTION6` - Click at (x, y) coordinates (0-63 range)
- `ACTION7` - Undo (in some games)
- Actions carry a `reasoning` field for metadata/tracing

### Pattern 3: Frame Data
- Grid is 64x64, values 0-15 (16 colors)
- Frame can contain multiple grids (layers)
- States: `NOT_PLAYED` -> `NOT_FINISHED` -> `WIN` or `GAME_OVER`
- Score tracked as `levels_completed` (games have multiple levels)

### Pattern 4: Two Approaches to LLM Integration
1. **Python SDK approach**: LLM as function-calling agent, game actions exposed as tools, text-based grid representation
2. **Claude Code SDK approach**: LLM as autonomous coding agent, CLI scripts as tools, can write custom analysis code

### Pattern 5: Multimodal Enhancement
- Grid-to-image conversion with 16-color palette
- Image diffs to highlight changes between frames
- Zone-based coordinate labeling for spatial reasoning
- Both raw grid text AND rendered images sent to vision models

### Pattern 6: Memory and Self-Programming
- `ReasoningAgent`: Structured hypothesis tracking with Pydantic models
- `MultiModalLLM`: Self-modifying memory prompt (LLM updates its own context)
- `GuidedLLM`: Hard-coded game-specific rules in prompt
- Claude SDK: Note files on disk, custom scripts written during gameplay

### Pattern 7: Concurrency
- `Swarm` runs one agent per game in parallel threads
- Each agent is independent with its own environment wrapper
- Scorecard system tracks results across all games

---

## 4. API Reference Summary

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/games` | GET | List available games |
| `/api/cmd/RESET` | POST | Start/restart game |
| `/api/cmd/ACTION{1-7}` | POST | Execute game action |
| `/api/scorecards` | POST | Open new scorecard |
| `/api/scorecards/{id}` | GET | View scorecard |
| `/api/scorecards/{id}/close` | POST | Close scorecard |

**Request body:** `{ game_id, guid, x?, y?, reasoning? }`
**Response:** `{ game_id, frame, state, score, win_score, levels_completed, guid, available_actions }`

---

## 5. Key Takeaways for Our Agent

1. **Must implement `is_done()` and `choose_action()` as the core interface**
2. **The `arc-agi` package handles environment communication** - install it or reimplement the HTTP API layer
3. **Frame grids are 64x64 with values 0-15** - both text and image representations are useful
4. **Hypothesis-driven exploration** (ReasoningAgent pattern) is the most sophisticated approach
5. **Self-modifying memory** (MultiModalLLM pattern) allows the agent to learn and adapt its strategy
6. **The Claude Code SDK approach** is uniquely powerful because Claude can write custom analysis scripts during gameplay
7. **Game-specific knowledge** (GuidedLLM) dramatically improves performance but requires per-game prompts
8. **MAX_ACTIONS varies**: 40-400 depending on agent complexity and game difficulty
