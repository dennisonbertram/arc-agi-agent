# Available Tools Investigation

**Date:** 2025-03-25

## agent-browser

**Status:** AVAILABLE

**Path:** `/Users/dennisonbertram/.nvm/versions/node/v22.21.1/bin/agent-browser`

**Skill file:** `~/.claude/skills/agent-browser.md` (exists, 26KB)

**Description:** Fast browser automation CLI for AI agents. Uses Chrome/Chromium via CDP directly. Supports headless mode, snapshots with element refs, authentication persistence, sessions, screenshots, and works in subagents.

**Core workflow:**
1. `agent-browser open <url>` -- navigate
2. `agent-browser snapshot -i` -- get element refs (@e1, @e2, etc.)
3. Interact using refs (click, fill, select)
4. Re-snapshot after DOM changes

**Key commands:** open, click, type, fill, press, hover, screenshot, pdf, snapshot, eval, connect, scroll, wait, find, get, is, mouse, drag, upload, download

**Command chaining:** Supports `&&` chaining between commands. Browser persists via background daemon.

---

## agent-mail (agentmail)

**Status:** AVAILABLE (CLI + MCP)

**Path:** `/Users/dennisonbertram/.local/bin/agent-mail`

**Skill file:** `~/.claude/skills/agentmail.md` (exists, ~3KB)

**Description:** Disposable email inbox management for testing, receiving verification codes, and email-based workflows. API key set via `AGENTMAIL_API_KEY` in `~/.zshrc`.

**Note:** Also available as MCP tools (`mcp__mcp-agent-mail__*`), but the CLI (`agentmail`) is the primary interface for subagents.

**Key capabilities:**
- Create/delete disposable inboxes (@agentmail.to)
- Send, receive, reply, forward emails
- Thread management
- Webhook support
- Domain management
- Pod-based inbox organization

**Command pattern:** `agentmail [resource] <command> [flags...]`

**Resources:** inboxes, inboxes:messages, inboxes:threads, inboxes:drafts, webhooks, domains, pods, pods:inboxes, api-keys

---

## Other Skills Available (~/.claude/skills/)

| Skill | Type |
|-------|------|
| agentic-hosting/ | directory |
| anti-slop-tweet-review.md | file |
| artcraft/ | directory |
| ask-questions-if-underspecified/ | directory |
| bird-twitter/ | directory |
| bluesky-api/ | directory |
| code-review/ | directory |
| codex-agent/ | directory |
| crusades-research.md | file |
| dev-browser/ | directory |
| docs-organizer/ | directory |
| dogfood -> symlink | directory |
| elevenlabs/ | directory |
| generate-crusades-episode.md | file |
| generate-podcast-script.md | file |
| heygen/ | directory |
| lgrep-search/ | directory |
| linkedin-api/ | directory |
| nano-banana/ | directory |
| react-best-practices/ | directory |
| realistic-ugc-video/ | directory |
| remotion-best-practices -> symlink | directory |
| setup-repo.md | file |
| steal-react-component/ | directory |
| swarm/ | directory |
| swarm-implementation.md | file |
| tally-podcraft/ | directory |
| tally-wallet/ | directory |
| tempo-development.md | file |
| tool-usability-test.md | file |
| ux-paths/ | directory |
| ux-tui-walker/ | directory |
| ux-walker/ | directory |
| verify-railway-deploy/ | directory |
| vestaboard/ | directory |
| video-editor/ | directory |
| walk-the-issues/ | directory |
| x-engagement-building.md | file |

---

## MCP Tools Also Available

- **claude-in-chrome** -- Browser automation via Chrome extension (MCP)
- **Gmail** -- Read/search Gmail messages, create drafts
- **Google Calendar** -- List/create/update calendar events
- **Pencil** -- Design tool for .pen files
- **Context7** -- Up-to-date library documentation lookup
- **Tally Wallet** -- Crypto wallet (send payments, check balances, trade tokens)
