# GAIA Benchmark Agent

Built with Anthropic Agent SDK (claude-agent-sdk).

## Architecture
- `agent.py` - Main agent with system prompt and task solving logic
- `benchmark.py` - Benchmark runner that loads GAIA dataset and evaluates
- `tools/` - Custom MCP tools (web search, code execution, file handling, math)

## Running
- Single question: `python3 agent.py "What is 2+2?"`
- Benchmark: `python3 benchmark.py --split validation --level 1`

## Key Design Decisions
- Uses `query()` (not ClaudeSDKClient) since each task is independent
- Custom MCP tools for web search, code execution, file reading, math
- bypassPermissions mode for autonomous operation
- Flexible answer matching with normalization
