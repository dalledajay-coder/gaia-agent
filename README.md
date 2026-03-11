# GAIA Benchmark Agent

An autonomous AI agent built with the **Anthropic Agent SDK** (`claude-agent-sdk`) that solves complex multi-step tasks from the [GAIA benchmark](https://arxiv.org/abs/2311.12983).

## Architecture

The agent uses Claude via the Agent SDK with custom MCP tools for:

- **Web Search & Browse** - DuckDuckGo search and full page content extraction
- **Wikipedia** - Direct Wikipedia API lookup and search
- **Code Execution** - Python execution via E2B sandbox or local fallback
- **File Processing** - Text, CSV, JSON, Excel, PDF, DOCX, PPTX, images, audio
- **Image Analysis** - Multimodal vision via Anthropic API
- **Math** - Safe mathematical expression evaluation
- **File Download** - Fetch remote resources

## Setup

```bash
pip install -r requirements.txt

# Set your API keys
cp .env.example .env
# Edit .env with your ANTHROPIC_API_KEY and optionally E2B_API_KEY
```

## Usage

### Single question
```bash
python3 agent.py "What is the capital of France?"
python3 agent.py "What is the average in this file?" data/scores.csv
```

### Run GAIA benchmark
```bash
# Run all validation tasks
python3 benchmark.py --split validation

# Run specific level
python3 benchmark.py --split validation --level 1

# Run limited number of tasks
python3 benchmark.py --split validation --max-tasks 10

# Resume from a specific task
python3 benchmark.py --split validation --resume-from 50
```

## Tech Stack

- **Anthropic Agent SDK** (`claude-agent-sdk`) - Agent orchestration
- **Claude Sonnet 4.6** - Primary model
- **E2B Sandboxes** - Secure code execution (with local fallback)
- **Python 3.10+**

## Project Structure

```
gaia-agent/
├── agent.py           # Main agent with system prompt and task solving
├── benchmark.py       # GAIA benchmark runner and evaluator
├── requirements.txt   # Python dependencies
├── tools/
│   ├── web_search.py     # Web search and browsing
│   ├── wikipedia_tool.py # Wikipedia lookup and search
│   ├── code_execution.py # Python code execution (E2B + local)
│   ├── file_tools.py     # Multi-format file reading
│   ├── math_tools.py     # Mathematical calculations
│   ├── vision_tool.py    # Image analysis via Claude vision
│   └── download_tool.py  # File downloading
└── CLAUDE.md          # Project context for Claude Code
```
