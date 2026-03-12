"""
GAIA Benchmark Agent - Built with Anthropic Agent SDK

An autonomous AI agent that solves complex multi-step tasks from the GAIA benchmark.
Uses web search, code execution, file processing, and mathematical reasoning.
"""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Prevent nested Claude Code session detection
os.environ.pop("CLAUDECODE", None)

from claude_agent_sdk import (
    query,
    ClaudeAgentOptions,
    AssistantMessage,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    tool,
    create_sdk_mcp_server,
)

from tools.web_search import web_search, web_browse
from tools.code_execution import execute_python
from tools.file_tools import read_file
from tools.math_tools import calculate
from tools.wikipedia_tool import wikipedia_lookup, wikipedia_search
from tools.download_tool import download_file
from tools.vision_tool import analyze_image


SYSTEM_PROMPT = """You are an expert AI assistant solving GAIA benchmark tasks. You MUST provide precise, exact answers.

## Strategy
1. ANALYZE the question carefully. Identify what type of answer is expected (number, name, date, list, etc.).
2. PLAN your approach: what tools do you need? What information do you need to find?
3. EXECUTE: Use tools systematically. If one approach fails, try another.
4. VERIFY: Double-check your answer before submitting.

## Tool Usage Guidelines
- **Files**: If a file is mentioned or attached, ALWAYS read it first with read_file or Read tool.
- **Web Search**: Use web_search for factual questions, current events, specific data lookups. Try multiple queries if first attempt fails.
- **Web Browse**: Use web_browse to get full content from specific URLs found via search.
- **Code Execution**: Use execute_python for calculations, data processing, parsing structured data, counting, sorting.
- **Calculate**: Use for simple math expressions. For complex computations, prefer execute_python.
- **Image Analysis**: Use analyze_image for questions about images, charts, diagrams, or text in images.
- **Download**: Use download_file to fetch files from URLs.
- **Bash**: Use for file system operations, running system commands, installing packages.

## Answer Rules
- Give ONLY the exact answer. No explanations, no qualifications, no "approximately".
- Numbers: exact value, no units unless the question asks for units. No commas in numbers unless they're part of the answer format.
- Names: full name as commonly known (e.g., "Albert Einstein" not "A. Einstein").
- Dates: use the format the question specifies, or YYYY-MM-DD if unspecified.
- Lists: comma-separated unless otherwise specified.
- If the question asks "how many", give just the number.
- If the question asks for a name, give just the name.
- Round numbers only if the question asks you to round.

## Common Pitfalls to Avoid
- Don't guess. If you can't find the answer, search more broadly.
- Don't confuse similar entities (e.g., cities with same name in different countries).
- Read the ENTIRE question - don't miss qualifiers like "as of 2023" or "in millions".
- For Wikipedia questions, browse the actual Wikipedia page rather than relying on search snippets.
- For file-based questions, make sure you process ALL the data, not just a sample.

## Final Answer Format
End your response with exactly:
FINAL ANSWER: <your answer here>

Examples:
FINAL ANSWER: 42
FINAL ANSWER: Marie Curie
FINAL ANSWER: 2024-01-15
FINAL ANSWER: hydrogen, helium, lithium
"""


def create_gaia_tools_server():
    """Create MCP server with all GAIA-solving tools."""
    return create_sdk_mcp_server(
        name="gaia_tools",
        version="1.0.0",
        tools=[web_search, web_browse, execute_python, read_file, calculate, wikipedia_lookup, wikipedia_search, download_file, analyze_image],
    )


async def solve_task(question: str, file_path: str | None = None, max_turns: int = 30) -> str:
    """Solve a single GAIA benchmark task.

    Args:
        question: The task question text
        file_path: Optional path to an attached file
        max_turns: Maximum number of agent turns

    Returns:
        The agent's final answer string
    """
    gaia_server = create_gaia_tools_server()

    # Build the prompt
    prompt_parts = []
    if file_path and os.path.exists(file_path):
        prompt_parts.append(
            f"An attachment file is provided at: {file_path}\n"
            f"You MUST read this file using the read_file tool before answering.\n\n"
        )
    prompt_parts.append(f"Question: {question}\n\n")
    prompt_parts.append(
        "Think step by step. Use tools as needed. "
        "End your response with 'FINAL ANSWER: <your exact answer>'"
    )

    prompt = "".join(prompt_parts)

    options = ClaudeAgentOptions(
        system_prompt=SYSTEM_PROMPT,
        mcp_servers={"gaia_tools": gaia_server},
        allowed_tools=[
            "mcp__gaia_tools__web_search",
            "mcp__gaia_tools__web_browse",
            "mcp__gaia_tools__execute_python",
            "mcp__gaia_tools__read_file",
            "mcp__gaia_tools__calculate",
            "mcp__gaia_tools__wikipedia_lookup",
            "mcp__gaia_tools__wikipedia_search",
            "mcp__gaia_tools__download_file",
            "mcp__gaia_tools__analyze_image",
            "Read",
            "Bash",
            "Glob",
            "Grep",
        ],
        permission_mode="bypassPermissions",
        max_turns=max_turns,
        model="claude-sonnet-4-6",
    )

    last_text = ""
    found_answer = ""
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        last_text = block.text
                        if "FINAL ANSWER:" in block.text and not found_answer:
                            found_answer = block.text.split("FINAL ANSWER:")[-1].strip()
            elif isinstance(message, ResultMessage):
                pass  # Let loop end naturally to avoid async cleanup issues
    except Exception as e:
        print(f"Agent error: {e}", file=sys.stderr)

    if found_answer:
        return found_answer

    # Try to extract answer from last text
    if "FINAL ANSWER:" in last_text:
        return last_text.split("FINAL ANSWER:")[-1].strip()

    # Return last text as fallback
    return last_text.strip().split("\n")[-1].strip() if last_text else "UNABLE TO DETERMINE"


async def solve_task_with_retry(question: str, file_path: str | None = None, max_retries: int = 2) -> str:
    """Solve a task with retry logic for robustness."""
    last_answer = ""
    for attempt in range(max_retries):
        try:
            answer = await solve_task(question, file_path, max_turns=25 + attempt * 5)
            if answer and answer != "UNABLE TO DETERMINE":
                return answer
            last_answer = answer
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}", file=sys.stderr)
            last_answer = f"ERROR: {e}"
            await asyncio.sleep(2)
    return last_answer


async def main():
    """Run the agent on a single question from command line."""
    if len(sys.argv) < 2:
        print("Usage: python agent.py '<question>' [file_path]")
        sys.exit(1)

    question = sys.argv[1]
    file_path = sys.argv[2] if len(sys.argv) > 2 else None

    print(f"Question: {question}")
    if file_path:
        print(f"File: {file_path}")
    print("---")

    answer = await solve_task(question, file_path)
    print(f"\nFINAL ANSWER: {answer}")


if __name__ == "__main__":
    asyncio.run(main())
