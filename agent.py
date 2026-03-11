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


SYSTEM_PROMPT = """You are an expert AI assistant designed to solve complex, multi-step tasks from the GAIA benchmark. You must provide precise, exact answers.

CRITICAL RULES:
1. Your final answer must be EXACT - not approximate, not explained, just the precise answer.
2. When the question asks for a specific format (number, name, date, etc.), match that format exactly.
3. Use tools strategically: search the web, execute code, read files, and calculate as needed.
4. Break complex problems into steps. Think carefully before answering.
5. For numerical answers: give the exact number, no units unless asked.
6. For names: give the exact name as commonly known.
7. For dates: use the format requested or the most standard format.
8. Do NOT hedge or say "approximately". Give the definitive answer.
9. If a file is attached to the task, ALWAYS read it first using read_file.
10. For mathematical/computational questions, use the calculate tool or execute_python for verification.
11. When searching the web, try multiple search queries if the first doesn't yield results.
12. Cross-verify important facts from multiple sources when possible.

ANSWER FORMAT:
- Your final response must contain ONLY the answer on the last line
- Prefix your final answer with "FINAL ANSWER: " followed by just the answer
- Examples: "FINAL ANSWER: 42", "FINAL ANSWER: Paris", "FINAL ANSWER: 2024-01-15"
"""


def create_gaia_tools_server():
    """Create MCP server with all GAIA-solving tools."""
    return create_sdk_mcp_server(
        name="gaia_tools",
        version="1.0.0",
        tools=[web_search, web_browse, execute_python, read_file, calculate],
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
    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        last_text = block.text
                        if "FINAL ANSWER:" in block.text:
                            # Extract answer immediately
                            answer = block.text.split("FINAL ANSWER:")[-1].strip()
                            return answer
            elif isinstance(message, ResultMessage):
                break
    except Exception as e:
        print(f"Agent error: {e}", file=sys.stderr)

    # Try to extract answer from last text
    if "FINAL ANSWER:" in last_text:
        return last_text.split("FINAL ANSWER:")[-1].strip()

    # Return last text as fallback
    return last_text.strip().split("\n")[-1].strip() if last_text else "UNABLE TO DETERMINE"


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
