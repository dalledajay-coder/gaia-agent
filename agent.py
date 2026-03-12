"""
GAIA Benchmark Agent - Built with Anthropic Agent SDK

An autonomous AI agent that solves complex multi-step tasks from the GAIA benchmark.
Uses web search, code execution, file processing, and mathematical reasoning.
"""

import asyncio
import json
import os
import re
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


SYSTEM_PROMPT = """You are an expert AI agent solving GAIA benchmark tasks. You MUST provide a precise, exact final answer for EVERY question. NEVER give up.

## Core Principles
1. EVERY question has an answer. Your job is to find it. NEVER say "UNABLE TO DETERMINE" or "I cannot find".
2. Be PERSISTENT. If one approach fails, try 2-3 different approaches before giving up.
3. Be EFFICIENT. Don't waste turns on verbose reasoning. Act quickly and decisively.
4. ALWAYS end with FINAL ANSWER, even if you're not 100% sure — your best guess is better than no answer.

## Strategy
1. ANALYZE: What type of answer is expected? (number, name, date, list, etc.)
2. PLAN: Which tools will get the answer fastest?
3. EXECUTE: Use tools. If blocked, try alternatives immediately.
4. ANSWER: Give your best answer. Never leave a question unanswered.

## Tool Usage — Be Smart
- **Files first**: If a file is attached, ALWAYS read it immediately with read_file.
- **Web search**: Try specific, targeted queries. If DuckDuckGo fails, try different query phrasings. Use web_browse to visit promising URLs.
- **Wikipedia**: For factual/historical questions, use wikipedia_lookup with the exact topic. More reliable than web search for established facts.
- **Code execution**: Use execute_python for ANY computation, data processing, counting, parsing. Don't do math in your head.
- **Bash**: Use for installing packages (pip install), running commands, file operations. Very powerful.
- **Multiple attempts**: If a search returns no results, rephrase the query. Try at least 2-3 different search strategies.

## Critical Answer Rules
- ONLY the exact answer. No explanations, no hedging, no "approximately".
- Numbers: exact value. No units unless asked. No commas unless part of format.
- Names: full name as commonly known.
- Lists: comma-separated, alphabetical unless otherwise specified.
- If asked "how many" → just the number.
- If asked for a name → just the name.
- NEVER include your reasoning in the final answer.
- If you must guess, GUESS. A wrong answer is better than no answer.

## Avoiding Common Failures
- Don't say "I need more information" — USE YOUR TOOLS to find it.
- Don't return partial reasoning as your answer — extract just the answer.
- If you're running low on turns, give your best answer immediately.
- For complex multi-step problems: solve each step, verify with code execution.
- For web research: visit the actual source pages, don't rely on search snippets alone.
- For file questions: process the ENTIRE file, not just a preview.

## MANDATORY Final Answer Format
You MUST end EVERY response with exactly this format:
FINAL ANSWER: <your exact answer>

This is non-negotiable. Every response must end with FINAL ANSWER.
"""


def create_gaia_tools_server():
    """Create MCP server with all GAIA-solving tools."""
    return create_sdk_mcp_server(
        name="gaia_tools",
        version="1.0.0",
        tools=[web_search, web_browse, execute_python, read_file, calculate, wikipedia_lookup, wikipedia_search, download_file, analyze_image],
    )


def extract_answer(text: str) -> str:
    """Extract the final answer from agent text, handling various formats."""
    if not text:
        return ""

    # Try FINAL ANSWER: pattern
    if "FINAL ANSWER:" in text:
        answer = text.split("FINAL ANSWER:")[-1].strip()
        # Clean up: remove trailing explanation
        lines = answer.split("\n")
        return lines[0].strip()

    # Try "The answer is" pattern
    patterns = [
        r"(?:the answer is|answer:)\s*(.+?)(?:\.|$)",
        r"(?:result is|equals)\s*(.+?)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()

    # Return last non-empty line as fallback
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    return lines[-1] if lines else ""


async def solve_task(question: str, file_path: str | None = None, max_turns: int = 35) -> str:
    """Solve a single GAIA benchmark task."""
    gaia_server = create_gaia_tools_server()

    # Build the prompt
    prompt_parts = []
    if file_path and os.path.exists(file_path):
        ext = os.path.splitext(file_path)[1].lower()
        prompt_parts.append(
            f"IMPORTANT: A file is attached at: {file_path} (type: {ext})\n"
            f"You MUST read this file using the read_file tool FIRST before doing anything else.\n\n"
        )
    prompt_parts.append(f"Question: {question}\n\n")
    prompt_parts.append(
        "Solve this step by step. Use tools as needed. "
        "You MUST end with 'FINAL ANSWER: <your exact answer>'. "
        "Never give up — always provide your best answer."
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
            "Write",
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
    all_text_blocks = []

    try:
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock):
                        last_text = block.text
                        all_text_blocks.append(block.text)
                        if "FINAL ANSWER:" in block.text and not found_answer:
                            found_answer = extract_answer(block.text)
            elif isinstance(message, ResultMessage):
                pass  # Let loop end naturally to avoid async cleanup issues
    except Exception as e:
        print(f"Agent error: {e}", file=sys.stderr)

    if found_answer:
        return found_answer

    # Try to extract answer from all text blocks (agent may have answered earlier)
    for text in reversed(all_text_blocks):
        if "FINAL ANSWER:" in text:
            return extract_answer(text)

    # Try to extract any answer-like pattern from last text
    extracted = extract_answer(last_text)
    if extracted and len(extracted) < 200:
        return extracted

    return "UNABLE TO DETERMINE"


async def solve_task_with_retry(question: str, file_path: str | None = None, max_retries: int = 2) -> str:
    """Solve a task with retry logic for robustness."""
    last_answer = ""
    for attempt in range(max_retries):
        try:
            turns = 30 + attempt * 10
            answer = await solve_task(question, file_path, max_turns=turns)
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
