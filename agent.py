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
from tools.audio_tool import transcribe_audio


SYSTEM_PROMPT = """You are an expert AI agent solving GAIA benchmark tasks. You MUST provide a precise, exact final answer for EVERY question. NEVER give up.

## ABSOLUTE RULES
1. EVERY question has an answer. NEVER say "UNABLE TO DETERMINE", "I cannot find", or "I'm not sure". These are FORBIDDEN responses.
2. ALWAYS end your FINAL message with: FINAL ANSWER: <exact answer>
3. A wrong guess is 100x better than no answer. ALWAYS GUESS if unsure.
4. Be PERSISTENT — try at least 3 different approaches before resorting to a guess.
5. Be CONCISE — don't waste turns on verbose reasoning.

## Strategy
1. ANALYZE: What type/format of answer is expected? (number, name, date, list, etc.)
2. PLAN: Which tools will get the answer fastest?
3. EXECUTE: Use tools. If one approach fails, IMMEDIATELY try alternatives.
4. VERIFY: Double-check your answer if possible.
5. ANSWER: State FINAL ANSWER: <exact answer>

## Tool Usage — Be Smart
- **Files first**: If a file is attached, ALWAYS read it IMMEDIATELY with read_file. For audio files, use transcribe_audio. For images, use analyze_image.
- **Web search**: Use specific, targeted queries. Try 2-3 different phrasings if first fails. ALWAYS use web_browse to visit the actual source pages — search snippets are often incomplete/wrong.
- **Wikipedia**: For factual/historical questions, use wikipedia_lookup with the exact topic name. More reliable than web search for established facts.
- **Code execution**: Use execute_python for ANY computation, data processing, counting, sorting, parsing, encoding/decoding. NEVER do math or counting in your head.
- **Bash**: Use for installing packages (pip install --break-system-packages), running commands, file operations, downloading files with wget/curl.
- **Vision**: Use analyze_image for any image understanding. Use execute_python with PIL for pixel-level analysis.
- **Audio**: Use transcribe_audio for speech-to-text. Use Bash with ffprobe for metadata.

## Critical Answer Format Rules
- ONLY the exact answer value. No explanations, no hedging, no "approximately".
- Numbers: exact value, NO UNITS unless the question explicitly asks for units.
- Names: full name as commonly known.
- Lists: comma-separated, in the order asked (alphabetical if not specified).
- Dates: use the format shown in the question, or MM/DD/YY if not specified.
- If asked "how many" → just the number (e.g., "42").
- If asked for a name → just the name (e.g., "Albert Einstein").
- NEVER include reasoning, explanations, or qualifiers in FINAL ANSWER.
- Remove trailing periods, quotes, or extra whitespace from answers.

## Common Task Patterns
- **"According to this file..."**: Read the file completely. Use execute_python with pandas for CSV/Excel analysis.
- **"What is the name/title/author..."**: Search web and Wikipedia. Visit actual source pages.
- **Counting tasks**: ALWAYS use execute_python. Never count manually. Even for "how many words in..."
- **Multi-hop research**: Break into sub-questions. Solve each with targeted searches.
- **Reversed/encoded text**: Use execute_python to decode. Never decode mentally.
- **Audio content questions**: Use transcribe_audio first, then analyze the transcript.
- **Image analysis**: Use analyze_image with a specific question about what to find.
- **Date/time calculations**: Use execute_python with datetime module.
- **Scientific/academic questions**: Check Wikipedia first, then search Google Scholar or specific databases.
- **Video game/pop culture**: Search multiple sources — Wikipedia, fandom wikis, etc.
- **Legal/government documents**: Browse specific government websites directly.

## MANDATORY Final Answer
Your LAST message MUST end with exactly:
FINAL ANSWER: <your exact answer>

If you cannot find the answer after exhausting all approaches, MAKE YOUR BEST EDUCATED GUESS.
Do NOT say "UNABLE TO DETERMINE". Do NOT say "I cannot find the answer."
ALWAYS provide FINAL ANSWER with your best guess.
"""


def create_gaia_tools_server():
    """Create MCP server with all GAIA-solving tools."""
    return create_sdk_mcp_server(
        name="gaia_tools",
        version="1.0.0",
        tools=[web_search, web_browse, execute_python, read_file, calculate, wikipedia_lookup, wikipedia_search, download_file, analyze_image, transcribe_audio],
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
        answer = lines[0].strip()
        # Remove surrounding quotes if present
        if len(answer) > 2 and answer[0] in ('"', "'") and answer[-1] == answer[0]:
            answer = answer[1:-1]
        return answer

    # Try "The answer is" pattern
    patterns = [
        r"(?:the (?:final )?answer is)[:\s]*(.+?)(?:\n|$)",
        r"(?:^answer:)\s*(.+?)(?:\n|$)",
        r"(?:result is|equals)\s*(.+?)(?:\.|$)",
    ]
    for pat in patterns:
        m = re.search(pat, text, re.IGNORECASE | re.MULTILINE)
        if m:
            return m.group(1).strip()

    # Filter out non-answer lines (reasoning, errors, tool calls)
    non_answer_prefixes = [
        "let me", "i found", "i need", "i'll", "i will", "searching",
        "api error", "error:", "browse error", "fatal error",
        "now let me", "trying", "checking", "unfortunately",
        "i'm going to", "i should", "i can", "i cannot", "i was unable",
        "looking at", "based on", "according to", "the search",
        "unable to", "i don't", "i couldn't", "i haven't",
        "you've hit", "limit", "no results",
    ]

    # Return last non-empty, non-reasoning line as fallback
    lines = [l.strip() for l in text.strip().split("\n") if l.strip()]
    for line in reversed(lines):
        lower = line.lower()
        if any(lower.startswith(p) for p in non_answer_prefixes):
            continue
        if len(line) > 200:  # Too long to be a concise answer
            continue
        return line

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
            "mcp__gaia_tools__transcribe_audio",
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

    # Try to extract any answer-like pattern from the combined text
    combined = "\n".join(all_text_blocks[-3:]) if all_text_blocks else ""
    extracted = extract_answer(combined)
    if extracted and len(extracted) < 200 and not extracted.lower().startswith(("let me", "i need", "searching", "i'll")):
        return extracted

    # Last resort: try to extract from last text block
    if last_text:
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
