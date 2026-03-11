"""
GAIA Benchmark Runner

Loads the GAIA dataset, runs the agent on each task, and evaluates results.
Outputs results in JSONL format compatible with the GAIA leaderboard.
"""

import asyncio
import json
import os
import sys
import time
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from agent import solve_task


def normalize_answer(answer: str) -> str:
    """Normalize an answer for comparison."""
    answer = answer.strip()
    # Remove common prefixes
    for prefix in ["FINAL ANSWER:", "Answer:", "The answer is"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()

    # Normalize whitespace
    answer = " ".join(answer.split())

    # Remove trailing periods
    answer = answer.rstrip(".")

    return answer


def answers_match(predicted: str, gold: str) -> bool:
    """Check if predicted answer matches gold answer.

    Uses flexible matching: exact match, case-insensitive, numeric equivalence.
    """
    pred = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    # Exact match
    if pred == gold_norm:
        return True

    # Case-insensitive
    if pred.lower() == gold_norm.lower():
        return True

    # Numeric comparison
    try:
        pred_num = float(pred.replace(",", "").replace("$", "").replace("%", ""))
        gold_num = float(gold_norm.replace(",", "").replace("$", "").replace("%", ""))
        if abs(pred_num - gold_num) < 1e-6:
            return True
        # Relative tolerance for larger numbers
        if gold_num != 0 and abs((pred_num - gold_num) / gold_num) < 0.001:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    # Contained match (for short answers in longer responses)
    if len(gold_norm) > 2 and gold_norm.lower() in pred.lower():
        return True

    return False


async def load_gaia_dataset(data_dir: str | None = None, split: str = "validation", level: int | None = None) -> list[dict]:
    """Load GAIA dataset tasks.

    Args:
        data_dir: Path to downloaded GAIA dataset. If None, tries to download.
        split: Dataset split ('validation' or 'test')
        level: Filter by difficulty level (1, 2, or 3). None for all levels.

    Returns:
        List of task dictionaries
    """
    tasks = []

    # Try loading from local directory first
    if data_dir and os.path.exists(data_dir):
        return _load_from_local(data_dir, split, level)

    # Try loading from HuggingFace
    try:
        from datasets import load_dataset

        # Load all levels from validation split
        dataset_configs = []
        levels_to_load = [level] if level else [1, 2, 3]

        for lvl in levels_to_load:
            config_name = f"2023_level{lvl}"
            try:
                ds = load_dataset("gaia-benchmark/GAIA", config_name, split=split, trust_remote_code=True)
                for item in ds:
                    task = {
                        "task_id": item.get("task_id", ""),
                        "question": item.get("Question", ""),
                        "level": item.get("Level", lvl),
                        "final_answer": item.get("Final answer", ""),
                        "file_name": item.get("file_name", ""),
                        "file_path": item.get("file_path", ""),
                    }
                    tasks.append(task)
            except Exception as e:
                print(f"Warning: Could not load level {lvl}: {e}", file=sys.stderr)

        return tasks

    except ImportError:
        print("datasets library not available. Please provide a local data directory.", file=sys.stderr)
        return []


def _load_from_local(data_dir: str, split: str, level: int | None) -> list[dict]:
    """Load tasks from local GAIA dataset directory."""
    tasks = []

    # Look for metadata files
    for pattern in ["metadata.jsonl", "*.jsonl", "metadata.parquet"]:
        import glob
        for f in glob.glob(os.path.join(data_dir, "**", pattern), recursive=True):
            if split in f or "validation" in f or "test" in f:
                if f.endswith(".jsonl"):
                    with open(f) as fh:
                        for line in fh:
                            task = json.loads(line)
                            if level is None or task.get("Level") == level:
                                tasks.append({
                                    "task_id": task.get("task_id", ""),
                                    "question": task.get("Question", ""),
                                    "level": task.get("Level", 0),
                                    "final_answer": task.get("Final answer", ""),
                                    "file_name": task.get("file_name", ""),
                                    "file_path": task.get("file_path", ""),
                                })

    return tasks


async def run_benchmark(
    tasks: list[dict],
    output_file: str = "results.jsonl",
    max_tasks: int | None = None,
    resume_from: int = 0,
    data_dir: str | None = None,
) -> dict:
    """Run the agent on benchmark tasks and save results.

    Args:
        tasks: List of task dictionaries
        output_file: Path to save JSONL results
        max_tasks: Maximum number of tasks to run (None for all)
        resume_from: Index to resume from
        data_dir: Base directory for file attachments

    Returns:
        Summary statistics dictionary
    """
    if max_tasks:
        tasks = tasks[:max_tasks]

    total = len(tasks)
    correct = 0
    errors = 0
    results = []

    # Load existing results if resuming
    if resume_from > 0 and os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                results.append(json.loads(line))
                r = results[-1]
                if r.get("correct"):
                    correct += 1

    print(f"\n{'='*60}")
    print(f"GAIA Benchmark Run")
    print(f"Tasks: {total} | Resuming from: {resume_from}")
    print(f"{'='*60}\n")

    for i, task in enumerate(tasks):
        if i < resume_from:
            continue

        task_id = task["task_id"]
        question = task["question"]
        gold_answer = task["final_answer"]
        level = task.get("level", "?")

        print(f"\n[{i+1}/{total}] Level {level} | Task: {task_id}")
        print(f"Q: {question[:120]}...")

        # Resolve file path
        file_path = None
        if task.get("file_name"):
            if data_dir:
                candidate = os.path.join(data_dir, task.get("file_path", task["file_name"]))
                if os.path.exists(candidate):
                    file_path = candidate
            if not file_path:
                # Try common locations
                for base in [".", data_dir or ".", "data"]:
                    candidate = os.path.join(base, task["file_name"])
                    if os.path.exists(candidate):
                        file_path = candidate
                        break

        start_time = time.time()
        try:
            predicted = await solve_task(question, file_path, max_turns=25)
        except Exception as e:
            predicted = f"ERROR: {str(e)}"
            errors += 1
            print(f"  ERROR: {e}")

        elapsed = time.time() - start_time
        is_correct = answers_match(predicted, gold_answer)

        if is_correct:
            correct += 1

        result = {
            "task_id": task_id,
            "question": question,
            "level": level,
            "gold_answer": gold_answer,
            "predicted_answer": predicted,
            "correct": is_correct,
            "time_seconds": round(elapsed, 1),
        }
        results.append(result)

        # Save incrementally
        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        status = "CORRECT" if is_correct else "WRONG"
        print(f"  Predicted: {predicted}")
        print(f"  Gold:      {gold_answer}")
        print(f"  Status:    {status} ({elapsed:.1f}s)")
        print(f"  Running:   {correct}/{i+1-resume_from+correct} = {correct/(i+1-resume_from)*100:.1f}%")

    # Summary
    attempted = total - resume_from
    accuracy = (correct / attempted * 100) if attempted > 0 else 0

    summary = {
        "total_tasks": total,
        "attempted": attempted,
        "correct": correct,
        "errors": errors,
        "accuracy": round(accuracy, 2),
        "timestamp": datetime.now().isoformat(),
    }

    # Save summary
    summary_file = output_file.replace(".jsonl", "_summary.json")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Total tasks:  {total}")
    print(f"Attempted:    {attempted}")
    print(f"Correct:      {correct}")
    print(f"Errors:       {errors}")
    print(f"Accuracy:     {accuracy:.1f}%")
    print(f"Results:      {output_file}")
    print(f"Summary:      {summary_file}")
    print(f"{'='*60}\n")

    return summary


async def main():
    """Main entry point for benchmark runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Run GAIA Benchmark")
    parser.add_argument("--data-dir", type=str, help="Path to GAIA dataset directory")
    parser.add_argument("--split", type=str, default="validation", choices=["validation", "test"])
    parser.add_argument("--level", type=int, choices=[1, 2, 3], help="Filter by difficulty level")
    parser.add_argument("--max-tasks", type=int, help="Maximum number of tasks to run")
    parser.add_argument("--resume-from", type=int, default=0, help="Task index to resume from")
    parser.add_argument("--output", type=str, default="results.jsonl", help="Output file path")
    args = parser.parse_args()

    # Load dataset
    print("Loading GAIA dataset...")
    tasks = await load_gaia_dataset(args.data_dir, args.split, args.level)

    if not tasks:
        print("No tasks loaded. Check your data directory or HuggingFace credentials.")
        sys.exit(1)

    print(f"Loaded {len(tasks)} tasks")

    # Run benchmark
    summary = await run_benchmark(
        tasks=tasks,
        output_file=args.output,
        max_tasks=args.max_tasks,
        resume_from=args.resume_from,
        data_dir=args.data_dir,
    )


if __name__ == "__main__":
    asyncio.run(main())
