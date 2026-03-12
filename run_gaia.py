"""Run GAIA benchmark with per-task timeout and resume support."""

import os
import asyncio
import json
import time
import sys
import re

os.environ.pop("CLAUDECODE", None)

from datasets import load_dataset
from agent import solve_task

TASK_TIMEOUT = 300  # 5 minutes max per task (was 3 min)
MAX_TURNS = 35  # More turns for complex tasks (was 20)


async def solve_with_timeout(question, file_path, timeout=TASK_TIMEOUT):
    """Solve a task with a timeout."""
    try:
        return await asyncio.wait_for(
            solve_task(question, file_path=file_path, max_turns=MAX_TURNS),
            timeout=timeout,
        )
    except asyncio.TimeoutError:
        return "TIMEOUT"
    except Exception as e:
        return f"ERROR: {e}"


def normalize_answer(text: str) -> str:
    """Normalize answer for comparison."""
    text = text.strip().lower().rstrip(".")
    # Remove common prefixes
    for prefix in ["final answer:", "answer:", "the answer is "]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    # Normalize whitespace
    text = " ".join(text.split())
    return text


def answers_match(predicted: str, gold: str) -> bool:
    """Flexible answer matching."""
    pred = normalize_answer(predicted)
    gold_n = normalize_answer(gold)

    # Exact match
    if pred == gold_n:
        return True

    # Numeric match
    try:
        pred_num = float(pred.replace(",", "").replace("$", "").replace("%", "").strip())
        gold_num = float(gold_n.replace(",", "").replace("$", "").replace("%", "").strip())
        if abs(pred_num - gold_num) < 0.01:
            return True
        if gold_num != 0 and abs((pred_num - gold_num) / gold_num) < 0.005:
            return True
    except (ValueError, TypeError, ZeroDivisionError):
        pass

    # Containment match (gold answer found within prediction)
    if len(gold_n) > 2 and gold_n in pred:
        return True

    # Reverse containment (prediction found within gold, for partial answers)
    if len(pred) > 2 and pred in gold_n:
        # Only if pred is substantial portion of gold
        if len(pred) > len(gold_n) * 0.7:
            return True

    return False


async def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    resume_from = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    output_file = f"gaia_results_l{level}_v2.jsonl"

    ds = load_dataset("gaia-benchmark/GAIA", f"2023_level{level}", split="validation")
    total = len(ds)

    # Load existing results
    existing = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                existing[r["task_id"]] = r

    correct = sum(1 for r in existing.values() if r.get("correct"))
    done = len(existing)

    print(f"\nGAIA Level {level} Benchmark v2 | {total} tasks | Resuming from {resume_from} | {done} already done")
    print(f"Timeout: {TASK_TIMEOUT}s | Max turns: {MAX_TURNS}")
    print(f"{'='*70}\n")

    for i in range(resume_from, total):
        task = ds[i]
        tid = task["task_id"]
        q = task["Question"]
        gold = task["Final answer"]
        fname = task.get("file_name", "")

        # Skip already completed
        if tid in existing:
            continue

        file_path = None
        if fname:
            candidate = f"/home/ubuntu/gaia-data/2023/validation/{fname}"
            if os.path.exists(candidate):
                file_path = candidate

        start = time.time()
        ans = await solve_with_timeout(q, file_path)
        elapsed = time.time() - start

        match = answers_match(ans, gold)

        if match:
            correct += 1
        done += 1

        result = {
            "task_id": tid,
            "predicted": ans,
            "gold": gold,
            "correct": match,
            "time": round(elapsed, 1),
            "level": level,
        }
        existing[tid] = result

        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        status = "PASS" if match else "FAIL"
        print(f"[{done}/{total}] {status} | Got: {ans[:80]} | Gold: {gold} | {elapsed:.0f}s | Running: {correct}/{done}={correct/done*100:.0f}%")
        sys.stdout.flush()

        await asyncio.sleep(0.5)

    accuracy = correct / done * 100 if done else 0
    print(f"\nLEVEL {level} FINAL: {correct}/{done} = {accuracy:.1f}%")

    summary = {"level": level, "correct": correct, "total": done, "accuracy": round(accuracy, 2)}
    with open(f"gaia_summary_l{level}_v2.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
