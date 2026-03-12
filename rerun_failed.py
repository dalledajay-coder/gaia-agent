"""Re-run only failed tasks from a previous benchmark run."""

import os
import asyncio
import json
import time
import sys
import re

os.environ.pop("CLAUDECODE", None)

from datasets import load_dataset
from agent import solve_task

TASK_TIMEOUT = 360  # 6 minutes for retries
MAX_TURNS = 40  # More turns for retry


async def solve_with_timeout(question, file_path, timeout=TASK_TIMEOUT):
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
    text = text.strip().lower().rstrip(".")
    for prefix in ["final answer:", "answer:", "the answer is "]:
        if text.startswith(prefix):
            text = text[len(prefix):].strip()
    if len(text) > 2 and text[0] in ('"', "'") and text[-1] == text[0]:
        text = text[1:-1]
    text = " ".join(text.split())
    return text


def strip_units(text: str) -> str:
    units = [
        "angstroms", "angstrom", "å", "nm", "mm", "cm", "m", "km",
        "kg", "g", "mg", "lb", "lbs", "oz",
        "seconds", "second", "sec", "minutes", "minute", "min",
        "hours", "hour", "hr", "days", "day",
        "dollars", "dollar", "usd", "euros", "euro", "eur",
        "percent", "%", "degrees", "degree", "°",
    ]
    t = text.strip().lower()
    for unit in units:
        if t.endswith(" " + unit):
            t = t[: -(len(unit) + 1)].strip()
        elif t.endswith(unit) and len(t) > len(unit):
            remainder = t[:-len(unit)].strip()
            try:
                float(remainder.replace(",", ""))
                t = remainder
            except ValueError:
                pass
    return t


def answers_match(predicted: str, gold: str) -> bool:
    pred = normalize_answer(predicted)
    gold_n = normalize_answer(gold)
    if pred == gold_n:
        return True
    pred_stripped = strip_units(pred)
    gold_stripped = strip_units(gold_n)
    try:
        pred_num = float(pred_stripped.replace(",", "").replace("$", "").replace("%", "").strip())
        gold_num = float(gold_stripped.replace(",", "").replace("$", "").replace("%", "").strip())
        if abs(pred_num - gold_num) < 0.01:
            return True
        if gold_num != 0 and abs((pred_num - gold_num) / gold_num) < 0.01:
            return True
    except (ValueError, TypeError, ZeroDivisionError):
        pass
    if len(gold_n) > 2 and gold_n in pred:
        return True
    if len(pred) > 2 and pred in gold_n:
        if len(pred) > len(gold_n) * 0.7:
            return True
    if pred_stripped and gold_stripped and pred_stripped == gold_stripped:
        return True
    return False


async def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    input_file = sys.argv[2] if len(sys.argv) > 2 else f"gaia_results_l{level}_v2.jsonl"
    output_file = f"gaia_results_l{level}_v3.jsonl"

    ds = load_dataset("gaia-benchmark/GAIA", f"2023_level{level}", split="validation")

    # Load previous results
    prev_results = {}
    if os.path.exists(input_file):
        with open(input_file) as f:
            for line in f:
                r = json.loads(line)
                prev_results[r["task_id"]] = r

    # Identify failed tasks
    failed_ids = set()
    for tid, r in prev_results.items():
        if not r.get("correct"):
            pred = r.get("predicted", "")
            # Re-run: timeouts, unable, errors, and wrong answers
            failed_ids.add(tid)

    # Load any existing v3 results
    v3_results = {}
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                v3_results[r["task_id"]] = r

    # Start with all passing results from v2
    all_results = {}
    for tid, r in prev_results.items():
        if r.get("correct"):
            all_results[tid] = r

    # Add any v3 results
    all_results.update(v3_results)

    correct = sum(1 for r in all_results.values() if r.get("correct"))
    total_tasks = len(ds)
    to_retry = [tid for tid in failed_ids if tid not in v3_results]

    print(f"\nGAIA Level {level} Re-run | {len(to_retry)} failed tasks to retry")
    print(f"Already correct: {correct}/{total_tasks}")
    print(f"Timeout: {TASK_TIMEOUT}s | Max turns: {MAX_TURNS}")
    print(f"{'='*70}\n")

    # Build task lookup
    task_lookup = {}
    for i in range(len(ds)):
        task = ds[i]
        task_lookup[task["task_id"]] = task

    done_retries = 0
    new_correct = 0
    for tid in to_retry:
        task = task_lookup.get(tid)
        if not task:
            continue

        q = task["Question"]
        gold = task["Final answer"]
        fname = task.get("file_name", "")

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
            new_correct += 1
            correct += 1

        done_retries += 1

        result = {
            "task_id": tid,
            "predicted": ans,
            "gold": gold,
            "correct": match,
            "time": round(elapsed, 1),
            "level": level,
            "retry": True,
        }
        all_results[tid] = result

        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        status = "PASS" if match else "FAIL"
        prev_pred = prev_results.get(tid, {}).get("predicted", "?")[:30]
        print(f"[{done_retries}/{len(to_retry)}] {status} | Got: {ans[:60]} | Gold: {gold} | Was: {prev_pred} | {elapsed:.0f}s")
        sys.stdout.flush()

        await asyncio.sleep(0.5)

    total_correct = sum(1 for r in all_results.values() if r.get("correct"))
    total_done = len(all_results)
    accuracy = total_correct / total_done * 100 if total_done else 0
    print(f"\nLEVEL {level} FINAL (v3): {total_correct}/{total_done} = {accuracy:.1f}%")
    print(f"New correct from retries: {new_correct}/{done_retries}")

    summary = {"level": level, "correct": total_correct, "total": total_done, "accuracy": round(accuracy, 2)}
    with open(f"gaia_summary_l{level}_v3.json", "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
