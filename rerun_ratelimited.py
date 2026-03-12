"""Re-run only rate-limited tasks from a previous benchmark run."""

import os
import asyncio
import json
import time
import sys
import re

os.environ.pop("CLAUDECODE", None)

from datasets import load_dataset
from agent import solve_task

TASK_TIMEOUT = 300
MAX_TURNS = 35


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


def is_rate_limited(predicted: str) -> bool:
    lower = predicted.lower()
    return "you've hit your limit" in lower or ("limit" in lower and "reset" in lower)


async def main():
    level = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    input_file = sys.argv[2] if len(sys.argv) > 2 else f"gaia_results_l{level}_v2.jsonl"
    output_file = f"gaia_results_l{level}_rl.jsonl"

    ds = load_dataset("gaia-benchmark/GAIA", f"2023_level{level}", split="validation")

    # Load previous results
    prev_results = {}
    with open(input_file) as f:
        for line in f:
            r = json.loads(line)
            prev_results[r["task_id"]] = r

    # Identify rate-limited tasks
    rl_ids = [tid for tid, r in prev_results.items() if is_rate_limited(r.get("predicted", ""))]

    # Load any existing rl results (for resume)
    done_ids = set()
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                done_ids.add(r["task_id"])

    to_run = [tid for tid in rl_ids if tid not in done_ids]

    print(f"\nGAIA Level {level} Rate-Limited Re-run | {len(to_run)} tasks to retry (of {len(rl_ids)} rate-limited)")
    print(f"Timeout: {TASK_TIMEOUT}s | Max turns: {MAX_TURNS}")
    print(f"{'='*70}\n")

    # Build task lookup
    task_lookup = {}
    for i in range(len(ds)):
        task = ds[i]
        task_lookup[task["task_id"]] = task

    done = 0
    correct = 0
    for tid in to_run:
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

        # If still rate limited, stop
        if is_rate_limited(ans):
            print(f"\nStill rate limited! Stopping. Completed {done} tasks.")
            break

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

        with open(output_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        status = "PASS" if match else "FAIL"
        print(f"[{done}/{len(to_run)}] {status} | Got: {ans[:60]} | Gold: {gold[:40]} | {elapsed:.0f}s | Running: {correct}/{done}={correct/done*100:.0f}%")
        sys.stdout.flush()

        await asyncio.sleep(0.5)

    print(f"\nDone: {correct}/{done} = {correct/done*100:.1f}%" if done else "\nNo tasks completed")

    # Merge results
    all_results = dict(prev_results)
    if os.path.exists(output_file):
        with open(output_file) as f:
            for line in f:
                r = json.loads(line)
                all_results[r["task_id"]] = r

    total_correct = sum(1 for r in all_results.values() if r.get("correct"))
    still_rl = sum(1 for r in all_results.values() if is_rate_limited(r.get("predicted", "")))
    actual_total = len(all_results) - still_rl
    print(f"\nLevel {level} MERGED: {total_correct}/{actual_total} non-RL tasks = {total_correct/actual_total*100:.1f}%")
    print(f"Still rate-limited: {still_rl}")


if __name__ == "__main__":
    asyncio.run(main())
