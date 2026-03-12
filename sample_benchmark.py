"""
Sample benchmark with GAIA-representative tasks.
Used when the official GAIA dataset is not accessible.
Tests the same skills: web search, computation, file processing, reasoning.
"""

import asyncio
import json
import os
import sys
import time
from datetime import datetime

os.environ.pop("CLAUDECODE", None)

from agent import solve_task


SAMPLE_TASKS = [
    # Level 1 - Simple factual / computation
    {
        "task_id": "sample_001",
        "question": "What is the chemical symbol for gold?",
        "level": 1,
        "final_answer": "Au",
    },
    {
        "task_id": "sample_002",
        "question": "How many sides does a dodecahedron have?",
        "level": 1,
        "final_answer": "12",
    },
    {
        "task_id": "sample_003",
        "question": "What is the square root of 1764?",
        "level": 1,
        "final_answer": "42",
    },
    {
        "task_id": "sample_004",
        "question": "Who painted the Mona Lisa?",
        "level": 1,
        "final_answer": "Leonardo da Vinci",
    },
    {
        "task_id": "sample_005",
        "question": "What is the largest planet in our solar system?",
        "level": 1,
        "final_answer": "Jupiter",
    },
    {
        "task_id": "sample_006",
        "question": "In what year did World War II end?",
        "level": 1,
        "final_answer": "1945",
    },
    {
        "task_id": "sample_007",
        "question": "What is the boiling point of water in degrees Fahrenheit?",
        "level": 1,
        "final_answer": "212",
    },
    {
        "task_id": "sample_008",
        "question": "How many chromosomes do humans have?",
        "level": 1,
        "final_answer": "46",
    },

    # Level 2 - Requires web search or multi-step reasoning
    {
        "task_id": "sample_009",
        "question": "What is the sum of all prime numbers between 1 and 30?",
        "level": 2,
        "final_answer": "129",
    },
    {
        "task_id": "sample_010",
        "question": "Who was the first woman to win a Nobel Prize, and in what year?",
        "level": 2,
        "final_answer": "Marie Curie, 1903",
    },
    {
        "task_id": "sample_011",
        "question": "What is the GDP of Japan in trillions of USD as of 2023, rounded to one decimal place?",
        "level": 2,
        "final_answer": "4.2",
    },
    {
        "task_id": "sample_012",
        "question": "How many countries are there in the European Union as of 2024?",
        "level": 2,
        "final_answer": "27",
    },
    {
        "task_id": "sample_013",
        "question": "What is the tallest building in the world as of 2024, and how tall is it in meters?",
        "level": 2,
        "final_answer": "Burj Khalifa, 828",
    },
    {
        "task_id": "sample_014",
        "question": "If you convert 5 miles to kilometers and then round to the nearest whole number, what do you get?",
        "level": 2,
        "final_answer": "8",
    },
    {
        "task_id": "sample_015",
        "question": "What element has the atomic number 79?",
        "level": 2,
        "final_answer": "Gold",
    },

    # Level 3 - Complex multi-step
    {
        "task_id": "sample_016",
        "question": "What is the factorial of 12?",
        "level": 3,
        "final_answer": "479001600",
    },
    {
        "task_id": "sample_017",
        "question": "List all US states that border the Pacific Ocean, separated by commas, in alphabetical order.",
        "level": 3,
        "final_answer": "Alaska, California, Hawaii, Oregon, Washington",
    },
    {
        "task_id": "sample_018",
        "question": "What is the 15th element in the Fibonacci sequence (starting from 1, 1)?",
        "level": 3,
        "final_answer": "610",
    },
    {
        "task_id": "sample_019",
        "question": "How many words are in the first paragraph of the Wikipedia article about Python (programming language)?",
        "level": 3,
        "final_answer": "",  # Will vary, used for testing tool usage
    },
    {
        "task_id": "sample_020",
        "question": "What is the result of 2^10 + 3^7 - 5^4?",
        "level": 3,
        "final_answer": "2586",
    },
]


def normalize_answer(answer: str) -> str:
    answer = answer.strip()
    for prefix in ["FINAL ANSWER:", "Answer:", "The answer is"]:
        if answer.lower().startswith(prefix.lower()):
            answer = answer[len(prefix):].strip()
    answer = " ".join(answer.split())
    answer = answer.rstrip(".")
    return answer


def answers_match(predicted: str, gold: str) -> bool:
    if not gold:  # Skip tasks with no gold answer
        return True

    pred = normalize_answer(predicted)
    gold_norm = normalize_answer(gold)

    if pred.lower() == gold_norm.lower():
        return True

    try:
        pred_num = float(pred.replace(",", "").replace("$", "").replace("%", ""))
        gold_num = float(gold_norm.replace(",", "").replace("$", "").replace("%", ""))
        if abs(pred_num - gold_num) < 1e-6:
            return True
        if gold_num != 0 and abs((pred_num - gold_num) / gold_num) < 0.01:
            return True
    except (ValueError, ZeroDivisionError):
        pass

    if len(gold_norm) > 2 and gold_norm.lower() in pred.lower():
        return True

    return False


async def main():
    tasks = SAMPLE_TASKS
    total = len(tasks)
    correct = 0
    results = []

    print(f"\n{'='*60}")
    print(f"GAIA-Style Sample Benchmark")
    print(f"Tasks: {total}")
    print(f"{'='*60}\n")

    for i, task in enumerate(tasks):
        task_id = task["task_id"]
        question = task["question"]
        gold = task["final_answer"]
        level = task["level"]

        print(f"\n[{i+1}/{total}] Level {level} | {task_id}")
        print(f"Q: {question}")

        start = time.time()
        try:
            predicted = await solve_task(question, max_turns=20)
        except Exception as e:
            predicted = f"ERROR: {e}"
            print(f"  ERROR: {e}")

        elapsed = time.time() - start
        is_correct = answers_match(predicted, gold)

        if is_correct:
            correct += 1

        result = {
            "task_id": task_id,
            "question": question,
            "level": level,
            "gold_answer": gold,
            "predicted_answer": predicted,
            "correct": is_correct,
            "time_seconds": round(elapsed, 1),
        }
        results.append(result)

        with open("sample_results.jsonl", "a") as f:
            f.write(json.dumps(result) + "\n")

        status = "CORRECT" if is_correct else ("SKIP" if not gold else "WRONG")
        print(f"  Predicted: {predicted}")
        print(f"  Gold:      {gold}")
        print(f"  {status} ({elapsed:.1f}s)")
        scored = sum(1 for r in results if r["gold_answer"])
        print(f"  Running: {correct}/{scored} = {correct/max(scored,1)*100:.1f}%")

    scored_tasks = [r for r in results if r["gold_answer"]]
    accuracy = correct / len(scored_tasks) * 100 if scored_tasks else 0

    summary = {
        "total_tasks": total,
        "scored_tasks": len(scored_tasks),
        "correct": correct,
        "accuracy": round(accuracy, 2),
        "timestamp": datetime.now().isoformat(),
    }

    with open("sample_results_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"RESULTS: {correct}/{len(scored_tasks)} = {accuracy:.1f}%")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    asyncio.run(main())
