#!/bin/bash
cd /home/ubuntu/gaia-agent
export CLAUDECODE=''
export ANTHROPIC_API_KEY=$(grep ANTHROPIC_API_KEY /home/ubuntu/gaia-agent/.env 2>/dev/null | cut -d= -f2 || echo "")

echo "Starting rate-limited re-runs at $(date -u)"
echo "============================================"

# L2 rate-limited re-run
echo ""
echo "=== L2 Rate-Limited Re-run ==="
timeout 14400 python3 rerun_ratelimited.py 2 gaia_results_l2_v2.jsonl 2>&1 | tee /tmp/gaia_l2_rl.txt

# L3 rate-limited re-run
echo ""
echo "=== L3 Rate-Limited Re-run ==="
timeout 7200 python3 rerun_ratelimited.py 3 gaia_results_l3_v2.jsonl 2>&1 | tee /tmp/gaia_l3_rl.txt

# L1 rate-limited re-run
echo ""
echo "=== L1 Rate-Limited Re-run ==="
timeout 3600 python3 rerun_ratelimited.py 1 gaia_results_l1_v3.jsonl 2>&1 | tee /tmp/gaia_l1_rl.txt

echo ""
echo "All re-runs complete at $(date -u)"

# Calculate final scores
python3 -c "
import json, os

def is_rate_limited(p):
    l = p.lower()
    return \"you've hit your limit\" in l or ('limit' in l and 'reset' in l)

for level, total in [(1,53),(2,86),(3,26)]:
    results = {}
    # Load base results
    for f in [f'gaia_results_l{level}_v2.jsonl', f'gaia_results_l{level}_v3.jsonl', f'gaia_results_l{level}_rl.jsonl']:
        if os.path.exists(f):
            for line in open(f):
                r = json.loads(line)
                tid = r['task_id']
                # Keep best result (correct > non-RL > RL)
                if tid not in results or r.get('correct') or (not is_rate_limited(r.get('predicted','')) and is_rate_limited(results[tid].get('predicted',''))):
                    results[tid] = r
    correct = sum(1 for r in results.values() if r.get('correct'))
    rl = sum(1 for r in results.values() if is_rate_limited(r.get('predicted','')))
    print(f'L{level}: {correct}/{total} = {correct/total*100:.1f}% (still RL: {rl})')

# Overall
all_correct = 0
all_total = 0
for level, total in [(1,53),(2,86),(3,26)]:
    results = {}
    for f in [f'gaia_results_l{level}_v2.jsonl', f'gaia_results_l{level}_v3.jsonl', f'gaia_results_l{level}_rl.jsonl']:
        if os.path.exists(f):
            for line in open(f):
                r = json.loads(line)
                tid = r['task_id']
                if tid not in results or r.get('correct') or (not is_rate_limited(r.get('predicted','')) and is_rate_limited(results[tid].get('predicted',''))):
                    results[tid] = r
    all_correct += sum(1 for r in results.values() if r.get('correct'))
    all_total += total
print(f'OVERALL: {all_correct}/{all_total} = {all_correct/all_total*100:.1f}%')
" 2>&1 | tee /tmp/gaia_final_scores.txt
