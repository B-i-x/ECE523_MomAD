"""Summarize selector weight distribution from selector_weights.jsonl.

Run after `dist_test.sh` to see what the AdaptiveHistorySelector actually picked.

Usage:
    python tools/analyze_selector_weights.py [path/to/selector_weights.jsonl]
"""
import json
import sys
from collections import Counter, defaultdict


def main(path):
    with open(path) as f:
        rows = [json.loads(line) for line in f if line.strip()]

    if not rows:
        print(f"No entries in {path}")
        return

    n = len(rows)
    print(f"Total samples logged: {n}")
    print()

    branch_names = {0: 'no-history', 1: 't-1', 2: 't-1+t-2'}
    cmd_names = {0: 'TURN_RIGHT', 1: 'TURN_LEFT', 2: 'STRAIGHT'}

    # Argmax distribution overall
    argmax_counts = Counter(r['argmax'] for r in rows)
    print("Argmax (which branch the selector picked):")
    for branch, count in sorted(argmax_counts.items()):
        pct = 100.0 * count / n
        print(f"  branch {branch} ({branch_names.get(branch, '?')}): {count} ({pct:.1f}%)")
    print()

    # Mean weights overall
    mean_w = [sum(r['weights'][i] for r in rows) / n for i in range(3)]
    print(f"Mean weights:  w0={mean_w[0]:.3f}  w1={mean_w[1]:.3f}  w2={mean_w[2]:.3f}")
    print()

    # Validity-state breakdown — how often were branches actually masked off?
    validity_counts = Counter(
        (r.get('last_valid'), r.get('last2_valid')) for r in rows
    )
    print("Validity state of (last, last2):")
    for state, count in sorted(validity_counts.items(), key=lambda kv: -kv[1]):
        print(f"  last={state[0]}, last2={state[1]}: {count} samples ({100.0*count/n:.1f}%)")
    print()

    # Argmax distribution conditioned on ego command
    by_cmd = defaultdict(list)
    for r in rows:
        cmd = r.get('cmd')
        if cmd is not None:
            by_cmd[cmd].append(r)
    if by_cmd:
        print("Argmax distribution per ego command:")
        for cmd_idx in sorted(by_cmd.keys()):
            cmd_rows = by_cmd[cmd_idx]
            cmd_n = len(cmd_rows)
            cmd_argmax = Counter(r['argmax'] for r in cmd_rows)
            cmd_mean = [sum(r['weights'][i] for r in cmd_rows) / cmd_n for i in range(3)]
            print(f"  cmd={cmd_idx} ({cmd_names.get(cmd_idx, '?')}, n={cmd_n}):")
            for branch in (0, 1, 2):
                c = cmd_argmax.get(branch, 0)
                print(f"      branch {branch}: {c} ({100.0*c/cmd_n:.1f}%), mean w={cmd_mean[branch]:.3f}")
        print()
        print("(Hypothesis from the History Paradox: STRAIGHT scenes should lean to higher w_2;")
        print(" TURN scenes should lean to higher w_0 or w_1.)")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "./selector_weights.jsonl"
    main(path)