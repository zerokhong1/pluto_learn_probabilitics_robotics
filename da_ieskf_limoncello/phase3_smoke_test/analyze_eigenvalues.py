#!/usr/bin/env python3
"""
Analyze DA-IESKF eigenvalue CSV logs.
Compares box room (non-degenerate) vs corridor (degenerate).

Usage:
    python3 analyze_eigenvalues.py <log1.csv> [<log2.csv> ...]
    python3 analyze_eigenvalues.py ~/LIMOncello_ws/logs/box_eigenvalues.csv \
                                   ~/LIMOncello_ws/logs/corridor_eigenvalues.csv
"""

import csv
import sys
import os
from pathlib import Path


def load_log(path: str) -> list[dict]:
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def summarize(rows: list[dict], label: str) -> dict:
    if not rows:
        print(f"\n[{label}] EMPTY LOG")
        return {}

    n = len(rows)
    degen = sum(1 for r in rows if (r.get('is_degenerate') or '0').strip() == '1')
    degen_pct = 100.0 * degen / n

    ratios = []
    for r in rows:
        try:
            val = r.get('ratio') or r.get('eigenvalue_ratio')
            if val is not None:
                ratios.append(float(val))
        except (ValueError, TypeError):
            pass

    ndims = []
    for r in rows:
        try:
            val = r.get('n_degen_dims') or r.get('n_degenerate_dims') or '0'
            ndims.append(int(val))
        except (ValueError, TypeError):
            pass

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(f"  Frames:       {n}")
    print(f"  Degenerate:   {degen}  ({degen_pct:.1f}%)")
    if ratios:
        print(f"  Ratio min:    {min(ratios):.4f}")
        print(f"  Ratio max:    {max(ratios):.4f}")
        print(f"  Ratio mean:   {sum(ratios)/len(ratios):.4f}")
    if ndims:
        print(f"  Degen dims:   max={max(ndims)}, mean={sum(ndims)/len(ndims):.2f}")

    return {'n': n, 'degen': degen, 'degen_pct': degen_pct, 'ratios': ratios}


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    results = {}
    for path in sys.argv[1:]:
        label = Path(path).stem
        rows = load_log(path)
        results[label] = summarize(rows, label)

    # If exactly 2 logs passed, run comparison (box vs corridor)
    if len(sys.argv) == 3:
        keys = list(results.keys())
        r0, r1 = results[keys[0]], results[keys[1]]

        print(f"\n{'='*55}")
        print("  PASS/FAIL comparison")
        print(f"{'='*55}")

        # Determine which is non-degen and which is degen
        # Heuristic: lower degenerate% = non-degen
        if r0.get('degen_pct', 0) <= r1.get('degen_pct', 0):
            non_degen, degen = r0, r1
            non_label, deg_label = keys[0], keys[1]
        else:
            non_degen, degen = r1, r0
            non_label, deg_label = keys[1], keys[0]

        nd_pct = non_degen.get('degen_pct', -1)
        d_pct  = degen.get('degen_pct', -1)

        print(f"  Non-degen log: {non_label}  → {nd_pct:.1f}% degenerate")
        print(f"  Degen log:     {deg_label}  → {d_pct:.1f}% degenerate")
        print()

        nd_ratios = non_degen.get('ratios', [])
        d_ratios  = degen.get('ratios', [])
        nd_mean   = sum(nd_ratios) / len(nd_ratios) if nd_ratios else 0
        d_mean    = sum(d_ratios)  / len(d_ratios)  if d_ratios  else 1

        checks = [
            # Corridor must be flagged much more than box room
            ("Corridor degen%  >  box degen% + 30",   d_pct - nd_pct > 30.0),
            # Corridor ratio mean must be << box room ratio mean (10× or more)
            ("Corridor ratio <<  box ratio / 5",       d_mean < nd_mean / 5.0 if nd_mean > 0 else False),
            # Corridor must exceed 50% degenerate
            ("Corridor:  > 50% frames flagged",        d_pct >= 0 and d_pct > 50.0),
            ("No NaN in ratios",                       all(not (r != r) for r in nd_ratios + d_ratios)),
        ]

        all_pass = True
        for desc, ok in checks:
            status = "PASS" if ok else "FAIL"
            print(f"  [{status}] {desc}")
            if not ok:
                all_pass = False

        print()
        if all_pass:
            print("  >>> ALL CHECKS PASS: DA-IESKF correctly discriminates degeneracy")
        else:
            print("  >>> SOME CHECKS FAILED — tune da_eigenvalue_threshold in smoke_test.yaml")


if __name__ == "__main__":
    main()
