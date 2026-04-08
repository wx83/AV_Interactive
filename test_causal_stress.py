#!/usr/bin/env python3
"""Repeated stress runner for test_causal_conversion.py.

Runs the full conversion test multiple times and reports:
- pass/fail count
- average iteration latency
- whether NaN appeared in logged losses/grad norms
"""

import argparse
import io
import time
from contextlib import redirect_stdout

import test_causal_conversion as tcc


def run_once(iter_idx: int):
    buf = io.StringIO()
    t0 = time.time()
    with redirect_stdout(buf):
        ok = tcc.main()
    dt = time.time() - t0
    out = buf.getvalue()
    has_nan = "nan" in out.lower()
    print(f"ITER {iter_idx}: ok={ok} duration_s={dt:.2f} has_nan={has_nan}")
    if has_nan:
        for line in out.splitlines():
            if "loss" in line.lower() or "grad norm" in line.lower():
                print(f"  {line}")
    return ok, dt, has_nan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()

    oks = 0
    total = 0.0
    nan_count = 0

    for i in range(1, args.iterations + 1):
        ok, dt, has_nan = run_once(i)
        oks += int(ok)
        total += dt
        nan_count += int(has_nan)

    print("=" * 60)
    print(f"ITERATIONS={args.iterations}")
    print(f"PASS_COUNT={oks}")
    print(f"FAIL_COUNT={args.iterations - oks}")
    print(f"NAN_COUNT={nan_count}")
    print(f"AVG_DURATION_S={total / args.iterations:.2f}")
    print(f"TOTAL_DURATION_S={total:.2f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
