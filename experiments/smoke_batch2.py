"""
Smoke-test runner for Batch 2 algorithms (10-13) on their test problems.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
import numpy as np
from problems import P2, P3, P5, P7


def smoke_test_batch2():
    print("=== SMOKE TESTING BATCH 2 (Algorithms 10-13) ===")

    # 10. VEGA (P2, P3)
    alg10 = importlib.import_module("algorithms.10_vega")
    p2 = P2()
    X10, F10 = alg10.run(p2, pop_size=20, n_gen=10, seed=42)
    print(f"Alg 10 (VEGA) on P2: {len(X10)} front points found.")

    # 11. MOGA (P3, P5)
    alg11 = importlib.import_module("algorithms.11_moga")
    p3 = P3(n_var=5)
    X11, F11 = alg11.run(p3, pop_size=20, n_gen=10, seed=42)
    print(f"Alg 11 (MOGA) on P3: {len(X11)} front points found.")

    # 12. NSGA (P5, P7)
    alg12 = importlib.import_module("algorithms.12_nsga")
    p5 = P5()
    X12, F12 = alg12.run(p5, pop_size=20, n_gen=10, seed=42)
    print(f"Alg 12 (NSGA) on P5: {len(X12)} front points found.")

    # 13. NPGA (P3, P5)
    alg13 = importlib.import_module("algorithms.13_npga")
    X13, F13 = alg13.run(p3, pop_size=20, n_gen=10, seed=42)
    print(f"Alg 13 (NPGA) on P3: {len(X13)} front points found.")

    print("ALL BATCH 2 SMOKE TESTS PASSED!")


if __name__ == "__main__":
    smoke_test_batch2()
