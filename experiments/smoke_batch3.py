"""
Smoke-test runner for Batch 3 algorithms (14-20) on their test problems.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
import numpy as np
from problems import P3, P4, P5, P6, P8, P10


def smoke_test_batch3():
    print("=== SMOKE TESTING BATCH 3 (Algorithms 14-20) ===")

    # 14. NSGA-II (P3, P5, P6)
    alg14 = importlib.import_module("algorithms.14_nsga2")
    p5 = P5()
    X14, F14 = alg14.run(p5, pop_size=20, n_gen=5, seed=42)
    print(f"Alg 14 (NSGA-II) on P5: {len(X14)} front points found.")

    # 15. NSGA-III (P10)
    alg15 = importlib.import_module("algorithms.15_nsga3")
    p10 = P10(M=3)
    X15, F15 = alg15.run(p10, p=2, n_gen=3, seed=42)
    print(f"Alg 15 (NSGA-III) on P10: {len(X15)} front points found.")

    # 16. SPEA2 (P4, P5)
    alg16 = importlib.import_module("algorithms.16_spea2")
    X16, F16 = alg16.run(p5, pop_size=20, n_gen=5, seed=42)
    print(f"Alg 16 (SPEA2) on P5: {len(X16)} front points found.")

    # 17. PESA-II (P3, P5)
    alg17 = importlib.import_module("algorithms.17_pesa2")
    p3 = P3(n_var=5)
    X17, F17 = alg17.run(p3, pop_size=20, archive_size=20, n_gen=5, seed=42)
    print(f"Alg 17 (PESA-II) on P3: {len(X17)} front points found.")

    # 18. MOEA/D (P3, P10)
    alg18 = importlib.import_module("algorithms.18_moead")
    X18, F18 = alg18.run(p3, n_subproblems=10, T=3, n_gen=5, seed=42)
    print(f"Alg 18 (MOEA/D) on P3: {len(X18)} front points found.")

    # 19. MOEA/D-DE (P3, P5)
    alg19 = importlib.import_module("algorithms.19_moead_de")
    X19, F19 = alg19.run(p3, n_subproblems=10, T=3, n_gen=5, seed=42)
    print(f"Alg 19 (MOEA/D-DE) on P3: {len(X19)} front points found.")

    # 20. MOEA/D-ACO (P8)
    alg20 = importlib.import_module("algorithms.20_moead_aco")
    p8 = P8()
    X20, F20 = alg20.run(p8, n_subproblems=6, T=3, n_gen=3, seed=42)
    print(f"Alg 20 (MOEA/D-ACO) on P8: {len(X20)} front points found.")

    print("ALL BATCH 3 SMOKE TESTS PASSED!")


if __name__ == "__main__":
    smoke_test_batch3()
