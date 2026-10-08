"""
Smoke-test runner for Batch 1 algorithms (1-9) on their test problems.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
import numpy as np
from problems import P1, P2, P3, P4, P5, P7, P11, P12, P14
import moo_utils as U

def smoke_test_batch1():
    print("=== SMOKE TESTING BATCH 1 (Algorithms 1-9) ===")

    # 1. Weighted Sum (P1, P3, P4)
    alg01 = importlib.import_module("algorithms.01_weighted_sum")
    p1 = P1()
    X1, F1 = alg01.run(p1, n_weights=5, n_candidates=100, seed=42)
    print(f"Alg 01 on P1: {len(X1)} front points found.")

    # 2. Epsilon-Constraint (P2, P3, P5)
    alg02 = importlib.import_module("algorithms.02_epsilon_constraint")
    p2 = P2()
    X2, F2 = alg02.run(p2, n_points=5, n_candidates=100, seed=42)
    print(f"Alg 02 on P2: {len(X2)} front points found.")

    # 3. Benson's Method (P1, P7)
    alg03 = importlib.import_module("algorithms.03_bensons_method")
    p7 = P7()
    X3, F3 = alg03.run(p7, n_weights=5, n_candidates=100, seed=42)
    print(f"Alg 03 on P7: {len(X3)} front points found.")

    # 4. Weight Metric (P1, P3)
    alg04 = importlib.import_module("algorithms.04_weight_metric")
    p3 = P3(n_var=5)
    X4, F4 = alg04.run(p3, p=np.inf, n_weights=5, n_candidates=100, seed=42)
    print(f"Alg 04 on P3: {len(X4)} front points found.")

    # 5. Value Function (P4, P14)
    alg05 = importlib.import_module("algorithms.05_value_function")
    p14 = P14()
    X5, F5 = alg05.run(p14, n_candidates=100, seed=42)
    print(f"Alg 05 on P14: {len(X5)} solution found, F={F5[0]}.")

    # 6. Goal Programming (P11, P7)
    alg06 = importlib.import_module("algorithms.06_goal_programming")
    p11 = P11()
    X6, F6 = alg06.run(p11, n_candidates=100, seed=42)
    print(f"Alg 06 on P11: {len(X6)} solution found, x={X6[0]}.")

    # 7. Lexicographic (P11, P7)
    alg07 = importlib.import_module("algorithms.07_lexicographic")
    X7, F7 = alg07.run(p11, n_candidates=100, seed=42)
    print(f"Alg 07 on P11: {len(X7)} solution found, x={X7[0]}.")

    # 8. TOPSIS (P12, front from P4)
    alg08 = importlib.import_module("algorithms.08_topsis")
    p12 = P12()
    ranks8, C8 = alg08.run(p12)
    print(f"Alg 08 on P12: Supplier rankings = {[p12.suppliers[i] for i in ranks8]}.")

    # 9. VIKOR (P12, front from P4)
    alg09 = importlib.import_module("algorithms.09_vikor")
    ranks9, Q9, comp9 = alg09.run(p12, v=0.5)
    print(f"Alg 09 on P12: Supplier rankings = {[p12.suppliers[i] for i in ranks9]}, compromise = {[p12.suppliers[i] for i in comp9]}.")

    print("ALL BATCH 1 SMOKE TESTS PASSED!")

if __name__ == "__main__":
    smoke_test_batch1()
