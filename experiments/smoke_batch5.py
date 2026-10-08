"""
Smoke-test runner for Batch 5 algorithms (29-37) on their test problems.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
import numpy as np
from problems import P3, P4, P5, P9, P10, ToyTwoTask, P14
P13a = ToyTwoTask


def smoke_test_batch5():
    print("=== SMOKE TESTING BATCH 5 (Algorithms 29-37) ===")

    # 29. MOBA (P3, P4)
    alg29 = importlib.import_module("algorithms.29_moba")
    p4 = P4()
    X29, F29 = alg29.run(p4, n_bats=20, archive_size=20, n_gen=5, seed=42)
    print(f"Alg 29 (MOBA) on P4: {len(X29)} front points found.")

    # 30. MOMA (P5, P9)
    alg30 = importlib.import_module("algorithms.30_moma")
    p5 = P5()
    X30, F30 = alg30.run(p5, pop_size=20, n_gen=3, n_local=3, seed=42)
    print(f"Alg 30 (MOMA) on P5: {len(X30)} front points found.")

    # 31. RVEA (P10)
    alg31 = importlib.import_module("algorithms.31_rvea")
    p10 = P10(M=3)
    X31, F31 = alg31.run(p10, p=2, n_gen=3, seed=42)
    print(f"Alg 31 (RVEA) on P10: {len(X31)} front points found.")

    # 32. IBEA (P3, P14)
    alg32 = importlib.import_module("algorithms.32_ibea")
    p3 = P3(n_var=5)
    X32, F32 = alg32.run(p3, pop_size=20, n_gen=5, seed=42)
    print(f"Alg 32 (IBEA) on P3: {len(X32)} front points found.")

    # 33. SMS-EMOA (P3, P14)
    alg33 = importlib.import_module("algorithms.33_sms_emoa")
    p14 = P14()
    X33, F33 = alg33.run(p14, pop_size=15, n_evals=20, seed=42)
    print(f"Alg 33 (SMS-EMOA) on P14: {len(X33)} front points found.")

    # 34. HypE (P10)
    alg34 = importlib.import_module("algorithms.34_hype")
    X34, F34 = alg34.run(p10, pop_size=15, n_gen=3, n_samples=100, seed=42)
    print(f"Alg 34 (HypE) on P10: {len(X34)} front points found.")

    # 35. R-NSGA-II (P5, P14)
    alg35 = importlib.import_module("algorithms.35_r_nsga2")
    X35, F35 = alg35.run(p14, pop_size=20, n_gen=5, ref_point=np.array([1.8, -12.0]), seed=42)
    print(f"Alg 35 (R-NSGA-II) on P14: {len(X35)} front points found.")

    # 36. MGDA (P13a)
    alg36 = importlib.import_module("algorithms.36_mgda")
    p13a = P13a()
    traj_x, traj_f = alg36.run(p13a, max_steps=20, eta=0.1)
    print(f"Alg 36 (MGDA) on P13a: {len(traj_x)} trajectory steps, final F={traj_f[-1]}.")

    # 37. GradNorm (P13a)
    alg37 = importlib.import_module("algorithms.37_gradnorm")
    theta, hist_l, hist_w = alg37.run(p13a, n_steps=20, lr_theta=0.05, lr_w=0.025)
    print(f"Alg 37 (GradNorm) on P13a: {len(hist_l)} steps, final weights={hist_w[-1]}.")

    print("ALL BATCH 5 SMOKE TESTS PASSED!")


if __name__ == "__main__":
    smoke_test_batch5()
