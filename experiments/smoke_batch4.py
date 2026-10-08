"""
Smoke-test runner for Batch 4 algorithms (21-28) on their test problems.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import importlib
import numpy as np
from problems import P3, P4, P5, P6, P7, P8, P9


def smoke_test_batch4():
    print("=== SMOKE TESTING BATCH 4 (Algorithms 21-28) ===")

    # 21. Constrained NSGA-II (P5, P6, P7)
    alg21 = importlib.import_module("algorithms.21_constrained_nsga2")
    p5 = P5()
    X21, F21 = alg21.run(p5, pop_size=20, n_gen=5, seed=42)
    print(f"Alg 21 (Constrained NSGA-II) on P5: {len(X21)} front points found.")

    # 22. MOPSO (P4, P5)
    alg22 = importlib.import_module("algorithms.22_mopso")
    p4 = P4()
    X22, F22 = alg22.run(p4, swarm_size=20, archive_size=20, n_gen=5, seed=42)
    print(f"Alg 22 (MOPSO) on P4: {len(X22)} front points found.")

    # 23. MOGWO (P5, P9)
    alg23 = importlib.import_module("algorithms.23_mogwo")
    X23, F23 = alg23.run(p5, pack_size=20, archive_size=20, n_gen=5, seed=42)
    print(f"Alg 23 (MOGWO) on P5: {len(X23)} front points found.")

    # 24. MOABC (P3, P6)
    alg24 = importlib.import_module("algorithms.24_moabc")
    p3 = P3(n_var=5)
    X24, F24 = alg24.run(p3, colony_size=20, archive_size=20, n_gen=5, seed=42)
    print(f"Alg 24 (MOABC) on P3: {len(X24)} front points found.")

    # 25. MOACO (P8)
    alg25 = importlib.import_module("algorithms.25_moaco")
    p8 = P8()
    X25, F25 = alg25.run(p8, n_ants=10, archive_size=20, n_gen=3, seed=42)
    print(f"Alg 25 (MOACO) on P8: {len(X25)} front points found.")

    # 26. MOSA (P3, P5)
    alg26 = importlib.import_module("algorithms.26_mosa")
    X26, F26 = alg26.run(p3, n_iterations=30, seed=42)
    print(f"Alg 26 (MOSA) on P3: {len(X26)} front points found.")

    # 27. MODE (P3, P5)
    alg27 = importlib.import_module("algorithms.27_mode")
    X27, F27 = alg27.run(p3, pop_size=20, archive_size=20, n_gen=5, seed=42)
    print(f"Alg 27 (MODE) on P3: {len(X27)} front points found.")

    # 28. MO-TLBO (P3, P5)
    alg28 = importlib.import_module("algorithms.28_mo_tlbo")
    X28, F28 = alg28.run(p3, pop_size=20, archive_size=20, n_gen=5, seed=42)
    print(f"Alg 28 (MO-TLBO) on P3: {len(X28)} front points found.")

    print("ALL BATCH 4 SMOKE TESTS PASSED!")


if __name__ == "__main__":
    smoke_test_batch4()
