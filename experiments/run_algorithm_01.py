"""
Experiment for Algorithm 1: Weighted Sum
"""

import sys
from pathlib import Path
import importlib
import numpy as np

# Add project root folder to Python path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from problems import PROBLEMS


def load_weighted_sum():
    """Load Algorithm 1 from its numbered module."""
    module = importlib.import_module("algorithms.01_weighted_sum")
    return module.weighted_sum


def main():
    weighted_sum = load_weighted_sum()

    # Weight combinations for two-objective problems
    weight_sets_2 = [
        np.array([0.1, 0.9]),
        np.array([0.3, 0.7]),
        np.array([0.5, 0.5]),
        np.array([0.7, 0.3]),
        np.array([0.9, 0.1]),
    ]

    print("=" * 70)
    print("ALGORITHM 1: WEIGHTED SUM")
    print("=" * 70)

    for problem_name, problem_class in PROBLEMS.items():

        problem = problem_class()

        print(f"\n{problem_name}")
        print("-" * 50)

        n_obj = problem.n_objectives

        print(f"Number of objectives: {n_obj}")

        if n_obj != 2:
            print(
                "Skipping initial experiment because this experiment "
                "currently uses 2-objective weight sets."
            )
            continue

        rng = np.random.default_rng(42)

        candidates = []

        for _ in range(100):
            x = rng.uniform(
                problem.lower_bounds,
                problem.upper_bounds
            )
            candidates.append(x)

        for weights in weight_sets_2:

            best_x = None
            best_f = None
            best_value = np.inf

            for x in candidates:

                f = np.asarray(problem.evaluate(x), dtype=float)

                value = weighted_sum(f, weights)

                if value < best_value:
                    best_value = value
                    best_x = x.copy()
                    best_f = f.copy()

            print(
                f"weights={weights} "
                f"best_value={best_value:.6f} "
                f"objectives={best_f}"
            )


if __name__ == "__main__":
    main()