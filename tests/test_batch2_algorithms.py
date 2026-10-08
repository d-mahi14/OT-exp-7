import unittest
import importlib
import numpy as np
from problems import P2, P3, P5, P7
import moo_utils as U


class TestBatch2Algorithms(unittest.TestCase):

    def test_algorithm_10_vega(self):
        mod = importlib.import_module("algorithms.10_vega")
        rng = np.random.default_rng(42)
        # Test subpopulation selection:
        # 4 individuals with 2 objectives
        F = np.array([
            [1.0, 10.0],  # best on obj 0, worst on obj 1
            [5.0, 5.0],
            [10.0, 1.0],  # worst on obj 0, best on obj 1
            [8.0, 8.0]
        ])
        CV = np.zeros(4)
        X = np.arange(4)[:, None]
        mating_idx = mod.select_subpopulations(X, F, CV, k_sub=2, rng=rng)
        self.assertEqual(len(mating_idx), 4)

        # Test run on P2
        p2 = P2()
        X_front, F_front = mod.run(p2, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)
        self.assertGreater(len(F_front), 0)

    def test_algorithm_11_moga(self):
        mod = importlib.import_module("algorithms.11_moga")
        # Test dominance ranking:
        # A(1, 1), B(2, 2) -> A dominates B -> rank(A) = 1, rank(B) = 2
        # C(1, 2) -> A dominates C -> rank(C) = 2
        F = np.array([
            [1.0, 1.0],
            [2.0, 2.0],
            [1.0, 2.0]
        ])
        ranks = mod.compute_moga_ranks(F)
        self.assertEqual(ranks[0], 1)
        self.assertEqual(ranks[1], 3)  # dominated by A and C
        self.assertEqual(ranks[2], 2)  # dominated by A

        # Test fitness sharing
        shared_fit, nc = mod.compute_shared_fitness(F, ranks, sigma_share=0.5)
        self.assertEqual(len(shared_fit), 3)
        self.assertGreater(shared_fit[0], shared_fit[1])

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_12_nsga(self):
        mod = importlib.import_module("algorithms.12_nsga")
        # Test dummy fitness by front:
        F = np.array([
            [1.0, 4.0],  # Front 1
            [4.0, 1.0],  # Front 1
            [5.0, 5.0],  # Front 2
        ])
        shared_fit, fronts = mod.assign_nsga_shared_fitness(F, sigma_share=0.5, f_max=100.0, delta_f=10.0)
        self.assertEqual(len(fronts[0]), 2)  # 2 points in front 1
        self.assertEqual(len(fronts[1]), 1)  # 1 point in front 2
        # Front 1 members must have higher fitness than Front 2 members
        self.assertGreater(shared_fit[0], shared_fit[2])
        self.assertGreater(shared_fit[1], shared_fit[2])

        # Test run on P5
        p5 = P5()
        X_front, F_front = mod.run(p5, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_13_npga(self):
        mod = importlib.import_module("algorithms.13_npga")
        # Test Lecture 5B Slide 29 tournament tie-break by niche count:
        # A(cost=10, rel=0.80) -> min form: (10, -0.80)
        # B(cost=15, rel=0.90) -> min form: (15, -0.90)
        # Comparison set: C(12, -0.85), D(18, -0.95)
        F = np.array([
            [10.0, -0.80],  # 0: A
            [15.0, -0.90],  # 1: B
            [12.0, -0.85],  # 2: C
            [18.0, -0.95],  # 3: D
        ])
        CV = np.zeros(4)
        comp_set = np.array([2, 3])
        # Using the slide's normalization coordinates:
        # A: (0, 0), B: (0.5, 0.5), C: (0.2, 0.25), D: (0.8, 0.75)
        Fn_slide = np.array([
            [0.0, 0.0],
            [0.5, 0.5],
            [0.2, 0.25],
            [0.8, 0.75]
        ])
        # Niche count: NC(A) = 0.590 < NC(B) = 0.778 -> A wins
        winner = mod.pareto_tournament(F, CV, 0, 1, comp_set, sigma_share=0.5, Fn=Fn_slide)
        self.assertEqual(winner, 0)  # A wins!

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)


if __name__ == "__main__":
    unittest.main()
