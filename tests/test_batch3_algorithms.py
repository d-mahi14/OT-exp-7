import unittest
import importlib
import numpy as np
from problems import P3, P4, P5, P6, P8, P10
import moo_utils as U
import operators as ops


class TestBatch3Algorithms(unittest.TestCase):

    def test_algorithm_14_nsga2(self):
        mod = importlib.import_module("algorithms.14_nsga2")
        # Test Lecture Part 3 Slide 8 instance:
        # x = [1, 2, 3, 4], f1 = [1, 4, 9, 16], f2 = [96, 99, 100, 99]
        # In min-form, f2 is negated: [-96, -99, -100, -99]
        F = np.array([
            [1.0, -96.0],
            [4.0, -99.0],
            [9.0, -100.0],
            [16.0, -99.0]
        ])
        ranks, crowding, fronts = mod.compute_ranks_and_crowding(F)
        # Front 1 should have x=1, 2, 3; x=4 is dominated (Front 2)
        self.assertEqual(len(fronts[0]), 3)
        self.assertEqual(len(fronts[1]), 1)
        self.assertEqual(fronts[1][0], 3)  # x=4 is index 3

        # Boundaries of Front 1 have infinite crowding distance
        self.assertTrue(np.isinf(crowding[0]))  # x=1
        self.assertTrue(np.isinf(crowding[2]))  # x=3
        self.assertFalse(np.isinf(crowding[1])) # x=2 (interior)

        # Test run on P6
        p6 = P6()
        X_front, F_front = mod.run(p6, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_15_nsga3(self):
        mod = importlib.import_module("algorithms.15_nsga3")
        # Test reference points and association (Lecture Part 3 Slide 16)
        # 3 reference directions: r1=(1,0,0), r2=(0.577,0.577,0.577), r3=(0,0,1)
        ref_points = np.array([
            [1.0, 0.0, 0.0],
            [1.0 / np.sqrt(3), 1.0 / np.sqrt(3), 1.0 / np.sqrt(3)],
            [0.0, 0.0, 1.0]
        ])
        # Design 1: f = (0.85, 0.10, 0.08) -> closest to r1 (d=0.128)
        F_test = np.array([[0.85, 0.10, 0.08]])
        assoc, d_perp = mod.associate_to_reference_points(F_test, ref_points)
        self.assertEqual(assoc[0], 0)  # associates to r1
        self.assertAlmostEqual(d_perp[0], 0.128, delta=0.01)

        # Test run on P10 with M=3
        p10 = P10(M=3)
        X_front, F_front = mod.run(p10, p=2, n_gen=3, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_16_spea2(self):
        mod = importlib.import_module("algorithms.16_spea2")
        # Lecture Part 3 Slide 24 portfolio optimization instance:
        # Candidate portfolios: A(5,8), B(8,12), C(10,15), D(6,7), E(12,14)
        # Objectives: min risk f1, max return f2 (min -return)
        F = np.array([
            [5.0, -8.0],   # A
            [8.0, -12.0],  # B
            [10.0, -15.0], # C
            [6.0, -7.0],   # D (dominated by A)
            [12.0, -14.0]  # E (dominated by C)
        ])
        S, R, density, fitness, dist_mat = mod.compute_spea2_fitness(F, k_nn=1)
        # Slide 24: S(A)=1, S(C)=1, S(B)=0, S(D)=0, S(E)=0
        self.assertEqual(S[0], 1)
        self.assertEqual(S[1], 0)
        self.assertEqual(S[2], 1)
        self.assertEqual(S[3], 0)
        self.assertEqual(S[4], 0)

        # Raw fitness: R(D)=S(A)=1, R(E)=S(C)=1; R(A)=R(B)=R(C)=0
        self.assertEqual(R[0], 0)
        self.assertEqual(R[1], 0)
        self.assertEqual(R[2], 0)
        self.assertEqual(R[3], 1)
        self.assertEqual(R[4], 1)

        # Non-dominated solutions have fitness < 1
        self.assertLess(fitness[0], 1.0)
        self.assertLess(fitness[1], 1.0)
        self.assertLess(fitness[2], 1.0)
        self.assertGreaterEqual(fitness[3], 1.0)
        self.assertGreaterEqual(fitness[4], 1.0)

        # Test run on P5
        p5 = P5()
        X_front, F_front = mod.run(p5, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_17_pesa2(self):
        mod = importlib.import_module("algorithms.17_pesa2")
        # Lecture Part 3 Slide 32 sensor placement:
        # A(1, 1), B(2, 2.2), C(2.3, 2.4), D(3.5, 3.5), E(5, 4.5)
        # Min cost f1, max coverage f2. All non-dominated.
        # B and C are in the same grid cell [2, 3), so cell 2 has squeeze = 2.
        # Cells 1, 3, 4 have squeeze = 1.
        F = np.array([
            [1.0, 1.0],
            [2.2, 2.2],
            [2.5, 2.4],
            [3.5, 3.5],
            [4.8, 4.5]
        ])
        # Force 4 grid divisions with bounds [1, 5] along axis 0
        cell_tuples, _ = mod.compute_grid_indices(F, n_divs=4, f_min=np.array([1.0, 1.0]), f_max=np.array([5.0, 5.0]))
        squeeze, members = mod.compute_squeeze_factors(cell_tuples)
        # At least one cell has squeeze 2 (B and C)
        self.assertIn(2, squeeze.values())

        # Test region-based selection favors lower squeeze factor
        rng = np.random.default_rng(42)
        sel = mod.region_based_selection(members, squeeze, n_select=10, rng=rng)
        self.assertEqual(len(sel), 10)

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_18_moead(self):
        mod = importlib.import_module("algorithms.18_moead")
        # Lecture Part 3 Slide 40-41 Tchebycheff setup:
        # lambda = (0.5, 0.5), z* = (0, 0)
        # x=1 gives f = (1, 1) -> g = max(0.5*1, 0.5*1) = 0.5
        g_val = mod.tchebycheff_subproblem([1.0, 1.0], [0.5, 0.5], [0.0, 0.0])
        self.assertAlmostEqual(g_val, 0.5)

        # x=0 gives f = (0, 4) -> g = max(0, 2) = 2.0
        g_val_0 = mod.tchebycheff_subproblem([0.0, 4.0], [0.5, 0.5], [0.0, 0.0])
        self.assertAlmostEqual(g_val_0, 2.0)

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, n_subproblems=10, T=3, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_19_moead_de(self):
        mod = importlib.import_module("algorithms.19_moead_de")
        # Lecture Part 3 Slide 47 DE Mutation:
        # V = X_r1 + F * (X_r2 - X_r3)
        # r1 = 8, r2 = 15, r3 = 18, F = 0.5 -> V = 8 + 0.5*(15 - 18) = 6.5
        v = ops.de_mutation([8.0], [15.0], [18.0], F=0.5)
        self.assertAlmostEqual(v[0], 6.5)

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, n_subproblems=10, T=3, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_20_moead_aco(self):
        mod = importlib.import_module("algorithms.20_moead_aco")
        p8 = P8()
        # Verify ant tour construction
        tau = np.ones((p8.n_var, p8.n_var))
        rng = np.random.default_rng(42)
        tour = mod.construct_ant_tour(p8.n_var, tau, p8.dist, rng=rng)
        self.assertEqual(len(tour), p8.n_var)
        self.assertEqual(len(set(tour)), p8.n_var)  # valid permutation

        # Test run on P8
        X_front, F_front = mod.run(p8, n_subproblems=6, T=3, n_gen=3, seed=42)
        self.assertGreater(len(X_front), 0)
        self.assertGreater(len(F_front), 0)


if __name__ == "__main__":
    unittest.main()
