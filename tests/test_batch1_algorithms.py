import unittest
import importlib
import numpy as np
from problems import P1, P2, P3, P4, P5, P7, P11, P12, P14


class TestBatch1Algorithms(unittest.TestCase):

    def test_algorithm_02_epsilon_constraint(self):
        mod = importlib.import_module("algorithms.02_epsilon_constraint")
        # Test feasibility check
        objs = np.array([50.0, 4.0])
        viol, is_feas = mod.check_epsilon_feasibility(objs, epsilons=[4.0], primary_idx=0)
        self.assertEqual(viol, 0.0)
        self.assertTrue(is_feas)

        viol, is_feas = mod.check_epsilon_feasibility(objs, epsilons=[3.5], primary_idx=0)
        self.assertAlmostEqual(viol, 0.5)
        self.assertFalse(is_feas)

        # Test on P2: f1 = 100/x, f2 = x^2, x in [1, 10]
        p2 = P2()
        x_cands = np.linspace(1.0, 10.0, 100)[:, None]
        # At eps2 = 4, optimal x is 2 (f1=50, f2=4)
        bx, bf, score = mod.solve_epsilon_constraint(p2, epsilons=[4.0], candidates=x_cands, primary_idx=0)
        self.assertAlmostEqual(bx[0], 2.0, delta=0.2)
        self.assertAlmostEqual(bf[0], 50.0, delta=5.0)
        self.assertAlmostEqual(bf[1], 4.0, delta=1.0)

        # Test run method
        X, F = mod.run(p2, n_points=5, candidates=x_cands)
        self.assertGreater(len(X), 0)
        self.assertGreater(len(F), 0)

    def test_algorithm_03_bensons_method(self):
        mod = importlib.import_module("algorithms.03_bensons_method")
        # Reference point z* = [10.0, 10.0], candidate f = [4.0, 6.0]
        s = mod.benson_achievement([4.0, 6.0], [10.0, 10.0])
        np.testing.assert_allclose(s, [6.0, 4.0])

        p1 = P1()
        cands = np.linspace(1.0, 10.0, 100)[:, None]
        ref = np.array([16.0, 16.0])
        w = np.array([0.5, 0.5])
        bx, bf, score = mod.solve_benson(p1, ref, w, cands)
        # On P1 with equal weights, x=5 gives f1=4, f2=4
        self.assertAlmostEqual(bx[0], 5.0, delta=0.2)

        # Test run method
        X, F = mod.run(p1, ref_point=ref, n_weights=5, candidates=cands)
        self.assertGreater(len(X), 0)

    def test_algorithm_04_weight_metric(self):
        mod = importlib.import_module("algorithms.04_weight_metric")
        f = np.array([4.0, 4.0])
        z = np.array([0.0, 0.0])
        w = np.array([0.5, 0.5])

        # p = 1: 0.5*4 + 0.5*4 = 4.0
        val_1 = mod.weight_metric(f, z, w, p=1)
        self.assertAlmostEqual(val_1, 4.0)

        # p = 2: sqrt(0.5*16 + 0.5*16) = 4.0
        val_2 = mod.weight_metric(f, z, w, p=2)
        self.assertAlmostEqual(val_2, 4.0)

        # p = inf (Tchebycheff): max(0.5*4, 0.5*4) = 2.0
        val_inf = mod.weight_metric(f, z, w, p=np.inf)
        self.assertAlmostEqual(val_inf, 2.0)

        # Solve on P1
        p1 = P1()
        cands = np.linspace(1.0, 10.0, 100)[:, None]
        bx, bf, val = mod.solve_weight_metric(p1, p1.ideal, np.array([0.5, 0.5]), p=2, candidates=cands)
        self.assertAlmostEqual(bx[0], 5.0, delta=0.2)

        # Run with Tchebycheff
        X, F = mod.run(p1, p=np.inf, n_weights=5, candidates=cands)
        self.assertGreater(len(X), 0)

    def test_algorithm_05_value_function(self):
        mod = importlib.import_module("algorithms.05_value_function")
        p4 = P4()
        cands = p4.repair(p4.random_solutions(200, np.random.default_rng(42)))
        # Custom value function: higher return, lower risk
        # In min form, F = [risk, -return]
        # utility = return - 2 * risk = -F[1] - 2 * F[0]
        def utility(F):
            return -float(F[1]) - 2.0 * float(F[0])

        bx, bf, cost = mod.solve_value_function(p4, utility, cands)
        self.assertIsNotNone(bx)
        self.assertAlmostEqual(np.sum(bx), 1.0, delta=1e-5)

        X, F = mod.run(p4, value_func=utility, candidates=cands)
        self.assertEqual(len(X), 1)

    def test_algorithm_06_goal_programming(self):
        mod = importlib.import_module("algorithms.06_goal_programming")
        # Lecture 5E Slide 24 test:
        # xA=150, xB=100 -> profit = 5(150)+4(100) = 1150 (tgt 1000) -> d1-=0, d1+=150
        # output = 250 (tgt 300) -> d2-=50, d2+=0
        d1_m, d1_p = mod.compute_deviations(1150, 1000)
        self.assertEqual(d1_m, 0.0)
        self.assertEqual(d1_p, 150.0)

        d2_m, d2_p = mod.compute_deviations(250, 300)
        self.assertEqual(d2_m, 50.0)
        self.assertEqual(d2_p, 0.0)

        # Solve P11
        p11 = P11()
        X, F = mod.run(p11)
        # Optimal for P11 is (300, 200) giving profit 2400, units 500
        xA, xB = X[0]
        self.assertAlmostEqual(4 * xA + 6 * xB, 2400.0, delta=1e-3)

    def test_algorithm_07_lexicographic(self):
        mod = importlib.import_module("algorithms.07_lexicographic")
        p11 = P11()
        X, F = mod.run(p11)
        xA, xB = X[0]
        profit = 4 * xA + 6 * xB
        self.assertAlmostEqual(profit, 2400.0, delta=1e-3)
        # Tie break on priority 2 selects maximum units (500)
        self.assertAlmostEqual(xA + xB, 500.0, delta=1e-3)

    def test_algorithm_08_topsis(self):
        mod = importlib.import_module("algorithms.08_topsis")
        # Lecture 5E Slide 30 exact numerical test:
        # 4 suppliers x 3 criteria: Cost (min), Quality (max), Delivery (min)
        matrix = np.array([
            [50, 7, 10],
            [40, 6, 14],
            [60, 9, 8],
            [45, 8, 12]
        ], dtype=float)
        weights1 = np.array([0.4, 0.4, 0.2])
        maximize = np.array([False, True, False])

        C, rankings, d_plus, d_minus, A_plus, A_minus = mod.topsis(matrix, weights1, maximize)
        # Slide values: S1=0.466, S2=0.459, S3=0.541, S4=0.629 (best)
        # Rank: S4 > S3 > S1 > S2 (0-indexed: 3, 2, 0, 1)
        self.assertAlmostEqual(C[0], 0.466, delta=0.01)
        self.assertAlmostEqual(C[1], 0.459, delta=0.01)
        self.assertAlmostEqual(C[2], 0.541, delta=0.01)
        self.assertAlmostEqual(C[3], 0.629, delta=0.01)
        np.testing.assert_array_equal(rankings, [3, 2, 0, 1])

        # Iteration 2 (Slide 31): weights [0.3, 0.5, 0.2] flips rank to S3 > S4 > S1 > S2
        weights2 = np.array([0.3, 0.5, 0.2])
        C2, rankings2, _, _, _, _ = mod.topsis(matrix, weights2, maximize)
        self.assertAlmostEqual(C2[2], 0.649, delta=0.01)
        self.assertAlmostEqual(C2[3], 0.618, delta=0.01)
        np.testing.assert_array_equal(rankings2, [2, 3, 0, 1])

        # Test on P12 problem
        p12 = P12()
        rank_p12, C_p12 = mod.run(p12)
        self.assertEqual(len(rank_p12), 5)

    def test_algorithm_09_vikor(self):
        mod = importlib.import_module("algorithms.09_vikor")
        # Lecture 5E Slide 33 exact numerical test:
        matrix = np.array([
            [50, 7, 10],
            [40, 6, 14],
            [60, 9, 8],
            [45, 8, 12]
        ], dtype=float)
        weights = np.array([0.4, 0.4, 0.2])
        maximize = np.array([False, True, False])

        # Iteration 1: v = 0.5
        S, R, Q, rankings, comp_set = mod.vikor(matrix, weights, v=0.5, maximize=maximize)
        # Slide values:
        # S: S1=0.533, S2=0.600, S3=0.400, S4=0.367
        # R: S1=0.267, S2=0.400, S3=0.400, S4=0.133
        # Q: S1=0.607, S2=1.000, S3=0.571, S4=0.000
        # Rank: S4 > S3 > S1 > S2
        self.assertAlmostEqual(S[0], 0.533, delta=0.01)
        self.assertAlmostEqual(S[1], 0.600, delta=0.01)
        self.assertAlmostEqual(S[2], 0.400, delta=0.01)
        self.assertAlmostEqual(S[3], 0.367, delta=0.01)

        self.assertAlmostEqual(R[0], 0.267, delta=0.01)
        self.assertAlmostEqual(R[1], 0.400, delta=0.01)
        self.assertAlmostEqual(R[2], 0.400, delta=0.01)
        self.assertAlmostEqual(R[3], 0.133, delta=0.01)

        self.assertAlmostEqual(Q[0], 0.607, delta=0.01)
        self.assertAlmostEqual(Q[1], 1.000, delta=0.01)
        self.assertAlmostEqual(Q[2], 0.571, delta=0.01)
        self.assertAlmostEqual(Q[3], 0.000, delta=0.01)

        np.testing.assert_array_equal(rankings, [3, 2, 0, 1])
        self.assertEqual(comp_set, [3])  # S4 is single winner

        # Iteration 2: v = 0.3 -> S1 and S3 swap
        S2, R2, Q2, rankings2, comp_set2 = mod.vikor(matrix, weights, v=0.3, maximize=maximize)
        self.assertAlmostEqual(Q2[0], 0.564, delta=0.01)
        self.assertAlmostEqual(Q2[2], 0.743, delta=0.01)
        np.testing.assert_array_equal(rankings2, [3, 0, 2, 1])

        # Test on P12 problem
        p12 = P12()
        rank_p12, Q_p12, comp_p12 = mod.run(p12)
        self.assertEqual(len(rank_p12), 5)


if __name__ == "__main__":
    unittest.main()
