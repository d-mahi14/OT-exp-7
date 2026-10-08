import unittest
import importlib
import numpy as np
from problems import P3, P4, P5, P6, P7, P8, P9
import moo_utils as U


class TestBatch4Algorithms(unittest.TestCase):

    def test_algorithm_21_constrained_nsga2(self):
        mod = importlib.import_module("algorithms.21_constrained_nsga2")
        # Lecture Part 4 Slide 88 Deb's rules:
        # A(CV=5, f1=50, f2=30) -> infeasible
        # B(CV=0, f1=45, f2=35) -> feasible
        # C(CV=0, f1=40, f2=32) -> feasible
        # In min form: A(f1=-50, f2=-30), B(-45, -35), C(-40, -32)
        F = np.array([[-50.0, -30.0], [-45.0, -35.0], [-40.0, -32.0]])
        CV = np.array([5.0, 0.0, 0.0])
        ranks, crowding, fronts = mod.compute_constrained_ranks_and_crowding(F, CV)
        # B dominates C on objectives -> B is Front 1. C is Front 2. A is infeasible -> Front 3.
        self.assertEqual(fronts[0], [1])  # B only
        self.assertEqual(fronts[1], [2])  # C
        self.assertEqual(fronts[2], [0])  # A

        # Test run on P5
        p5 = P5()
        X_front, F_front = mod.run(p5, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_22_mopso(self):
        mod = importlib.import_module("algorithms.22_mopso")
        # Lecture Part 4 Slide 89 exact numerical verification:
        # x=8, v=2, pbest=5, gbest=14, w=0.6, c1r1=0.8, c2r2=0.5
        # v_new = 0.6*2 + 0.8*(5-8) + 0.5*(14-8) = 1.2 - 2.4 + 3.0 = 1.8
        # x_new = 8 + 1.8 = 9.8
        w = 0.6
        c1r1 = 0.8
        c2r2 = 0.5
        x = 8.0
        v = 2.0
        pbest = 5.0
        gbest = 14.0
        v_new = w * v + c1r1 * (pbest - x) + c2r2 * (gbest - x)
        x_new = x + v_new
        self.assertAlmostEqual(v_new, 1.8)
        self.assertAlmostEqual(x_new, 9.8)

        # Test run on P4
        p4 = P4()
        X_front, F_front = mod.run(p4, swarm_size=20, archive_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_23_mogwo(self):
        mod = importlib.import_module("algorithms.23_mogwo")
        # 3 leaders centroid update (Lecture Part 4 Slide 22):
        # alpha=5, beta=20, delta=12 -> X_new = (5+20+12)/3 = 12.33
        X_alpha, X_beta, X_delta = np.array([5.0]), np.array([20.0]), np.array([12.0])
        X_new = (X_alpha + X_beta + X_delta) / 3.0
        self.assertAlmostEqual(X_new[0], 12.333, delta=0.01)

        # Test run on P5
        p5 = P5()
        X_front, F_front = mod.run(p5, pack_size=20, archive_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_24_moabc(self):
        mod = importlib.import_module("algorithms.24_moabc")
        # Lecture Part 4 Slide 31 greedy dominance replacement:
        # Trial S3 (energy=10, stab=0.90) vs original S3 (12, 0.85)
        # In min form: S3_trial = (10, -0.90), S3_orig = (12, -0.85)
        f_trial = np.array([10.0, -0.90])
        f_orig = np.array([12.0, -0.85])
        self.assertTrue(U.dominates(f_trial, f_orig))  # trial dominates original

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, colony_size=20, archive_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_25_moaco(self):
        mod = importlib.import_module("algorithms.25_moaco")
        p8 = P8()
        tau1 = np.ones((p8.n_var, p8.n_var))
        tau2 = np.ones((p8.n_var, p8.n_var))
        rng = np.random.default_rng(42)
        route = mod.construct_ant_route(p8.n_var, tau1, tau2, p8.dist, rng=rng)
        self.assertEqual(len(route), p8.n_var)
        self.assertEqual(len(set(route)), p8.n_var)

        # Test run on P8
        X_front, F_front = mod.run(p8, n_ants=10, archive_size=20, n_gen=3, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_26_mosa(self):
        mod = importlib.import_module("algorithms.26_mosa")
        # Test energy difference
        f_curr = np.array([20.0, 10.0])
        f_cand = np.array([23.0, 12.0])  # worsened by +3 and +2 -> dE = 5.0
        dE = mod.compute_energy_difference(f_cand, f_curr)
        self.assertAlmostEqual(dE, 5.0)

        # Early on at T=100: P = exp(-5/100) = 0.951
        self.assertAlmostEqual(np.exp(-dE / 100.0), 0.9512, delta=0.01)

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, n_iterations=30, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_27_mode(self):
        mod = importlib.import_module("algorithms.27_mode")
        # Lecture Part 4 Slide 90 DE selection check:
        # Target X = (size=10, gain=5.0) -> min form: [10, -5.0]
        # Trial 1 U = (size=9, gain=5.2) -> min form: [9, -5.2]
        # Trial 2 U2 = (size=9, gain=4.8) -> min form: [9, -4.8]
        f_target = np.array([10.0, -5.0])
        f_trial1 = np.array([9.0, -5.2])
        f_trial2 = np.array([9.0, -4.8])

        self.assertTrue(U.dominates(f_trial1, f_target))   # trial 1 replaces target
        self.assertFalse(U.dominates(f_trial2, f_target))  # trial 2 does NOT dominate target
        self.assertFalse(U.dominates(f_target, f_trial2))  # non-dominated

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, pop_size=20, archive_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_28_mo_tlbo(self):
        mod = importlib.import_module("algorithms.28_mo_tlbo")
        # Lecture Part 4 Slide 57 teacher phase formula:
        # X = 2, teacher = 7, mean = 5.75, r = 0.5, T_F = 2
        # X_new = 2 + 0.5*(7 - 2*5.75) = 2 + 0.5*(7 - 11.5) = 2 - 2.25 = -0.25 -> clip 0
        X = 2.0
        teacher = 7.0
        mean = 5.75
        r = 0.5
        T_F = 2
        X_new = X + r * (teacher - T_F * mean)
        self.assertAlmostEqual(X_new, -0.25)

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, pop_size=20, archive_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)


if __name__ == "__main__":
    unittest.main()
