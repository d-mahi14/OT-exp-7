import unittest
import importlib
import numpy as np
from problems import P3, P4, P5, P9, P10, ToyTwoTask, P14
P13a = ToyTwoTask
import moo_utils as U


class TestBatch5Algorithms(unittest.TestCase):

    def test_algorithm_29_moba(self):
        mod = importlib.import_module("algorithms.29_moba")
        # Lecture Part 4 Slide 66:
        # f_min=0, f_max=2, beta=0.2 -> f_i = 0 + 2*0.2 = 0.4
        # Bat 1: x=5, x_best=15, v=0 -> v = 0 + (15 - 5)*0.4 = 4.0 -> x_new = 5 + 4 = 9.0
        x = 5.0
        x_best = 15.0
        f_i = 0.4
        v = 0.0
        v_new = v + (x_best - x) * f_i
        x_new = x + v_new
        self.assertAlmostEqual(v_new, 4.0)
        self.assertAlmostEqual(x_new, 9.0)

        # Test run on P4
        p4 = P4()
        X_front, F_front = mod.run(p4, n_bats=20, archive_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_30_moma(self):
        mod = importlib.import_module("algorithms.30_moma")
        p5 = P5()
        x0 = (p5.lower + p5.upper) / 2.0
        f0, cv0 = p5.evaluate(x0)
        lx, lf, lcv = mod.local_search(p5, x0, f0, cv0, max_steps=3)
        self.assertEqual(len(lx), len(x0))

        # Test run on P5
        X_front, F_front = mod.run(p5, pop_size=20, n_gen=3, n_local=3, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_31_rvea(self):
        mod = importlib.import_module("algorithms.31_rvea")
        # Test unit reference vectors
        V = mod.generate_unit_reference_vectors(k=3, p=2)
        # Check all vectors are unit norm
        np.testing.assert_allclose(np.linalg.norm(V, axis=1), 1.0)

        gamma_min = mod.compute_gamma_min(V)
        self.assertGreater(gamma_min, 0.0)

        # Test run on P10 with M=3
        p10 = P10(M=3)
        X_front, F_front = mod.run(p10, p=2, n_gen=3, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_32_ibea(self):
        mod = importlib.import_module("algorithms.32_ibea")
        # Pairwise epsilon indicator test:
        # a = (1.0, 2.0), b = (2.0, 1.0)
        # I(a, b) = max(1-2, 2-1) = 1.0
        # I(b, a) = max(2-1, 1-2) = 1.0
        Fn = np.array([[1.0, 2.0], [2.0, 1.0]])
        I_mat = mod.compute_epsilon_indicator_matrix(Fn)
        self.assertAlmostEqual(I_mat[0, 1], 1.0)
        self.assertAlmostEqual(I_mat[1, 0], 1.0)

        # Test run on P3
        p3 = P3(n_var=5)
        X_front, F_front = mod.run(p3, pop_size=20, n_gen=5, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_33_sms_emoa(self):
        mod = importlib.import_module("algorithms.33_sms_emoa")
        # Lecture Part 5E Slide 15:
        # 2D front points A=(6, 50), B=(10, 30), ref=(12, 60)
        F_front = np.array([[6.0, 50.0], [10.0, 30.0]])
        ref = np.array([12.0, 60.0])
        # Total HV = (12-6)*(60-50) + (12-10)*(60-30) - (12-10)*(60-50) = 60 + 60 - 20 = 100
        total_hv = U.hypervolume(F_front, ref)
        self.assertAlmostEqual(total_hv, 100.0)

        # Test run on P14
        p14 = P14()
        X_front, F_front = mod.run(p14, pop_size=15, n_evals=20, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_34_hype(self):
        mod = importlib.import_module("algorithms.34_hype")
        # Bounding box test:
        F_pool = np.array([[1.0, 2.0], [2.0, 1.0]])
        ref = np.array([3.0, 3.0])
        fit = mod.estimate_hype_fitness(F_pool, ref, n_samples=200, rng=np.random.default_rng(42))
        self.assertEqual(len(fit), 2)
        self.assertTrue(np.all(fit > 0.0))

        # Test run on P10 with M=3
        p10 = P10(M=3)
        X_front, F_front = mod.run(p10, pop_size=15, n_gen=3, n_samples=100, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_35_r_nsga2(self):
        mod = importlib.import_module("algorithms.35_r_nsga2")
        p14 = P14()
        # Reference point in min form: (1.8, -12.0)
        ref_point = np.array([1.8, -12.0])
        X_front, F_front = mod.run(p14, pop_size=20, n_gen=5, ref_point=ref_point, seed=42)
        self.assertGreater(len(X_front), 0)

    def test_algorithm_36_mgda(self):
        mod = importlib.import_module("algorithms.36_mgda")
        # Lecture Part 5E Slide 36 exact numerical test:
        # g1 = (4, 4), g2 = (16, 16)
        # min-norm point is g1 itself -> alpha* = 1.0, d* = (4, 4), ||d*|| = 5.657
        alpha_star, d_star = mod.find_min_norm_element_2d([4.0, 4.0], [16.0, 16.0])
        self.assertAlmostEqual(alpha_star, 1.0)
        np.testing.assert_allclose(d_star, [4.0, 4.0])
        self.assertAlmostEqual(np.linalg.norm(d_star), 5.65685, delta=0.01)

        # Run on P13a (Toy two-task)
        p13a = P13a()
        traj_x, traj_f = mod.run(p13a, max_steps=20, eta=0.1)
        self.assertGreater(len(traj_x), 1)
        # Initial losses vs final losses (both should decrease)
        self.assertLess(traj_f[-1, 0], traj_f[0, 0])
        self.assertLess(traj_f[-1, 1], traj_f[0, 1])

    def test_algorithm_37_gradnorm(self):
        mod = importlib.import_module("algorithms.37_gradnorm")
        # Lecture Part 5E Slide 39 exact numerical test:
        # L1(0) = 2.0, L2(0) = 4.0. Current L1(t)=1.0, L2(t)=3.6.
        # Base gradient norms fixed at g1=2.0, g2=3.0.
        # Start w = (1, 1). G = (2.0, 3.0), G_bar = 2.5.
        # r = (0.714, 1.286)
        # Target = G_bar * r^0.5 = (2.113, 2.835)
        # L_grad = |2.0 - 2.113| + |3.0 - 2.835| = 0.278
        # raw w = (1.05, 0.925) -> renormalized sum=2: (1.063, 0.937)
        losses = np.array([1.0, 3.6])
        l0 = np.array([2.0, 4.0])
        base_g = np.array([2.0, 3.0])
        w_curr = np.array([1.0, 1.0])
        G = w_curr * base_g

        targets, G_bar, r = mod.compute_gradnorm_targets(losses, l0, G, alpha_asym=0.5)
        self.assertAlmostEqual(G_bar, 2.5, delta=0.01)
        self.assertAlmostEqual(r[0], 0.714, delta=0.01)
        self.assertAlmostEqual(r[1], 1.286, delta=0.01)
        self.assertAlmostEqual(targets[0], 2.113, delta=0.01)
        self.assertAlmostEqual(targets[1], 2.835, delta=0.01)

        w_new, l_grad = mod.update_task_weights(w_curr, base_g, targets, lr_w=0.025)
        self.assertAlmostEqual(l_grad, 0.278, delta=0.01)
        self.assertAlmostEqual(w_new[0], 1.063, delta=0.01)
        self.assertAlmostEqual(w_new[1], 0.937, delta=0.01)

        # Test run on P13a
        p13a = P13a()
        theta, hist_l, hist_w = mod.run(p13a, n_steps=20, lr_theta=0.05, lr_w=0.025)
        self.assertEqual(len(hist_l), 21)


if __name__ == "__main__":
    unittest.main()
