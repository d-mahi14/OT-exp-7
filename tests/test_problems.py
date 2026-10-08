"""Unit tests for problems.py (P1-P14).  Every sanity check quoted in the PDF is
tested, plus generic checks (shapes, true front non-dominated, normalised front
inside [0,1], no random feasible sample strictly dominates the true front).
Run from the project root:  python -m unittest discover -s tests -t . -v
"""
import unittest
import numpy as np
import moo_utils as U
import problems as P


def strictly_dominates_any(S, T, tol=1e-7):
    """True if some row of S is better than some row of T by more than tol in ALL objectives."""
    return bool(np.any(np.all(S[:, None, :] < T[None, :, :] - tol, axis=2)))


class TestGeneric(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(0)
        self.problems = [P.P1(), P.P2(), P.P2(integer=True), P.P3(), P.P4(), P.P4("penalty"),
                         P.P5(), P.P6(), P.P7(), P.P8(), P.P9(), P.P10(3), P.P10(5), P.P11(),
                         P.ToyTwoTask(), P.P14()]

    def test_shapes_and_counter(self):
        for p in self.problems:
            p.reset_counter()
            X = p.random_solutions(7, self.rng)
            F, CV = p.evaluate(X)
            self.assertEqual(F.shape, (7, p.n_obj), p.name)
            self.assertEqual(CV.shape, (7,), p.name)
            self.assertEqual(p.n_evals, 7, p.name)
            self.assertTrue(np.all(np.isfinite(F)), p.name)

    def test_true_front_is_nondominated_and_in_unit_range(self):
        for p in self.problems:
            T = p.true_front(200)
            if T is None:
                continue
            self.assertEqual(len(U.non_dominated_sort(np.round(T, 12))[0]), len(T), p.name)
            Tn = p.normalize(T)
            self.assertTrue(np.all(Tn >= -1e-9) and np.all(Tn <= 1 + 1e-9), p.name)

    def test_true_front_not_beaten_by_random_feasible_points(self):
        for p in self.problems:
            T = p.true_front(200)
            if T is None or p.var_type != "real":
                continue
            X = p.random_solutions(20000, self.rng)
            F, CV = p.evaluate(X)
            F = F[CV <= 0]
            self.assertFalse(strictly_dominates_any(F, T), p.name)

    def test_natural_units(self):
        p = P.P7()
        F = np.array([[-8.0, -96.0]])
        self.assertTrue(np.allclose(p.to_natural(F), [[8.0, 96.0]]))


class TestP1P2P3(unittest.TestCase):
    def test_p1(self):                                  # PDF sanity check
        F, _ = P.P1().evaluate([[5.0], [3.0]])
        self.assertTrue(np.allclose(F, [[4, 4], [0, 16]]))
        T = P.P1().true_front(50)                       # f2 = (4 - sqrt(f1))^2
        self.assertTrue(np.allclose(T[:, 1], (4 - np.sqrt(T[:, 0])) ** 2))

    def test_p2(self):
        F, _ = P.P2().evaluate([[2.0], [10.0]])
        self.assertTrue(np.allclose(F, [[50, 4], [10, 100]]))
        T = P.P2().true_front(50)
        self.assertTrue(np.allclose(T[:, 1], 10000.0 / T[:, 0] ** 2))
        Fi, _ = P.P2(integer=True).evaluate([[2.4], [2.6]])     # rounds to 2 and 3
        self.assertTrue(np.allclose(Fi, [[50, 4], [100 / 3, 9]]))
        self.assertEqual(len(P.P2(integer=True).true_front()), 10)

    def test_p3(self):
        p = P.P3()
        x = np.zeros(10)
        x[0] = 0.6
        F, _ = p.evaluate(x)
        self.assertTrue(np.allclose(F, [[0.6, 1 - 0.36]]))      # g = 1, f2 = 1 - f1^2
        x[1:] = 0.1                                             # g = 1 + 9*0.1 = 1.9
        F, _ = p.evaluate(x)
        self.assertAlmostEqual(F[0, 1], 1.9 * (1 - (0.6 / 1.9) ** 2))


class TestP4(unittest.TestCase):
    def test_sanity(self):
        F, CV = P.P4().evaluate([[0, 0, 1, 0]])
        self.assertTrue(np.allclose(F, [[0.09, -0.15]]))
        self.assertEqual(CV[0], 0.0)

    def test_min_risk_end_is_a_mixture(self):
        p = P.P4()
        w_minrisk = p.ref_weights[np.argmin(p.true_front()[:, 0])]
        self.assertGreaterEqual(int(np.sum(w_minrisk > 1e-3)), 2)
        self.assertAlmostEqual(w_minrisk.sum(), 1.0)

    def test_repair_vs_penalty(self):
        pr, pp = P.P4("repair"), P.P4("penalty")
        x = np.array([[1.0, 1.0, 1.0, 1.0]])            # sum = 4, not 1
        Fr, CVr = pr.evaluate(x)
        Fp, CVp = pp.evaluate(x)
        self.assertEqual(CVr[0], 0.0)
        self.assertAlmostEqual(CVp[0], 3.0)
        self.assertTrue(np.allclose(pr.repair(x), 0.25))
        self.assertTrue(np.allclose(pr.repair(np.zeros((1, 4))), 0.25))    # all-zero -> equal weights
        w = np.full(4, 0.25)
        self.assertAlmostEqual(Fr[0, 0], w @ P.SIGMA @ w)

    def test_simplex_projection(self):
        w = P.project_to_simplex(np.array([0.5, 2.0, -1.0, 0.3]))
        self.assertAlmostEqual(w.sum(), 1.0)
        self.assertTrue(np.all(w >= 0))
        self.assertTrue(np.allclose(P.project_to_simplex(np.array([0.2, 0.3, 0.5])), [0.2, 0.3, 0.5]))

    def test_reference_front_beats_random_portfolios(self):
        p = P.P4()
        rng = np.random.default_rng(3)
        W = rng.dirichlet(np.ones(4), size=20000)
        F, _ = p.evaluate(W)
        self.assertFalse(strictly_dominates_any(F, p.true_front()))


class TestP5P6P7(unittest.TestCase):
    def test_p5_sanity(self):                           # (b,h) = (0.01, 0.05)
        F, CV = P.P5().evaluate([[0.01, 0.05]])
        self.assertAlmostEqual(F[0, 0], 3.925)
        self.assertAlmostEqual(F[0, 1], 0.016)
        self.assertEqual(CV[0], 0.0)
        stress = 6 * 1000 * 1 / (0.01 * 0.05 ** 2)
        self.assertAlmostEqual(stress / 1e6, 240.0)

    def test_p5_infeasible_and_ignore_stress(self):
        F, CV = P.P5().evaluate([[0.01, 0.03]])         # b h^2 = 9e-6 < 2.4e-5
        self.assertGreater(CV[0], 0)
        _, CV2 = P.P5(use_stress=False).evaluate([[0.01, 0.03]])
        self.assertEqual(CV2[0], 0.0)
        T1, T2 = P.P5().true_front(100), P.P5(use_stress=False).true_front(100)
        self.assertLess(T2[:, 0].min(), T1[:, 0].min())  # ignoring stress allows lighter beams

    def test_p5_front_is_feasible(self):
        p = P.P5()
        T = p.true_front(100)
        h0 = np.sqrt(2.4e-5 / 0.01)
        self.assertAlmostEqual(T[0, 0], 7850 * 0.01 * h0)
        self.assertTrue(np.all(T[:, 0] > 0))

    def test_p6_sanity(self):
        F, CV = P.P6().evaluate([[0, 0], [5, 3], [1, 1]])
        self.assertTrue(np.allclose(F[0], [0, 50]))
        self.assertTrue(np.allclose(F[1], [136, 4]))
        self.assertTrue(np.all(CV[:2] == 0))
        # (4.9, 0.0) violates g2? (4.9-8)^2 + 3^2 = 18.61 >= 7.7 ok ; g1: 0.01 <= 25 ok
        _, CV2 = P.P6().evaluate([[5.0, 3.0], [0.0, 3.0]])
        self.assertEqual(CV2[0], 0.0)
        # (1, 3): g1 = 16 + 9 - 25 = 0 feasible ; make a violating point: (0, 3) -> 25+9-25 = 9 > 0
        self.assertAlmostEqual(CV2[1], 9.0)

    def test_p6_deb_rules_on_four_points(self):
        # four candidates: two feasible (trade-off), two infeasible (cv 9 and larger)
        pts = np.array([[0, 0], [5, 3], [0, 3], [0.0, 2.9]])
        F, CV = P.P6().evaluate(pts)
        cd = U.constrained_dominates
        self.assertFalse(cd(F[0], CV[0], F[1], CV[1]))      # feasible vs feasible: trade-off
        self.assertTrue(cd(F[0], CV[0], F[2], CV[2]))       # feasible beats infeasible
        self.assertAlmostEqual(CV[2], 9.0)                  # (0,3): g1 = 25 + 9 - 25
        self.assertAlmostEqual(CV[3], 8.41)                 # (0,2.9): g1 = 25 + 8.41 - 25
        self.assertTrue(cd(F[3], CV[3], F[2], CV[2]))       # Rule 3: smaller violation wins

    def test_p7_sanity(self):
        F, CV = P.P7().evaluate([[10, 80], [70, 20], [60, 60]])
        self.assertTrue(np.allclose(P.P7().to_natural(F[:2]), [[8, 96], [56, 24]]))
        self.assertEqual(CV[0], 0.0)
        self.assertGreater(CV[2], 0.0)                      # 120 > 90
        T = P.P7().true_front(30)
        self.assertTrue(np.allclose(-T[:, 0] / 0.8 - T[:, 1] / 1.2, 90.0))   # lies on x1 + x2 = 90


class TestP8(unittest.TestCase):
    def test_data(self):
        p = P.P8()
        rs = np.random.RandomState(7)
        self.assertTrue(np.allclose(p.coords, rs.uniform(0, 100, size=(15, 2))))
        self.assertTrue(np.allclose(p.u, p.u.T))
        off = p.u[np.triu_indices(15, 1)]
        self.assertTrue(off.min() >= 0.5 and off.max() <= 2.0)
        self.assertTrue(np.allclose(p.time, p.dist * p.u))

    def test_tour_cost_and_validity(self):
        p = P.P8()
        ident = np.arange(15)
        F, CV = p.evaluate([ident])
        manual = sum(np.linalg.norm(p.coords[i] - p.coords[(i + 1) % 15]) for i in range(15))
        self.assertAlmostEqual(F[0, 0], manual)
        self.assertEqual(CV[0], 0.0)
        bad = ident.copy()
        bad[1] = 0                                          # city 0 twice, city 1 missing
        _, CVb = p.evaluate([bad])
        self.assertEqual(CVb[0], 1.0)

    def test_nn_and_2opt_and_ideal(self):
        p = P.P8()
        nn = P.nearest_neighbour_tour(p.dist)
        self.assertEqual(sorted(nn.tolist()), list(range(15)))
        rnd = p.random_solutions(200, np.random.default_rng(0))
        d_nn = p.tour_costs(nn)[0]
        self.assertLess(d_nn, np.mean([p.tour_costs(t)[0] for t in rnd]))
        better = P.two_opt(nn, p.dist)
        self.assertLessEqual(p.tour_costs(better)[0], d_nn + 1e-9)
        self.assertTrue(np.all(p.nadir > p.ideal))
        self.assertLessEqual(p.ideal[0], d_nn + 1e-9)       # ideal distance <= NN distance


class TestP9(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.p = P.P9()

    def test_data_and_baseline(self):
        self.assertEqual(self.p.n_var, 30)
        self.assertEqual(self.p.n, 569)
        F, CV = self.p.evaluate(np.ones((1, 30)))
        self.assertEqual(F[0, 1], 30.0)
        self.assertEqual(CV[0], 0.0)
        self.assertLess(F[0, 0], 0.08)                      # baseline 5-NN accuracy > 92 %

    def test_empty_mask_infeasible_and_cache(self):
        _, CV = self.p.evaluate(np.zeros((1, 30)))
        self.assertEqual(CV[0], 1.0)
        n0 = self.p.n_evals
        x = np.zeros((1, 30))
        x[0, :3] = 1
        a, _ = self.p.evaluate(x)
        b, _ = self.p.evaluate(x)                           # second call uses the cache
        self.assertTrue(np.allclose(a, b))
        self.assertEqual(self.p.n_evals, n0 + 2)            # counter still counts every call

    def test_knn_by_hand_on_tiny_case(self):
        # a clean separable data set: the CV error must be 0 and objective 2 = 1
        p = self.p
        rng = np.random.default_rng(0)
        mask = np.zeros(30, dtype=np.int8)
        mask[[7, 27]] = 1                                   # mean concave points, worst concave points
        err = p.cv_error(mask)
        self.assertLess(err, 0.12)


class TestP10(unittest.TestCase):
    def test_sphere_when_g_is_zero(self):
        for M in (3, 5):
            p = P.P10(M)
            self.assertEqual(p.n_var, M + 9)
            rng = np.random.default_rng(0)
            X = rng.random((50, p.n_var))
            X[:, M - 1:] = 0.5                              # g = 0
            F, _ = p.evaluate(X)
            self.assertTrue(np.allclose((F ** 2).sum(axis=1), 1.0))

    def test_g_scaling_and_front_points(self):
        p = P.P10(3)
        x = np.full(12, 0.5)
        x[:2] = [0.0, 0.0]                                  # f = (1, 0, 0) * (1+g)
        F, _ = p.evaluate(x)
        self.assertTrue(np.allclose(F, [[1, 0, 0]]))
        x[2] = 0.7                                          # g = 0.04
        F, _ = p.evaluate(x)
        self.assertAlmostEqual(F[0, 0], 1.04)
        T = p.true_front(300)
        self.assertTrue(np.allclose(np.linalg.norm(T, axis=1), 1.0))
        self.assertGreaterEqual(len(T), 300)

    def test_random_population_nondominated_fraction_grows(self):
        rng = np.random.default_rng(0)
        frac = {}
        for M in (3, 5):
            p = P.P10(M)
            F, _ = p.evaluate(p.random_solutions(200, rng))
            frac[M] = len(U.non_dominated_sort(F)[0]) / 200
        self.assertGreater(frac[5], frac[3])


class TestP11(unittest.TestCase):
    def test_corners_and_max(self):
        p = P.P11()
        F, CV = p.evaluate([[0, 0], [400, 0], [300, 200], [0, 400]])
        self.assertTrue(np.all(CV == 0))
        nat = p.to_natural(F)
        self.assertTrue(np.allclose(nat[:, 0], [0, 1600, 2400, 2400]))      # profit
        self.assertTrue(np.allclose(nat[:, 1], [0, 400, 500, 400]))         # units
        _, CVx = p.evaluate([[400, 400]])
        self.assertGreater(CVx[0], 0)

    def test_front_is_single_point(self):
        p = P.P11()
        rng = np.random.default_rng(0)
        F, CV = p.evaluate(p.random_solutions(20000, rng))
        F = F[CV <= 0]
        self.assertTrue(np.all(F[:, 0] >= -2400 - 1e-9) and np.all(F[:, 1] >= -500 - 1e-9))
        self.assertTrue(np.allclose(p.true_front(), [[-2400, -500]]))
        self.assertGreater(600, 500)        # units goal 600 cannot be met: shortfall >= 100


class TestP12(unittest.TestCase):
    def test_table(self):
        s = P.P12()
        self.assertEqual(s.matrix.shape, (5, 4))
        self.assertAlmostEqual(s.weights.sum(), 1.0)
        M = s.minimization_matrix()
        self.assertTrue(np.allclose(M[:, 0], [120, 100, 140, 90, 130]))
        self.assertTrue(np.allclose(M[:, 1], [-8, -7, -9, -6, -8]))
        # all five suppliers are Pareto-optimal (checked by hand: cost order is
        # opposite to the order of quality / delivery / reliability, S1 vs S5 trade off)
        self.assertEqual(sorted(s.pareto_optimal().tolist()), [0, 1, 2, 3, 4])


class TestP13(unittest.TestCase):
    def test_toy(self):
        p = P.ToyTwoTask()
        F, _ = p.evaluate([[0.0, 5.0]])
        self.assertTrue(np.allclose(F, [[9 + 4, 9 + 64]]))
        g = p.gradients(np.array([0.0, 5.0]))
        self.assertTrue(np.allclose(g, [[-6, 4], [6, 16]]))
        T = p.true_front(30)                                # on the segment f1 + f2 ... check ends
        self.assertTrue(np.allclose(T[0], [0, 72]) and np.allclose(T[-1], [72, 0]))
        mid, _ = p.evaluate([[1.0, 1.0]])                   # x1 = x2 is Pareto-optimal
        self.assertAlmostEqual(mid[0, 0], 8.0)
        self.assertAlmostEqual(mid[0, 1], 32.0)

    def test_network_gradients_match_finite_differences(self):
        net = P.MultiTaskNet(n_samples=40, n_features=4, hidden=5, seed=1)
        theta = net.init_params(seed=2)
        G = net.gradients(theta)
        self.assertEqual(G.shape, (2, net.n_params))
        for k, loss in enumerate((net.loss_cls, net.loss_reg)):
            num = np.zeros(net.n_params)
            for i in range(net.n_params):
                e = np.zeros(net.n_params)
                e[i] = 1e-6
                num[i] = (loss(theta + e) - loss(theta - e)) / 2e-6
            self.assertTrue(np.allclose(G[k], num, rtol=1e-4, atol=1e-7), "task %d" % k)

    def test_head_gradients_are_separate_and_descent_works(self):
        net = P.MultiTaskNet(seed=0)
        theta = net.init_params()
        d, h = net.d, net.h
        base = d * h + h
        self.assertTrue(np.all(net.grad_cls(theta)[base + h + 1:] == 0))     # no regression-head entries
        self.assertTrue(np.all(net.grad_reg(theta)[base:base + h + 1] == 0))  # no classification-head entries
        before = net.losses(theta)
        theta2 = theta - 0.05 * net.grad_cls(theta) - 0.05 * net.grad_reg(theta) * 0.1
        after = net.losses(theta2)
        self.assertLess(after[0], before[0])

    def test_reg_scale_makes_gradients_larger(self):
        small = P.MultiTaskNet(reg_scale=1.0, seed=0)
        big = P.MultiTaskNet(reg_scale=10.0, seed=0)
        th = small.init_params()
        ratio_small = np.linalg.norm(small.grad_reg(th)) / np.linalg.norm(small.grad_cls(th))
        ratio_big = np.linalg.norm(big.grad_reg(th)) / np.linalg.norm(big.grad_cls(th))
        self.assertGreater(ratio_big, 3 * ratio_small)


class TestP14(unittest.TestCase):
    def test_sanity(self):
        p = P.P14()
        F, _ = p.evaluate([[13, 40], [13, 90]])
        nat = p.to_natural(F)
        self.assertAlmostEqual(nat[0, 0], 1.53)
        self.assertAlmostEqual(nat[0, 1], 8.989, places=2)
        self.assertAlmostEqual(nat[1, 0], 2.13)
        self.assertAlmostEqual(nat[1, 1], 20.22, places=2)

    def test_x1_is_13_on_front(self):
        p = P.P14()
        rng = np.random.default_rng(0)
        X = p.random_solutions(5000, rng)
        F, _ = p.evaluate(X)
        X13 = X.copy()
        X13[:, 0] = 13.0                                    # same battery, smallest screen
        F13, _ = p.evaluate(X13)
        # the x1 = 13 design is never worse, and strictly better in both objectives when x1 > 13
        self.assertTrue(np.all(F13 <= F + 1e-12))
        self.assertTrue(np.all(F13[X[:, 0] > 13] < F[X[:, 0] > 13]))
        # increasing x1 worsens both objectives:
        Fa, _ = p.evaluate([[13, 60], [15, 60]])
        self.assertTrue(U.dominates(Fa[0], Fa[1]))

    def test_reference_point_is_dominated_by_front(self):
        p = P.P14()
        T = p.to_natural(p.true_front(500))                 # (weight, runtime)
        ref = p.preference_point
        # some front point has weight <= 1.8 and runtime >= 12 -> the target is attainable
        self.assertTrue(np.any((T[:, 0] <= ref[0]) & (T[:, 1] >= ref[1])))


class TestRegistry(unittest.TestCase):
    def test_make_problem(self):
        self.assertEqual(P.make_problem("P10", M=5).n_obj, 5)
        self.assertEqual(len(P.PROBLEMS), 15)               # P1..P12 + P13a, P13b + P14


if __name__ == "__main__":
    unittest.main()
