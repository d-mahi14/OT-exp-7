"""Unit tests for moo_utils.py.  Every expected value is checkable by hand and
most come from the lecture slides.  Run from the project root:
    python -m unittest discover -s tests -t . -v
"""
import unittest
import numpy as np
import moo_utils as U


class TestDominance(unittest.TestCase):
    def test_basic(self):
        self.assertTrue(U.dominates([1, 2], [2, 2]))       # better in one, equal in other
        self.assertFalse(U.dominates([1, 3], [2, 2]))      # trade-off
        self.assertFalse(U.dominates([2, 2], [2, 2]))      # equal does not dominate

    def test_lecture_car_example(self):
        # slide 8: Min cost, Max mileage -> store (cost, -mileage)
        cars = {"A": (5, 15), "B": (8, 22), "C": (6, 18), "D": (10, 25), "E": (7, 12), "F": (9, 20)}
        names = list(cars)
        F = np.array([[c, -m] for c, m in cars.values()], dtype=float)
        self.assertTrue(U.dominates(F[names.index("A")], F[names.index("E")]))   # A dominates E
        self.assertTrue(U.dominates(F[names.index("B")], F[names.index("F")]))   # B dominates F
        front = U.non_dominated_sort(F)[0]
        self.assertEqual(sorted(names[i] for i in front), ["A", "B", "C", "D"])

    def test_lecture_portfolio_example(self):
        # slide 11: Min risk, Max return.  Front = {P1,P2,P3}.
        # (The slide says P5 is dominated by P2; it is really dominated by P3:
        #  P2 has return 12 < 14.)  The front is the same either way.
        pts = {"P1": (5, 8), "P2": (8, 12), "P3": (10, 15), "P4": (6, 7), "P5": (12, 14)}
        names = list(pts)
        F = np.array([[r, -ret] for r, ret in pts.values()], dtype=float)
        front = U.non_dominated_sort(F)[0]
        self.assertEqual(sorted(names[i] for i in front), ["P1", "P2", "P3"])
        self.assertTrue(U.dominates(F[2], F[4]))          # P3 dominates P5
        self.assertFalse(U.dominates(F[1], F[4]))         # P2 does NOT dominate P5


class TestConstraints(unittest.TestCase):
    def test_cv(self):
        self.assertAlmostEqual(U.constraint_violation(g=[-1, 2], h=[0.5]), 2.5)
        self.assertEqual(U.constraint_violation(), 0.0)

    def test_deb_table(self):
        # slide 22: A, D feasible ; B (cv 2), C (cv 5) infeasible
        A, D = ([3, 6], 0.0), ([4, 5], 0.0)
        B, C = ([1, 2], 2.0), ([0.5, 1], 5.0)
        cd = U.constrained_dominates
        # Rule 1: feasible beats infeasible, whatever the objectives
        for feas in (A, D):
            for infe in (B, C):
                self.assertTrue(cd(feas[0], feas[1], infe[0], infe[1]))
                self.assertFalse(cd(infe[0], infe[1], feas[0], feas[1]))
        # Rule 2: A vs D is a genuine trade-off
        self.assertFalse(cd(A[0], A[1], D[0], D[1]))
        self.assertFalse(cd(D[0], D[1], A[0], A[1]))
        # Rule 3: B beats C although C's objectives look better
        self.assertTrue(cd(B[0], B[1], C[0], C[1]))
        self.assertFalse(cd(C[0], C[1], B[0], B[1]))

    def test_sort_with_constraints(self):
        F = np.array([[3, 6], [4, 5], [1, 2], [0.5, 1]], dtype=float)   # A, D, B, C
        CV = np.array([0, 0, 2, 5], dtype=float)
        fronts = U.non_dominated_sort(F, CV)
        self.assertEqual(sorted(fronts[0]), [0, 1])     # A, D
        self.assertEqual(list(fronts[1]), [2])          # B
        self.assertEqual(list(fronts[2]), [3])          # C


class TestSorting(unittest.TestCase):
    def test_three_fronts(self):
        F = np.array([[1, 4], [2, 2], [4, 1],      # front 1
                      [2, 4], [3, 3], [5, 2],      # front 2
                      [4, 5]], dtype=float)        # front 3
        fronts = U.non_dominated_sort(F)
        self.assertEqual([sorted(f.tolist()) for f in fronts], [[0, 1, 2], [3, 4, 5], [6]])
        rank = U.ranks_from_fronts(fronts, len(F))
        self.assertEqual(rank.tolist(), [0, 0, 0, 1, 1, 1, 2])

    def test_matches_brute_force(self):
        rng = np.random.default_rng(1)
        F = rng.random((60, 3))
        brute = [i for i in range(60) if not any(U.dominates(F[j], F[i]) for j in range(60))]
        self.assertEqual(sorted(U.non_dominated_sort(F)[0].tolist()), brute)

    def test_pareto_front_drops_infeasible_and_duplicates(self):
        F = np.array([[1, 1], [1, 1], [0.5, 0.5], [2, 0.1]], dtype=float)
        CV = np.array([0, 0, 1.0, 0])
        idx = U.pareto_front(F, CV)             # (0.5,0.5) infeasible, duplicate removed
        self.assertEqual(sorted(idx.tolist()), [0, 3])
        self.assertEqual(len(U.pareto_front(F, np.ones(4))), 0)


class TestCrowding(unittest.TestCase):
    def test_hand_example(self):
        F = np.array([[0, 4], [1, 2], [2, 1], [4, 0]], dtype=float)
        d = U.crowding_distance(F)
        self.assertTrue(np.isinf(d[0]) and np.isinf(d[3]))
        # point (1,2): f1 term (2-0)/4 = 0.5, f2 term (4-1)/4 = 0.75
        self.assertAlmostEqual(d[1], 1.25)
        self.assertAlmostEqual(d[2], 1.25)

    def test_small_fronts(self):
        self.assertTrue(np.all(np.isinf(U.crowding_distance(np.array([[1.0, 2.0]])))))
        self.assertTrue(np.all(np.isinf(U.crowding_distance(np.array([[1.0, 2.0], [2.0, 1.0]])))))

    def test_all(self):
        F = np.array([[0, 4], [1, 2], [2, 1], [4, 0], [5, 5]], dtype=float)
        fronts = U.non_dominated_sort(F)
        d = U.crowding_distance_all(F, fronts)
        self.assertAlmostEqual(d[1], 1.25)
        self.assertTrue(np.isinf(d[4]))          # single-point front 2


class TestNormalize(unittest.TestCase):
    def test_given_and_default(self):
        Fn, ideal, nadir = U.normalize_objectives([[2, 10], [4, 30]], [0, 0], [4, 20])
        self.assertTrue(np.allclose(Fn, [[0.5, 0.5], [1.0, 1.5]]))
        Fn, ideal, nadir = U.normalize_objectives([[2, 10], [4, 30]])
        self.assertTrue(np.allclose(Fn, [[0, 0], [1, 1]]))
        Fn, _, _ = U.normalize_objectives([[1, 5], [1, 7]])      # zero range -> no NaN
        self.assertFalse(np.any(np.isnan(Fn)))


class TestHypervolume(unittest.TestCase):
    def test_2d_hand(self):
        self.assertAlmostEqual(U.hv_2d([[1, 3], [2, 2], [3, 1]], [4, 4]), 6.0)
        self.assertAlmostEqual(U.hv_2d([[1, 3], [2, 2], [3, 1], [3.5, 3.5]], [4, 4]), 6.0)  # dominated point
        self.assertAlmostEqual(U.hv_2d([[0, 0]], [1, 1]), 1.0)
        self.assertEqual(U.hv_2d([[5, 5]], [4, 4]), 0.0)                    # outside ref point

    def test_mc_3d_known_values(self):
        self.assertAlmostEqual(U.hv_monte_carlo([[0.5, 0.5, 0.5]], [1, 1, 1]), 0.125)
        F = [[0, .5, .5], [.5, 0, .5]]         # 0.25 + 0.25 - 0.125 = 0.375 by inclusion-exclusion
        self.assertAlmostEqual(U.hv_monte_carlo(F, [1, 1, 1], 300000), 0.375, delta=0.01)

    def test_mc_matches_exact_2d(self):
        rng = np.random.default_rng(0)
        F = rng.random((30, 2))
        exact = U.hv_2d(F, [1.1, 1.1])
        approx = U.hv_monte_carlo(F, [1.1, 1.1], 300000)
        self.assertAlmostEqual(approx, exact, delta=0.01)

    def test_dispatcher_and_reproducible(self):
        F = np.array([[0.2, 0.3, 0.4], [0.4, 0.2, 0.3]])
        self.assertEqual(U.hypervolume(F, [1, 1, 1], seed=3), U.hypervolume(F, [1, 1, 1], seed=3))
        self.assertAlmostEqual(U.hypervolume([[1, 3], [2, 2], [3, 1]], [4, 4]), 6.0)


class TestIGDSpacing(unittest.TestCase):
    def test_igd(self):
        self.assertAlmostEqual(U.igd([[0, 0], [1, 1]], [[0, 1]]), 1.0)
        T = np.array([[0, 1], [1, 0]])
        self.assertAlmostEqual(U.igd(T, T), 0.0)
        self.assertEqual(U.igd(T, np.empty((0, 2))), float("inf"))

    def test_spacing(self):
        even = np.array([[0, 3], [1, 2], [2, 1], [3, 0]], dtype=float)
        self.assertAlmostEqual(U.spacing(even), 0.0)
        line = np.array([[0, 0], [1, 0], [3, 0]], dtype=float)   # nn distances 1, 1, 2
        self.assertAlmostEqual(U.spacing(line), np.sqrt(1 / 3))
        self.assertEqual(U.spacing(np.array([[1.0, 2.0]])), 0.0)


class TestDasDennis(unittest.TestCase):
    def test_counts_and_sums(self):
        self.assertTrue(np.allclose(U.das_dennis(2, 2), [[0, 1], [0.5, 0.5], [1, 0]]))
        W = U.das_dennis(3, 4)
        self.assertEqual(len(W), 15)                       # C(6, 2)
        self.assertTrue(np.allclose(W.sum(axis=1), 1.0))
        self.assertEqual(len(U.das_dennis(5, 6)), 210)     # C(10, 4)


class TestHelpers(unittest.TestCase):
    def test_population_and_clip(self):
        rng = np.random.default_rng(0)
        P = U.random_population([0, 10], [1, 20], 50, rng)
        self.assertEqual(P.shape, (50, 2))
        self.assertTrue(np.all(P[:, 0] >= 0) and np.all(P[:, 1] <= 20))
        self.assertTrue(np.allclose(U.clip_to_bounds([-1, 5], [0, 0], [1, 1]), [0, 1]))

    def test_lazy_reexport(self):
        from moo_utils import P3, P1
        self.assertEqual(P3().n_var, 10)
        self.assertEqual(P1().n_obj, 2)
        with self.assertRaises(AttributeError):
            U.does_not_exist


if __name__ == "__main__":
    unittest.main()
