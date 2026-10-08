"""
problems.py -- the 14 problems of Lab Assignment 7 (Section 4 of the PDF).

Every optimisation problem is a subclass of `Problem` and works like this:

    prob = P3()
    F, CV = prob.evaluate(X)      # X: (N, n_var)  ->  F: (N, n_obj), CV: (N,)

* F is ALWAYS in MINIMIZATION form.  Maximization objectives are stored as -f.
  `prob.to_natural(F)` flips the sign back (e.g. for plots in natural units).
* CV is the total constraint violation (0 = feasible), see moo_utils.
* prob.n_evals counts the evaluated solutions (use it for the equal budget).
* prob.ideal / prob.nadir are used to normalise objectives before HV / IGD /
  Spacing; prob.hv_ref() is the HV reference point in normalised space (1.1).
* prob.true_front(n) returns points of the true Pareto front (min-form) or
  None when it is not known (P8, P9).

DESIGN CHOICES that are NOT given in the PDF are marked "DESIGN CHOICE".
"""
import functools
import math
import os
import numpy as np

import moo_utils as U


# =============================================================================
# Base class
# =============================================================================
class Problem:
    """Common interface shared by all problems."""
    name = "problem"
    var_type = "real"            # "real", "int", "perm" or "binary"

    def __init__(self, n_var, n_obj, lower, upper, maximize=None):
        self.n_var = n_var
        self.n_obj = n_obj
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        # which ORIGINAL objectives are maximised (stored internally as -f)
        self.maximize = np.zeros(n_obj, dtype=bool) if maximize is None \
            else np.asarray(maximize, dtype=bool)
        self.ideal = np.zeros(n_obj)
        self.nadir = np.ones(n_obj)
        self.n_evals = 0

    # ---- to be written by each problem -------------------------------------
    def _evaluate_one(self, x):
        """Return (f array in min-form, constraint violation float)."""
        raise NotImplementedError

    # ---- common behaviour ---------------------------------------------------
    def evaluate(self, X):
        """Evaluate many solutions. X shape (N, n_var) or a single (n_var,)."""
        X = np.atleast_2d(X)
        F = np.zeros((len(X), self.n_obj))
        CV = np.zeros(len(X))
        for i in range(len(X)):
            F[i], CV[i] = self._evaluate_one(X[i])
        self.n_evals += len(X)
        return F, CV

    def reset_counter(self):
        self.n_evals = 0

    def random_solutions(self, N, rng):
        """N random solutions (rng = np.random.default_rng(seed))."""
        return U.random_population(self.lower, self.upper, N, rng)

    def true_front(self, n=200):
        return None

    def normalize(self, F):
        """(F - ideal) / (nadir - ideal): the true front maps into [0, 1]."""
        return U.normalize_objectives(F, self.ideal, self.nadir)[0]

    def hv_ref(self):
        """Hypervolume reference point in NORMALISED space (DESIGN CHOICE: 1.1)."""
        return np.full(self.n_obj, 1.1)

    def to_natural(self, F):
        """Undo the sign flip of maximised objectives."""
        F = np.array(F, dtype=float)
        F[..., self.maximize] *= -1.0
        return F


# =============================================================================
# P1 -- Factory production speed (convex, 1 variable)
# =============================================================================
class P1(Problem):
    """min f1=(x-3)^2, min f2=(x-7)^2, x in [1,10].  Pareto set 3<=x<=7."""
    name = "P1 Factory production speed"

    def __init__(self):
        super().__init__(1, 2, [1.0], [10.0])
        self.ideal = np.array([0.0, 0.0])
        self.nadir = np.array([16.0, 16.0])      # f1(7)=f2(3)=16

    def _evaluate_one(self, x):
        x = x[0]
        return np.array([(x - 3.0) ** 2, (x - 7.0) ** 2]), 0.0

    def true_front(self, n=200):
        x = np.linspace(3.0, 7.0, n)
        return np.column_stack(((x - 3.0) ** 2, (x - 7.0) ** 2))


# =============================================================================
# P2 -- Delivery fleet sizing (real or integer)
# =============================================================================
class P2(Problem):
    """min f1=100/x, min f2=x^2, x in [1,10].  EVERY x is Pareto-optimal.
    P2(integer=True): x is rounded to the nearest integer before evaluation
    (DESIGN CHOICE), so only 10 distinct front points exist."""
    name = "P2 Delivery fleet sizing"

    def __init__(self, integer=False):
        super().__init__(1, 2, [1.0], [10.0])
        self.integer = integer
        self.var_type = "int" if integer else "real"
        self.ideal = np.array([10.0, 1.0])
        self.nadir = np.array([100.0, 100.0])

    def _evaluate_one(self, x):
        x = x[0]
        if self.integer:
            x = float(np.round(x))
        return np.array([100.0 / x, x * x]), 0.0

    def random_solutions(self, N, rng):
        if self.integer:
            return rng.integers(1, 11, size=(N, 1)).astype(float)
        return super().random_solutions(N, rng)

    def true_front(self, n=200):
        x = np.arange(1, 11, dtype=float) if self.integer else np.linspace(1, 10, n)
        return np.column_stack((100.0 / x, x * x))


# =============================================================================
# P3 -- ZDT2 (non-convex, n = 10)
# =============================================================================
class P3(Problem):
    """f1 = x1 ; g = 1 + sum_{i=2..10} x_i ; f2 = g (1 - (f1/g)^2).
    (sum x_i  ==  9 * mean(x_2..x_10)  because n-1 = 9.)  True front: f2 = 1 - f1^2."""
    name = "P3 ZDT2"

    def __init__(self, n_var=10):
        super().__init__(n_var, 2, np.zeros(n_var), np.ones(n_var))
        self.ideal = np.array([0.0, 0.0])
        self.nadir = np.array([1.0, 1.0])

    def _evaluate_one(self, x):
        f1 = x[0]
        g = 1.0 + 9.0 * np.sum(x[1:]) / (self.n_var - 1)
        f2 = g * (1.0 - (f1 / g) ** 2)
        return np.array([f1, f2]), 0.0

    def true_front(self, n=200):
        f1 = np.linspace(0.0, 1.0, n)
        return np.column_stack((f1, 1.0 - f1 ** 2))


# =============================================================================
# P4 -- Investment portfolio (equality constraint)
# =============================================================================
MU = np.array([0.08, 0.12, 0.15, 0.10])
SIGMA = np.array([[0.010, 0.002, 0.001, 0.003],
                  [0.002, 0.040, 0.006, 0.004],
                  [0.001, 0.006, 0.090, 0.005],
                  [0.003, 0.004, 0.005, 0.020]])


def project_to_simplex(v):
    """Closest point (Euclidean) to v with w >= 0 and sum(w) = 1.
    Standard sort-based algorithm; used only to build the reference front."""
    u = np.sort(v)[::-1]
    css = np.cumsum(u)
    k = np.arange(1, len(v) + 1)
    rho = k[u - (css - 1.0) / k > 0][-1]
    theta = (css[rho - 1] - 1.0) / rho
    return np.maximum(v - theta, 0.0)


@functools.lru_cache(maxsize=1)
def _portfolio_reference():
    """True efficient frontier of P4 (the problem is a convex QP).

    For many values t >= 0 we minimise  risk(w) - t*return(w)  over the simplex
    by projected gradient descent (own code, step 1/L).  Every t gives one
    point of the front.  Returns (F_front [risk, -return], weights)."""
    L = 2.0 * np.max(np.linalg.eigvalsh(SIGMA))        # Lipschitz constant
    ts = np.concatenate(([0.0], np.logspace(-3, 2.5, 400)))
    w = np.full(4, 0.25)
    W = []
    for t in ts:
        for _ in range(5000):
            grad = 2.0 * SIGMA @ w - t * MU
            w_new = project_to_simplex(w - grad / L)
            if np.max(np.abs(w_new - w)) < 1e-13:
                w = w_new
                break
            w = w_new
        W.append(w.copy())
    W = np.array(W)
    F = np.column_stack((np.einsum("ij,jk,ik->i", W, SIGMA, W), -(W @ MU)))
    keep = U.pareto_front(F)
    return F[keep], W[keep]


class P4(Problem):
    """Portfolio: min risk = w'Sw, max return = mu'w (stored as -mu'w),
    w >= 0, sum(w) = 1.

    handling="repair"  : evaluate() first replaces x by x/sum(x) (negative
                         values clipped to 0).  The equality constraint then
                         always holds, CV = 0.  Use prob.repair(X) to write the
                         repaired weights back into your population.
    handling="penalty" : x is used as it is and CV = |sum(x) - 1|; the
                         algorithm's constraint rule (Deb) punishes violators.
    """
    name = "P4 Investment portfolio"

    def __init__(self, handling="repair"):
        super().__init__(4, 2, np.zeros(4), np.ones(4), maximize=[False, True])
        assert handling in ("repair", "penalty")
        self.handling = handling
        F, self.ref_weights = _portfolio_reference()
        self._front = F
        self.ideal = np.array([F[:, 0].min(), F[:, 1].min()])
        self.nadir = np.array([F[:, 0].max(), F[:, 1].max()])

    @staticmethod
    def repair(X):
        """Clip negatives to 0 and divide by the sum (equal weights if sum = 0)."""
        X = np.maximum(np.asarray(X, dtype=float), 0.0)
        s = X.sum(axis=-1, keepdims=True)
        return np.where(s > 0, X / np.where(s > 0, s, 1.0), 1.0 / X.shape[-1])

    def _evaluate_one(self, x):
        if self.handling == "repair":
            w, cv = self.repair(x), 0.0
        else:
            w, cv = x, abs(float(np.sum(x)) - 1.0)
        return np.array([w @ SIGMA @ w, -(MU @ w)]), cv

    def true_front(self, n=200):
        return self._front.copy()


# =============================================================================
# P5 -- Cantilever beam (constrained)
# =============================================================================
class P5(Problem):
    """b in [0.01,0.10], h in [0.02,0.20].  min mass = 7850*L*b*h,
    min deflection = 4PL^3/(E b h^3), constraint b*h^2 >= 2.4e-5.
    The constraint is written in normalised form g = (2.4e-5 - b h^2)/2.4e-5 <= 0
    (DESIGN CHOICE: dividing by 2.4e-5 keeps CV on an O(1) scale).
    P5(use_stress=False) ignores the stress limit (for the 'what if' question)."""
    name = "P5 Cantilever beam"
    L, P, E, RHO, C = 1.0, 1000.0, 200e9, 7850.0, 2.4e-5

    def __init__(self, use_stress=True):
        super().__init__(2, 2, [0.01, 0.02], [0.10, 0.20])
        self.use_stress = use_stress
        T = self.true_front(400)
        self.ideal = T.min(axis=0)
        self.nadir = T.max(axis=0)

    def _evaluate_one(self, x):
        b, h = x
        mass = self.RHO * self.L * b * h
        defl = 4.0 * self.P * self.L ** 3 / (self.E * b * h ** 3)
        cv = max(0.0, (self.C - b * h * h) / self.C) if self.use_stress else 0.0
        return np.array([mass, defl]), cv

    def true_front(self, n=200):
        """Analytic front.  For a given area A=b*h the deflection is smallest
        for the largest h.  So the front is: b = 0.01 with h growing from its
        lowest feasible value up to 0.2, then h = 0.2 with b growing to 0.1."""
        h0 = math.sqrt(self.C / 0.01) if self.use_stress else 0.02
        h0 = max(h0, 0.02)
        h1 = np.linspace(h0, 0.20, n // 2)
        b1 = np.full_like(h1, 0.01)
        b2 = np.linspace(0.01, 0.10, n - n // 2 + 1)[1:]     # skip the shared junction point
        h2 = np.full_like(b2, 0.20)
        b, h = np.concatenate((b1, b2)), np.concatenate((h1, h2))
        return np.column_stack((self.RHO * self.L * b * h,
                                4 * self.P * self.L ** 3 / (self.E * b * h ** 3)))


# =============================================================================
# P6 -- Binh-Korn
# =============================================================================
class P6(Problem):
    """x1 in [0,5], x2 in [0,3]; f1=4x1^2+4x2^2, f2=(x1-5)^2+(x2-5)^2;
    g1=(x1-5)^2+x2^2-25 <= 0 ; g2=7.7-((x1-8)^2+(x2+3)^2) <= 0."""
    name = "P6 Binh-Korn"

    def __init__(self):
        super().__init__(2, 2, [0.0, 0.0], [5.0, 3.0])
        self.ideal = np.array([0.0, 4.0])
        self.nadir = np.array([136.0, 50.0])

    def _evaluate_one(self, x):
        x1, x2 = x
        f = np.array([4 * x1 ** 2 + 4 * x2 ** 2, (x1 - 5) ** 2 + (x2 - 5) ** 2])
        g = [(x1 - 5) ** 2 + x2 ** 2 - 25.0, 7.7 - ((x1 - 8) ** 2 + (x2 + 3) ** 2)]
        return f, U.constraint_violation(g=g)

    def true_front(self, n=200):
        """Known Pareto set: x1 = x2 for 0<=x1<=3, then x2 = 3 for 3<=x1<=5."""
        t = np.linspace(0.0, 3.0, n // 2)
        s = np.linspace(3.0, 5.0, n - n // 2 + 1)[1:]        # skip the shared junction point
        x1 = np.concatenate((t, s))
        x2 = np.concatenate((t, np.full_like(s, 3.0)))
        return np.column_stack((4 * x1 ** 2 + 4 * x2 ** 2,
                                (x1 - 5) ** 2 + (x2 - 5) ** 2))


# =============================================================================
# P7 -- Water allocation (maximisation, linear)
# =============================================================================
class P7(Problem):
    """max yield 0.8*x1 and output 1.2*x2 (stored as negatives);
    x1 + x2 <= 90, x1 >= 10, x2 >= 20.
    Bounds x1 in [10,70], x2 in [20,80] follow from the three constraints
    (DESIGN CHOICE: the sum constraint is kept as g, normalised by 90).
    prob.preference_point = (40, 50) is the reference point of the PDF
    (natural units: yield, output)."""
    name = "P7 Water allocation"
    preference_point = np.array([40.0, 50.0])

    def __init__(self):
        super().__init__(2, 2, [10.0, 20.0], [70.0, 80.0], maximize=[True, True])
        self.ideal = np.array([-56.0, -96.0])
        self.nadir = np.array([-8.0, -24.0])

    def _evaluate_one(self, x):
        x1, x2 = x
        cv = max(0.0, (x1 + x2 - 90.0) / 90.0)
        return np.array([-0.8 * x1, -1.2 * x2]), cv

    def true_front(self, n=200):
        x1 = np.linspace(10.0, 70.0, n)
        return np.column_stack((-0.8 * x1, -1.2 * (90.0 - x1)))


# =============================================================================
# P8 -- Bi-objective TSP (15 cities)
# =============================================================================
def nearest_neighbour_tour(cost, start=0):
    """Greedy tour: always go to the closest unvisited city."""
    n = len(cost)
    tour, left = [start], set(range(n)) - {start}
    while left:
        last = tour[-1]
        nxt = min(left, key=lambda c: cost[last, c])
        tour.append(nxt)
        left.remove(nxt)
    return np.array(tour)


def two_opt(tour, cost):
    """Simple 2-opt local search for a SYMMETRIC cost matrix (own code)."""
    tour = np.array(tour)
    n = len(tour)
    improved = True
    while improved:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                a, b = tour[i - 1], tour[i]
                c, d = tour[j], tour[(j + 1) % n]
                delta = cost[a, c] + cost[b, d] - cost[a, b] - cost[c, d]
                if delta < -1e-12:
                    tour[i:j + 1] = tour[i:j + 1][::-1].copy()
                    improved = True
    return tour


class P8(Problem):
    """15 cities, coordinates uniform in [0,100]^2 from seed 7.

    DESIGN CHOICE (the PDF fixes only the seed): with
    rs = np.random.RandomState(7) we first draw coords = rs.uniform(0,100,(15,2))
    and then U = rs.uniform(0.5,2.0,(15,15)); the symmetric matrix is
    u = triu(U,1) + triu(U,1).T.  This gives the same numbers as
    np.random.seed(7) but does not disturb the global generator.
    Time t_ij = d_ij * u_ij.  A solution is a permutation of 0..14 (closed tour).
    ideal/nadir (DESIGN CHOICE): ideal = (shortest distance, shortest time) found
    by 2-opt from NN + 30 random starts for each cost alone; nadir = (distance
    of the shortest-time tour, time of the shortest-distance tour)."""
    name = "P8 Bi-objective TSP"
    var_type = "perm"

    def __init__(self, n_cities=15, seed=7):
        super().__init__(n_cities, 2, np.zeros(n_cities), np.full(n_cities, n_cities - 1.0))
        rs = np.random.RandomState(seed)
        self.coords = rs.uniform(0.0, 100.0, size=(n_cities, 2))
        diff = self.coords[:, None, :] - self.coords[None, :, :]
        self.dist = np.linalg.norm(diff, axis=2)
        u = np.triu(rs.uniform(0.5, 2.0, size=(n_cities, n_cities)), 1)
        self.u = u + u.T
        self.time = self.dist * self.u
        self._estimate_normalization()

    def tour_costs(self, tour):
        t = np.asarray(tour, dtype=int)
        nxt = np.roll(t, -1)
        return float(self.dist[t, nxt].sum()), float(self.time[t, nxt].sum())

    def _estimate_normalization(self):
        rs = np.random.RandomState(0)
        best = {}
        for key, cost in (("dist", self.dist), ("time", self.time)):
            starts = [nearest_neighbour_tour(cost)] + \
                     [rs.permutation(self.n_var) for _ in range(30)]
            tours = [two_opt(s, cost) for s in starts]
            best[key] = min(tours, key=lambda t: self.tour_costs(t)[0 if key == "dist" else 1])
        d_best, t_of_d = self.tour_costs(best["dist"])
        d_of_t, t_best = self.tour_costs(best["time"])
        self.best_dist_tour, self.best_time_tour = best["dist"], best["time"]
        self.ideal = np.array([d_best, t_best])
        self.nadir = np.array([d_of_t, t_of_d])

    def _evaluate_one(self, x):
        t = np.asarray(x, dtype=int)
        missing = self.n_var - len(np.unique(t))      # 0 for a valid permutation
        d, tm = self.tour_costs(t) if missing == 0 else (1e9, 1e9)
        return np.array([d, tm]), float(missing)

    def random_solutions(self, N, rng):
        return np.array([rng.permutation(self.n_var) for _ in range(N)])


# =============================================================================
# P9 -- Feature selection (binary)
# =============================================================================
def load_breast_cancer_data():
    """Wisconsin breast-cancer data: (X (569,30), y (569,), feature_names).
    Uses scikit-learn only to LOAD the data; if it is missing the CSV
    data/breast_cancer.csv (same data) is read instead."""
    try:
        from sklearn.datasets import load_breast_cancer
        d = load_breast_cancer()
        return d.data.astype(float), d.target.astype(int), list(d.feature_names)
    except ImportError:
        path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "breast_cancer.csv")
        with open(path) as fh:
            names = fh.readline().strip().split(",")[:-1]
        arr = np.loadtxt(path, delimiter=",", skiprows=1)
        return arr[:, :-1], arr[:, -1].astype(int), names


class P9(Problem):
    """Solution = 0/1 mask of length 30 (value > 0.5 means 'selected').
    Objectives: min error = 1 - 5-fold CV accuracy of our own 5-NN classifier,
    min number of selected features.  Constraint: at least one feature
    (CV = 1 for the empty mask).

    DESIGN CHOICES: features are z-scored with the statistics of the whole data
    set (no labels used); the 5 folds are a fixed random split (seed 0) so the
    objective is deterministic; results are cached per mask (the evaluation
    counter still counts every call).  ideal = (0, 1), nadir = (0.4, 30)."""
    name = "P9 Feature selection"
    var_type = "binary"

    def __init__(self, k=5, n_folds=5, fold_seed=0):
        X, y, self.feature_names = load_breast_cancer_data()
        n, d = X.shape
        super().__init__(d, 2, np.zeros(d), np.ones(d))
        Z = (X - X.mean(axis=0)) / X.std(axis=0)
        sq = ((Z[:, None, :] - Z[None, :, :]) ** 2).astype(np.float32)   # (n, n, d)
        self._sq = sq.reshape(n * n, d)
        self.n, self.k, self.y = n, k, y
        perm = np.random.RandomState(fold_seed).permutation(n)
        self.folds = np.array_split(perm, n_folds)
        self.train = [np.setdiff1d(np.arange(n), f) for f in self.folds]
        self._cache = {}
        self.ideal = np.array([0.0, 1.0])
        self.nadir = np.array([0.4, float(d)])

    def cv_error(self, mask):
        """1 - 5-fold cross-validated accuracy of 5-NN using the selected features."""
        key = mask.tobytes()
        if key not in self._cache:
            D = (self._sq @ mask.astype(np.float32)).reshape(self.n, self.n)  # squared distances
            correct = 0
            for test, train in zip(self.folds, self.train):
                sub = D[np.ix_(test, train)]
                nn = np.argpartition(sub, self.k, axis=1)[:, :self.k]   # k nearest
                votes = self.y[train][nn].sum(axis=1)                   # class-1 votes
                pred = (2 * votes > self.k).astype(int)                 # majority
                correct += int(np.sum(pred == self.y[test]))
            self._cache[key] = 1.0 - correct / self.n
        return self._cache[key]

    def _evaluate_one(self, x):
        mask = (np.asarray(x) > 0.5).astype(np.int8)
        count = int(mask.sum())
        if count == 0:
            return np.array([1.0, 0.0]), 1.0
        return np.array([self.cv_error(mask), float(count)]), 0.0

    def random_solutions(self, N, rng):
        return (rng.random((N, self.n_var)) < 0.5).astype(float)


# =============================================================================
# P10 -- DTLZ2 (many objectives)
# =============================================================================
class P10(Problem):
    """DTLZ2 with M objectives and n = M + 9 variables in [0,1].
    g = sum_{i>=M} (x_i - 0.5)^2 ;  f_m = (1+g) * prod_{i=1..M-m} cos(x_i pi/2)
    * sin(x_{M-m+1} pi/2)  (no sin factor for m = 1).
    True front: sum f_m^2 = 1, f_m >= 0 (unit sphere), sampled with Das-Dennis points."""
    name = "P10 DTLZ2"

    def __init__(self, M=3):
        n = M + 9
        super().__init__(n, M, np.zeros(n), np.ones(n))
        self.M = M
        self.ideal = np.zeros(M)
        self.nadir = np.ones(M)

    def _evaluate_one(self, x):
        M = self.M
        g = np.sum((x[M - 1:] - 0.5) ** 2)
        f = np.zeros(M)
        for m in range(1, M + 1):
            val = 1.0 + g
            val *= np.prod(np.cos(x[:M - m] * np.pi / 2.0))
            if m > 1:
                val *= np.sin(x[M - m] * np.pi / 2.0)
            f[m - 1] = val
        return f, 0.0

    def true_front(self, n=500):
        H = 1
        while math.comb(H + self.M - 1, self.M - 1) < n:
            H += 1
        W = U.das_dennis(self.M, H)
        return W / np.linalg.norm(W, axis=1, keepdims=True)


# =============================================================================
# P11 -- Production planning (goals and priorities)
# =============================================================================
class P11(Problem):
    """xA, xB >= 0.  max profit 4xA+6xB, max units xA+xB (stored as negatives).
    Labour 2xA+3xB<=1200, material 4xA+2xB<=1600 (each divided by its limit).
    Goals (Goal Programming): profit >= 2400, units >= 600.
    Priorities (Lexicographic): profit first, units second.
    NOTE: the two objectives do NOT conflict -- (300,200) gives the maximum of
    both (2400, 500), so the true Pareto front is that single point.
    nadir (DESIGN CHOICE) = (0,0) = profit/units at the origin."""
    name = "P11 Production planning"
    goals = {"profit": 2400.0, "units": 600.0}
    priorities = ("profit", "units")

    def __init__(self):
        super().__init__(2, 2, [0.0, 0.0], [400.0, 400.0], maximize=[True, True])
        self.ideal = np.array([-2400.0, -500.0])
        self.nadir = np.array([0.0, 0.0])

    def _evaluate_one(self, x):
        xa, xb = x
        cv = U.constraint_violation(g=[(2 * xa + 3 * xb - 1200.0) / 1200.0,
                                       (4 * xa + 2 * xb - 1600.0) / 1600.0])
        return np.array([-(4 * xa + 6 * xb), -(xa + xb)]), cv

    def true_front(self, n=200):
        return np.array([[-2400.0, -500.0]])


# =============================================================================
# P12 -- Supplier selection (MCDM table, no optimisation)
# =============================================================================
class P12:
    """Decision matrix of 5 suppliers x 4 criteria (not a search problem)."""
    name = "P12 Supplier selection"
    suppliers = ["S1", "S2", "S3", "S4", "S5"]
    criteria = ["Cost ($k)", "Quality (/10)", "Delivery (days)", "Reliability (%)"]
    matrix = np.array([[120, 8, 10, 92],
                       [100, 7, 14, 88],
                       [140, 9, 7, 95],
                       [90, 6, 18, 80],
                       [130, 8, 9, 90]], dtype=float)
    maximize = np.array([False, True, False, True])
    weights = np.array([0.3, 0.3, 0.2, 0.2])

    def minimization_matrix(self):
        """Same table with maximised criteria multiplied by -1."""
        return np.where(self.maximize, -self.matrix, self.matrix)

    def pareto_optimal(self):
        """Indices of the Pareto-optimal suppliers."""
        return U.non_dominated_sort(self.minimization_matrix())[0]


# =============================================================================
# P13 -- Multi-task learning (differentiable)
# =============================================================================
class ToyTwoTask(Problem):
    """P13(a): x in R^2, f1=||x-a||^2, f2=||x-b||^2, a=(3,3), b=(-3,-3).
    Pareto set: the segment from a to b.  Start point (0,5).
    gradients(x) returns a (2,2) array; row k = gradient of f_k."""
    name = "P13a Toy two-task"
    a = np.array([3.0, 3.0])
    b = np.array([-3.0, -3.0])
    start = np.array([0.0, 5.0])

    def __init__(self):
        super().__init__(2, 2, [-5.0, -5.0], [5.0, 5.0])
        self.ideal = np.zeros(2)
        self.nadir = np.array([72.0, 72.0])      # f1(b) = f2(a) = 72

    def _evaluate_one(self, x):
        return np.array([np.sum((x - self.a) ** 2), np.sum((x - self.b) ** 2)]), 0.0

    def gradients(self, x):
        return np.array([2.0 * (x - self.a), 2.0 * (x - self.b)])

    def true_front(self, n=200):
        t = np.linspace(0.0, 1.0, n)[:, None]
        X = self.a + t * (self.b - self.a)
        return np.column_stack((np.sum((X - self.a) ** 2, axis=1),
                                np.sum((X - self.b) ** 2, axis=1)))


class MultiTaskNet:
    """P13(b): shared-body network with a classification and a regression head,
    forward AND backward pass coded by hand (NumPy only).

        Z = X W1 + b1 ;  H = tanh(Z)                       (shared body)
        classification: logit = H wc + bc, loss = binary cross-entropy
        regression    : pred  = H wr + br, loss = mean squared error

    All parameters live in ONE flat vector theta:
        [W1 (d*h) | b1 (h) | wc (h) | bc (1) | wr (h) | br (1)]
    `shared_idx` marks the shared-body entries (W1, b1).
    DESIGN CHOICES (synthetic data): X ~ N(0,1) with n=300, d=6; class label =
    1[X w_c + 0.3*noise > 0]; regression target = reg_scale*(X w_r + 0.5 sin(2 x_1)
    + 0.1*noise).  reg_scale > 1 makes the regression gradients much larger
    than the classification gradients (used to test MGDA / GradNorm)."""
    name = "P13b Multi-task network"

    def __init__(self, n_samples=300, n_features=6, hidden=8, reg_scale=1.0, seed=0):
        rs = np.random.RandomState(seed)
        self.n, self.d, self.h = n_samples, n_features, hidden
        self.X = rs.normal(size=(self.n, self.d))
        w_c, w_r = rs.normal(size=self.d), rs.normal(size=self.d)
        self.y_cls = (self.X @ w_c + 0.3 * rs.normal(size=self.n) > 0).astype(float)
        self.y_reg = reg_scale * (self.X @ w_r + 0.5 * np.sin(2 * self.X[:, 0])
                                  + 0.1 * rs.normal(size=self.n))
        d, h = self.d, self.h
        self.n_params = d * h + h + h + 1 + h + 1
        self.shared_idx = np.arange(d * h + h)

    def init_params(self, seed=0):
        return 0.5 * np.random.RandomState(seed).normal(size=self.n_params)

    def unpack(self, theta):
        d, h = self.d, self.h
        i = 0
        W1 = theta[i:i + d * h].reshape(d, h); i += d * h
        b1 = theta[i:i + h]; i += h
        wc = theta[i:i + h]; i += h
        bc = theta[i]; i += 1
        wr = theta[i:i + h]; i += h
        br = theta[i]
        return W1, b1, wc, bc, wr, br

    def _body(self, theta):
        W1, b1, wc, bc, wr, br = self.unpack(theta)
        H = np.tanh(self.X @ W1 + b1)
        return H, wc, bc, wr, br

    # ---- losses ------------------------------------------------------------
    def loss_cls(self, theta):
        H, wc, bc, _, _ = self._body(theta)
        z = H @ wc + bc
        return float(np.mean(np.maximum(z, 0) - z * self.y_cls + np.log1p(np.exp(-np.abs(z)))))

    def loss_reg(self, theta):
        H, _, _, wr, br = self._body(theta)
        return float(np.mean((H @ wr + br - self.y_reg) ** 2))

    def losses(self, theta):
        return np.array([self.loss_cls(theta), self.loss_reg(theta)])

    def accuracy(self, theta):
        H, wc, bc, _, _ = self._body(theta)
        return float(np.mean(((H @ wc + bc) > 0) == (self.y_cls > 0.5)))

    # ---- backward passes (chain rule written out by hand) -------------------
    def _backprop(self, theta, dout, head):
        """dout = dLoss/d(head output), shape (n,). head = 'cls' or 'reg'."""
        W1, b1, wc, bc, wr, br = self.unpack(theta)
        H = np.tanh(self.X @ W1 + b1)
        w_head = wc if head == "cls" else wr
        g_head_w = H.T @ dout                       # gradient of the head weights
        g_head_b = dout.sum()                       # gradient of the head bias
        dH = np.outer(dout, w_head)                 # (n, h)
        dZ = dH * (1.0 - H ** 2)                    # tanh'(z) = 1 - tanh(z)^2
        gW1 = self.X.T @ dZ
        gb1 = dZ.sum(axis=0)
        g = np.zeros(self.n_params)
        d, h = self.d, self.h
        g[:d * h] = gW1.ravel()
        g[d * h:d * h + h] = gb1
        base = d * h + h
        if head == "cls":
            g[base:base + h] = g_head_w
            g[base + h] = g_head_b
        else:
            g[base + h + 1:base + 2 * h + 1] = g_head_w
            g[base + 2 * h + 1] = g_head_b
        return g

    def grad_cls(self, theta):
        H, wc, bc, _, _ = self._body(theta)
        z = H @ wc + bc
        p = 1.0 / (1.0 + np.exp(-np.clip(z, -60, 60)))
        return self._backprop(theta, (p - self.y_cls) / self.n, "cls")

    def grad_reg(self, theta):
        H, _, _, wr, br = self._body(theta)
        pred = H @ wr + br
        return self._backprop(theta, 2.0 * (pred - self.y_reg) / self.n, "reg")

    def gradients(self, theta):
        """(2, n_params) array: row 0 = classification, row 1 = regression."""
        return np.array([self.grad_cls(theta), self.grad_reg(theta)])


# =============================================================================
# P14 -- Laptop design with a decision-maker preference
# =============================================================================
class P14(Problem):
    """x1 (screen) in [13,17], x2 (battery) in [40,90].
    min weight f1 = 0.4+0.05x1+0.012x2 ; max runtime f2 = 0.2x2/(0.5+0.03x1).
    Preference (reference) point in natural units: (1.8 kg, 12 h)."""
    name = "P14 Laptop design"
    preference_point = np.array([1.8, 12.0])

    def __init__(self):
        super().__init__(2, 2, [13.0, 40.0], [17.0, 90.0], maximize=[False, True])
        self.ideal = np.array([1.53, -20.2247191])
        self.nadir = np.array([2.13, -8.98876404])

    def _evaluate_one(self, x):
        x1, x2 = x
        weight = 0.4 + 0.05 * x1 + 0.012 * x2
        runtime = 0.2 * x2 / (0.5 + 0.03 * x1)
        return np.array([weight, -runtime]), 0.0

    def true_front(self, n=200):
        x2 = np.linspace(40.0, 90.0, n)
        x1 = np.full_like(x2, 13.0)
        return np.column_stack((0.4 + 0.05 * x1 + 0.012 * x2,
                                -0.2 * x2 / (0.5 + 0.03 * x1)))


# =============================================================================
# Registry
# =============================================================================
PROBLEMS = {"P1": P1, "P2": P2, "P3": P3, "P4": P4, "P5": P5, "P6": P6, "P7": P7,
            "P8": P8, "P9": P9, "P10": P10, "P11": P11, "P12": P12,
            "P13a": ToyTwoTask, "P13b": MultiTaskNet, "P14": P14}


def make_problem(name, **kwargs):
    """make_problem("P10", M=5) -> a fresh problem object."""
    return PROBLEMS[name](**kwargs)
