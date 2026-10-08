"""
moo_utils.py -- shared toolkit for Lab Assignment 7 (Task A1).

Only NumPy and the standard library are used.

CONVENTION (very important): every function here assumes that ALL objectives
are MINIMIZED.  A maximization objective f must be stored as -f
(Max f == Min -f, lecture Part 1, slide 7).  Problems in problems.py already
return objectives in this "minimization form".

Contents (slide numbers refer to the Part 1 lecture deck)
---------------------------------------------------------
 1. dominates                 Pareto dominance               (slide 5)
 2. constraint_violation      CV(x) = sum max(0,g) + sum|h|  (slide 22)
 3. constrained_dominates     Deb's three rules              (slide 22)
 4. dominance_matrix          all pairwise dominance at once
 5. non_dominated_sort        fast non-dominated sorting     (slide 21)
 6. pareto_front              indices of the final feasible front
 7. crowding_distance         diversity measure              (slide 23)
 8. normalize_objectives      scale objectives to [0, 1]
 9. hypervolume               exact in 2-D, Monte-Carlo for 3+ objectives
10. igd                       Inverted Generational Distance
11. spacing                   Schott-style spacing
12. front_metrics             HV + IGD + Spacing in one call
13. das_dennis                uniformly spread reference points
14. small helpers             random_population, clip_to_bounds

Problems P1-P14 live in problems.py and can also be imported from here, e.g.
    from moo_utils import P3          (re-exported lazily at the bottom of file)
"""
import itertools
import numpy as np


# ----------------------------------------------------------------------------
# 1. Pareto dominance
# ----------------------------------------------------------------------------
def dominates(fa, fb):
    """True if solution A dominates solution B (minimization).

    A dominates B  <=>  fa_i <= fb_i for every i  AND  fa_j < fb_j for some j.
    Inputs : fa, fb -- 1-D arrays (objective vectors, minimization form).
    Output : bool.
    Example: dominates([1, 2], [2, 2]) -> True ; dominates([1, 3], [2, 2]) -> False
    """
    fa = np.asarray(fa, dtype=float)
    fb = np.asarray(fb, dtype=float)
    return bool(np.all(fa <= fb) and np.any(fa < fb))


# ----------------------------------------------------------------------------
# 2. Constraint violation
# ----------------------------------------------------------------------------
def constraint_violation(g=None, h=None):
    """Total constraint violation  CV = sum max(0, g_i) + sum |h_j|.

    g : values of inequality constraints written as g(x) <= 0 (may be None)
    h : values of equality constraints written as h(x) = 0     (may be None)
    CV == 0 means the solution is feasible.
    Example: constraint_violation(g=[-1, 2], h=[0.5]) -> 2.5
    """
    cv = 0.0
    if g is not None:
        cv += float(np.sum(np.maximum(0.0, np.asarray(g, dtype=float))))
    if h is not None:
        cv += float(np.sum(np.abs(np.asarray(h, dtype=float))))
    return cv


# ----------------------------------------------------------------------------
# 3. Deb's constrained dominance (three rules, checked in this order)
# ----------------------------------------------------------------------------
def constrained_dominates(fa, cva, fb, cvb):
    """Does A constrained-dominate B?   (Deb's rules, slide 22)

    Rule 1: feasible beats infeasible, always.
    Rule 2: both feasible   -> ordinary Pareto dominance.
    Rule 3: both infeasible -> the smaller constraint violation wins
            (objective values are ignored).
    Inputs : fa, fb objective vectors; cva, cvb constraint violations (>= 0).
    Output : bool.
    """
    a_feasible = cva <= 0.0
    b_feasible = cvb <= 0.0
    if a_feasible and not b_feasible:        # Rule 1
        return True
    if (not a_feasible) and b_feasible:      # Rule 1 (the other way round)
        return False
    if a_feasible and b_feasible:            # Rule 2
        return dominates(fa, fb)
    return cva < cvb                         # Rule 3


# ----------------------------------------------------------------------------
# 4. All pairwise dominance relations in one NumPy step
# ----------------------------------------------------------------------------
def dominance_matrix(F, CV=None):
    """Boolean matrix D with D[i, j] == True  <=>  i dominates j.

    If CV is given, Deb's constrained dominance is used, otherwise plain
    Pareto dominance.  F is (N, M); CV is (N,).
    The 3-D comparison F[:, None, :] <= F[None, :, :] has shape (N, N, M):
    entry (i, j, k) asks "is f_k(i) <= f_k(j)?".
    """
    F = np.asarray(F, dtype=float)
    no_worse = np.all(F[:, None, :] <= F[None, :, :], axis=2)
    better_somewhere = np.any(F[:, None, :] < F[None, :, :], axis=2)
    dom = no_worse & better_somewhere
    if CV is None:
        return dom
    CV = np.asarray(CV, dtype=float)
    feas = CV <= 0.0
    fi, fj = feas[:, None], feas[None, :]
    rule1 = fi & ~fj                                  # feasible beats infeasible
    rule2 = fi & fj & dom                             # both feasible
    rule3 = (~fi) & (~fj) & (CV[:, None] < CV[None, :])  # both infeasible
    return rule1 | rule2 | rule3


# ----------------------------------------------------------------------------
# 5. Fast non-dominated sorting (slide 21)
# ----------------------------------------------------------------------------
def non_dominated_sort(F, CV=None):
    """Split a population into fronts F1, F2, ... of decreasing quality.

    Steps (exactly the slide):
      n_p = number of solutions that dominate p ; S_p = set p dominates.
      n_p == 0 -> p is in the current front.
      For every q in S_p: n_q -= 1 ; if n_q == 0, q joins the NEXT front.
    Inputs : F (N, M) objectives, CV (N,) violations or None.
    Output : list of integer index arrays, fronts[0] is the best front.
    """
    dom = dominance_matrix(F, CV)
    n = dom.sum(axis=0).astype(int)          # n[p] = how many dominate p
    fronts = []
    current = np.where(n == 0)[0]
    while len(current) > 0:
        fronts.append(current)
        nxt = []
        for p in current:
            for q in np.where(dom[p])[0]:    # q in S_p
                n[q] -= 1
                if n[q] == 0:
                    nxt.append(q)
        current = np.array(nxt, dtype=int)
    return fronts


def ranks_from_fronts(fronts, N):
    """Array rank[i] = front number of solution i (0 = best front)."""
    rank = np.zeros(N, dtype=int)
    for r, f in enumerate(fronts):
        rank[f] = r
    return rank


# ----------------------------------------------------------------------------
# 6. Final front extraction
# ----------------------------------------------------------------------------
def pareto_front(F, CV=None, unique=True):
    """Indices of the non-dominated FEASIBLE solutions.

    Infeasible rows (CV > 0) are dropped first.  If unique=True, duplicate
    objective vectors are reported only once.  Returns an index array
    (empty if nothing is feasible).
    """
    F = np.asarray(F, dtype=float)
    idx = np.arange(len(F))
    if CV is not None:
        idx = idx[np.asarray(CV, dtype=float) <= 0.0]
    if len(idx) == 0:
        return idx
    first = non_dominated_sort(F[idx])[0]
    idx = idx[first]
    if unique:
        _, keep = np.unique(np.round(F[idx], 12), axis=0, return_index=True)
        idx = idx[np.sort(keep)]
    return idx


# ----------------------------------------------------------------------------
# 7. Crowding distance (slide 23)
# ----------------------------------------------------------------------------
def crowding_distance(F):
    """Crowding distance of every point of ONE front.

    For each objective m: sort the points; boundary points get infinity;
    an inner point i adds ( f_m(i+1) - f_m(i-1) ) / (f_m_max - f_m_min).
    Input  : F (n, M) objectives of one front.
    Output : array (n,) ; larger = more isolated = better for diversity.
    """
    F = np.asarray(F, dtype=float)
    n, m = F.shape
    d = np.zeros(n)
    if n <= 2:
        d[:] = np.inf
        return d
    for k in range(m):
        order = np.argsort(F[:, k], kind="stable")
        fmin, fmax = F[order[0], k], F[order[-1], k]
        d[order[0]] = np.inf
        d[order[-1]] = np.inf
        if fmax - fmin <= 0:                  # all equal in this objective
            continue
        d[order[1:-1]] += (F[order[2:], k] - F[order[:-2], k]) / (fmax - fmin)
    return d


def crowding_distance_all(F, fronts):
    """Crowding distance of every solution, computed front by front."""
    d = np.zeros(len(F))
    for f in fronts:
        d[f] = crowding_distance(np.asarray(F)[f])
    return d


# ----------------------------------------------------------------------------
# 8. Normalization
# ----------------------------------------------------------------------------
def normalize_objectives(F, ideal=None, nadir=None):
    """Scale objectives:  (f - ideal) / (nadir - ideal).

    If ideal / nadir are not given, the column minima / maxima of F are used.
    Zero ranges are replaced by 1 to avoid division by zero.
    Returns (F_normalized, ideal, nadir).
    """
    F = np.asarray(F, dtype=float)
    ideal = F.min(axis=0) if ideal is None else np.asarray(ideal, dtype=float)
    nadir = F.max(axis=0) if nadir is None else np.asarray(nadir, dtype=float)
    span = nadir - ideal
    span = np.where(span <= 0, 1.0, span)
    return (F - ideal) / span, ideal, nadir


# ----------------------------------------------------------------------------
# 9. Hypervolume
# ----------------------------------------------------------------------------
def hv_2d(F, ref):
    """EXACT hypervolume (area) for 2 objectives.

    Keep points that are strictly better than ref, sort by f1, and sweep:
    each point adds the rectangle (ref1 - f1) x (previous_f2 - f2).
    Example: hv_2d([[1,3],[2,2],[3,1]], [4,4]) -> 6.0
    """
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if F.size == 0:
        return 0.0
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[np.argsort(F[:, 0], kind="stable")]
    hv, prev_f2 = 0.0, ref[1]
    for f1, f2 in F:
        if f2 < prev_f2:                      # dominated points are skipped
            hv += (ref[0] - f1) * (prev_f2 - f2)
            prev_f2 = f2
    return float(hv)


def hv_monte_carlo(F, ref, n_samples=200000, seed=0):
    """Monte-Carlo hypervolume estimate for 3 or more objectives.

    Throw random points into the box [min(F), ref]; the fraction that is
    dominated by at least one front point times the box volume is the HV.
    A fixed seed makes the estimate reproducible (design choice: seed 0).
    """
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if F.size == 0:
        return 0.0
    F = F[np.all(F < ref, axis=1)]
    if len(F) == 0:
        return 0.0
    F = F[non_dominated_sort(F)[0]]           # fewer points = faster test
    lower = F.min(axis=0)
    box = float(np.prod(ref - lower))
    rng = np.random.default_rng(seed)
    hits, chunk = 0, 20000
    for start in range(0, n_samples, chunk):
        s = min(chunk, n_samples - start)
        S = lower + rng.random((s, F.shape[1])) * (ref - lower)
        dominated = np.any(np.all(F[None, :, :] <= S[:, None, :], axis=2), axis=1)
        hits += int(dominated.sum())
    return box * hits / n_samples


def hypervolume(F, ref, n_samples=200000, seed=0):
    """HV dispatcher: exact for M = 2 (and M = 1), Monte-Carlo for M >= 3.
    Larger is better.  Use the SAME ref (in normalized space) for all methods."""
    F = np.asarray(F, dtype=float)
    ref = np.asarray(ref, dtype=float)
    if F.size == 0:
        return 0.0
    m = F.shape[1]
    if m == 1:
        return float(max(0.0, ref[0] - F[:, 0].min()))
    if m == 2:
        return hv_2d(F, ref)
    return hv_monte_carlo(F, ref, n_samples, seed)


# ----------------------------------------------------------------------------
# 10. IGD and 11. Spacing
# ----------------------------------------------------------------------------
def igd(true_front, F):
    """Inverted Generational Distance (smaller is better).

    Average, over sampled TRUE-front points, of the distance to the nearest
    OBTAINED point.  Measures convergence and spread together.
    Both inputs must be normalized with the same ideal / nadir.
    Example: igd([[0,0],[1,1]], [[0,1]]) -> 1.0
    """
    T = np.asarray(true_front, dtype=float)
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return float("inf")
    d = np.linalg.norm(T[:, None, :] - F[None, :, :], axis=2)
    return float(d.min(axis=1).mean())


def spacing(F):
    """Spacing (smaller = more evenly spread).

    d_i = Euclidean distance of point i to its nearest neighbour;
    Spacing = standard deviation of the d_i (sample std, n-1 in the
    denominator, as in Schott's definition).  Fewer than 2 points -> 0.
    """
    F = np.asarray(F, dtype=float)
    n = len(F)
    if n < 2:
        return 0.0
    D = np.linalg.norm(F[:, None, :] - F[None, :, :], axis=2)
    np.fill_diagonal(D, np.inf)
    d = D.min(axis=1)
    return float(np.std(d, ddof=1))


# ----------------------------------------------------------------------------
# 12. All three metrics for one obtained set
# ----------------------------------------------------------------------------
def front_metrics(problem, F, n_true=500, hv_samples=200000, seed=0):
    """Compute HV, IGD (if the true front is known) and Spacing.

    Steps: keep the non-dominated points of F -> normalize with the problem's
    ideal / nadir -> HV with ref point problem.hv_ref() -> IGD against
    problem.true_front(n_true) -> Spacing.
    Returns a dict {"HV", "IGD", "Spacing", "n_points"}; IGD is nan when the
    true front is unknown.
    """
    F = np.asarray(F, dtype=float)
    if len(F) == 0:
        return {"HV": 0.0, "IGD": float("nan"), "Spacing": float("nan"), "n_points": 0}
    F = F[pareto_front(F)]
    Fn = problem.normalize(F)
    hv = hypervolume(Fn, problem.hv_ref(), hv_samples, seed)
    T = problem.true_front(n_true)
    igd_value = igd(problem.normalize(T), Fn) if T is not None else float("nan")
    return {"HV": hv, "IGD": igd_value, "Spacing": spacing(Fn), "n_points": len(F)}


# ----------------------------------------------------------------------------
# 13. Das-Dennis reference points (needed by NSGA-III, MOEA/D, RVEA, DTLZ2)
# ----------------------------------------------------------------------------
def das_dennis(M, H):
    """All weight vectors w with w_i = k_i / H, k_i integers, sum w_i = 1.
    Count = C(H + M - 1, M - 1).  Returns array (count, M).
    Trick (stars and bars): choose M-1 'bar' positions among H+M-1 slots;
    the gaps between bars are the integers k_i.
    Example: das_dennis(2, 2) -> [[0, 1], [0.5, 0.5], [1, 0]]
    """
    points = []
    for bars in itertools.combinations(range(H + M - 1), M - 1):
        edges = (-1,) + bars + (H + M - 1,)
        ks = [edges[i + 1] - edges[i] - 1 for i in range(M)]
        points.append(np.array(ks, dtype=float) / H)
    return np.array(points)


# ----------------------------------------------------------------------------
# 14. Small helpers
# ----------------------------------------------------------------------------
def random_population(lower, upper, N, rng):
    """N random real-coded solutions inside [lower, upper]. rng = np.random.default_rng(seed)."""
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    return lower + rng.random((N, len(lower))) * (upper - lower)


def clip_to_bounds(X, lower, upper):
    """Force every variable back inside its bounds."""
    return np.clip(X, lower, upper)


# ----------------------------------------------------------------------------
# Lazy re-export of the problems (avoids a circular import with problems.py)
# ----------------------------------------------------------------------------
_PROBLEM_NAMES = {"P%d" % i for i in range(1, 15) if i != 13} | {
    "P13a", "P13b", "ToyTwoTask", "MultiTaskNet", "Problem", "make_problem", "PROBLEMS"}


def __getattr__(name):
    """Called by Python only when `name` is not found in this module."""
    if name in _PROBLEM_NAMES:
        import problems
        if name == "P13a":
            return problems.ToyTwoTask
        if name == "P13b":
            return problems.MultiTaskNet
        return getattr(problems, name)
    raise AttributeError("module 'moo_utils' has no attribute %r" % name)
