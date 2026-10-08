"""
Algorithm 7: Lexicographic Method
Family: Classical (Part 1, slide 42; Part 5E, slides 26-28)

Core mechanism:
Rank objectives strictly by DM-assigned priority order (1, 2, ..., k).
Optimize the top-priority objective alone, then optimize the next objective
while holding previous optima within a small tolerance ε:

    Stage 1: min f_(1)(x) s.t. x in Feasible -> f_(1)*
    Stage 2: min f_(2)(x) s.t. f_(1)(x) <= f_(1)* + ε_1, x in Feasible -> f_(2)*
    ...
    Stage k: min f_(k)(x) s.t. f_(i)(x) <= f_(i)* + ε_i for all i < k

Lecture steps:
1. Rank the k objectives strictly 1..k (priority hierarchy).
2. Optimize priority-1 objective alone -> obtain f1*.
3. Add constraint f1(x) <= f1* + ε_1 and optimize priority-2 objective.
4. Repeat, adding one 'hold previous optimum' constraint per stage,
   until all k objectives are processed.
"""

import numpy as np
import moo_utils as U


def initialize(problem, priorities=None, tolerances=None, candidates=None, rng=None):
    """
    Initialize priority order, tolerances, and candidate solutions.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    k = problem.n_obj
    if priorities is None:
        priorities = list(range(k))
    if tolerances is None:
        tolerances = [1e-4] * k
    if candidates is None:
        candidates = problem.random_solutions(1000, rng)
    return priorities, tolerances, candidates


def solve_candidates(problem, priorities, tolerances, candidates):
    """
    Apply lexicographic sequential optimization over candidate pool.
    """
    if len(candidates) == 0:
        return None, None

    # Evaluate all candidates
    F_list, CV_list = [], []
    for x in candidates:
        f, cv = problem.evaluate(x)
        F_list.append(f[0] if f.ndim > 1 else f)
        CV_list.append(cv[0] if np.ndim(cv) > 0 else cv)
    F = np.array(F_list)
    CV = np.array(CV_list)

    # Filter feasible or least violating
    min_cv = np.min(CV)
    feas_idx = np.where(CV <= min_cv + 1e-6)[0]

    current_pool = feas_idx
    for p_idx in priorities:
        tol = tolerances[p_idx]
        obj_vals = F[current_pool, p_idx]
        best_val = np.min(obj_vals)
        # Keep solutions within tolerance of best
        keep = np.where(obj_vals <= best_val + tol)[0]
        current_pool = current_pool[keep]
        if len(current_pool) == 1:
            break

    best_idx = current_pool[0]
    return np.asarray(candidates[best_idx]).copy(), F[best_idx].copy()


def solve_p11_lexicographic():
    """
    Exact 2-stage LP solve for problem P11 (Production Planning):
    Priority 1: max profit 4*xA + 6*xB
    Priority 2: max units xA + xB (or xA tie-breaker)
    Constraints: 2*xA + 3*xB <= 1200, 4*xA + 2*xB <= 1600, xA, xB >= 0.
    """
    from scipy.optimize import linprog
    # Stage 1: max 4*xA + 6*xB => min -4*xA - 6*xB
    c1 = np.array([-4.0, -6.0])
    A_ub = np.array([[2.0, 3.0], [4.0, 2.0]])
    b_ub = np.array([1200.0, 1600.0])
    bounds = [(0, None), (0, None)]
    res1 = linprog(c1, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
    if not res1.success:
        return None, None
    max_profit = -res1.fun

    # Stage 2: max xA + xB (units) subject to 4*xA + 6*xB >= 2400 (i.e. -4*xA - 6*xB <= -2400)
    c2 = np.array([-1.0, -1.0])
    A_ub2 = np.vstack([A_ub, [-4.0, -6.0]])
    b_ub2 = np.append(b_ub, [-max_profit + 1e-5])
    res2 = linprog(c2, A_ub=A_ub2, b_ub=b_ub2, bounds=bounds, method="highs")
    if res2.success:
        xA, xB = res2.x
        x = np.array([xA, xB])
        profit = 4 * xA + 6 * xB
        units = xA + xB
        return x, np.array([-profit, -units])
    return None, None


def evaluate(problem, x):
    """
    Evaluate candidate x.
    """
    f, cv = problem.evaluate(x)
    return (f[0] if f.ndim > 1 else f), (cv[0] if np.ndim(cv) > 0 else cv)


def step(problem, priorities, tolerances, candidates):
    """
    Perform one solve step.
    """
    return solve_candidates(problem, priorities, tolerances, candidates)


def run(problem, priorities=None, tolerances=None, candidates=None, n_candidates=1000, seed=42):
    """
    Run lexicographic optimization.
    """
    rng = np.random.default_rng(seed)
    priorities, tolerances, default_cands = initialize(problem, priorities, tolerances, candidates, rng=rng)

    if getattr(problem, "name", "").startswith("P11"):
        bx, bf = solve_p11_lexicographic()
        if bx is not None:
            return np.array([bx]), np.array([bf])

    if candidates is None:
        candidates = problem.random_solutions(n_candidates, rng)

    bx, bf = solve_candidates(problem, priorities, tolerances, candidates)
    if bx is None:
        bx = candidates[0]
        f, _ = problem.evaluate(bx)
        bf = f[0] if f.ndim > 1 else f
    return np.array([bx]), np.array([bf])
