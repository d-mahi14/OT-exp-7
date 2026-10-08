"""
Algorithm 6: Goal Programming
Family: Classical (Part 1, slides 40-41; Part 5E, slides 23-25)

Core mechanism:
Convert each objective into a GOAL with a target T_i, then minimize the
(weighted) sum of undesired deviations from those targets -- satisficing
rather than optimizing:

    f_i(x) + d_i^- - d_i^+ = T_i
    d_i^-, d_i^+ >= 0

For a goal "at least T_i" (natural maximization): shortfall d_i^- is undesirable.
For a goal "at most T_i" (natural minimization): overshoot d_i^+ is undesirable.
For a goal "exactly T_i": both d_i^- and d_i^+ are undesirable.

Lecture steps:
1. For k goals, introduce target T_i and deviational variables d_i^-, d_i^+ >= 0.
2. Decide which deviation is undesirable per goal and penalize only those.
3. Add original constraints unchanged.
4. Solve the resulting single-objective LP (or nonlinear/candidate program).
5. Read off deviations: 0 = goal exactly or over-achieved.
"""

import numpy as np
import moo_utils as U


def compute_deviations(value, target):
    """
    Compute under-achievement (d_minus) and over-achievement (d_plus).
    value + d_minus - d_plus = target
    d_minus = max(0, target - value)
    d_plus  = max(0, value - target)
    """
    val = float(value)
    tgt = float(target)
    d_minus = max(0.0, tgt - val)
    d_plus = max(0.0, val - tgt)
    return d_minus, d_plus


def goal_programming_score(natural_objs, targets, goal_types=None, weights=None):
    """
    Compute total weighted undesired deviation.

    Parameters
    ----------
    natural_objs : array-like
        Objective values in natural problem units (higher is better for max, etc.).
    targets : array-like
        Targets T_i for each objective.
    goal_types : list of str
        '>=' for at-least (shortfall d_minus penalized)
        '<=' for at-most (overshoot d_plus penalized)
        '==' for exact (both penalized)
    weights : array-like
        Weights on the deviations.
    """
    objs = np.asarray(natural_objs, dtype=float)
    tgts = np.asarray(targets, dtype=float)
    k = len(objs)
    if goal_types is None:
        goal_types = [">="] * k
    if weights is None:
        weights = np.ones(k) / k
    else:
        weights = np.asarray(weights, dtype=float)

    total_deviation = 0.0
    deviations = []
    for i in range(k):
        d_minus, d_plus = compute_deviations(objs[i], tgts[i])
        gtype = goal_types[i]
        if gtype == ">=":
            undesired = d_minus
        elif gtype == "<=":
            undesired = d_plus
        else:
            undesired = d_minus + d_plus
        total_deviation += weights[i] * undesired
        deviations.append((d_minus, d_plus, undesired))

    return total_deviation, deviations


def initialize(problem, targets=None, goal_types=None, candidates=None, rng=None):
    """
    Initialize targets, goal types, and candidate solutions.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if targets is None:
        if hasattr(problem, "goals") and isinstance(problem.goals, dict):
            targets = list(problem.goals.values())
        else:
            targets = np.abs(problem.ideal).tolist()
    if goal_types is None:
        goal_types = [">=" if m else "<=" for m in problem.maximize]
    if candidates is None:
        candidates = problem.random_solutions(1000, rng)
    return targets, goal_types, candidates


def evaluate(problem, x, targets, goal_types=None, weights=None, penalty_factor=1e6):
    """
    Evaluate candidate x and compute goal programming deviation + constraint violation.
    """
    f_min, cv = problem.evaluate(x)
    f_min = f_min[0] if f_min.ndim > 1 else f_min
    cv = cv[0] if np.ndim(cv) > 0 else cv
    f_nat = problem.to_natural(f_min)
    dev_score, _ = goal_programming_score(f_nat, targets, goal_types, weights)
    fitness = dev_score + penalty_factor * cv
    return fitness, f_min, cv


def solve_candidates(problem, targets, goal_types=None, weights=None, candidates=None, penalty_factor=1e6):
    """
    Select candidate minimizing undesired deviation among candidate pool.
    """
    best_x = None
    best_f = None
    best_score = np.inf

    for x in candidates:
        score, f, _ = evaluate(problem, x, targets, goal_types, weights, penalty_factor)
        if score < best_score:
            best_score = score
            best_x = np.asarray(x).copy()
            best_f = f.copy()

    return best_x, best_f, best_score


def step(problem, targets, goal_types=None, weights=None, candidates=None):
    """
    Perform one solve iteration.
    """
    return solve_candidates(problem, targets, goal_types, weights, candidates)


def solve_p11_lp(targets=None, weights=None):
    """
    Solve P11 exact linear goal programming using scipy.optimize.linprog.
    P11: max profit 4*xA + 6*xB (>= 2400), max units xA + xB (>= 600)
    subject to: 2*xA + 3*xB <= 1200, 4*xA + 2*xB <= 1600, xA, xB >= 0.
    """
    from scipy.optimize import linprog
    if targets is None:
        targets = [2400.0, 600.0]
    if weights is None:
        weights = [1.0, 1.0]

    # Variables: xA, xB, d1_minus, d1_plus, d2_minus, d2_plus
    # Minimize: weights[0]*d1_minus + weights[1]*d2_minus
    c = np.array([0.0, 0.0, weights[0], 0.0, weights[1], 0.0])

    # Equalities:
    # 4*xA + 6*xB + d1_minus - d1_plus = targets[0]
    # 1*xA + 1*xB + d2_minus - d2_plus = targets[1]
    A_eq = np.array([
        [4.0, 6.0, 1.0, -1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0, 0.0, 1.0, -1.0]
    ])
    b_eq = np.array([targets[0], targets[1]])

    # Inequalities:
    # 2*xA + 3*xB <= 1200
    # 4*xA + 2*xB <= 1600
    A_ub = np.array([
        [2.0, 3.0, 0.0, 0.0, 0.0, 0.0],
        [4.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    ])
    b_ub = np.array([1200.0, 1600.0])

    bounds = [(0, None)] * 6
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    if res.success:
        xA, xB = res.x[0], res.x[1]
        x = np.array([xA, xB])
        profit = 4 * xA + 6 * xB
        units = xA + xB
        f_min = np.array([-profit, -units])
        return x, f_min, res.fun
    return None, None, np.inf


def run(problem, targets=None, goal_types=None, weights=None, candidates=None, n_candidates=1000, seed=42):
    """
    Run Goal Programming on problem.
    """
    rng = np.random.default_rng(seed)
    targets, goal_types, default_cands = initialize(problem, targets=targets, goal_types=goal_types, rng=rng)

    # For P11 problem, we can use exact LP solver if applicable
    if getattr(problem, "name", "").startswith("P11"):
        bx, bf, _ = solve_p11_lp(targets, weights)
        if bx is not None:
            return np.array([bx]), np.array([bf])

    if candidates is None:
        candidates = problem.random_solutions(n_candidates, rng)

    bx, bf, _ = solve_candidates(problem, targets, goal_types, weights, candidates)
    if bx is None:
        bx = candidates[0]
        f, _ = problem.evaluate(bx)
        bf = f[0] if f.ndim > 1 else f
    return np.array([bx]), np.array([bf])
