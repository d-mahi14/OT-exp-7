"""
Algorithm 5: Value Function Method
Family: Classical (Part 1, slide 39)

Core mechanism:
Optimize the decision-maker's own utility / value function directly:

    max v(f_1(x), f_2(x), ..., f_k(x))
    subject to constraints

In minimization form:
    min L(x) = -v(f_1(x), ..., f_k(x)) + penalty * CV(x)

Lecture steps:
1. Elicit the decision-maker's value function v(.) -- how they trade off objectives.
2. Substitute F(x) into v and solve as a single-objective problem.
3. The resulting x* is the decision-maker's most-preferred point on the front --
   no further ranking needed.

Default value functions:
- Linear utility / risk-return trade-off (e.g. for P4: expected return - risk penalty)
- Laptop satisfaction (e.g. for P14: battery runtime / weight)
"""

import numpy as np
import moo_utils as U


def default_value_function(objectives, weights=None):
    """
    Default linear utility function over minimization objectives:
    v(F) = - sum(weights * F)
    (Higher is better).
    """
    f = np.asarray(objectives, dtype=float)
    if weights is None:
        weights = np.ones(len(f)) / len(f)
    else:
        weights = np.asarray(weights, dtype=float)
    return -float(np.dot(weights, f))


def initialize(problem, value_func=None, candidates=None, rng=None):
    """
    Initialize value function and candidate solutions.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    if value_func is None:
        value_func = default_value_function
    if candidates is None:
        candidates = problem.random_solutions(1000, rng)
    return value_func, candidates


def evaluate(problem, x, value_func, penalty_factor=1e6):
    """
    Evaluate candidate solution x under the decision-maker's value function.
    Returns scalar cost to minimize (lower is better), objectives, and CV.
    """
    f, cv = problem.evaluate(x)
    f = f[0] if f.ndim > 1 else f
    cv = cv[0] if np.ndim(cv) > 0 else cv
    # Value function is maximized, so cost = -v + penalty * cv
    utility = value_func(f)
    cost = -utility + penalty_factor * cv
    return cost, f, cv


def solve_value_function(problem, value_func, candidates, penalty_factor=1e6):
    """
    Find candidate maximizing decision-maker's utility among candidates.
    """
    best_x = None
    best_f = None
    best_cost = np.inf

    for x in candidates:
        cost, f, _ = evaluate(problem, x, value_func, penalty_factor=penalty_factor)
        if cost < best_cost:
            best_cost = cost
            best_x = np.asarray(x).copy()
            best_f = f.copy()

    return best_x, best_f, best_cost


def step(problem, value_func, candidates):
    """
    Perform one solve / evaluation iteration.
    """
    return solve_value_function(problem, value_func, candidates)


def run(problem, value_func=None, candidates=None, n_candidates=1000, seed=42):
    """
    Run value function optimization.
    Returns:
        best_x : (1, n_var) array
        best_f : (1, n_obj) array
    """
    rng = np.random.default_rng(seed)
    if value_func is None:
        value_func = default_value_function
    if candidates is None:
        candidates = problem.random_solutions(n_candidates, rng)

    bx, bf, _ = solve_value_function(problem, value_func, candidates)
    if bx is None:
        bx = candidates[0]
        f, _ = problem.evaluate(bx)
        bf = f[0] if f.ndim > 1 else f
    return np.array([bx]), np.array([bf])
