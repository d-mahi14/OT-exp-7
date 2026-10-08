"""
Algorithm 2: ε-Constraint Method
Family: Classical (Part 1, slides 33-35)

Core mechanism:
Optimize one primary objective directly while bounding all other (secondary)
objectives by user-specified tolerances ε_i:

    min f_1(x)
    subject to f_i(x) <= ε_i,  i = 2, ..., k
               g_j(x) <= 0
               h_l(x) == 0

Lecture steps:
1. Choose the primary objective f_1 to optimize directly.
2. Fix ε values (upper bounds) for every other objective.
3. Solve the constrained single-objective problem.
4. Vary ε_i across achievable ranges and re-solve to sweep out the Pareto front.

Default parameters:
- primary_idx = 0 (primary objective index to minimize)
- n_points = 11 (number of ε grid points per secondary objective)
- penalty_factor = 1e6 (penalty multiplier for violated ε and problem constraints)
"""

import numpy as np
import moo_utils as U


def check_epsilon_feasibility(objectives, epsilons, primary_idx=0):
    """
    Check if secondary objectives satisfy the epsilon bounds.
    All objectives are assumed in minimization form.
    """
    objectives = np.asarray(objectives, dtype=float)
    epsilons = np.asarray(epsilons, dtype=float)
    k = len(objectives)
    sec_indices = [i for i in range(k) if i != primary_idx]
    if len(sec_indices) != len(epsilons):
        raise ValueError(f"Expected {len(sec_indices)} epsilon bounds for {k} objectives")
    sec_objs = objectives[sec_indices]
    violations = np.maximum(0.0, sec_objs - epsilons)
    return float(np.sum(violations)), bool(np.all(sec_objs <= epsilons))


def epsilon_constraint_fitness(objectives, epsilons, cv=0.0, primary_idx=0, penalty_factor=1e6):
    """
    Scalarized fitness: primary objective + penalty for secondary objective violations
    and problem constraint violations.
    """
    eps_viol, _ = check_epsilon_feasibility(objectives, epsilons, primary_idx)
    return float(objectives[primary_idx] + penalty_factor * (eps_viol + cv))


def initialize(problem, n_points=11, primary_idx=0, candidates=None, rng=None):
    """
    Initialize epsilon grid across secondary objective ranges.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    k = problem.n_obj
    sec_indices = [i for i in range(k) if i != primary_idx]
    eps_grid = []
    for idx in sec_indices:
        lo = float(problem.ideal[idx])
        hi = float(problem.nadir[idx])
        if np.isclose(lo, hi):
            hi = lo + 1.0
        grid_1d = np.linspace(lo, hi, n_points)
        eps_grid.append(grid_1d)
    # Cartesian product if multi-secondary, or simple 1D if single secondary
    if len(eps_grid) == 1:
        epsilons_list = eps_grid[0][:, None]
    else:
        grids = np.meshgrid(*eps_grid, indexing="ij")
        epsilons_list = np.column_stack([g.ravel() for g in grids])

    if candidates is None:
        candidates = problem.random_solutions(1000, rng)
    return epsilons_list, candidates


def evaluate(problem, x, epsilons, primary_idx=0, penalty_factor=1e6):
    """
    Evaluate candidate x against epsilon constraint.
    """
    f, cv = problem.evaluate(x)
    f = f[0] if f.ndim > 1 else f
    cv = cv[0] if np.ndim(cv) > 0 else cv
    score = epsilon_constraint_fitness(f, epsilons, cv, primary_idx, penalty_factor)
    return score, f, cv


def solve_epsilon_constraint(problem, epsilons, candidates, primary_idx=0, penalty_factor=1e6):
    """
    Solve subproblem for one fixed epsilon setting over a candidate set.
    """
    best_x = None
    best_f = None
    best_score = np.inf

    for x in candidates:
        score, f, _ = evaluate(problem, x, epsilons, primary_idx, penalty_factor)
        if score < best_score:
            best_score = score
            best_x = np.asarray(x).copy()
            best_f = f.copy()

    return best_x, best_f, best_score


def step(problem, epsilons, candidates, primary_idx=0):
    """
    Perform one iteration (solve for one epsilon vector).
    """
    return solve_epsilon_constraint(problem, epsilons, candidates, primary_idx)


def run(problem, epsilons_list=None, n_points=11, primary_idx=0, candidates=None, n_candidates=1000, seed=42):
    """
    Run epsilon-constraint sweep across all epsilon settings.
    """
    rng = np.random.default_rng(seed)
    if epsilons_list is None:
        epsilons_list, _ = initialize(problem, n_points=n_points, primary_idx=primary_idx, rng=rng)
    if candidates is None:
        candidates = problem.random_solutions(n_candidates, rng)

    X_list, F_list = [], []
    for eps in epsilons_list:
        bx, bf, score = solve_epsilon_constraint(problem, eps, candidates, primary_idx)
        # Check if actually feasible
        if bx is not None and score < 1e5:
            X_list.append(bx)
            F_list.append(bf)

    if not X_list:
        # Fallback: keep lowest score candidate
        bx, bf, _ = solve_epsilon_constraint(problem, epsilons_list[0], candidates, primary_idx)
        return np.array([bx]), np.array([bf])

    X_arr = np.array(X_list)
    F_arr = np.array(F_list)
    nd_idx = U.pareto_front(F_arr)
    return X_arr[nd_idx], F_arr[nd_idx]
