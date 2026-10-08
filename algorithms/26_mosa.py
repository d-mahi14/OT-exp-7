"""
Algorithm 26: MOSA (Multi-Objective Simulated Annealing)
Family: Physics-Inspired (Part 4, slides 42-48)

Core idea:
A search agent explores by proposing random neighbor moves:
- Non-dominated neighbors are always accepted.
- Dominated (worse) neighbors are accepted with Boltzmann probability:
      P(accept) = exp(-Delta E / T)
  where Delta E = sum_{worsened m} (f_m(neighbor) - f_m(current)).
- Cooling schedule gradually reduces temperature T:
      T_{k+1} = alpha * T_k
  shifting the search from broad exploration to fine exploitation.
- An external archive stores all non-dominated solutions encountered.

Default parameters:
- T_0 = 100.0, alpha = 0.9 (cooling factor)
"""

import numpy as np
import moo_utils as U


def update_archive(archive_X, archive_F, archive_CV, new_X, new_F, new_CV, max_size=100):
    if len(archive_X) == 0:
        comb_X = np.atleast_2d(new_X)
        comb_F = np.atleast_2d(new_F)
        comb_CV = np.atleast_1d(new_CV)
    else:
        comb_X = np.vstack([archive_X, np.atleast_2d(new_X)])
        comb_F = np.vstack([archive_F, np.atleast_2d(new_F)])
        comb_CV = np.concatenate([archive_CV, np.atleast_1d(new_CV)])

    nd_idx = U.pareto_front(comb_F, comb_CV)
    arch_X = comb_X[nd_idx]
    arch_F = comb_F[nd_idx]
    arch_CV = comb_CV[nd_idx]

    while len(arch_X) > max_size:
        cd = U.crowding_distance(arch_F)
        worst_idx = np.argmin(cd)
        arch_X = np.delete(arch_X, worst_idx, axis=0)
        arch_F = np.delete(arch_F, worst_idx, axis=0)
        arch_CV = np.delete(arch_CV, worst_idx, axis=0)

    return arch_X, arch_F, arch_CV


def compute_energy_difference(f_cand, f_curr):
    """
    Sum of objective increases across worsened objectives (minimization).
    """
    diff = np.asarray(f_cand, float) - np.asarray(f_curr, float)
    worsened = np.maximum(0.0, diff)
    return float(np.sum(worsened))


def initialize(problem, T0=100.0, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    x = problem.random_solutions(1, rng)[0]
    f, cv = problem.evaluate(x)
    f = f[0] if f.ndim > 1 else f
    cv = cv[0] if np.ndim(cv) > 0 else cv
    arch_X, arch_F, arch_CV = update_archive(
        np.empty((0, problem.n_var)), np.empty((0, problem.n_obj)), np.empty(0),
        x, f, cv, archive_size
    )
    return x, f, cv, T0, arch_X, arch_F, arch_CV


def evaluate(problem, x):
    f, cv = problem.evaluate(x)
    return (f[0] if f.ndim > 1 else f), (cv[0] if np.ndim(cv) > 0 else cv)


def step(problem, current_x, current_f, current_cv, T, arch_X, arch_F, arch_CV,
         step_size=0.1, alpha=0.9, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    d = len(current_x)

    # 1. Generate neighbor
    delta = rng.uniform(-step_size, step_size, size=d) * (problem.upper - problem.lower)
    cand_x = np.clip(current_x + delta, problem.lower, problem.upper)
    if problem.var_type == "int":
        cand_x = np.round(cand_x)

    cand_f, cand_cv = evaluate(problem, cand_x)

    # 2. Pareto acceptance logic
    if U.constrained_dominates(cand_f, cand_cv, current_f, current_cv):
        accepted = True
    elif not U.constrained_dominates(current_f, current_cv, cand_f, cand_cv):
        # Non-dominated move -> accept unconditionally
        accepted = True
    else:
        # Dominated move -> Boltzmann acceptance
        delta_E = compute_energy_difference(cand_f, current_f) + 1e3 * max(0.0, cand_cv - current_cv)
        p_accept = np.exp(-delta_E / max(T, 1e-6))
        accepted = bool(rng.random() < p_accept)

    if accepted:
        current_x = cand_x.copy()
        current_f = cand_f.copy()
        current_cv = cand_cv

    # Update archive
    arch_X, arch_F, arch_CV = update_archive(arch_X, arch_F, arch_CV, cand_x, cand_f, cand_cv, archive_size)

    # Cool temperature
    T_next = T * alpha

    return current_x, current_f, current_cv, T_next, arch_X, arch_F, arch_CV


def run(problem, n_iterations=200, T0=100.0, alpha=0.95, step_size=0.1, archive_size=100, seed=42):
    rng = np.random.default_rng(seed)
    (x, f, cv, T,
     arch_X, arch_F, arch_CV) = initialize(problem, T0=T0, archive_size=archive_size, rng=rng)

    for i in range(n_iterations):
        (x, f, cv, T,
         arch_X, arch_F, arch_CV) = step(
            problem, x, f, cv, T, arch_X, arch_F, arch_CV,
            step_size=step_size, alpha=alpha, archive_size=archive_size, rng=rng
        )

    return arch_X, arch_F
