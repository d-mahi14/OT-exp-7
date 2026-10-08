"""
Algorithm 24: MOABC (Multi-Objective Artificial Bee Colony)
Proposed by: Akbari, Hedayatzadeh, Ziarati & Hassanizadeh (2012)
Family: Swarm (Part 4, slides 28-34)

Core idea:
Extends ABC to multiobjective problems:
- Employed bees exploit known food sources via neighbor searches:
      x_new = x_i + phi * (x_i - x_k),  phi in [-1, 1], k != i
  and replace the food source via greedy Pareto dominance.
- Onlooker bees select food sources probabilistically (favoring less-crowded archive members)
  and perform local exploitation.
- Scout bees abandon food sources that fail to improve after a threshold 'limit' of trials,
  replacing them with uniformly random exploration.
- An external archive preserves the best non-dominated solutions found so far.
"""

import numpy as np
import moo_utils as U


def update_archive(archive_X, archive_F, archive_CV, new_X, new_F, new_CV, max_size):
    if len(archive_X) == 0:
        comb_X = new_X
        comb_F = new_F
        comb_CV = new_CV
    else:
        comb_X = np.vstack([archive_X, new_X])
        comb_F = np.vstack([archive_F, new_F])
        comb_CV = np.concatenate([archive_CV, new_CV])

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


def initialize(problem, colony_size=40, archive_size=100, limit=20, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    # Number of food sources = colony_size // 2
    n_sources = max(2, colony_size // 2)
    X = problem.random_solutions(n_sources, rng)
    F, CV = problem.evaluate(X)
    trials = np.zeros(n_sources, dtype=int)

    arch_X, arch_F, arch_CV = update_archive(
        np.empty((0, problem.n_var)), np.empty((0, problem.n_obj)), np.empty(0),
        X, F, CV, archive_size
    )
    return X, F, CV, trials, arch_X, arch_F, arch_CV


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, trials, arch_X, arch_F, arch_CV, limit=20, archive_size=100, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    n_sources, d = X.shape

    # 1. Employed Bees Phase
    for i in range(n_sources):
        # Choose partner k != i
        k = rng.integers(n_sources - 1)
        if k >= i:
            k += 1
        phi = rng.uniform(-1.0, 1.0)
        j = rng.integers(d)  # perturb one random dimension

        v = X[i].copy()
        v[j] = v[j] + phi * (v[j] - X[k, j])
        v = np.clip(v, problem.lower, problem.upper)
        if problem.var_type == "int":
            v = np.round(v)

        fv, cvv = problem.evaluate(v)
        fv = fv[0] if fv.ndim > 1 else fv
        cvv = cvv[0] if np.ndim(cvv) > 0 else cvv

        # Greedy dominance test
        if U.constrained_dominates(fv, cvv, F[i], CV[i]):
            X[i] = v.copy()
            F[i] = fv.copy()
            CV[i] = cvv
            trials[i] = 0
        else:
            trials[i] += 1

    # 2. Onlooker Bees Phase
    # Selection probabilities based on dominance rank
    ranks = U.non_dominated_sort(F, CV)
    prob_weights = np.zeros(n_sources)
    for r_idx, front in enumerate(ranks):
        prob_weights[front] = 1.0 / (r_idx + 1)
    probs = prob_weights / np.sum(prob_weights)

    for _ in range(n_sources):
        i = rng.choice(n_sources, p=probs)
        k = rng.integers(n_sources - 1)
        if k >= i:
            k += 1
        phi = rng.uniform(-1.0, 1.0)
        j = rng.integers(d)

        v = X[i].copy()
        v[j] = v[j] + phi * (v[j] - X[k, j])
        v = np.clip(v, problem.lower, problem.upper)
        if problem.var_type == "int":
            v = np.round(v)

        fv, cvv = problem.evaluate(v)
        fv = fv[0] if fv.ndim > 1 else fv
        cvv = cvv[0] if np.ndim(cvv) > 0 else cvv

        if U.constrained_dominates(fv, cvv, F[i], CV[i]):
            X[i] = v.copy()
            F[i] = fv.copy()
            CV[i] = cvv
            trials[i] = 0
        else:
            trials[i] += 1

    # 3. Scout Bees Phase
    for i in range(n_sources):
        if trials[i] >= limit:
            X[i] = problem.random_solutions(1, rng)[0]
            fi, cvi = problem.evaluate(X[i])
            F[i] = fi[0] if fi.ndim > 1 else fi
            CV[i] = cvi[0] if np.ndim(cvi) > 0 else cvi
            trials[i] = 0

    # 4. Update archive
    arch_X, arch_F, arch_CV = update_archive(arch_X, arch_F, arch_CV, X, F, CV, archive_size)
    return X, F, CV, trials, arch_X, arch_F, arch_CV


def run(problem, colony_size=40, archive_size=100, n_gen=50, limit=20, seed=42):
    rng = np.random.default_rng(seed)
    (X, F, CV, trials,
     arch_X, arch_F, arch_CV) = initialize(problem, colony_size=colony_size, archive_size=archive_size, limit=limit, rng=rng)

    for gen in range(n_gen):
        (X, F, CV, trials,
         arch_X, arch_F, arch_CV) = step(
            problem, X, F, CV, trials, arch_X, arch_F, arch_CV,
            limit=limit, archive_size=archive_size, rng=rng
        )

    return arch_X, arch_F
