"""
Algorithm 30: MOMA (Multi-Objective Memetic Algorithm)
Family: Hybrid (Evolutionary + Local Search) (Part 4, slides 70-76)

Core idea:
Combines NSGA-II global evolutionary exploration with local hill-climbing exploitation:
1. Run NSGA-II crossover and mutation to produce new candidates.
2. Identify elite / least-crowded individuals from Front 1.
3. Apply local hill-climbing search to fine-tune decision variables:
       x_cand = x ± step
   A local step is accepted if it improves a combined Pareto trade-off score:
       score = -Delta f_1 - Delta f_2 (minimization: both decreases give positive score)
   or if x_cand strictly Pareto-dominates the current solution.
4. Elitist survivor selection preserves the enhanced solutions.
"""

import numpy as np
import moo_utils as U
import operators as ops


def local_search(problem, x, f, cv, step_size=0.05, max_steps=5, rng=None):
    """
    Coordinate hill-climbing local search around solution x.
    """
    if rng is None:
        rng = np.random.default_rng(42)
    current_x = np.array(x, dtype=float).ravel()
    current_f = np.array(f, dtype=float).ravel()
    current_cv = float(np.ravel(cv)[0]) if np.ndim(cv) > 0 else float(cv)

    d = len(current_x)
    span = problem.upper - problem.lower

    for _ in range(max_steps):
        improved = False
        dim = rng.integers(d)
        direction = 1.0 if rng.random() < 0.5 else -1.0
        delta = direction * step_size * span[dim]

        trial_x = current_x.copy()
        trial_x[dim] = np.clip(trial_x[dim] + delta, problem.lower[dim], problem.upper[dim])
        if problem.var_type in ("int", "binary"):
            trial_x[dim] = np.round(trial_x[dim])

        trial_f, trial_cv = problem.evaluate(trial_x)
        trial_f = trial_f[0] if trial_f.ndim > 1 else trial_f
        trial_cv = trial_cv[0] if np.ndim(trial_cv) > 0 else trial_cv

        # Accept if Pareto dominates or weakly improves both
        if U.constrained_dominates(trial_f, trial_cv, current_f, current_cv):
            current_x = trial_x
            current_f = trial_f
            current_cv = trial_cv
            improved = True
        else:
            # Score: net decrease across objectives
            delta_f = trial_f - current_f
            net_score = -float(np.sum(delta_f)) - 1e3 * max(0.0, trial_cv - current_cv)
            if net_score > 0:
                current_x = trial_x
                current_f = trial_f
                current_cv = trial_cv
                improved = True

        if not improved:
            break

    return current_x, current_f, current_cv


def initialize(problem, pop_size=50, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    return problem.random_solutions(pop_size, rng)


def evaluate(problem, X):
    return problem.evaluate(X)


def step(problem, X, F, CV, n_local=5, pc=0.9, pm=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    N = len(X)

    # 1. NSGA-II selection and offspring creation
    fronts = U.non_dominated_sort(F, CV)
    ranks = np.zeros(N, dtype=int)
    crowding = np.zeros(N, dtype=float)
    for r, fr in enumerate(fronts):
        ranks[fr] = r
        if len(fr) > 0:
            crowding[fr] = U.crowding_distance(F[fr])

    mating_idx = ops.crowded_tournament(ranks, crowding, N, rng)
    parents = X[mating_idx]

    offspring = []
    for i in range(0, N, 2):
        p1 = parents[i]
        p2 = parents[(i + 1) % N]
        c1, c2 = ops.sbx_crossover(p1, p2, problem.lower, problem.upper, rng, pc=pc)
        c1 = ops.polynomial_mutation(c1, problem.lower, problem.upper, rng, pm=pm)
        c2 = ops.polynomial_mutation(c2, problem.lower, problem.upper, rng, pm=pm)
        offspring.append(c1)
        if len(offspring) < N:
            offspring.append(c2)

    X_off = np.array(offspring[:N])
    if problem.var_type in ("int", "binary"):
        X_off = np.round(X_off)
    F_off, CV_off = evaluate(problem, X_off)

    # 2. Local search on top candidates
    fronts_off = U.non_dominated_sort(F_off, CV_off)
    elite_idx = fronts_off[0][:min(n_local, len(fronts_off[0]))]

    for idx in elite_idx:
        lx, lf, lcv = local_search(problem, X_off[idx], F_off[idx], CV_off[idx], rng=rng)
        X_off[idx] = lx
        F_off[idx] = lf
        CV_off[idx] = lcv

    # 3. Elitist survivor selection
    X_comb = np.vstack([X, X_off])
    F_comb = np.vstack([F, F_off])
    CV_comb = np.concatenate([CV, CV_off])

    fronts_comb = U.non_dominated_sort(F_comb, CV_comb)
    survivors = []
    for fr in fronts_comb:
        if len(survivors) + len(fr) <= N:
            survivors.extend(fr)
        else:
            needed = N - len(survivors)
            cd = U.crowding_distance(F_comb[fr])
            chosen = [fr[i] for i in np.argsort(-cd)[:needed]]
            survivors.extend(chosen)
            break

    survivors = np.array(survivors, dtype=int)
    return X_comb[survivors], F_comb[survivors], CV_comb[survivors]


def run(problem, pop_size=50, n_gen=30, n_local=5, pc=0.9, pm=None, seed=42):
    rng = np.random.default_rng(seed)
    X = initialize(problem, pop_size=pop_size, rng=rng)
    F, CV = evaluate(problem, X)

    for gen in range(n_gen):
        X, F, CV = step(problem, X, F, CV, n_local=n_local, pc=pc, pm=pm, rng=rng)

    nd_idx = U.pareto_front(F, CV)
    return X[nd_idx], F[nd_idx]
