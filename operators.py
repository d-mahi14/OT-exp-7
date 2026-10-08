"""
operators.py -- common building blocks shared by the algorithms (Stage 2).

Only NumPy is used.  Every random operator receives a NumPy Generator
`rng = np.random.default_rng(seed)` so that runs are reproducible.

Contents
--------
Weights            weight_grid
Real-coded         arithmetic_crossover (lecture), sbx_crossover,
                   uniform_mutation (lecture), polynomial_mutation
Differential Evol. de_mutation, de_binomial_crossover       (MOEA/D-DE, MODE)
Selection          crowded_tournament (NSGA-II), roulette_wheel (MOGA, NSGA)
Sharing / niching  minmax_normalize, sharing_function, niche_counts
                   (formulas of Part 2, slide 6)
Permutation (P8)   order_crossover, swap_mutation, inversion_mutation
Binary (P9)        one_point_crossover, uniform_crossover, bitflip_mutation
"""
import math
import numpy as np

import moo_utils as U


# ----------------------------------------------------------------------------
# Weight vectors
# ----------------------------------------------------------------------------
def weight_grid(n_obj, n_weights=11, include_ends=True):
    """Evenly spread weight vectors (rows sum to 1).

    n_obj == 2 : w1 = linspace(0, 1, n_weights), w2 = 1 - w1.
    n_obj >= 3 : Das-Dennis lattice with the smallest H giving >= n_weights rows.
    include_ends=False removes every row that contains a zero weight
    (a zero weight makes a method ignore one objective completely)."""
    if n_obj == 2:
        w1 = np.linspace(0.0, 1.0, n_weights + (0 if include_ends else 2))
        W = np.column_stack((w1, 1.0 - w1))
    else:
        H = 1
        while math.comb(H + n_obj - 1, n_obj - 1) < n_weights + (0 if include_ends else 2 * n_obj):
            H += 1
        W = U.das_dennis(n_obj, H)
    if not include_ends:
        W = W[np.all(W > 1e-12, axis=1)]
    return W


# ----------------------------------------------------------------------------
# Real-coded crossover / mutation
# ----------------------------------------------------------------------------
def arithmetic_crossover(p1, p2, alpha=0.5):
    """Lecture crossover: child = p1 + alpha * (p2 - p1).
    alpha = 0.5 is the average (Part 2); alpha = 0.6 gives the NSGA-II slide's
    'blend' 1 + 0.6 * (3 - 1) = 2.2.  Returns ONE child."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    return p1 + alpha * (p2 - p1)


def sbx_crossover(p1, p2, lower, upper, rng, eta=15.0, pc=0.9):
    """Simulated Binary Crossover (Deb & Agrawal 1995).  Returns two children.

    With probability pc the parents are crossed; each gene is crossed with
    probability 0.5.  For a crossed gene, u ~ U(0,1) and
        beta = (2u)^(1/(eta+1))               if u <= 0.5
             = (1/(2(1-u)))^(1/(eta+1))       otherwise
        c1 = 0.5[(1+beta) p1 + (1-beta) p2],  c2 = 0.5[(1-beta) p1 + (1+beta) p2]
    (the mean of the two children equals the mean of the parents).
    Large eta -> children close to the parents.  Children are clipped to bounds."""
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    c1, c2 = p1.copy(), p2.copy()
    if rng.random() > pc:
        return c1, c2
    n = len(p1)
    u = rng.random(n)
    beta = np.where(u <= 0.5, (2.0 * u) ** (1.0 / (eta + 1.0)),
                    (1.0 / (2.0 * (1.0 - u) + 1e-300)) ** (1.0 / (eta + 1.0)))
    mask = rng.random(n) < 0.5
    a = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
    b = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
    c1[mask], c2[mask] = a[mask], b[mask]
    return np.clip(c1, lower, upper), np.clip(c2, lower, upper)


def uniform_mutation(x, lower, upper, rng, delta=0.3, pm=None):
    """Lecture mutation: every gene is chosen with probability pm (default 1/n)
    and perturbed by d ~ U(-delta, +delta)  (NSGA-II slide: 2.2 -> 2.3).  Clipped."""
    x = np.array(x, dtype=float)
    pm = 1.0 / len(x) if pm is None else pm
    mask = rng.random(len(x)) < pm
    x[mask] += rng.uniform(-delta, delta, size=int(mask.sum()))
    return np.clip(x, lower, upper)


def polynomial_mutation(x, lower, upper, rng, eta=20.0, pm=None):
    """Polynomial mutation (Deb).  For a chosen gene (probability pm, default 1/n):
        u ~ U(0,1);  d = (2u)^(1/(eta+1)) - 1  if u < 0.5  else  1 - (2(1-u))^(1/(eta+1))
        x <- x + d * (upper - lower)
    Large eta -> smaller moves.  Result is clipped to the bounds."""
    x = np.array(x, dtype=float)
    lower = np.broadcast_to(np.asarray(lower, dtype=float), x.shape)
    upper = np.broadcast_to(np.asarray(upper, dtype=float), x.shape)
    pm = 1.0 / len(x) if pm is None else pm
    for i in range(len(x)):
        if rng.random() < pm:
            u = rng.random()
            if u < 0.5:
                d = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
            else:
                d = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
            x[i] += d * (upper[i] - lower[i])
    return np.clip(x, lower, upper)


# ----------------------------------------------------------------------------
# Differential evolution pieces
# ----------------------------------------------------------------------------
def de_mutation(x_r1, x_r2, x_r3, F=0.5):
    """V = X_r1 + F * (X_r2 - X_r3)   (MOEA/D-DE slide: 8 + 0.5(15-18) = 6.5)."""
    return np.asarray(x_r1, float) + F * (np.asarray(x_r2, float) - np.asarray(x_r3, float))


def de_binomial_crossover(target, mutant, CR, rng):
    """U_j = V_j if rand_j < CR (or j == j_rand, which is forced), else target_j.
    With one variable the trial is therefore always the mutant."""
    target = np.asarray(target, float)
    mutant = np.asarray(mutant, float)
    n = len(target)
    mask = rng.random(n) < CR
    mask[rng.integers(n)] = True
    return np.where(mask, mutant, target)


# ----------------------------------------------------------------------------
# Selection
# ----------------------------------------------------------------------------
def crowded_tournament(rank, crowding, n_select, rng):
    """NSGA-II binary tournament with the crowded-comparison operator:
    the lower rank wins; if the ranks are equal, the larger crowding distance wins;
    if both are equal, a random one of the two.  Returns n_select indices."""
    rank = np.asarray(rank)
    crowding = np.asarray(crowding, dtype=float)
    N = len(rank)
    winners = np.empty(n_select, dtype=int)
    for k in range(n_select):
        i, j = rng.integers(N, size=2)
        if rank[i] != rank[j]:
            winners[k] = i if rank[i] < rank[j] else j
        elif crowding[i] != crowding[j]:
            winners[k] = i if crowding[i] > crowding[j] else j
        else:
            winners[k] = i if rng.random() < 0.5 else j
    return winners


def roulette_wheel(fitness, n_select, rng):
    """Fitness-proportional selection: P(i) = fitness_i / sum(fitness).
    Fitness must be >= 0 (larger = better).  Returns n_select indices."""
    f = np.asarray(fitness, dtype=float)
    total = f.sum()
    if total <= 0:
        return rng.integers(len(f), size=n_select)
    cum = np.cumsum(f / total)
    cum[-1] = 1.0
    return np.searchsorted(cum, rng.random(n_select), side="right")


# ----------------------------------------------------------------------------
# Fitness sharing (Part 2, slide 6)
# ----------------------------------------------------------------------------
def minmax_normalize(F):
    """f' = (f - f_min) / (f_max - f_min) per column (zero range -> 0)."""
    F = np.asarray(F, dtype=float)
    lo, hi = F.min(axis=0), F.max(axis=0)
    span = np.where(hi - lo <= 0, 1.0, hi - lo)
    return (F - lo) / span


def sharing_function(d, sigma_share):
    """sh(d) = 1 - (d / sigma)^2 if d < sigma else 0."""
    d = np.asarray(d, dtype=float)
    return np.where(d < sigma_share, 1.0 - (d / sigma_share) ** 2, 0.0)


def niche_counts(Fn, sigma_share):
    """NC(i) = sum_j sh(d_ij) over ALL j including i itself (sh(0) = 1, so
    NC >= 1).  Fn must already be normalized.  Shared fitness = F / NC."""
    Fn = np.asarray(Fn, dtype=float)
    D = np.linalg.norm(Fn[:, None, :] - Fn[None, :, :], axis=2)
    return sharing_function(D, sigma_share).sum(axis=1)


# ----------------------------------------------------------------------------
# Permutation operators (P8, TSP)
# ----------------------------------------------------------------------------
def order_crossover(p1, p2, rng):
    """Order crossover (OX).  Copy a random slice of p1 into the child at the
    same positions, then fill the remaining positions with the missing cities
    in the order they appear in p2 (starting after the slice)."""
    p1 = np.asarray(p1, dtype=int)
    p2 = np.asarray(p2, dtype=int)
    n = len(p1)
    a, b = sorted(rng.choice(n + 1, size=2, replace=False))
    child = -np.ones(n, dtype=int)
    child[a:b] = p1[a:b]
    kept = set(child[a:b].tolist())
    order = np.concatenate((p2[b:], p2[:b]))                 # p2 read from position b, wrapping
    fill = [c for c in order if c not in kept]
    positions = list(range(b, n)) + list(range(0, a))
    for pos, city in zip(positions, fill):
        child[pos] = city
    return child


def swap_mutation(tour, rng):
    """Swap two randomly chosen cities."""
    t = np.array(tour, dtype=int)
    i, j = rng.choice(len(t), size=2, replace=False)
    t[i], t[j] = t[j], t[i]
    return t


def inversion_mutation(tour, rng):
    """Reverse a random segment (like one 2-opt move)."""
    t = np.array(tour, dtype=int)
    a, b = sorted(rng.choice(len(t) + 1, size=2, replace=False))
    t[a:b] = t[a:b][::-1].copy()
    return t


# ----------------------------------------------------------------------------
# Binary operators (P9)
# ----------------------------------------------------------------------------
def one_point_crossover(a, b, rng):
    """Cut at c in {1..n-1}; genes 1..c from a, c+1..n from b (and vice versa)."""
    a = np.asarray(a)
    b = np.asarray(b)
    c = int(rng.integers(1, len(a)))
    return np.concatenate((a[:c], b[c:])), np.concatenate((b[:c], a[c:]))


def uniform_crossover(a, b, rng):
    """Each gene is taken from a or b with probability 0.5."""
    a = np.asarray(a)
    b = np.asarray(b)
    m = rng.random(len(a)) < 0.5
    return np.where(m, a, b), np.where(m, b, a)


def bitflip_mutation(x, rng, pm=None):
    """Flip every bit independently with probability pm (default 1/n)."""
    x = np.array(x)
    pm = 1.0 / len(x) if pm is None else pm
    flip = rng.random(len(x)) < pm
    x[flip] = 1 - x[flip]
    return x