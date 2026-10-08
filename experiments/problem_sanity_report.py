"""Print (and save to results/problem_sanity.csv) every sanity check quoted in the PDF
plus a few facts the report needs (P5 infeasible fraction, P10 non-dominated fraction,
P8 coordinates).  Run from the project root:
    python experiments/problem_sanity_report.py
"""
import csv
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import moo_utils as U          # noqa: E402
import problems as P           # noqa: E402

rows = []   # (problem, quantity, computed, expected_from_PDF)


def add(problem, what, got, expected=""):
    rows.append((problem, what, str(got), str(expected)))


def r(a, nd=4):
    return np.round(np.asarray(a, dtype=float), nd).tolist()


# P1, P2, P3
F, _ = P.P1().evaluate([[5.0], [3.0]])
add("P1", "F(x=5), F(x=3)", r(F), "(4,4), (0,16)")
F, _ = P.P2().evaluate([[2.0], [10.0]])
add("P2", "F(x=2), F(x=10)", r(F), "(50,4), (10,100)")
x = np.zeros(10); x[0] = 0.5
F, _ = P.P3().evaluate(x)
add("P3", "F at x1=0.5, rest 0", r(F), "(0.5, 1-0.25=0.75)")
# P4
F, CV = P.P4().evaluate([[0, 0, 1, 0]])
add("P4", "w=(0,0,1,0): (risk, -return)", r(F), "(0.09, -0.15)")
p4 = P.P4()
w = p4.ref_weights[np.argmin(p4.true_front()[:, 0])]
add("P4", "min-risk weights", r(w, 3), "a mixture, not a single asset")
# P5
p5 = P.P5()
F, CV = p5.evaluate([[0.01, 0.05]])
add("P5", "(0.01,0.05): mass, deflection, CV", r(list(F[0]) + list(CV), 5), "3.925, 0.016, feasible")
rng = np.random.default_rng(0)
_, CVr = p5.evaluate(p5.random_solutions(100000, rng))
add("P5", "fraction of random designs infeasible (100000, seed 0)", round(float(np.mean(CVr > 0)), 4))
# P6
F, CV = P.P6().evaluate([[0, 0], [5, 3]])
add("P6", "(0,0),(5,3)", r(F), "(0,50),(136,4); both feasible, CV=%s" % CV.tolist())
# P7
p7 = P.P7()
F, CV = p7.evaluate([[10, 80], [70, 20]])
add("P7", "natural units of (10,80),(70,20)", r(p7.to_natural(F)), "(8,96),(56,24)")
# P8
p8 = P.P8()
add("P8", "coordinates (seed 7)", r(p8.coords, 2))
add("P8", "ideal (dist, time)", r(p8.ideal, 2))
add("P8", "nadir (dist, time)", r(p8.nadir, 2))
nn_d = P.nearest_neighbour_tour(p8.dist)
nn_t = P.nearest_neighbour_tour(p8.time)
add("P8", "nearest-neighbour tour on distance -> (dist, time)", r(p8.tour_costs(nn_d), 2))
add("P8", "nearest-neighbour tour on time -> (dist, time)", r(p8.tour_costs(nn_t), 2))
rnd = p8.random_solutions(1000, np.random.default_rng(0))
Fr, _ = p8.evaluate(rnd)
add("P8", "mean random tour (dist, time)", r(Fr.mean(axis=0), 2))
add("P8", "shortest-distance tour (2-opt)", p8.best_dist_tour.tolist())
add("P8", "shortest-time tour (2-opt)", p8.best_time_tour.tolist())
# P9
p9 = P.P9()
F, _ = p9.evaluate(np.ones((1, 30)))
add("P9", "all-ones mask: (error, #features)", r(F, 4), "accuracy baseline")
# P10
for M in (3, 5):
    p = P.P10(M)
    Fm, _ = p.evaluate(p.random_solutions(200, np.random.default_rng(0)))
    add("P10", "M=%d: non-dominated fraction of 200 random points" % M,
        round(len(U.non_dominated_sort(Fm)[0]) / 200, 3))
    X = p.random_solutions(5, np.random.default_rng(1)); X[:, M - 1:] = 0.5
    add("P10", "M=%d: sum f^2 when g=0" % M, r(np.sum(p.evaluate(X)[0] ** 2, axis=1)), "1.0")
# P11
p11 = P.P11()
F, CV = p11.evaluate([[0, 0], [400, 0], [300, 200], [0, 400]])
add("P11", "corners (profit, units)", r(p11.to_natural(F), 1), "(0,0),(1600,400),(2400,500),(2400,400)")
# P12
s = P.P12()
add("P12", "Pareto-optimal suppliers", [s.suppliers[i] for i in s.pareto_optimal()])
# P13
net = P.MultiTaskNet()
th = net.init_params()
add("P13b", "initial (cls loss, reg loss)", r(net.losses(th)))
add("P13a", "gradients at (0,5)", r(P.ToyTwoTask().gradients(np.array([0.0, 5.0]))))
# P14
p14 = P.P14()
F, _ = p14.evaluate([[13, 40], [13, 90]])
add("P14", "(13,40),(13,90) natural", r(p14.to_natural(F), 3), "(1.53, 9.0), (2.13, 20.2)")
x2 = (1.8 - 0.4 - 0.05 * 13) / 0.012
Ff, _ = p14.evaluate([[13, x2]])
add("P14", "front point with weight 1.8 kg: x2, runtime", r([x2, p14.to_natural(Ff)[0, 1]], 2),
    "runtime > 12 h -> reference point (1.8, 12) is attainable")

os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
path = os.path.join(ROOT, "results", "problem_sanity.csv")
with open(path, "w", newline="") as fh:
    w_ = csv.writer(fh)
    w_.writerow(["problem", "quantity", "computed", "expected_from_PDF"])
    w_.writerows(rows)
for row in rows:
    print("%-5s %-58s %s   %s" % (row[0], row[1], row[2], ("[PDF: " + row[3] + "]") if row[3] else ""))
print("\nSaved:", path)
