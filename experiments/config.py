"""Single source of truth for every experiment setting (budgets, seeds, pops, mappings)."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
FRONTS = RESULTS / "fronts"
PLOTS = RESULTS / "plots"

SEEDS = list(range(101, 111))      # 10 independent runs (PDF Sec. 2)
BUDGET = 10_000                    # function evaluations (PDF Sec. 2, "fair budget")
DEFAULT_POP = 100
POP_BY_PROBLEM = {"P10_M3": 91, "P10_M5": 210}   # Das-Dennis sizes: C(14,2), C(10,4)
HV_SAMPLES = 200_000
N_TRUE = 500
SWEEP_N = 11                       # >= 10 settings for algorithms 1-4 (Task D)
SWEEP_CANDIDATES = 1000

# id: (module stem, name, kind, test problems from the PDF catalogue)
# kind: evo (pop/n_gen driven) | sweep (algs 1-4) | custom (needs its own adapter)
CATALOGUE = {
    1: ("01_weighted_sum", "Weighted Sum", "sweep", ["P1", "P3", "P4"]),
    2: ("02_epsilon_constraint", "Epsilon-Constraint", "sweep", ["P2", "P3", "P5"]),
    3: ("03_bensons_method", "Benson's Method", "sweep", ["P1", "P7"]),
    4: ("04_weight_metric", "Weight Metric", "sweep", ["P1", "P3"]),
    5: ("05_value_function", "Value Function", "custom", ["P4", "P14"]),
    6: ("06_goal_programming", "Goal Programming", "custom", ["P11", "P7"]),
    7: ("07_lexicographic", "Lexicographic", "custom", ["P11", "P7"]),
    8: ("08_topsis", "TOPSIS", "custom", ["P12"]),
    9: ("09_vikor", "VIKOR", "custom", ["P12"]),
    10: ("10_vega", "VEGA", "evo", ["P2", "P3"]),
    11: ("11_moga", "MOGA", "evo", ["P3", "P5"]),
    12: ("12_nsga", "NSGA", "evo", ["P5", "P7"]),
    13: ("13_npga", "NPGA", "evo", ["P3", "P5"]),
    14: ("14_nsga2", "NSGA-II", "evo", ["P3", "P5", "P6"]),
    15: ("15_nsga3", "NSGA-III", "evo", ["P10_M3", "P10_M5"]),
    16: ("16_spea2", "SPEA2", "evo", ["P4", "P5"]),
    17: ("17_pesa2", "PESA-II", "evo", ["P3", "P5"]),
    18: ("18_moead", "MOEA/D", "evo", ["P3", "P10_M5"]),
    19: ("19_moead_de", "MOEA/D-DE", "evo", ["P3", "P5"]),
    20: ("20_moead_aco", "MOEA/D-ACO", "evo", ["P8"]),
    21: ("21_constrained_nsga2", "Constrained NSGA-II", "evo", ["P5", "P6", "P7"]),
    22: ("22_mopso", "MOPSO", "evo", ["P4", "P5"]),
    23: ("23_mogwo", "MOGWO", "evo", ["P5", "P9"]),
    24: ("24_moabc", "MOABC", "evo", ["P3", "P6"]),
    25: ("25_moaco", "MOACO", "evo", ["P8"]),
    26: ("26_mosa", "MOSA", "evo", ["P3", "P5"]),
    27: ("27_mode", "MODE", "evo", ["P3", "P5"]),
    28: ("28_mo_tlbo", "MO-TLBO", "evo", ["P3", "P5"]),
    29: ("29_moba", "MOBA", "evo", ["P3", "P4"]),
    30: ("30_moma", "MOMA", "evo", ["P5", "P9"]),
    31: ("31_rvea", "RVEA", "evo", ["P10_M5"]),
    32: ("32_ibea", "IBEA", "evo", ["P3", "P14"]),
    33: ("33_sms_emoa", "SMS-EMOA", "evo", ["P3", "P14"]),
    34: ("34_hype", "HypE", "evo", ["P10_M5"]),
    35: ("35_r_nsga2", "R-NSGA-II", "evo", ["P5", "P14"]),
    36: ("36_mgda", "MGDA", "custom", ["P13"]),
    37: ("37_gradnorm", "GradNorm", "custom", ["P13"]),
}

# Task C, roll number 17 -> algorithms 17, 29, 4.
# Parameter NAMES are guesses: the runner checks them against each run() signature
# and records status=param_not_accepted instead of silently running with a default.
ROLL = 17
TASK_C = {
    17: {"param": "n_divs", "values": [4, 8, 16]},          # PESA-II grid divisions
    29: {"param": "alpha", "values": [0.7, 0.9, 0.99]},  # MOBA
    4: {"param": "p", "values": [1, 2, float("inf")]},     # Weight Metric norm
}
TASK_C_PROBLEMS = {17: ["P3", "P5"], 29: ["P3", "P4"], 4: ["P1", "P3"]}

TASK_D2 = {"problems": ["P10_M5"], "algs": [14, 15, 18, 31, 34]}
TASK_D3 = {"problems": ["P8"], "algs": [20, 25, 14, 30]}


def jobs_for(task, seeds=None):
    """Return the list of job dicts for a task. Job = one (alg, problem, seed, params) run."""
    seeds = seeds or SEEDS
    runnable = lambda a: True
    jobs = []
    if task == "A":      # every algorithm on every problem in its Test-problems column
        for a, (_, _, _, probs) in CATALOGUE.items():
            jobs += [dict(task="A", alg=a, pid=p, seed=s, params={}) for p in probs for s in seeds]
    elif task == "C":
        for a, spec in TASK_C.items():
            for p in TASK_C_PROBLEMS[a]:
                for v in spec["values"]:
                    jobs += [dict(task="C", alg=a, pid=p, seed=s, params={spec["param"]: v}) for s in seeds]
    elif task == "D1":   # continuous bi-objective: P3 (unconstrained) and P5 (constrained)
        for a, (_, _, _, probs) in CATALOGUE.items():
            if runnable(a):
                jobs += [dict(task="D1", alg=a, pid=p, seed=s, params={})
                         for p in probs if p in ("P3", "P5") for s in seeds]
    elif task == "D2":
        jobs = [dict(task="D2", alg=a, pid=p, seed=s, params={})
                for a in TASK_D2["algs"] for p in TASK_D2["problems"] for s in seeds]
    elif task == "D3":
        jobs = [dict(task="D3", alg=a, pid=p, seed=s, params={})
                for a in TASK_D3["algs"] for p in TASK_D3["problems"] for s in seeds]
    else:
        raise ValueError(f"unknown task {task}")
    return jobs