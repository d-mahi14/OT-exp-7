"""Plots built only from files the runner wrote (fronts/*.csv, runs/summary/convergence CSVs)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from experiments import config as C
from experiments import runner as R


def plot_fronts(tag, pid, seed, fname=None):
    files = sorted((C.FRONTS / tag).glob(f"a*_{pid}_s{seed}.csv"))
    if not files:
        return None
    fig, ax = plt.subplots(figsize=(7, 5))
    try:
        tf = R.get_true_front(R.load_problem(pid))
    except Exception:
        tf = None
    if tf is not None:
        ax.scatter(tf[:, 0], tf[:, 1], s=4, c="lightgray", label="true front", zorder=0)
    for f in files:
        F = np.atleast_2d(np.loadtxt(f, delimiter=","))
        aid = int(f.name[1:3])
        ax.scatter(F[:, 0], F[:, 1], s=14, label=C.CATALOGUE[aid][1])
    ax.set_xlabel("f1"); ax.set_ylabel("f2")
    ax.set_title(f"{pid}, seed {seed} (f1-f2 projection)")
    ax.legend(fontsize=7)
    out = C.PLOTS / (fname or f"{tag}_fronts_{pid}_s{seed}.png")
    C.PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
    return out


def plot_metric_bars(tag, metric="hv"):
    summ = R.read_csv(C.RAW / f"summary_{tag}.csv")
    outs = []
    for pid in sorted({r["problem"] for r in summ}):
        rs = [r for r in summ if r["problem"] == pid and r[metric + "_mean"] != ""]
        if not rs:
            continue
        labels = [f'{r["alg_name"]} {r["params"] if r["params"] != "{}" else ""}'.strip() for r in rs]
        m = [float(r[metric + "_mean"]) for r in rs]
        s = [float(r[metric + "_std"]) if r[metric + "_std"] != "" else 0.0 for r in rs]
        fig, ax = plt.subplots(figsize=(max(6, 0.6 * len(rs)), 4))
        ax.bar(range(len(rs)), m, yerr=s, capsize=3)
        ax.set_xticks(range(len(rs))); ax.set_xticklabels(labels, rotation=60, ha="right", fontsize=7)
        ax.set_ylabel(metric.upper() + " (mean ± std over seeds)"); ax.set_title(f"{pid}")
        out = C.PLOTS / f"{tag}_{metric}_bars_{pid}.png"
        C.PLOTS.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        outs.append(out)
    return outs


def plot_convergence(tag):
    p = C.RAW / f"convergence_{tag}.csv"
    if not p.exists():
        return []
    rows = [r for r in R.read_csv(p) if r["status"] == "ok" and r["hv"] != ""]
    outs = []
    for pid in sorted({r["problem"] for r in rows}):
        fig, ax = plt.subplots(figsize=(6, 4))
        for name in sorted({r["alg_name"] for r in rows if r["problem"] == pid}):
            rs = [r for r in rows if r["problem"] == pid and r["alg_name"] == name]
            bs = sorted({int(r["budget"]) for r in rs})
            mean = [np.mean([float(r["hv"]) for r in rs if int(r["budget"]) == b]) for b in bs]
            std = [np.std([float(r["hv"]) for r in rs if int(r["budget"]) == b]) for b in bs]
            ax.errorbar(bs, mean, yerr=std, marker="o", capsize=3, label=name)
        ax.set_xlabel("evaluation budget (independent runs, same seed)")
        ax.set_ylabel("HV"); ax.set_title(f"HV vs budget, {pid}"); ax.legend(fontsize=7)
        out = C.PLOTS / f"{tag}_convergence_{pid}.png"
        C.PLOTS.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)
        outs.append(out)
    return outs


def plot_task_c_sensitivity(tag):
    p = C.RAW / f"summary_{tag}.csv"
    if not p.exists():
        return []
    summ = R.read_csv(p)
    outs = []
    c_rows = [r for r in summ if r["task"] == "C"]
    if not c_rows:
        return []
    for aid, spec in C.TASK_C.items():
        name = C.CATALOGUE[aid][1]
        param_name = spec["param"]
        for pid in sorted({r["problem"] for r in c_rows if int(r["alg_id"]) == aid}):
            rs = [r for r in c_rows if int(r["alg_id"]) == aid and r["problem"] == pid]
            if not rs:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            labels = [r["params"] for r in rs]
            hvs = [float(r["hv_mean"]) if r["hv_mean"] != "" else 0.0 for r in rs]
            errs = [float(r["hv_std"]) if r["hv_std"] != "" else 0.0 for r in rs]
            ax.bar(range(len(rs)), hvs, yerr=errs, capsize=4, color="teal", alpha=0.8)
            ax.set_xticks(range(len(rs)))
            ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
            ax.set_ylabel("Hypervolume (mean ± std)")
            ax.set_title(f"Task C: {name} on {pid} - Sensitivity ({param_name})")
            out = C.PLOTS / f"{tag}_task_c_alg{aid}_{pid}_sensitivity.png"
            fig.savefig(out, dpi=150, bbox_inches="tight")
            plt.close(fig)
            outs.append(out)
    return outs


def plot_task_d2_many_obj(tag):
    p = C.RAW / f"summary_{tag}.csv"
    if not p.exists():
        return []
    summ = R.read_csv(p)
    rs = [r for r in summ if r["problem"] == "P10_M5" and (r["task"] in ("D2", "A"))]
    if not rs:
        return []
    # Deduplicate by alg_id
    seen = {}
    for r in rs:
        aid = int(r["alg_id"])
        if aid in C.TASK_D2["algs"] and aid not in seen:
            seen[aid] = r
    d2_rows = list(seen.values())
    if not d2_rows:
        return []
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    names = [r["alg_name"] for r in d2_rows]
    hvs = [float(r["hv_mean"]) if r["hv_mean"] != "" else 0.0 for r in d2_rows]
    hv_err = [float(r["hv_std"]) if r["hv_std"] != "" else 0.0 for r in d2_rows]
    ax1.bar(names, hvs, yerr=hv_err, capsize=4, color="royalblue", alpha=0.8)
    ax1.set_ylabel("Hypervolume (mean ± std)")
    ax1.set_title("P10 (M=5) - Hypervolume")
    ax1.tick_params(axis="x", rotation=45)

    sps = [float(r["spacing_mean"]) if r["spacing_mean"] != "" else 0.0 for r in d2_rows]
    sp_err = [float(r["spacing_std"]) if r["spacing_std"] != "" else 0.0 for r in d2_rows]
    ax2.bar(names, sps, yerr=sp_err, capsize=4, color="coral", alpha=0.8)
    ax2.set_ylabel("Spacing (mean ± std)")
    ax2.set_title("P10 (M=5) - Spacing")
    ax2.tick_params(axis="x", rotation=45)

    out = C.PLOTS / f"{tag}_task_d2_p10_m5.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


def plot_task_d3_discrete(tag):
    p = C.RAW / f"summary_{tag}.csv"
    if not p.exists():
        return []
    summ = R.read_csv(p)
    rs = [r for r in summ if r["problem"] == "P8" and (r["task"] in ("D3", "A"))]
    if not rs:
        return []
    seen = {}
    for r in rs:
        aid = int(r["alg_id"])
        if aid in C.TASK_D3["algs"] and aid not in seen:
            seen[aid] = r
    d3_rows = list(seen.values())
    if not d3_rows:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    names = [r["alg_name"] for r in d3_rows]
    hvs = [float(r["hv_mean"]) if r["hv_mean"] != "" else 0.0 for r in d3_rows]
    hv_err = [float(r["hv_std"]) if r["hv_std"] != "" else 0.0 for r in d3_rows]
    ax.bar(names, hvs, yerr=hv_err, capsize=4, color="darkseagreen", alpha=0.8)
    ax.set_ylabel("Hypervolume (mean ± std)")
    ax.set_title("Task D3: Discrete TSP (P8) Comparison")
    ax.tick_params(axis="x", rotation=45)
    out = C.PLOTS / f"{tag}_task_d3_p8.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return [out]


def plot_mcdm_p12(tag):
    """Plot TOPSIS and VIKOR results on P12 supplier selection."""
    try:
        p12 = R.load_problem("P12")
        top_mod = R.load_algorithm(8)
        vik_mod = R.load_algorithm(9)
        _, C_base = top_mod.run(p12)
        w_shift = np.array([0.2, 0.4, 0.2, 0.2])
        _, C_shift = top_mod.run(p12, weights=w_shift)
        _, Q_base, _ = vik_mod.run(p12)
        _, Q_shift, _ = vik_mod.run(p12, weights=w_shift)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        x = np.arange(len(p12.suppliers))
        w = 0.35
        ax1.bar(x - w/2, C_base, w, label="Base w=[0.3,0.3,0.2,0.2]", color="steelblue")
        ax1.bar(x + w/2, C_shift, w, label="Shifted w=[0.2,0.4,0.2,0.2]", color="orange")
        ax1.set_xticks(x); ax1.set_xticklabels(p12.suppliers)
        ax1.set_ylabel("Closeness C_i (higher is better)")
        ax1.set_title("TOPSIS (Alg 8) on P12")
        ax1.legend(fontsize=7)

        ax2.bar(x - w/2, Q_base, w, label="Base w=[0.3,0.3,0.2,0.2]", color="seagreen")
        ax2.bar(x + w/2, Q_shift, w, label="Shifted w=[0.2,0.4,0.2,0.2]", color="tomato")
        ax2.set_xticks(x); ax2.set_xticklabels(p12.suppliers)
        ax2.set_ylabel("Compromise Index Q_i (lower is better)")
        ax2.set_title("VIKOR (Alg 9) on P12")
        ax2.legend(fontsize=7)

        out = C.PLOTS / f"{tag}_mcdm_p12.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return [out]
    except Exception:
        return []


def plot_gradient_dynamics_p13(tag):
    """Plot multi-task learning dynamics for MGDA (#36) and GradNorm (#37) on P13."""
    try:
        p13 = R.load_problem("P13")
        mgda_mod = R.load_algorithm(36)
        gn_mod = R.load_algorithm(37)

        _, mgda_f = mgda_mod.run(p13, max_steps=60, eta=0.1)
        _, gn_losses, gn_w = gn_mod.run(p13, n_steps=60, lr_theta=0.01, lr_w=0.025, seed=101)

        # Baseline: Equal weights gradient descent
        theta_eq = np.array(p13.start, dtype=float)
        eq_losses = [p13.evaluate(theta_eq)[0]]
        for _ in range(60):
            grads = p13.gradients(theta_eq)
            avg_g = 0.5 * grads[0] + 0.5 * grads[1]
            theta_eq = theta_eq - 0.1 * avg_g
            eq_losses.append(p13.evaluate(theta_eq)[0])
        eq_losses = np.array(eq_losses)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        # Total loss trace
        ax1.plot(np.sum(mgda_f, axis=1), label="MGDA (Alg 36)", color="navy", lw=1.8)
        ax1.plot(np.sum(gn_losses, axis=1), label="GradNorm (Alg 37)", color="crimson", lw=1.8)
        ax1.plot(np.sum(eq_losses, axis=1), label="Equal Weights (baseline)", color="gray", ls="--", lw=1.5)
        ax1.set_xlabel("Optimization Step")
        ax1.set_ylabel("Total Loss (L1 + L2)")
        ax1.set_title("P13 Multi-Task Optimization Loss Dynamics")
        ax1.legend(fontsize=8)

        # GradNorm adaptive weights
        ax2.plot(gn_w[:, 0], label="w1 (Task 1 / classification)", color="purple", lw=1.8)
        ax2.plot(gn_w[:, 1], label="w2 (Task 2 / regression)", color="teal", lw=1.8)
        ax2.axhline(1.0, color="gray", ls=":", label="Equal Weight (1.0)")
        ax2.set_xlabel("Optimization Step")
        ax2.set_ylabel("GradNorm Task Weights")
        ax2.set_title("GradNorm (Alg 37) Weight Balancing on P13")
        ax2.legend(fontsize=8)

        out = C.PLOTS / f"{tag}_gradient_dynamics_p13.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        plt.close(fig)
        return [out]
    except Exception:
        return []


def make_all(tag, seed=None):
    outs = []
    runs = R.read_csv(C.RAW / f"runs_{tag}.csv")
    if not runs:
        return outs
    seed = seed or sorted({r["seed"] for r in runs})[0]
    for pid in sorted({r["problem"] for r in runs if r["status"] == "ok"}):
        o = plot_fronts(tag, pid, seed)
        if o:
            outs.append(o)
    R.aggregate(tag)
    for m in ("hv", "spacing", "igd"):
        outs += plot_metric_bars(tag, m)
    outs += plot_convergence(tag)
    outs += plot_task_c_sensitivity(tag)
    outs += plot_task_d2_many_obj(tag)
    outs += plot_task_d3_discrete(tag)
    outs += plot_mcdm_p12(tag)
    outs += plot_gradient_dynamics_p13(tag)
    return outs