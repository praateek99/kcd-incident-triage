"""
Reproduce every number, table and figure in the paper:   python run_experiments.py
Quick development check on separate seeds (writes nothing): python run_experiments.py --dev
Redraw figures from saved CSVs:                          python run_experiments.py --figures-only

Outputs go to results/. Settings below were frozen before the confirmation run, and the
confirmation seeds (CONFIRM_BASE, SENS_BASE) were not used during development.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t as student_t, wilcoxon

from kcd import (RULES, CostModel, TriageAgent, apply_defects, calibration_metrics, company_offsets,
                 completeness_score, constrained_decision, defer_top, deferral_cost, evaluate,
                 generate_latent, knowledge_bounds, rules_only_decision, simulate_reviewer)

OUT = Path(__file__).parent / "results"
OUT.mkdir(exist_ok=True)

# ---- frozen settings --------------------------------------------------------------
R = 30                                   # replications in the main study
N_TRAIN = [150, 400, 1000, 2400]         # nested subsets of one 2,400-incident training pool
N_TEST = 1600                            # one shared latent test set per replication
TRAIN_DEFECT = 0.30
TEST_DEFECTS = (0.30, 0.60)              # common random numbers: 60% is a superset of 30%
BUDGET = 0.20                            # matched human-review quota
BUDGETS = np.round(np.arange(0.05, 0.51, 0.05), 2)
CM = CostModel()
CONFIRM_BASE, SENS_BASE, DEV_BASE = 70_000, 90_000, 10_000
R_SENS = 15
N_BOOT = 10_000

DATA, KB = "Data agent, always answer", "KB agent, always answer"
RA, KCD = "Reviewer-aware deferral", "KCD (proposed)"


def seeds(rep_seed: int) -> dict:
    s = 10 * rep_seed
    return dict(company=s + 1, train=s + 2, train_def=s + 3, test=s + 4, test_def=s + 5, reviewer=s + 6, tie=s + 7)


def one_replication(rep_seed, cm=CM, sizes=N_TRAIN, defect_rates=TEST_DEFECTS, budgets_by_rate=None,
                    noise_sd=0.35, company_sd=0.25, calibrate=False, methods="all"):
    """Returns (metric rows, calibration rows, rule rows) for one replication."""
    sd = seeds(rep_seed)
    off = company_offsets(sd["company"], sd=company_sd)
    pool = apply_defects(generate_latent(max(sizes), sd["train"], off, noise_sd), TRAIN_DEFECT, sd["train_def"])
    latent_test = generate_latent(N_TEST, sd["test"], off, noise_sd)
    y = latent_test.label.values
    human = simulate_reviewer(y, cm, sd["reviewer"])            # one reviewer outcome per incident
    tie = np.random.default_rng(sd["tie"]).random(N_TEST)        # fixed per-incident tie-break / random score
    C = cm.matrix()
    budgets_by_rate = budgets_by_rate or {r: (BUDGET,) for r in defect_rates}
    rows, cal_rows, rule_rows = [], [], []
    tests = {r: apply_defects(latent_test, r, sd["test_def"]) for r in defect_rates}
    for rate, test in tests.items():
        lo, hi, fired = knowledge_bounds(test)
        for name, (kind, level, _) in RULES.items():
            m = fired[name].values
            agree = ((y[m] >= level) if kind == "floor" else (y[m] <= level)).mean() if m.any() else np.nan
            rule_rows.append(dict(seed=rep_seed, test_defect=rate, rule=name, firing_rate=m.mean(), agreement=agree))
    for n in sizes:
        agent = TriageAgent(rep_seed, calibrate=calibrate).fit(pool.iloc[:n])
        for rate, test in tests.items():
            P = agent.predict_proba(test)
            cal_rows.append(dict(seed=rep_seed, n_train=n, test_defect=rate, **calibration_metrics(P, y)))
            d_data, r_data = agent.decide(P, C)
            lo, hi, _ = knowledge_bounds(test)
            d_kb, r_kb = constrained_decision(P, C, lo, hi)
            D = deferral_cost(P, cm)
            none = np.zeros(N_TEST, bool)

            def add(method, decision, deferred, b):
                rows.append(dict(seed=rep_seed, n_train=n, test_defect=rate, budget=b, method=method,
                                 **evaluate(y, decision, deferred, human, cm)))

            add(DATA, d_data, none, 0.0)
            add(KB, d_kb, none, 0.0)
            if methods == "all":
                add("Rules-only agent", rules_only_decision(test), none, 0.0)
                add("Reviewer-aware, no budget", d_data, r_data > D, -1)
                add("KCD, no budget", d_kb, r_kb > D, -1)
            for b in budgets_by_rate[rate]:
                pick = lambda s: defer_top(s, b, tie)
                m_ra = pick(r_data - D)
                add(RA, d_data, m_ra, b)
                add(KCD, d_kb, pick(r_kb - D), b)
                if methods == "all":
                    add("Random deferral", d_data, pick(tie), b)
                    add("Completeness deferral", d_data, pick(completeness_score(test).astype(float)), b)
                    add("Max-probability deferral", d_data, pick(1 - P.max(1)), b)
                    add("Risk-ranked deferral (heuristic)", d_data, pick(r_data), b)
                    add("KCD, risk-ranked (heuristic)", d_kb, pick(r_kb), b)
                    add("KCD, fixed mask (diagnostic)", d_kb, m_ra, b)
    return rows, cal_rows, rule_rows


def run_main(base):
    rows, cal, rules = [], [], []
    for r in range(R):
        a, b, c = one_replication(base + r, budgets_by_rate={0.30: BUDGETS, 0.60: (BUDGET,)})
        rows += a; cal += b; rules += c
    return pd.DataFrame(rows), pd.DataFrame(cal), pd.DataFrame(rules)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------
def holm(p):
    p = np.asarray(p, float)
    order, adj, running = np.argsort(p), np.empty(len(p)), 0.0
    for rank, i in enumerate(order):
        running = max(running, (len(p) - rank) * p[i])
        adj[i] = min(running, 1.0)
    return adj


def wilcox(a, b):
    d = np.asarray(a) - np.asarray(b)
    return 1.0 if np.allclose(d, 0) else wilcoxon(a, b).pvalue


def boot_ci(x, n=N_BOOT, seed=0):
    x = np.asarray(x)
    idx = np.random.default_rng(seed).integers(0, len(x), (n, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


def t_ci(x):
    x = np.asarray(x)
    h = student_t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    return x.mean() - h, x.mean() + h


def cost(df, n, rate, method, b=BUDGET):
    bb = 0.0 if method in (DATA, KB, "Rules-only agent") else b
    s = df[(df.n_train == n) & (df.test_defect == rate) & (df.method == method) & np.isclose(df.budget, bb)]
    return s.set_index("seed").cost_per_incident.sort_index()


def ke(df, n, rate):
    """Per-replication knowledge effect within reviewer-aware deferral (% change in cost)."""
    return 100 * (cost(df, n, rate, KCD) / cost(df, n, rate, RA) - 1)


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
ORDER = ["Rules-only agent", DATA, "Random deferral", "Completeness deferral", "Max-probability deferral",
         "Risk-ranked deferral (heuristic)", RA, KCD]


def table1(df):
    out = []
    for n in (150, 2400):
        base = cost(df, n, 0.30, DATA)
        for m in ORDER:
            c = cost(df, n, 0.30, m)
            bb = 0.0 if m in (DATA, "Rules-only agent") else BUDGET
            g = df[(df.n_train == n) & (df.test_defect == 0.30) & (df.method == m) & np.isclose(df.budget, bb)]
            out.append({"Training incidents": n, "Method": m, "Deferred (%)": 100 * g.deferred.mean(),
                        "Error on answered (%)": 100 * g.answered_error.mean(),
                        "Severe under-triage (%)": 100 * g.severe_undertriage.mean(),
                        "Cost per incident": c.mean(), "Cost SD": c.std(),
                        "Mean saving vs always-answer (%)": (100 * (1 - c / base)).mean()})
    t = pd.DataFrame(out)
    t.round(3).to_csv(OUT / "table1_main_comparison.csv", index=False)
    return t


def table2(df):
    out = []
    for rate in TEST_DEFECTS:
        for n in N_TRAIN:
            k = ke(df, n, rate)
            lo, hi = boot_ci(k)
            de = 100 * (1 - cost(df, n, rate, RA) / cost(df, n, rate, DATA))
            out.append({"Test defect rate": rate, "Training incidents": n,
                        "Neither": cost(df, n, rate, DATA).mean(), "Knowledge only": cost(df, n, rate, KB).mean(),
                        "Deferral only": cost(df, n, rate, RA).mean(), "Both (KCD)": cost(df, n, rate, KCD).mean(),
                        "Knowledge effect (%)": k.mean(), "KE bootstrap CI low": lo, "KE bootstrap CI high": hi,
                        "KE replications improved": int((k < 0).sum()),
                        "p knowledge (raw)": wilcox(cost(df, n, rate, KCD), cost(df, n, rate, RA)),
                        "Deferral saving (%)": de.mean(),
                        "p deferral (raw)": wilcox(cost(df, n, rate, RA), cost(df, n, rate, DATA))})
    t = pd.DataFrame(out)
    t["p knowledge (Holm)"] = holm(t["p knowledge (raw)"])
    t["p deferral (Holm)"] = holm(t["p deferral (raw)"])
    t.to_csv(OUT / "table2_ablation.csv", index=False)
    return t


def interaction(df):
    rows = []
    for rate in TEST_DEFECTS:
        d = ke(df, 150, rate) - ke(df, 2400, rate)
        lo, hi = boot_ci(d)
        rows.append({"Test defect rate": rate, "KE(150) - KE(2400) mean (pp)": d.mean(), "CI low": lo, "CI high": hi,
                     "p raw": 1.0 if np.allclose(d, 0) else wilcoxon(d).pvalue})
    t = pd.DataFrame(rows)
    t["p Holm"] = holm(t["p raw"])
    t.to_csv(OUT / "interaction_test.csv", index=False)
    return t


def diagnostics(df, cal, rules):
    """Decision-vs-ranking decomposition, heuristic comparison, calibration, rule agreement."""
    rows = []
    for rate in TEST_DEFECTS:
        for n in N_TRAIN:
            ra, kcd = cost(df, n, rate, RA), cost(df, n, rate, KCD)
            fm = cost(df, n, rate, "KCD, fixed mask (diagnostic)")
            rr = cost(df, n, rate, "Risk-ranked deferral (heuristic)")
            kr = cost(df, n, rate, "KCD, risk-ranked (heuristic)")
            rows.append({"Test defect rate": rate, "Training incidents": n,
                         "Decision pathway (%)": (100 * (fm / ra - 1)).mean(),
                         "Ranking pathway (%)": (100 * (kcd / fm - 1)).mean(),
                         "Reviewer-aware vs risk-ranked (%)": (100 * (ra / rr - 1)).mean(),
                         "KE under risk ranking (%)": (100 * (kr / rr - 1)).mean()})
    pd.DataFrame(rows).to_csv(OUT / "diagnostics_pathways.csv", index=False)
    cal.groupby(["n_train", "test_defect"])[["brier", "log_loss", "ece"]].mean().round(4).to_csv(OUT / "calibration.csv")
    rd = rules.groupby(["test_defect", "rule"]).agg(firing_rate_pct=("firing_rate", lambda x: 100 * x.mean()),
                                                    agreement_pct=("agreement", lambda x: 100 * x.mean()),
                                                    agreement_min_pct=("agreement", lambda x: 100 * x.min()),
                                                    agreement_max_pct=("agreement", lambda x: 100 * x.max()))
    rd.round(2).to_csv(OUT / "rule_diagnostics.csv")
    nat = df[df.budget == -1].groupby(["n_train", "test_defect", "method"])[["deferred", "cost_per_incident"]].mean()
    nat.round(4).to_csv(OUT / "no_budget_operating_point.csv")
    return pd.DataFrame(rows), rd


# ---------------------------------------------------------------------------
# Sensitivity analyses at scarce (150) and ample (2,400) data
# ---------------------------------------------------------------------------
SENS = [
    ("Baseline", {}),
    ("Higher label noise (sd 0.60)", dict(noise_sd=0.60)),
    ("Stronger company heterogeneity (sd 0.50)", dict(company_sd=0.50)),
    ("Reviewer accuracy 0.85", dict(cm=CostModel(human_acc=0.85))),
    ("Reviewer accuracy 0.98", dict(cm=CostModel(human_acc=0.98))),
    ("Over-triage cost 150", dict(cm=CostModel(over=150))),
    ("Over-triage cost 450", dict(cm=CostModel(over=450))),
    ("Calibrated classifier (sigmoid, train-only CV)", dict(calibrate=True)),
]


def sensitivity():
    rows = []
    for name, kw in SENS:
        for r in range(R_SENS):
            a, cal, rules = one_replication(SENS_BASE + r, sizes=[150, 2400], defect_rates=(0.30,), methods="core", **kw)
            for x in a:
                rows.append({**x, "setting": name})
            for c in cal:
                rows.append({**c, "setting": name, "method": "_calibration"})
            for rr in rules:
                rows.append({**rr, "setting": name, "method": "_rules"})
    raw = pd.DataFrame(rows)
    raw.to_csv(OUT / "sensitivity_runs.csv", index=False)
    summ = []
    for name, _ in SENS:
        s = raw[(raw.setting == name) & ~raw.method.str.startswith("_")]
        for n in (150, 2400):
            k = ke(s, n, 0.30)
            de = 100 * (1 - cost(s, n, 0.30, RA) / cost(s, n, 0.30, DATA))
            lo, hi = boot_ci(k)
            cal = raw[(raw.setting == name) & (raw.method == "_calibration") & (raw.n_train == n)]
            rl = raw[(raw.setting == name) & (raw.method == "_rules")]
            summ.append({"Setting": name, "Training incidents": n, "Deferral saving (%)": de.mean(),
                         "Knowledge effect (%)": k.mean(), "KE CI low": lo, "KE CI high": hi,
                         "p knowledge (raw)": wilcox(cost(s, n, 0.30, KCD), cost(s, n, 0.30, RA)),
                         "Mean rule agreement (%)": 100 * rl.agreement.mean(), "ECE": cal.ece.mean(),
                         "Brier": cal.brier.mean()})
    t = pd.DataFrame(summ)
    t.to_csv(OUT / "sensitivity_summary.csv", index=False)
    return t


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False, "savefig.dpi": 1000, "pdf.fonttype": 42})
# Monochrome, print-friendly styles: policies are distinguished by line style and marker, not colour.
POLICY_STYLE = {
    "Random deferral":          dict(color="#9a9a9a", ls="--", marker=None),
    "Max-probability deferral": dict(color="#6e6e6e", ls="-.", marker=None),
    "Completeness deferral":    dict(color="#444444", ls=":",  marker=None),
    RA:                         dict(color="#222222", ls="-",  marker="o"),
    KCD:                        dict(color="#000000", ls="-",  marker="s"),
}
SENS_LABEL = {
    "Baseline": "Baseline",
    "Higher label noise (sd 0.60)": "Higher noise",
    "Stronger company heterogeneity (sd 0.50)": "More company variation",
    "Reviewer accuracy 0.85": "Reviewer acc. 0.85",
    "Reviewer accuracy 0.98": "Reviewer acc. 0.98",
    "Over-triage cost 150": "Over-triage cost 150",
    "Over-triage cost 450": "Over-triage cost 450",
    "Calibrated classifier (sigmoid, train-only CV)": "Calibrated",
}


def figure1(df):
    """Cost against review budget with scarce (150) and ample (2,400) training data, 30% defects."""
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.6), sharey=False, gridspec_kw={"wspace": 0.28})
    for ax, n, tag in ((axes[0], 150, "(a) 150 training incidents"), (axes[1], 2400, "(b) 2,400 training incidents")):
        base = cost(df, n, 0.30, DATA)
        blo, bhi = t_ci(base)
        for m, st in POLICY_STYLE.items():
            xs, mu, lo, hi = [0.0], [base.mean()], [blo], [bhi]
            for bud in BUDGETS:
                c = cost(df, n, 0.30, m, bud)
                l, h = t_ci(c)
                xs.append(bud); mu.append(c.mean()); lo.append(l); hi.append(h)
            xs = 100 * np.array(xs)
            ax.plot(xs, mu, color=st["color"], ls=st["ls"], marker=st["marker"], ms=3.2, lw=1.3,
                    markevery=2, label=m.replace(" (proposed)", "").replace(" deferral", ""))
            ax.fill_between(xs, lo, hi, color="#000000", alpha=0.06, lw=0)
        ax.set_xlabel("Human review budget (% of incidents)")
        ax.set_title(tag, fontsize=9, loc="left")
    axes[0].set_ylabel("Simulated cost per incident")
    axes[1].legend(frameon=False, fontsize=7, loc="lower left")
    fig.savefig(OUT / "figure1_budget.png", bbox_inches="tight")
    fig.savefig(OUT / "figure1_budget.pdf", bbox_inches="tight")
    plt.close(fig)


def figure2(df):
    """(a) Knowledge effect by training size; (b) knowledge effect across sensitivity settings."""
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"wspace": 0.62, "width_ratios": [1, 1]})
    for rate, mk, fill, off in ((0.30, "o", "black", 0.97), (0.60, "s", "white", 1.03)):
        mus, err = [], []
        for n in N_TRAIN:
            k = ke(df, n, rate)
            lo, hi = boot_ci(k)
            mus.append(k.mean()); err.append((k.mean() - lo, hi - k.mean()))
        a.errorbar(np.array(N_TRAIN) * off, mus, yerr=np.array(err).T, color="black", mfc=fill, marker=mk, ms=4.5,
                   lw=1.1, ls="-" if rate == 0.30 else "--", capsize=2.5, label=f"{int(rate * 100)}% test defects")
    a.axhline(0, color="#888888", lw=0.8)
    a.set_xscale("log"); a.minorticks_off(); a.set_xticks(N_TRAIN); a.set_xticklabels(N_TRAIN)
    a.set_xlabel("Labelled training incidents (log scale)")
    a.set_ylabel("Cost change from knowledge rules (%)")
    a.set_title("(a) By training-set size", fontsize=9, loc="left")
    a.legend(frameon=False, fontsize=7, loc="upper left")
    ss = pd.read_csv(OUT / "sensitivity_summary.csv")
    names = list(SENS_LABEL)
    y = np.arange(len(names))[::-1]
    for n, mk, fill, dy in ((150, "o", "black", 0.15), (2400, "s", "white", -0.15)):
        s = ss[ss["Training incidents"] == n].set_index("Setting").loc[names]
        b.errorbar(s["Knowledge effect (%)"], y + dy,
                   xerr=[s["Knowledge effect (%)"] - s["KE CI low"], s["KE CI high"] - s["Knowledge effect (%)"]],
                   fmt=mk, color="black", mfc=fill, ms=4.5, lw=1.0, capsize=2.5, label=f"{n:,} training incidents")
    b.axvline(0, color="#888888", lw=0.8)
    b.set_yticks(y); b.set_yticklabels([SENS_LABEL[k] for k in names], fontsize=7.5)
    b.set_xlabel("Cost change from knowledge rules (%)")
    b.set_title("(b) Sensitivity analyses", fontsize=9, loc="left")
    b.legend(frameon=False, fontsize=7, loc="upper center", bbox_to_anchor=(0.45, -0.2), ncol=2)
    fig.savefig(OUT / "figure2_results.png", bbox_inches="tight")
    fig.savefig(OUT / "figure2_results.pdf", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    if "--figures-only" in sys.argv:
        d = pd.read_csv(OUT / "all_runs.csv"); figure1(d); figure2(d); raise SystemExit
    if "--dev" in sys.argv:                       # development seeds only; nothing is saved
        global_R = 5
        rows = []
        for r in range(global_R):
            rows += one_replication(DEV_BASE + r)[0]
        d = pd.DataFrame(rows)
        print(d[d.budget.isin([0.0, 0.2])].groupby(["test_defect", "n_train", "method"]).cost_per_incident.mean().unstack().round(1))
        raise SystemExit
    df, cal, rules = run_main(CONFIRM_BASE)
    df.to_csv(OUT / "all_runs.csv", index=False)
    t1, t2, it = table1(df), table2(df), interaction(df)
    diag, rd = diagnostics(df, cal, rules)
    sens = sensitivity()
    figure1(df); figure2(df)
    print(t1.round(2).to_string(index=False)); print(); print(t2.round(4).to_string(index=False)); print()
    print(it.round(4).to_string(index=False)); print(); print(diag.round(3).to_string(index=False)); print()
    print(rd); print(); print(pd.read_csv(OUT / "calibration.csv")); print()
    print(pd.read_csv(OUT / "no_budget_operating_point.csv")); print(); print(sens.round(3).to_string(index=False))
