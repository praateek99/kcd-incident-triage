# KCD: Knowledge-guided Cost-aware Deferral for Cloud Incident Triage

Code and results for the short communication
**"Cost-Aware Deferral in Knowledge-Based AI Agents: Data-Driven Decision-Making and Business Value in Cloud Incident Triage"** (Praateek Goud Gunday).

KCD combines three parts:

1. **Data-driven agent**: TF-IDF + one-hot multinomial logistic regression estimating P(impact | first status update).
2. **Knowledge base**: five study-defined triage rules that set severity floors and ceilings (`KR1` to `KR5` in `kcd.py`).
3. **Reviewer-aware deferral**: choose the lowest-expected-cost impact level within the rule bounds, then send to human review the incidents with the largest expected benefit of review, `r*(x) - D(x)`, where `D(x)` is the review cost plus the expected reviewer error cost.

## Reproduce everything

```bash
pip install -r requirements.txt
python run_experiments.py              # about 2 minutes; writes results/
python run_experiments.py --figures-only
python run_experiments.py --dev        # development seeds only; writes nothing
```

All data are synthetic and seeded. Results were produced with Python 3.13, NumPy 2.5.3, pandas 3.0.5, SciPy 1.18.1, scikit-learn 1.9.1 and Matplotlib 3.11.2.

## Files

| File | Purpose |
|---|---|
| `kcd.py` | Generator, defect injection, reviewer model, cost model, agent, knowledge rules, deferral, evaluation |
| `run_experiments.py` | Main study, tables, tests, diagnostics, sensitivity analyses, figures |
| `results/table1_main_comparison.csv` | Table 1 (20% quota, 150 vs 2,400 training incidents, 30% test defects) |
| `results/table2_ablation.csv` | Table 2 (2x2 ablation, bootstrap CIs, Holm-adjusted Wilcoxon tests) |
| `results/interaction_test.csv` | Knowledge effect at 150 minus 2,400 incidents (paired) |
| `results/diagnostics_pathways.csv` | Fixed-mask decomposition (decision vs ranking) and heuristic comparisons |
| `results/calibration.csv` | Brier score, log loss and top-label ECE by condition |
| `results/rule_diagnostics.csv` | Rule firing rate and rule-bound agreement over 30 company draws |
| `results/no_budget_operating_point.csv` | Deferral without a quota (defer when expected benefit > 0) |
| `results/sensitivity_summary.csv`, `results/sensitivity_runs.csv` | Eight sensitivity settings at 150 and 2,400 incidents |
| `results/all_runs.csv` | Every replication, condition, policy and budget |
| `results/figure1_framework.*`, `results/figure2_results.*` | Figures (1000 dpi PNG and vector PDF) |

## Experimental design

- Per replication: one draw of company offsets, one 2,400-incident training pool (30% defects), one shared 1,600-incident test set.
- Learning curves use nested subsets of the pool (150, 400, 1,000, 2,400).
- Test defects use common random numbers: the 60% condition corrupts a superset of the 30% condition's incidents.
- One simulated reviewer outcome per test incident, reused by every policy. Reviewer accuracy 0.92 with valid adjacent errors.
- Company identity is not a model feature (pooled multi-company learning).
- 30 confirmation replications on seeds not used in development; 15 fresh replications per sensitivity setting.

## Main findings (simulated)

- Reviewer-aware deferral at a 20% quota lowered cost per incident by 9.6% to 11.3% in all eight conditions.
- With 150 training incidents, adding the rules lowered cost by a further 1.8% (30% defects) and 1.4% (60% defects), Holm-adjusted p < 0.001.
- With 400 or more incidents the rules gave no significant saving (and +0.3% cost at 1,000 incidents, 30% defects).
- The scarce-data saving held in all eight sensitivity settings, including a calibrated classifier.

## Version and development history

This repository contains the reviewed v2.1 source snapshot. Earlier development versions are retained in the author's local archive; their original commits and tags were not transferred to this GitHub repository. The snapshot name does not imply that a GitHub release or tag has been published.

- An independent pre-submission review found that the original reviewer simulation was more accurate at the scale ends than stated, KR5 could fire with a missing component, defect rate and training size changed the underlying cases, and the routing score ignored reviewer error. These issues were corrected before the v2 confirmation experiments, and all results were regenerated on fresh seeds.
- The v2.1 snapshot adds Figure 1 label fixes to v2.0; scientific code and result CSVs are unchanged.
- Earlier pilots used rules as deferral triggers without improving on cost-aware deferral, so rules became decision bounds. Cost coefficients were changed from 500/150 to 400/300 after the first pilot.
- Limitations: synthetic data generated under the same domain assumptions as the rules, illustrative costs and reviewer model, and a linear classifier.

## License

MIT
