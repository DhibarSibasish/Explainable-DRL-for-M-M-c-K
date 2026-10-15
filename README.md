# Explainable Deep Reinforcement Learning for Admission Control in M/M/c/K Queueing Systems

Code and results for the paper

> S. Dhibar, *Explainable Deep Reinforcement Learning for Admission Control in M/M/c/K Queueing Systems*, submitted to the Asia-Pacific Journal of Operational Research.

The repository reproduces every table and figure in Section 6 (Numerical Experiments and Results) and Figure 1. The raw outputs are included, so all tables and figures can be regenerated in a few minutes without re-running the experiments.

## What the code does

| Part | Model | Method |
|---|---|---|
| Exact solution | Base M/M/c/K model (Sections 3.1–3.4) | Closed-form evaluation of any occupancy policy (Proposition 3, Corollary 1) and relative value iteration on the uniformized chain (Eq. 20); state-by-state check of threshold structure and convexity (Proposition 2) |
| Tabular learning | Base model | RVI Q-learning (Section 4.2.1, Eq. 9) and a legacy R-learning ablation with constant step sizes |
| Deep learning | Extended model with hidden arrival regimes and two customer classes (Section 3.5) | DQN with a semi-Markov average-reward target (Section 4.2.2, Eqs. 10–13), implemented in NumPy |
| Baselines | Extended model | FCFS, best static thresholds, robust static rule, certainty-equivalent DP, plug-in regime controller, regime-aware oracle (exact) |
| Explanation layer | Trained DQN | Tree and logistic surrogates, held-out and near-boundary fidelity, permutation importance, counterfactuals, threshold profiles, patch test, seed stability, explanation-guided retraining (Section 4.3) |

## Requirements

Python 3.10 or later. Tested with Python 3.13.16 and:

```
numpy==2.5.3
scipy==1.18.1
scikit-learn==1.9.1
matplotlib==3.11.2
```

```bash
pip install -r requirements.txt
```

No deep-learning framework is needed; the DQN is written directly in NumPy. A LaTeX installation is only needed to compile `section6.tex`.

## Repository layout

```
base.py                 Base model: exact policy evaluation, relative value iteration, CTMC simulator
rl_tabular.py           Tabular RVI Q-learning (Sec. 4.2.1) and legacy R-learning ablation
extended.py             Extended model: regime-aware oracle DP, exact (s,z) evaluation, simulators
dqn.py                  NumPy DQN (Sec. 4.2.2)
policies.py             Baseline controllers for the extended model
figstyle.py             Shared plotting style

exp1_exact.py           Exp. 1: exact solution, structure checks, Prop. 3 check, sensitivity grid
exp2_tabular.py         Exp. 2: tabular learners, 10 seeds x 6 scenarios, sensitivity sweep
exp3_train_dqn.py       Exp. 3: DQN training, 5 seeds
exp4_eval.py            Exp. 4: all controllers in the training environment and four shifts
exp5_explain.py         Exp. 5: explanation layer on the trained DQN
exp6_patch.py           Exp. 6: patch test and seed stability
exp7_curve_retrain.py   Exp. 7: DQN learning curves and explanation-guided retraining

figs_base.py            Figures 2-4 and the base-model summary table
figs_ext.py             Figures 5-6
fig_model.py            Figure 1 (model diagram)
collect.py              Collects every reported number into results/numbers.json
render.py               Fills section6_template.tex with the numbers -> section6.tex
section6_template.tex   Text of Section 6 with <<placeholders>>

results/                Raw outputs of all experiments (JSON and pickle)
figs/                   Generated figures (PDF and PNG)
```

## Reproducing the results

### Quick: regenerate tables and figures from the included results

```bash
python figs_base.py
python figs_ext.py
python fig_model.py
python collect.py
python render.py
```

This takes about a minute. It writes the figures to `figs/`, the numbers to `results/numbers.json`, and the filled-in Section 6 to `section6.tex`.

### Full: re-run all experiments

Run the scripts in this order; each step uses the outputs of the previous ones. Times are wall-clock on a 2-core machine.

| Step | Command | Time | Writes |
|---|---|---|---|
| 1 | `python exp1_exact.py` | < 1 min | `results/exp1.json` |
| 2 | `python exp2_tabular.py` | ~ 6 min | `results/exp2.pkl` |
| 3 | `python figs_base.py` | < 1 min | `results/table_base.json`, Figures 2–4 |
| 4 | `python exp3_train_dqn.py` | ~ 35 min | `results/exp3_dqn.pkl` |
| 5 | `python exp4_eval.py` | ~ 15 min | `results/exp4.json`, `results/controllers.pkl` |
| 6 | `python exp5_explain.py plugin` | ~ 1 min | `results/exp5_plugin.json`, `results/surr_seed*.pkl` |
| 7 | `python exp6_patch.py plugin` | ~ 15 min | `results/exp6_plugin.json` |
| 8 | `python exp7_curve_retrain.py plugin` | ~ 15 min | `results/exp7_plugin.json` |
| 9 | `python figs_ext.py plugin` | < 1 min | Figures 5–6 |
| 10 | `python fig_model.py` | < 1 min | Figure 1 |
| 11 | `python collect.py plugin && python render.py` | < 1 min | `results/numbers.json`, `section6.tex` |

The argument `plugin` selects the reference controller used by the explanation layer (the plug-in regime controller, as in the paper); `ce_dp` (certainty-equivalent DP) is the alternative.

All random seeds are fixed. Exact results (Tables 2–4 and 6) are reproduced to the printed precision. Simulated and learned results (Tables 5 and 7–9, Figures 2–6) are reproduced exactly on the tested versions; with other NumPy versions or BLAS libraries the DQN may differ slightly, but within the reported confidence intervals.

## Where each table and figure comes from

| Paper | Content | Produced by | Data |
|---|---|---|---|
| Figure 1 | Model diagram | `fig_model.py` | – |
| Table 1 | Parameters and hyperparameters | set in `exp1`–`exp7`, `dqn.py` (`HP`), `exp3_train_dqn.py` (`HP_MAIN`) | – |
| Table 2 | Exact performance, FCFS vs. optimum | `exp1_exact.py` | `exp1.json` (`table`) |
| Table 3 | Proposition 3 vs. simulation | `exp1_exact.py` | `exp1.json` (`prop3`) |
| Table 4 | Sensitivity to R, P, h, (c, K) | `exp1_exact.py` | `exp1.json` (`sens`) |
| Table 5 | Tabular learners, 10 seeds | `exp2_tabular.py`, `figs_base.py` | `exp2.pkl`, `table_base.json` |
| Figure 2 | Learning curves, all scenarios | `figs_base.py` | `exp2.pkl` |
| Figure 3 | Visitation counts at λ = 7 | `figs_base.py` | `exp2.pkl` |
| Figure 4 | Step-size sensitivity | `figs_base.py` | `exp2.pkl` |
| Table 6 | Value of regime information | `collect.py` (computed exactly) | `numbers.json` (`voi_scan`) |
| Table 7 | Controllers under shifts | `exp4_eval.py` | `exp4.json` |
| Figure 5 | DQN learning curves | `exp7_curve_retrain.py`, `figs_ext.py` | `exp7_plugin.json` |
| Table 8 | Surrogate fidelity | `exp5_explain.py` | `exp5_plugin.json` |
| Figure 6 | Threshold profiles | `exp5_explain.py`, `figs_ext.py` | `exp5_plugin.json`, `exp6_plugin.json` |
| Table 9 | Patch test and guided retraining | `exp6_patch.py`, `exp7_curve_retrain.py` | `exp6_plugin.json`, `exp7_plugin.json` |

Every number quoted in the text of Section 6 is computed in `collect.py` and inserted by `render.py`; none is typed by hand.

## Evaluation protocol in brief

- **Exact where possible.** Any policy that depends only on the occupancy (FCFS, thresholds, the DP policy, and the greedy policies of the tabular learner) is evaluated exactly via the birth–death product form. Any policy that depends on (s, z, j) in the extended model is evaluated exactly by solving the balance equations of the CTMC on (s, z).
- **Simulation otherwise.** Policies that depend on the regime estimate λ̂ (DQN, certainty-equivalent DP, plug-in and patched controllers), and the server-failure scenario, are simulated with common random numbers across controllers and reported with 95% confidence intervals.
- **Gaps** are relative to the optimum of the same scenario, (g* − g)/g*.

## Citation

If you use this code, please cite the paper:

```bibtex
@article{dhibar_xdrl_admission,
  author  = {Dhibar, Sibasish},
  title   = {Explainable Deep Reinforcement Learning for Admission Control in {M/M/c/K} Queueing Systems},
  journal = {Asia-Pacific Journal of Operational Research},
  note    = {Under review},
  year    = {2026}
}
```

Archived release: [Zenodo DOI to be added]

## License

[Choose a license, e.g. MIT, and add a `LICENSE` file.]

## Contact

Sibasish Dhibar, TUM School of Management, Technical University of Munich — sibasish.dhibar@tum.de
