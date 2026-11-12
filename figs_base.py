"""Figures and tables for the base model (Exp 1-2)."""
import json
import pickle
import numpy as np
from figstyle import plt, COL, GRAY, MARK
from base import rvi_dp

E1 = json.load(open("results/exp1.json"))
E2 = pickle.load(open("results/exp2.pkl", "rb"))
LAMS = [2, 3, 4, 5, 6, 7]
OPT = {l: rvi_dp(l, 1.0, 5, 20, 10.0, 5.0, 1.0)["g"] for l in LAMS}
FCFS = {r["lam"]: r["g"] for r in E1["table"] if r["policy"] == "FCFS"}

# ---------- Fig: learning curves for all six scenarios ----------
fig, axes = plt.subplots(2, 3, figsize=(7.0, 4.2), sharex=True)
for ax, lam in zip(axes.flat, LAMS):
    for kind, col, lab in [("main", COL[0], "RVI-Q (Sec. 4.2.1)"), ("legacy", COL[1], "Legacy R-learning")]:
        cs = np.array([r["curve"] for r in E2 if r["kind"] == kind and r["lam"] == lam])
        x = cs[0, :, 0] / 1e6
        gap = 100 * (OPT[lam] - cs[:, :, 1]) / OPT[lam]
        ax.plot(x, np.median(gap, 0), color=col, label=lab)
        ax.fill_between(x, np.percentile(gap, 10, 0), np.percentile(gap, 90, 0), color=col, alpha=0.18, lw=0)
    fgap = 100 * (OPT[lam] - FCFS[lam]) / OPT[lam]
    if fgap > 0.05:
        ax.axhline(fgap, color=GRAY, ls="--", lw=1)
        ax.text(x[-1], fgap, "FCFS", ha="right", va="bottom", fontsize=7, color=GRAY)
    ax.set_title(r"$\lambda=%d$" % lam)
    if lam <= 4:
        ax.set_ylim(-0.05, 1.0)
    else:
        ax.set_ylim(bottom=-0.5)
for ax in axes[1]:
    ax.set_xlabel("Training steps (millions)")
for ax in axes[:, 0]:
    ax.set_ylabel("Gap to $g^*$ (%)")
axes[0, 0].legend(loc="upper right")
fig.tight_layout()
fig.savefig("figs/fig_learning_curves.pdf"); fig.savefig("figs/fig_learning_curves.png")

# ---------- Fig: visitation counts at lambda=7 ----------
fig, axes = plt.subplots(1, 2, figsize=(7.0, 2.4), sharey=True)
for ax, kind, ttl in [(axes[0], "main", "RVI-Q (Sec. 4.2.1)"), (axes[1], "legacy", "Legacy R-learning")]:
    N = np.array([r["N"] for r in E2 if r["kind"] == kind and r["lam"] == 7]).mean(0)
    s = np.arange(21)
    ax.bar(s - 0.2, N[:, 1] / 1e3, width=0.38, color=COL[0], label="admit (a=1)")
    ax.bar(s + 0.2, N[:, 0] / 1e3, width=0.38, color=COL[1], label="reject (a=0)")
    ax.axvspan(8.5, 19.5, color=GRAY, alpha=0.08, lw=0)
    ax.set_title(ttl + r", $\lambda=7$"); ax.set_xlabel("Occupancy $s$")
axes[0].set_ylabel("Updates per pair (thousands)")
axes[0].legend(loc="upper right")
fig.tight_layout()
fig.savefig("figs/fig_visits.pdf"); fig.savefig("figs/fig_visits.png")

# ---------- Fig: step-size / exploration sensitivity ----------
sens = [r for r in E2 if r["kind"] == "sens"]
fig, ax = plt.subplots(figsize=(3.4, 2.4))
for i, lam in enumerate([5, 7]):
    ks = [0.55, 0.6, 0.7, 0.8, 1.0]
    m, lo, hi = [], [], []
    for k in ks:
        v = [100 * (OPT[lam] - r["g"]) / OPT[lam] for r in sens if r["lam"] == lam and r["cfg"]["kappa"] == k
             and r["cfg"]["eps_min"] == 0.05 and r["cfg"]["start"] == "uniform" and r["cfg"]["q0"] == 0.0
             and r["cfg"]["n_episodes"] == 9000]
        m.append(np.mean(v)); lo.append(np.min(v)); hi.append(np.max(v))
    ax.errorbar(np.array(ks) + (i - 0.5) * 0.01, m, yerr=[np.array(m) - lo, np.array(hi) - m], color=COL[i],
                marker=MARK[i], capsize=2, label=r"$\lambda=%d$" % lam)
ax.set_xlabel(r"Step-size exponent $\kappa$"); ax.set_ylabel("Gap to $g^*$ (%)")
ax.legend()
fig.tight_layout(); fig.savefig("figs/fig_kappa.pdf"); fig.savefig("figs/fig_kappa.png")

# ---------- Fig: sensitivity of optimal-vs-FCFS gain ----------
S = E1["sens"]
fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2), sharey=True)
for ax, key, lab in zip(axes, ["h", "R", "P", "cK"], ["Holding cost $h$", "Revenue $R$", "Penalty $P$", "$(c,K)$"]):
    for i, rho in enumerate([1.0, 1.2, 1.4]):
        rows = [r for r in S if r["vary"] == key and r["rho"] == rho]
        xs = list(range(len(rows))) if key == "cK" else [r["value"] for r in rows]
        ax.plot(xs, [100 * r["fcfs_loss_rel_opt"] for r in rows], color=COL[i], marker=MARK[i],
                label=r"$\rho=%.1f$" % rho)
    if key == "cK":
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels(["(%d,%d)" % tuple(r["value"]) for r in rows], rotation=45, ha="right")
    elif key in ("h", "R"):
        ax.set_xscale("log", base=2)
    ax.set_xlabel(lab)
axes[0].set_ylabel("FCFS loss vs. optimum (%)")
axes[0].legend(loc="upper left")
fig.tight_layout(); fig.savefig("figs/fig_sensitivity.pdf"); fig.savefig("figs/fig_sensitivity.png")

# ---------- Tables ----------
tab = []
for lam in LAMS:
    ex = {r["policy"]: r for r in E1["table"] if r["lam"] == lam}
    rl = [r for r in E2 if r["kind"] == "main" and r["lam"] == lam]
    lg = [r for r in E2 if r["kind"] == "legacy" and r["lam"] == lam]
    def summ(rs):
        g = np.array([r["g"] for r in rs]); gap = 100 * (OPT[lam] - g) / OPT[lam]
        T = np.array([r["T"] for r in rs])
        return dict(g_mean=g.mean(), g_hw=2.262 * g.std(ddof=1) / np.sqrt(len(g)), gap_mean=gap.mean(),
                    gap_max=gap.max(), T_med=float(np.median(T)), T_min=int(T.min()), T_max=int(T.max()),
                    PB=np.mean([r["PB"] for r in rs]), W=np.mean([r["W"] for r in rs]), U=np.mean([r["U"] for r in rs]),
                    ghat=np.mean([r["ghat"] for r in rs]), ghat_sd=np.std([r["ghat"] for r in rs], ddof=1),
                    exact_hit=int(sum(r["T"] == ex["DP"]["T"] for r in rs)))
    tab.append(dict(lam=lam, exact=ex, rvi=summ(rl), legacy=summ(lg)))
json.dump(tab, open("results/table_base.json", "w"), indent=1, default=float)
for t in tab:
    print(t["lam"], {k: round(v, 3) if isinstance(v, float) else v for k, v in t["rvi"].items()})
    print("   legacy", {k: round(v, 3) if isinstance(v, float) else v for k, v in t["legacy"].items()})
