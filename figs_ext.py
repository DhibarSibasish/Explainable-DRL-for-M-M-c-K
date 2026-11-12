"""Figures for the extended model: threshold profiles (explanation) and DQN learning curves."""
import json
import pickle
import sys
import numpy as np
from figstyle import plt, COL, GRAY, MARK, INK

REF = sys.argv[1] if len(sys.argv) > 1 else "plugin"
E4 = json.load(open("results/exp4.json"))
E5 = json.load(open(f"results/exp5_{REF}.json"))
E6 = json.load(open(f"results/exp6_{REF}.json"))
E7 = json.load(open(f"results/exp7_{REF}.json"))
CTRL = pickle.load(open("results/controllers.pkl", "rb"))
MIDS = np.array(E5["mids"])
K = 20

# ---- threshold profile, standard class ----
fig, ax = plt.subplots(figsize=(3.6, 3.0))
Ta = np.array([r["T_agent"] for r in E5["seeds"]])
Ts = np.array([r["T_surr"] for r in E5["seeds"]])
Tr = np.array(E5["seeds"][0]["T_ref"])
ce = np.array([CTRL.ce_thr(m) for m in MIDS])
from collections import Counter
cnt = Counter(b for r in E6["patch"].values() for j, b in r["flagged"] if j == 0)
fl = {b for b, n in cnt.items() if n >= 3}
for b in fl:
    ax.axvspan(MIDS[b] - 0.5, MIDS[b] + 0.5, color=GRAY, alpha=0.07, lw=0)
for i in range(len(Ta)):
    ax.plot(MIDS, Ta[i, 0], color=COL[0], lw=0.7, alpha=0.45)
ax.plot(MIDS, Ta[:, 0].mean(0), color=COL[0], marker=MARK[0], label="DQN (mean of 5 seeds)")
ax.plot(MIDS, Ts[:, 0].mean(0), color=COL[0], ls="--", lw=1.2, label="Tree surrogate (mean)")
ax.step(MIDS, Tr[0], where="mid", color=COL[1], label="Plug-in regime reference")
ax.plot(MIDS, ce[:, 0], color=COL[2], marker=MARK[2], lw=1.2, label="Certainty-equivalent DP")
ax.set_xlabel(r"Regime estimate $\hat\lambda$ (bin midpoint)")
ax.set_ylabel(r"Standard-class threshold $T_\pi(L,b)$")
ax.set_ylim(4, 21)
ax.legend(loc="upper right", fontsize=7, bbox_to_anchor=(1.0, 1.0))
ax.set_ylim(4, 25)
fig.tight_layout(); fig.savefig("figs/fig_profile.pdf"); fig.savefig("figs/fig_profile.png")

# ---- DQN learning curves (simulated greedy reward rate) ----
fig, ax = plt.subplots(figsize=(3.6, 2.6))
cv = E7["curve"]
seeds = sorted(set(c["seed"] for c in cv))
steps = sorted(set(c["step"] for c in cv))
G = np.array([[next(c["g"] for c in cv if c["seed"] == s and c["step"] == st) for st in steps] for s in seeds])
x = np.array(steps) / 1e3
for i in range(len(seeds)):
    ax.plot(x, G[i], color=COL[0], lw=0.7, alpha=0.4)
ax.plot(x, G.mean(0), color=COL[0], marker=MARK[0], label="DQN greedy policy (mean)")
ex = E4["exact"]["S0_train"]
for val, lab, ls, va in [(ex["oracle_resolved"], "Regime-aware oracle", "-", "bottom"),
                         (ex["cls"], "Best static class thresholds", "--", "top")]:
    ax.axhline(val, color=GRAY, ls=ls, lw=1, label=lab)
ax.set_ylim(38, 48)
ax.legend(loc="lower right", fontsize=7)
ax.set_xlabel("Training decisions (thousands)"); ax.set_ylabel("Reward rate $g$")
fig.tight_layout(); fig.savefig("figs/fig_dqn_curve.pdf"); fig.savefig("figs/fig_dqn_curve.png")
print("ok")
