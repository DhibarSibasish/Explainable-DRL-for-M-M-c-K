"""Exp 2: tabular RVI-Q (Section 4.2.1) on the base model; legacy R-learning ablation;
one-factor sensitivity to step size, exploration, initial state, initialization, budget."""
import json
import pickle
from multiprocessing import Pool
import numpy as np
from base import eval_policy, rvi_dp
from rl_tabular import rvi_q, r_learning_legacy

C, K, MU, R, P, H = 5, 20, 1.0, 10.0, 5.0, 1.0
LAMS = [2, 3, 4, 5, 6, 7]
SEEDS = list(range(10))
MAIN = dict(kappa=0.6, eps_min=0.05, start="uniform", q0=0.0, n_episodes=9000)


def job(args):
    kind, lam, seed, cfg = args
    if kind == "legacy":
        o = r_learning_legacy(lam, MU, C, K, R, P, H, seed=seed)
    else:
        o = rvi_q(lam, MU, C, K, R, P, H, seed=seed, **cfg)
    ev = eval_policy(o["policy"], lam, MU, C, K, R, P, H)
    pol = o["policy"]
    first = int(np.argmin(pol)) if pol.min() == 0 else K
    mono = bool(np.all(pol[:first] == 1) and np.all(pol[first:] == 0))
    return dict(kind=kind, lam=lam, seed=seed, cfg=cfg, policy=pol.tolist(), T=ev["T"], g=ev["g"],
                PB=ev["PB"], W=ev["W"], U=ev["U"], ghat=float(o["ghat"]), monotone=mono,
                N=o["N"].tolist(), curve=o["curve"].tolist())


if __name__ == "__main__":
    jobs = []
    for lam in LAMS:
        for sd in SEEDS:
            jobs.append(("main", lam, sd, MAIN))
            jobs.append(("legacy", lam, sd, {}))
    variants = [("kappa", v) for v in [0.55, 0.7, 0.8, 1.0]] + \
               [("eps_min", v) for v in [0.01, 0.2]] + \
               [("start", "empty"), ("q0", 100.0), ("n_episodes", 45000)]
    for lam in [5, 7]:
        for key, v in variants:
            cfg = dict(MAIN); cfg[key] = v
            if key != "n_episodes":
                pass
            for sd in range(5):
                jobs.append(("sens", lam, 100 + sd, cfg))
        for sd in range(5):
            jobs.append(("sens", lam, 100 + sd, dict(MAIN)))
    for j in jobs:
        if j[3].get("n_episodes", 9000) > 9000:
            j[3]["eval_every"] = 250
    with Pool(2) as p:
        res = p.map(job, jobs, chunksize=1)
    pickle.dump(res, open("results/exp2.pkl", "wb"))
    opt = {lam: rvi_dp(lam, MU, C, K, R, P, H)["g"] for lam in LAMS}
    for kind in ["main", "legacy"]:
        for lam in LAMS:
            rs = [r for r in res if r["kind"] == kind and r["lam"] == lam]
            gaps = np.array([100 * (opt[lam] - r["g"]) / opt[lam] for r in rs])
            Ts = [r["T"] for r in rs]
            print(kind, lam, "gap mean %.3f max %.3f" % (gaps.mean(), gaps.max()), "T", Ts,
                  "mono", sum(r["monotone"] for r in rs), "ghat %.2f vs %.2f" % (np.mean([r["ghat"] for r in rs]), opt[lam]))
