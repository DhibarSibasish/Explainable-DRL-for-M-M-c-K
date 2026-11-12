"""Exp 6: patch test (Section 4.3.4) with random-cell control, and seed-to-seed stability (19)."""
import json
import pickle
import sys
from itertools import combinations
from multiprocessing import Pool
import numpy as np
from extended import Sim
from policies import load_agent

REF = sys.argv[1] if len(sys.argv) > 1 else "ce_dp"
E5 = json.load(open(f"results/exp5_{REF}.json"))
D = pickle.load(open("results/exp3_dqn.pkl", "rb"))
P0, HP = D["p"], D["hp"]
K = P0["K"]
CTRL = pickle.load(open("results/controllers.pkl", "rb"))
AGENTS = {r["seed"]: load_agent(P0, HP, r["final"], seed=r["seed"]) for r in D["runs"]}
BINS = np.arange(2.0, 14.0, 1.0)
NB = len(BINS) - 1
REPS, T_END, N_RAND, FLAG_DIFF, MIN_W = 12, 10000.0, 5, 2, 0.01


def bin_of(lh):
    return int(np.clip(np.searchsorted(BINS, lh, side="right") - 1, 0, NB - 1))


def make_patched(seed, cells):
    ag = AGENTS[seed]
    ref = getattr(CTRL, REF)
    cells = set(map(tuple, cells))

    def f(s, j, lh, z):
        if (j, bin_of(lh)) in cells:
            return ref(s, j, lh, z)
        return ag.act_greedy(s, j, lh)
    return f


def job(args):
    seed, label, cells, rep = args
    if label == "agent":
        f = AGENTS[seed].act_greedy
    elif label == "ref":
        f = getattr(CTRL, REF)
    else:
        f = make_patched(seed, cells)
    return dict(seed=seed, label=label, rep=rep, g=Sim(P0, seed=30_000 + rep).run(f, T_END)["g"])


if __name__ == "__main__":
    jobs, plan = [], {}
    for r in E5["seeds"]:
        sd = r["seed"]
        Ta, Tr, w = np.array(r["T_agent"]), np.array(r["T_ref"]), np.array(r["visit_w"])
        flagged = [(j, b) for j in range(2) for b in range(NB) if abs(Ta[j, b] - Tr[j, b]) >= FLAG_DIFF and w[j, b] >= MIN_W]
        visited = [(j, b) for j in range(2) for b in range(NB) if w[j, b] >= MIN_W and (j, b) not in flagged]
        rng = np.random.default_rng(sd)
        rands = []
        for k in range(N_RAND):
            n = min(len(flagged), len(visited))
            rands.append([visited[i] for i in rng.choice(len(visited), size=n, replace=False)] if n else [])
        plan[sd] = dict(flagged=flagged, flagged_w=float(sum(w[j, b] for j, b in flagged)), random=rands)
        for rep in range(REPS):
            jobs.append((sd, "agent", [], rep))
            jobs.append((sd, "patch", flagged, rep))
            for k, rc in enumerate(rands):
                jobs.append((sd, f"rand{k}", rc, rep))
    for rep in range(REPS):
        jobs.append((-1, "ref", [], rep))
    with Pool(2) as pl:
        res = pl.map(job, jobs, chunksize=2)
    g_ref = np.array([x["g"] for x in res if x["label"] == "ref"])
    summ = {}
    for sd in plan:
        ga = np.array([x["g"] for x in res if x["seed"] == sd and x["label"] == "agent"])
        gp = np.array([x["g"] for x in res if x["seed"] == sd and x["label"] == "patch"])
        gr = np.array([[x["g"] for x in res if x["seed"] == sd and x["label"] == f"rand{k}"] for k in range(N_RAND)])
        gap = g_ref.mean() - ga.mean()
        d_p = gp - ga; d_r = (gr - ga[None, :]).mean(0)
        summ[sd] = dict(plan[sd], g_agent=ga.mean(), g_patch=gp.mean(), g_rand=gr.mean(), g_ref=g_ref.mean(),
                        gap=gap, patch_gain=d_p.mean(), patch_gain_hw=2.201 * d_p.std(ddof=1) / np.sqrt(REPS),
                        rand_gain=d_r.mean(), rand_gain_hw=2.201 * d_r.std(ddof=1) / np.sqrt(REPS),
                        recovered=(d_p.mean() / gap) if gap > 0 else None)
        print(sd, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summ[sd].items() if k != "random"})
    # ---- stability (19) ----
    X0 = np.vstack([np.array(r["test_X"]) for r in E5["seeds"]])
    X0 = X0[np.random.default_rng(0).choice(len(X0), size=min(5000, len(X0)), replace=False)]
    sur = {r["seed"]: pickle.load(open(f"results/surr_seed{r['seed']}.pkl", "rb")) for r in E5["seeds"]}
    sd_list = sorted(sur)
    agr_s = [np.mean(sur[a].predict(X0) == sur[b].predict(X0)) for a, b in combinations(sd_list, 2)]
    agr_a = [np.mean(AGENTS[a].greedy_batch(X0[:, 0], X0[:, 1], X0[:, 2]) == AGENTS[b].greedy_batch(X0[:, 0], X0[:, 1], X0[:, 2]))
             for a, b in combinations(sd_list, 2)]
    Ta = np.array([r["T_agent"] for r in E5["seeds"]]); Ts = np.array([r["T_surr"] for r in E5["seeds"]])
    w = np.mean([r["visit_w"] for r in E5["seeds"]], 0)
    stab = dict(stab_surr=float(np.mean(agr_s)), stab_surr_min=float(np.min(agr_s)),
                stab_agent=float(np.mean(agr_a)), stab_agent_min=float(np.min(agr_a)),
                thr_sd_agent=float(np.average(Ta.std(0, ddof=1), weights=w)),
                thr_sd_surr=float(np.average(Ts.std(0, ddof=1), weights=w)),
                thr_sd_agent_cells=Ta.std(0, ddof=1).tolist(), thr_mean_agent=Ta.mean(0).tolist())
    print(stab)
    json.dump(dict(ref=REF, patch={str(k): v for k, v in summ.items()}, stability=stab, raw=res),
              open(f"results/exp6_{REF}.json", "w"), indent=1, default=float)
