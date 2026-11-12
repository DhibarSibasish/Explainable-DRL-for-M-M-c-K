"""Exp 7: (a) DQN learning curves from weight snapshots (simulated reward rate of greedy policy);
(b) explanation-guided retraining vs equal-budget control, starting from each trained agent."""
import json
import pickle
import sys
from multiprocessing import Pool
import numpy as np
from extended import Sim
from policies import load_agent
from dqn import DQNAgent

REF = sys.argv[1] if len(sys.argv) > 1 else "plugin"
D = pickle.load(open("results/exp3_dqn.pkl", "rb"))
P0, HP = D["p"], D["hp"]
K = P0["K"]
E6 = json.load(open(f"results/exp6_{REF}.json"))
E5 = json.load(open(f"results/exp5_{REF}.json"))
MIDS = np.array(E5["mids"])
CTRL = pickle.load(open("results/controllers.pkl", "rb"))
FT = dict(HP, steps=40_000, eps0=0.05, eps_end=0.05, lr=HP["lr"] * HP["lr_end_frac"], lr_end_frac=1.0,
          reset_every=200, warmup=2_000)
REPS, T_END = 12, 10000.0


def curve_job(args):
    seed, k = args
    run = [r for r in D["runs"] if r["seed"] == seed][0]
    ag = load_agent(P0, HP, run["snaps"][k], seed=seed)
    g = [Sim(P0, seed=40_000 + r).run(ag.act_greedy, 4000.0)["g"] for r in range(3)]
    return dict(seed=seed, step=run["log"][k]["step"], g=float(np.mean(g)))


def retrain_job(args):
    seed, arm = args
    run = [r for r in D["runs"] if r["seed"] == seed][0]
    ag = DQNAgent(P0, seed=seed + 500, hp=FT)
    for k_, v in run["final"].items():
        setattr(ag.net, k_, v.copy()); setattr(ag.tgt, k_, v.copy())
    cells = [tuple(c) for c in E6["patch"][str(seed)]["flagged"]]
    r5 = [r for r in E5["seeds"] if r["seed"] == seed][0]
    Ta, Tr = np.array(r5["T_agent"]), np.array(r5["T_ref"])

    def sampler(rng):
        j, b = cells[rng.integers(len(cells))]
        lo = max(0, min(Ta[j, b], Tr[j, b]) - 2); hi = min(K - 1, max(Ta[j, b], Tr[j, b]) + 1)
        z0 = 0 if MIDS[b] < CTRL.zcut else 1
        return int(rng.integers(lo, hi + 1)), z0
    ag.train(log_every=10_000, reset_sampler=sampler if (arm == "guided" and cells) else None,
             g_init=run["ghat"], env_seed=seed + 9999)
    w = {k_: getattr(ag.net, k_).copy() for k_ in ag.net.params}
    return dict(seed=seed, arm=arm, weights=w, n_cells=len(cells))


def eval_job(args):
    seed, arm, w, rep = args
    ag = load_agent(P0, HP, w, seed=seed)
    return dict(seed=seed, arm=arm, rep=rep, g=Sim(P0, seed=30_000 + rep).run(ag.act_greedy, T_END)["g"])


def prof(ag):
    T = np.zeros((2, len(MIDS)), int)
    for j in range(2):
        for b, m in enumerate(MIDS):
            a = ag.greedy_batch(np.arange(K + 1), np.full(K + 1, j), np.full(K + 1, m)); a[K] = 0
            T[j, b] = int(np.argmin(a))
    return T


if __name__ == "__main__":
    seeds = [r["seed"] for r in D["runs"]]
    with Pool(2) as pl:
        curve = pl.map(curve_job, [(s, k) for s in seeds for k in range(len(D["runs"][0]["snaps"]))])
        rt = pl.map(retrain_job, [(s, a) for s in seeds for a in ("control", "guided")])
        ev = pl.map(eval_job, [(r["seed"], r["arm"], r["weights"], rep) for r in rt for rep in range(REPS)], chunksize=2)
    raw6 = E6["raw"]
    out = {}
    for s in seeds:
        g0 = np.array(sorted([(x["rep"], x["g"]) for x in raw6 if x["seed"] == s and x["label"] == "agent"]))[:, 1]
        row = dict(g_orig=float(g0.mean()))
        for arm in ("control", "guided"):
            g = np.array(sorted([(x["rep"], x["g"]) for x in ev if x["seed"] == s and x["arm"] == arm]))[:, 1]
            d = g - g0
            row[arm] = dict(g=float(g.mean()), gain=float(d.mean()), hw=float(2.201 * d.std(ddof=1) / np.sqrt(REPS)))
            w = [r["weights"] for r in rt if r["seed"] == s and r["arm"] == arm][0]
            row[arm]["T_profile"] = prof(load_agent(P0, HP, w, seed=s)).tolist()
        gc = np.array(sorted([(x["rep"], x["g"]) for x in ev if x["seed"] == s and x["arm"] == "control"]))[:, 1]
        gg = np.array(sorted([(x["rep"], x["g"]) for x in ev if x["seed"] == s and x["arm"] == "guided"]))[:, 1]
        dd = gg - gc
        row["guided_minus_control"] = float(dd.mean()); row["gmc_hw"] = float(2.201 * dd.std(ddof=1) / np.sqrt(REPS))
        out[s] = row
        print(s, {k: v for k, v in row.items() if k not in ("control", "guided")},
              {a: (round(row[a]["g"], 2), round(row[a]["gain"], 2), round(row[a]["hw"], 2)) for a in ("control", "guided")})
    json.dump(dict(curve=curve, retrain={str(k): v for k, v in out.items()}, ft=FT),
              open(f"results/exp7_{REF}.json", "w"), indent=1, default=float)
