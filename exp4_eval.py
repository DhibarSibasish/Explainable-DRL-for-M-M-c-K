"""Exp 4: evaluate all controllers (no retraining) in the training environment and four shifts.
Replicated simulation with common seeds across policies; exact values where available."""
import json
import pickle
from multiprocessing import Pool
import numpy as np
from extended import Sim, oracle_dp, eval_szj
from policies import Controllers, load_agent, szj_from_thresholds

D = pickle.load(open("results/exp3_dqn.pkl", "rb"))
P0, HP = D["p"], D["hp"]


def scen(name):
    p = dict(P0); fail = None
    if name == "S1_fast_switch":
        p["gam"] = (0.2, 0.2)
    elif name == "S2_demand_shock":
        p["lams"] = (4.0, 12.0)
    elif name == "S3_service_degr":
        p["mu"] = 0.8
    elif name == "S4_server_fail":
        fail = (0.02, 0.1, 3)
    return p, fail


SCEN = ["S0_train", "S1_fast_switch", "S2_demand_shock", "S3_service_degr", "S4_server_fail"]
REPS, T_END = 10, 10000.0

try:
    CTRL = pickle.load(open("results/controllers.pkl", "rb"))
except FileNotFoundError:
    CTRL = Controllers(P0)
    models = []
    for lh in (7.0, 9.0, 12.0):
        for mu in (0.8, 1.0):
            m = dict(P0); m["lams"] = (4.0, lh); m["mu"] = mu; models.append(m)
    CTRL.set_robust(models)
    pickle.dump(CTRL, open("results/controllers.pkl", "wb"))

AGENTS = {r["seed"]: load_agent(P0, HP, r["final"], seed=r["seed"]) for r in D["runs"]}
BASEL = ["fcfs", "blind", "cls", "robust_pol", "ce_dp", "plugin", "oracle"]


def job(args):
    sc, pol, rep = args
    p, fail = scen(sc)
    if pol.startswith("dqn"):
        f = AGENTS[int(pol[3:])].act_greedy
    else:
        f = getattr(CTRL, pol)
    r = Sim(p, seed=10_000 + rep, fail=fail).run(f, T_END)
    return dict(scen=sc, policy=pol, rep=rep, g=r["g"], PB=r["PB"], L=r["L"])


if __name__ == "__main__":
    jobs = []
    for sc in SCEN:
        for rep in range(REPS):
            for pol in BASEL:
                jobs.append((sc, pol, rep))
            for sd in AGENTS:
                if rep < 5:
                    jobs.append((sc, f"dqn{sd}", rep))
    with Pool(2) as pl:
        res = pl.map(job, jobs, chunksize=4)
    exact = {}
    for sc in SCEN[:4]:
        p, _ = scen(sc)
        o = oracle_dp(p)
        K = p["K"]
        exact[sc] = dict(oracle_resolved=o["g"],
                         oracle_train_policy=eval_szj(CTRL.orc["policy"], p),
                         fcfs=eval_szj(szj_from_thresholds(K, K, K), p),
                         blind=eval_szj(szj_from_thresholds(CTRL.static_blind, CTRL.static_blind, K), p),
                         cls=eval_szj(szj_from_thresholds(*CTRL.static_cls, K), p),
                         robust=eval_szj(szj_from_thresholds(CTRL.robust[1], CTRL.robust[2], K), p))
    json.dump(dict(sim=res, exact=exact, static_blind=CTRL.static_blind, static_cls=CTRL.static_cls,
                   robust=CTRL.robust), open("results/exp4.json", "w"), indent=1, default=float)
    for sc in SCEN:
        print(sc, exact.get(sc, {}))
        rows = {}
        for r in res:
            if r["scen"] != sc:
                continue
            key = "dqn" if r["policy"].startswith("dqn") else r["policy"]
            rows.setdefault(key, []).append(r["g"])
        for k, v in rows.items():
            v = np.array(v)
            print("   %-11s g=%.2f +- %.2f (n=%d)" % (k, v.mean(), 1.96 * v.std(ddof=1) / np.sqrt(len(v)), len(v)))
