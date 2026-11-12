"""Collect every number reported in Section 6 into results/numbers.json (single source of truth)."""
import json
import pickle
import sys
import numpy as np
from base import eval_policy, threshold_policy, rvi_dp
from extended import BASE, oracle_dp, eval_szj
from policies import szj_from_thresholds, best_static

REF = sys.argv[1] if len(sys.argv) > 1 else "plugin"
E1 = json.load(open("results/exp1.json"))
E2 = pickle.load(open("results/exp2.pkl", "rb"))
TB = json.load(open("results/table_base.json"))
E4 = json.load(open("results/exp4.json"))
E5 = json.load(open(f"results/exp5_{REF}.json"))
E6 = json.load(open(f"results/exp6_{REF}.json"))
E7 = json.load(open(f"results/exp7_{REF}.json"))
D3 = pickle.load(open("results/exp3_dqn.pkl", "rb"))
N = {}
LAMS = [2, 3, 4, 5, 6, 7]
OPT = {l: rvi_dp(l, 1.0, 5, 20, 10.0, 5.0, 1.0)["g"] for l in LAMS}

# --- base: flatness of g(T) near T* ---
flat = {}
for lam in LAMS:
    gT = np.array([eval_policy(threshold_policy(T, 20), lam, 1.0, 5, 20, 10.0, 5.0, 1.0)["g"] for T in range(21)])
    flat[lam] = [int(T) for T in range(21) if gT[T] >= 0.99 * OPT[lam]]
N["flat_1pct"] = flat
# --- visits at lambda=7 in states 9..19 ---
for kind in ["main", "legacy"]:
    Ns = np.array([r["N"] for r in E2 if r["kind"] == kind and r["lam"] == 7])  # seeds x 21 x 2
    sub = Ns[:, 9:20, :]
    N[f"visits_{kind}_9_19_min"] = int(sub.min())
    N[f"visits_{kind}_9_19_mean"] = float(sub.mean())
    N[f"visits_{kind}_total"] = int(Ns[0].sum())
# --- monotone counts ---
N["monotone"] = {kind: {lam: int(sum(r["monotone"] for r in E2 if r["kind"] == kind and r["lam"] == lam)) for lam in LAMS}
                 for kind in ["main", "legacy"]}
# --- legacy best-during-training vs final ---
for lam in [6, 7]:
    cs = np.array([r["curve"] for r in E2 if r["kind"] == "legacy" and r["lam"] == lam])
    gaps = 100 * (OPT[lam] - cs[:, :, 1]) / OPT[lam]
    N[f"legacy_min_gap_during_{lam}"] = float(np.median(gaps.min(1)))
    N[f"legacy_step_of_min_{lam}"] = float(np.median(cs[0, gaps.argmin(1), 0]))
# --- sensitivity summary from exp2 ---
sens = [r for r in E2 if r["kind"] == "sens"]
def sgap(lam, **kw):
    v = []
    for r in sens:
        c = r["cfg"]
        ok = r["lam"] == lam and all(c.get(k) == val for k, val in
                                     dict(dict(kappa=0.6, eps_min=0.05, start="uniform", q0=0.0, n_episodes=9000), **kw).items())
        if ok:
            v.append(100 * (OPT[lam] - r["g"]) / OPT[lam])
    return dict(mean=float(np.mean(v)), max=float(np.max(v)), n=len(v))
N["sens"] = {lam: dict(base=sgap(lam), k055=sgap(lam, kappa=0.55), k07=sgap(lam, kappa=0.7), k08=sgap(lam, kappa=0.8),
                       k10=sgap(lam, kappa=1.0), e001=sgap(lam, eps_min=0.01), e02=sgap(lam, eps_min=0.2),
                       empty=sgap(lam, start="empty"), q100=sgap(lam, q0=100.0), x5=sgap(lam, n_episodes=45000))
             for lam in [5, 7]}
# --- Prop 3 coverage ---
cov = [(r[m + "_covered"] or abs(r[m + "_sim"] - r[m + "_exact"]) < 1e-4) for r in E1["prop3"] for m in ["PB", "W", "g"]]
N["prop3_covered"] = int(sum(cov)); N["prop3_total"] = len(cov)
N["structure"] = E1["structure"]
# --- extended: value-of-information scan ---
scan = []
for lams, g in [((3.0, 7.0), 0.05), ((4.0, 9.0), 0.05), ((4.0, 9.0), 0.02), ((4.0, 9.0), 0.01), ((4.0, 12.0), 0.01)]:
    p = dict(BASE); p["lams"] = lams; p["gam"] = (g, g)
    o = oracle_dp(p)["g"]
    (gb, Tb), (gc, TL, TH) = best_static(p)
    f = eval_szj(szj_from_thresholds(20, 20, 20), p)
    scan.append(dict(lams=lams, gam=g, oracle=o, cls=gc, TL=TL, TH=TH, blind=gb, Tb=Tb, fcfs=f,
                     gap_cls=100 * (o - gc) / o, gap_blind=100 * (o - gb) / o, gap_fcfs=100 * (o - f) / o))
N["voi_scan"] = scan
pe = dict(BASE); pe["lams"] = (4.0, 9.0); pe["gam"] = (0.02, 0.02)
orc = oracle_dp(pe)["policy"]
N["oracle_T"] = {f"z{z}_j{j}": int(np.argmin(np.append(orc[:20, z, j], 0))) for z in (0, 1) for j in (0, 1)}
# oracle threshold form check
thr_ok = True
for z in (0, 1):
    for j in (0, 1):
        col = orc[:, z, j]; T = int(np.argmin(np.append(col[:20], 0)))
        thr_ok &= bool(np.all(col[:T] == 1) and np.all(col[T:] == 0))
N["oracle_threshold_form"] = thr_ok
# --- extended: scenario table ---
sim = E4["sim"]
tab = {}
for sc in ["S0_train", "S1_fast_switch", "S2_demand_shock", "S3_service_degr", "S4_server_fail"]:
    row = {}
    for pol in ["fcfs", "blind", "cls", "robust_pol", "ce_dp", "plugin", "oracle"]:
        v = np.array([r["g"] for r in sim if r["scen"] == sc and r["policy"] == pol])
        row[pol] = dict(m=float(v.mean()), hw=float(2.262 * v.std(ddof=1) / np.sqrt(len(v))))
    sm = []
    for sd in range(5):
        v = [r["g"] for r in sim if r["scen"] == sc and r["policy"] == f"dqn{sd}"]
        sm.append(np.mean(v))
    sm = np.array(sm)
    row["dqn"] = dict(m=float(sm.mean()), hw=float(2.776 * sm.std(ddof=1) / np.sqrt(len(sm))),
                      min=float(sm.min()), max=float(sm.max()), seeds=sm.tolist())
    # paired comparisons DQN (seed-mean of reps 0-4) vs controllers on same reps
    for pol in ["cls", "ce_dp", "plugin"]:
        d = []
        for sd in range(5):
            dq = {r["rep"]: r["g"] for r in sim if r["scen"] == sc and r["policy"] == f"dqn{sd}"}
            bs = {r["rep"]: r["g"] for r in sim if r["scen"] == sc and r["policy"] == pol}
            d.append(np.mean([dq[k] - bs[k] for k in dq]))
        d = np.array(d)
        row[f"dqn_minus_{pol}"] = dict(m=float(d.mean()), hw=float(2.776 * d.std(ddof=1) / np.sqrt(len(d))))
    row["exact"] = E4["exact"].get(sc, {})
    tab[sc] = row
N["scen"] = tab
N["static_blind"] = E4["static_blind"]; N["static_cls"] = E4["static_cls"]; N["robust"] = E4["robust"]
# --- explanation ---
S5 = E5["seeds"]
def agg(vals):
    v = np.array([x for x in vals if x is not None], float)
    return dict(mean=float(v.mean()), min=float(v.min()), max=float(v.max()))
fid = {}
for name in ["tree1", "tree2", "tree3", "tree4", "tree5", "logit"]:
    fid[name] = {k: agg([r["fidelity"][name][k] for r in S5]) for k in
                 ["train", "test", "near_boundary", "grid", "z0_j0", "z0_j1", "z1_j0", "z1_j1"]}
N["fid"] = fid
N["frac_near"] = agg([r["frac_near_boundary"] for r in S5])
N["admit_rate"] = agg([r["admit_rate"] for r in S5])
N["n_test"] = agg([r["n_test"] for r in S5]); N["n_fit"] = agg([r["n_fit"] for r in S5])
N["pi"] = {f: agg([r["perm_importance"][f][0] for r in S5]) for f in S5[0]["perm_importance"]}
N["cf"] = {k: agg([r["cf"][k] for r in S5]) for k in S5[0]["cf"] if k != "n"}
N["cf_examples"] = S5[0]["cf_examples"]
N["tree_text_seed0"] = S5[0]["tree_text"]
N["profile_agree"] = agg([r["profile_agree_surr_agent"] for r in S5])
N["profile_absdiff"] = agg([r["profile_absdiff_surr_agent"] for r in S5])
N["T_agent"] = [r["T_agent"] for r in S5]; N["T_ref"] = S5[0]["T_ref"]; N["mids"] = E5["mids"]
N["patch"] = E6["patch"]; N["stability"] = E6["stability"]
P = E6["patch"]
gaps = np.array([P[s]["gap"] for s in P]); pg = np.array([P[s]["patch_gain"] for s in P]); rg = np.array([P[s]["rand_gain"] for s in P])
N["patch_pooled_recovered"] = float(pg.sum() / gaps.sum()); N["rand_pooled_recovered"] = float(rg.sum() / gaps.sum())
N["patch_flagged_w"] = agg([P[s]["flagged_w"] for s in P])
N["retrain"] = E7["retrain"]
R = E7["retrain"]
N["retrain_mean"] = dict(control=float(np.mean([R[s]["control"]["gain"] for s in R])),
                         guided=float(np.mean([R[s]["guided"]["gain"] for s in R])),
                         gmc=float(np.mean([R[s]["guided_minus_control"] for s in R])))
N["dqn_hp"] = {k: v for k, v in D3["hp"].items()}
N["dqn_ghat_final"] = [float(r["ghat"]) for r in D3["runs"]]
cv = E7["curve"]
N["dqn_curve_final"] = agg([c["g"] for c in cv if c["step"] == max(x["step"] for x in cv)])
N["dqn_curve_100k"] = agg([c["g"] for c in cv if c["step"] == 100000])
json.dump(N, open("results/numbers.json", "w"), indent=1, default=float)
print(json.dumps({k: N[k] for k in ["flat_1pct", "monotone", "prop3_covered", "prop3_total", "oracle_T",
                                     "oracle_threshold_form", "patch_pooled_recovered", "rand_pooled_recovered",
                                     "retrain_mean", "dqn_curve_final", "dqn_curve_100k", "profile_agree", "profile_absdiff",
                                     "frac_near", "patch_flagged_w"]}, indent=1, default=float))
for k in ["visits_main_9_19_min", "visits_main_9_19_mean", "visits_legacy_9_19_min", "visits_legacy_9_19_mean",
          "legacy_min_gap_during_6", "legacy_min_gap_during_7", "legacy_step_of_min_7"]:
    print(k, N[k])
for s in N["voi_scan"]:
    print(s)
print(json.dumps(N["sens"], indent=0))
print(json.dumps(N["cf"], indent=0)); print(json.dumps(N["pi"], indent=0))
print(json.dumps(N["fid"]["tree4"], indent=0)); print(json.dumps(N["fid"]["logit"], indent=0))
for sc, r in N["scen"].items():
    print(sc, {k: (round(v["m"], 2), round(v["hw"], 2)) for k, v in r.items() if isinstance(v, dict) and "m" in v})
