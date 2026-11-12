"""Exp 1: exact evaluation, structural checks of Prop 2, simulation-vs-analytic check of Prop 3,
and sensitivity of the optimal-vs-FCFS gap to (R,P,h,c,K)."""
import json
import numpy as np
from base import eval_policy, threshold_policy, best_threshold, rvi_dp, simulate_ctmc

C, K, MU, R, P, H = 5, 20, 1.0, 10.0, 5.0, 1.0
LAMS = [2, 3, 4, 5, 6, 7]
out = {"table": [], "structure": [], "prop3": [], "sens": []}

for lam in LAMS:
    fc = eval_policy(threshold_policy(K, K), lam, MU, C, K, R, P, H)
    Tb, th = best_threshold(lam, MU, C, K, R, P, H)
    dp = rvi_dp(lam, MU, C, K, R, P, H)
    dpe = eval_policy(dp["policy"], lam, MU, C, K, R, P, H)
    # structural checks on the DP solution itself (all states)
    pol = dp["policy"]
    first_rej = int(np.argmin(pol)) if pol.min() == 0 else K
    is_thr = bool(np.all(pol[:first_rej] == 1) and np.all(pol[first_rej:] == 0))
    a = np.diff(dp["J"])  # a_s = J(s)-J(s-1), s=1..K
    nondec = bool(np.all(a >= -1e-8))
    convex = bool(np.all(np.diff(a) >= -1e-8))
    s_star_formula = next((s for s in range(K) if dp["J"][s + 1] - dp["J"][s] > R + P), K)
    out["structure"].append(dict(lam=lam, is_threshold=is_thr, T_dp=first_rej, nondecreasing=nondec,
                                 convex=convex, min_a=float(a.min()), min_d=float(np.diff(a).min()),
                                 s_star_eq23=s_star_formula, g_rvi=dp["g"], g_exact_eval=dpe["g"]))
    for name, r in [("FCFS", fc), ("Threshold", th), ("DP", dpe)]:
        out["table"].append(dict(lam=lam, policy=name, T=r["T"], PB=r["PB"], W=r["W"], U=r["U"], L=r["L"], g=r["g"]))
    # Prop 3: replicated simulation vs closed form for FCFS and the optimal threshold
    for name, pi in [("FCFS", threshold_policy(K, K)), ("T*", threshold_policy(Tb, K))]:
        ex = eval_policy(pi, lam, MU, C, K, R, P, H)
        rng = np.random.default_rng(1000 + lam)
        sims = [simulate_ctmc(pi, lam, MU, C, K, R, P, H, 20000.0, rng) for _ in range(10)]
        row = dict(lam=lam, policy=name)
        for m in ["PB", "W", "g"]:
            v = np.array([x[m] for x in sims])
            hw = 2.262 * v.std(ddof=1) / np.sqrt(len(v))
            row[m + "_exact"] = ex[m]
            row[m + "_sim"] = float(v.mean())
            row[m + "_hw"] = float(hw)
            row[m + "_covered"] = bool(abs(v.mean() - ex[m]) <= hw)
        out["prop3"].append(row)

# Sensitivity: gain of optimal over FCFS, and T*, across parameters (exact)
base = dict(c=5, K=20, R=10.0, P=5.0, h=1.0)
grid = {
    "h": [0.25, 0.5, 1.0, 2.0, 4.0],
    "R": [5.0, 10.0, 20.0, 40.0],
    "P": [0.0, 2.5, 5.0, 10.0, 20.0],
    "cK": [(2, 8), (5, 10), (5, 20), (5, 40), (10, 40), (20, 80)],
}
for key, vals in grid.items():
    for v in vals:
        prm = dict(base)
        if key == "cK":
            prm["c"], prm["K"] = v
        else:
            prm[key] = v
        for rho in [0.8, 1.0, 1.2, 1.4]:
            lam = rho * prm["c"] * MU
            fc = eval_policy(threshold_policy(prm["K"], prm["K"]), lam, MU, prm["c"], prm["K"], prm["R"], prm["P"], prm["h"])
            Tb, th = best_threshold(lam, MU, prm["c"], prm["K"], prm["R"], prm["P"], prm["h"])
            gap = (th["g"] - fc["g"]) / abs(th["g"]) if th["g"] != 0 else np.nan
            out["sens"].append(dict(vary=key, value=v if key != "cK" else list(v), rho=rho, c=prm["c"], K=prm["K"],
                                    R=prm["R"], P=prm["P"], h=prm["h"], T_star=Tb, g_opt=th["g"], g_fcfs=fc["g"],
                                    rel_gain_vs_fcfs=(th["g"] - fc["g"]) / abs(fc["g"]) if fc["g"] != 0 else None,
                                    fcfs_loss_rel_opt=gap))

json.dump(out, open("results/exp1.json", "w"), indent=1, default=float)
for r in out["table"]:
    print(f"lam={r['lam']} {r['policy']:9s} T={r['T']:2d} PB={r['PB']:.4f} W={r['W']:.3f} U={r['U']:.3f} g={r['g']:.3f}")
for r in out["structure"]:
    print(r)
for r in out["prop3"]:
    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
