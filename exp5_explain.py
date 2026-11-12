"""Exp 5: explanation layer (Section 4.3) applied to the TRAINED DQN policies.
Held-out fidelity (overall, per regime x class, near-boundary), permutation importance,
agent- and surrogate-based counterfactuals with validity, threshold profiles vs a reference
controller, patch test with random-cell control, and seed-to-seed stability."""
import json
import pickle
import sys
from multiprocessing import Pool
import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from extended import Sim
from policies import load_agent

D = pickle.load(open("results/exp3_dqn.pkl", "rb"))
P0, HP = D["p"], D["hp"]
K = P0["K"]
CTRL = pickle.load(open("results/controllers.pkl", "rb"))
REF = sys.argv[1] if len(sys.argv) > 1 else "ce_dp"
AGENTS = {r["seed"]: load_agent(P0, HP, r["final"], seed=r["seed"]) for r in D["runs"]}
BINS = np.arange(2.0, 14.0, 1.0)            # lam_hat bin edges [2,3),...,[12,13)
MIDS = BINS[:-1] + 0.5
N_ROLL, T_ROLL, N_FIT = 10, 3000.0, 7
DEPTH = 4
FEATS = ["occupancy s", "class j", "regime estimate lam_hat"]


def bin_of(lh):
    return int(np.clip(np.searchsorted(BINS, lh, side="right") - 1, 0, len(MIDS) - 1))


def ref_decision(s, j, lh):
    return getattr(CTRL, REF)(s, j, lh, None)


def profile(fn):
    """T(j,b): smallest occupancy at which fn rejects, at the bin midpoint."""
    T = np.zeros((2, len(MIDS)), int)
    for j in range(2):
        for b, m in enumerate(MIDS):
            a = fn(np.arange(K + 1), np.full(K + 1, j), np.full(K + 1, m))
            a[K] = 0
            T[j, b] = int(np.argmin(a))
    return T


def rollouts(seed):
    ag = AGENTS[seed]
    R = []
    for r in range(N_ROLL):
        out = Sim(P0, seed=20_000 + 100 * seed + r).run(ag.act_greedy, T_ROLL, record=True, warm=100.0)
        rec = out["recs"]
        rec = rec[rec[:, 0] < K]  # s=K decisions are forced, excluded from fidelity
        R.append(np.column_stack([rec, np.full(len(rec), r)]))
    return np.vstack(R)  # cols: s, j, lh, z, a, rollout


def counterfactuals(ag, X, A, pred_fn):
    """Single-feature minimal counterfactuals (17) under pred_fn; returns deltas (nan = none)."""
    n = len(X)
    ds = np.full(n, np.nan); dl = np.full(n, np.nan); dj = np.zeros(n, bool)
    # occupancy: integer changes, feasible s' in [0, K-1]
    for d in range(1, K):
        for sgn in (-1, 1):
            s2 = X[:, 0] + sgn * d
            ok = (s2 >= 0) & (s2 <= K - 1) & np.isnan(ds)
            if ok.any():
                a2 = pred_fn(s2[ok], X[ok, 1], X[ok, 2])
                hit = np.where(ok)[0][a2 != A[ok]]
                ds[hit] = sgn * d
    grid = np.round(np.arange(0.5, 15.01, 0.1), 2)
    # lam_hat: nearest grid value (in either direction) that flips the action
    for i in range(n):
        cand = grid[np.argsort(np.abs(grid - X[i, 2]))]
        a2 = pred_fn(np.full(len(cand), X[i, 0]), np.full(len(cand), X[i, 1]), cand)
        f = np.where(a2 != A[i])[0]
        if len(f):
            dl[i] = cand[f[0]] - X[i, 2]
    a2 = pred_fn(X[:, 0], 1 - X[:, 1], X[:, 2])
    dj = a2 != A
    return ds, dl, dj


def analyse_seed(seed):
    ag = AGENTS[seed]
    rng = np.random.default_rng(seed)
    Dd = rollouts(seed)
    fit = Dd[Dd[:, 5] < N_FIT]; test = Dd[Dd[:, 5] >= N_FIT]
    Xf, yf = fit[:, :3], fit[:, 4].astype(int)
    Xt, yt = test[:, :3], test[:, 4].astype(int)
    out = dict(seed=seed, n_fit=len(fit), n_test=len(test), admit_rate=float(yt.mean()))
    # surrogates
    surr = {}
    for d in [1, 2, 3, 4, 5]:
        surr[f"tree{d}"] = DecisionTreeClassifier(max_depth=d, random_state=0).fit(Xf, yf)
    surr["logit"] = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xf, yf)
    agent_fn = lambda s, j, lh: ag.greedy_batch(s, j, lh)
    # near-boundary decisions: agent action flips within |ds|<=2
    ds_a, dl_a, dj_a = counterfactuals(ag, Xt, yt, agent_fn)
    near = np.abs(np.nan_to_num(ds_a, nan=99)) <= 2
    out["frac_near_boundary"] = float(near.mean())
    fid = {}
    for name, m in surr.items():
        pr = m.predict(Xt)
        row = dict(test=float((pr == yt).mean()), train=float((m.predict(Xf) == yf).mean()),
                   near_boundary=float((pr[near] == yt[near]).mean()))
        for z in (0, 1):
            for j in (0, 1):
                msk = (test[:, 3] == z) & (test[:, 1] == j)
                row[f"z{z}_j{j}"] = float((pr[msk] == yt[msk]).mean()) if msk.any() else None
        # off-policy grid fidelity (uniform over s<K, j, lam_hat in [2,13])
        S, J, L = np.meshgrid(np.arange(K), [0, 1], np.arange(2.0, 13.0, 0.25), indexing="ij")
        G = np.column_stack([S.ravel(), J.ravel(), L.ravel()])
        row["grid"] = float((m.predict(G) == ag.greedy_batch(G[:, 0], G[:, 1], G[:, 2])).mean())
        fid[name] = row
    out["fidelity"] = fid
    best = surr[f"tree{DEPTH}"]
    out["tree_text"] = export_text(best, feature_names=["s", "j", "lam_hat"], decimals=2)
    # permutation importance (16) on held-out data
    base = (best.predict(Xt) == yt).mean()
    pi = {}
    for f in range(3):
        drops = []
        for _ in range(30):
            Xp = Xt.copy(); Xp[:, f] = rng.permutation(Xp[:, f])
            drops.append(base - (best.predict(Xp) == yt).mean())
        pi[FEATS[f]] = (float(np.mean(drops)), float(np.std(drops)))
    out["perm_importance"] = pi
    # counterfactuals on a sample of held-out decisions
    idx = rng.choice(len(Xt), size=min(800, len(Xt)), replace=False)
    Xs, As = Xt[idx], yt[idx]
    ds_a, dl_a, dj_a = counterfactuals(ag, Xs, As, agent_fn)
    sur_fn = lambda s, j, lh: best.predict(np.column_stack([s, j, lh]))
    As_s = best.predict(Xs)
    ds_s, dl_s, dj_s = counterfactuals(ag, Xs, As_s, sur_fn)
    # validity: does the surrogate counterfactual also reverse the AGENT's action?
    v_s = ~np.isnan(ds_s)
    val_s = ag.greedy_batch(Xs[v_s, 0] + ds_s[v_s], Xs[v_s, 1], Xs[v_s, 2]) != As[v_s]
    v_l = ~np.isnan(dl_s)
    val_l = ag.greedy_batch(Xs[v_l, 0], Xs[v_l, 1], Xs[v_l, 2] + dl_s[v_l]) != As[v_l]
    val_j = ag.greedy_batch(Xs[dj_s, 0], 1 - Xs[dj_s, 1], Xs[dj_s, 2]) != As[dj_s]
    agree = As_s == As
    out["cf"] = dict(
        n=int(len(idx)),
        agent_s_exists=float(np.mean(~np.isnan(ds_a))), agent_s_med=float(np.nanmedian(np.abs(ds_a))),
        agent_s_le1=float(np.mean(np.abs(np.nan_to_num(ds_a, nan=99)) <= 1)),
        agent_l_exists=float(np.mean(~np.isnan(dl_a))), agent_l_med=float(np.nanmedian(np.abs(dl_a))),
        agent_j_flip=float(np.mean(dj_a)),
        agent_j_flip_given_reject=float(np.mean(dj_a[As == 0])) if (As == 0).any() else None,
        sur_valid_s=float(val_s.mean()) if len(val_s) else None,
        sur_valid_s_on_agree=float(val_s[agree[v_s]].mean()) if agree[v_s].any() else None,
        sur_valid_l=float(val_l.mean()) if len(val_l) else None,
        sur_valid_j=float(val_j.mean()) if len(val_j) else None,
        sur_exact_s=float(np.mean(ds_s[v_s & ~np.isnan(ds_a)] == ds_a[v_s & ~np.isnan(ds_a)])),
    )
    # example counterfactual statements (first rejected standard customer, first admitted)
    ex = []
    for want in (0, 1):
        k = np.where(As == want)[0]
        if len(k):
            k = k[0]
            ex.append(dict(s=int(Xs[k, 0]), j=int(Xs[k, 1]), lh=float(Xs[k, 2]), a=int(As[k]),
                           ds=None if np.isnan(ds_a[k]) else float(ds_a[k]),
                           dl=None if np.isnan(dl_a[k]) else float(dl_a[k]), dj=bool(dj_a[k])))
    out["cf_examples"] = ex
    # threshold profiles
    T_ag = profile(agent_fn)
    T_su = profile(sur_fn)
    T_rf = profile(lambda s, j, lh: np.array([ref_decision(int(a), int(b), float(c)) for a, b, c in zip(s, j, lh)]))
    T_or = np.array([[int(np.argmin(np.append(CTRL.orc["policy"][:K, z, j], 0))) for z in (0, 1)] for j in (0, 1)])
    w = np.zeros((2, len(MIDS)))
    for r in Dd:
        w[int(r[1]), bin_of(r[2])] += 1
    w /= w.sum()
    out.update(T_agent=T_ag.tolist(), T_surr=T_su.tolist(), T_ref=T_rf.tolist(), T_oracle_jz=T_or.tolist(),
               visit_w=w.tolist())
    out["profile_agree_surr_agent"] = float(np.average(T_ag == T_su, weights=w))
    out["profile_absdiff_surr_agent"] = float(np.average(np.abs(T_ag - T_su), weights=w))
    out["test_X"] = Xt[rng.choice(len(Xt), size=min(1500, len(Xt)), replace=False)].tolist()
    pickle.dump(best, open(f"results/surr_seed{seed}.pkl", "wb"))
    return out


if __name__ == "__main__":
    with Pool(2) as pl:
        res = pl.map(analyse_seed, sorted(AGENTS))
    json.dump(dict(ref=REF, depth=DEPTH, mids=MIDS.tolist(), seeds=res), open(f"results/exp5_{REF}.json", "w"),
              indent=1, default=float)
    for r in res:
        print("seed", r["seed"], "fid", {k: round(v["test"], 3) for k, v in r["fidelity"].items()},
              "near", round(r["fidelity"]["tree4"]["near_boundary"], 3), "grid", round(r["fidelity"]["tree4"]["grid"], 3))
        print("  PI", r["perm_importance"])
        print("  CF", r["cf"])
        print("  T_agent", r["T_agent"]); print("  T_ref  ", r["T_ref"])
