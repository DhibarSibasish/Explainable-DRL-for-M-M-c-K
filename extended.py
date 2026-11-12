"""Extended model (Section 3.5): hidden MMPP regimes + two customer classes.
Exact tools: regime-aware oracle (RVI on (s,z)), exact evaluation of (s,z,j)-policies,
two-class stationary DP (for certainty-equivalent control).
Simulation: arrival-epoch simulator for observation-based policies o=(s,j,lam_hat),
with optional shifts (faster switching, OOD demand, service degradation, server failures)."""
import numpy as np

# ---------------- default parameters ----------------
BASE = dict(c=5, K=20, mu=1.0, h=1.0,
            lams=(3.0, 7.0), gam=(0.05, 0.05),     # regime rates, leave-rates of regime 1,2
            qH=0.3, RH=20.0, PH=15.0, RL=10.0, PL=5.0, M=10)
CLS = (1, 0)  # j=1 high priority, j=0 standard


def cls_par(p):
    # index 0 = standard (L), 1 = high (H)
    return np.array([1 - p["qH"], p["qH"]]), np.array([p["RL"], p["RH"]]), np.array([p["PL"], p["PH"]])


def oracle_dp(p, tol=1e-11, maxit=400000):
    """Average-reward optimum for the fully observed (s,z) process; action per (s,z,j)."""
    c, K, mu, h = p["c"], p["K"], p["mu"], p["h"]
    lams = np.array(p["lams"]); gam = np.array(p["gam"])
    m = len(lams)
    q, Rj, Pj = cls_par(p)
    Lam = lams.max() + c * mu + gam.max()
    s = np.arange(K + 1)
    d = mu * np.minimum(s, c)
    V = np.zeros((K + 1, m))
    for it in range(maxit):
        up = np.vstack([V[1:], V[K:K + 1]])
        down = np.vstack([V[0:1], V[:-1]])
        newV = np.empty_like(V)
        for z in range(m):
            adm = Rj[None, :] + up[:, z][:, None]          # (K+1, 2)
            rej = -Pj[None, :] + V[:, z][:, None]
            best = np.maximum(adm, rej)
            best[K] = rej[K]
            arr = (best * q[None, :]).sum(1)
            zz = 1 - z if m == 2 else None
            newV[:, z] = (-h * s + lams[z] * arr + d * down[:, z] + gam[z] * V[:, zz]
                          + (Lam - lams[z] - d - gam[z]) * V[:, z]) / Lam
        rho = newV[0, 0]
        newV -= rho
        if np.max(np.abs(newV - V)) < tol:
            V = newV
            break
        V = newV
    up = np.vstack([V[1:], V[K:K + 1]])
    pol = np.zeros((K + 1, m, 2), dtype=int)
    for z in range(m):
        adm = Rj[None, :] + up[:, z][:, None]
        rej = -Pj[None, :] + V[:, z][:, None]
        pol[:, z, :] = (adm > rej + 1e-9).astype(int)
        pol[K, z, :] = 0
    return dict(g=rho * Lam, V=V, policy=pol, iters=it)


def eval_szj(pol, p):
    """Exact reward rate of a policy pol[s,z,j] on the (s,z) CTMC."""
    c, K, mu, h = p["c"], p["K"], p["mu"], p["h"]
    lams = np.array(p["lams"]); gam = np.array(p["gam"]); m = len(lams)
    q, Rj, Pj = cls_par(p)
    n = (K + 1) * m
    idx = lambda s, z: s * m + z
    Q = np.zeros((n, n))
    rew = np.zeros(n)
    for s in range(K + 1):
        for z in range(m):
            i = idx(s, z)
            a = pol[s, z, :].copy()
            if s == K:
                a[:] = 0
            pa = (q * a).sum()
            if s < K and pa > 0:
                Q[i, idx(s + 1, z)] += lams[z] * pa
            if s > 0:
                Q[i, idx(s - 1, z)] += mu * min(s, c)
            Q[i, idx(s, 1 - z)] += gam[z]
            rew[i] = lams[z] * (q * (a * Rj - (1 - a) * Pj)).sum() - h * s
    np.fill_diagonal(Q, -Q.sum(1))
    A = np.vstack([Q.T, np.ones(n)])
    b = np.zeros(n + 1); b[-1] = 1
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    return float(pi @ rew)


def stationary_two_class_dp(lam, p):
    """Two-class M/M/c/K with known constant lam: optimal class thresholds (T_L, T_H)."""
    pp = dict(p); pp["lams"] = (lam, lam); pp["gam"] = (0.0, 0.0)
    o = oracle_dp(pp)
    pol = o["policy"][:, 0, :]
    T = [int(np.argmin(pol[:, j])) if pol[:, j].min() == 0 else p["K"] for j in range(2)]
    return T, pol


def ce_table(p, grid):
    return np.array([stationary_two_class_dp(l, p)[0] for l in grid])  # (len, 2) [T_L, T_H]


# ---------------- simulator ----------------
class Sim:
    """Arrival-epoch simulator. policy(s, j, lam_hat, z) -> {0,1}. Shift options:
    lams/gam override; mu override (degradation); fail=(rate, repair, c_fail) server outages."""

    def __init__(self, p, seed, fail=None):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.fail = fail

    def run(self, policy, T_end, record=False, warm=200.0):
        p, rng = self.p, self.rng
        c0, K, mu, h, M = p["c"], p["K"], p["mu"], p["h"], p["M"]
        lams, gam = p["lams"], p["gam"]
        q, Rj, Pj = cls_par(p)
        s, z, t = 0, int(rng.integers(0, 2)), 0.0
        up = True
        c = c0
        taus = [1.0 / lams[z]] * M
        tau_sum = sum(taus); ti = 0
        last_arr = 0.0
        tot_r = 0.0; area = 0.0; nadm = narr = nblk = 0
        recs = []
        t_meas = 0.0
        while t < T_end + warm:
            rate_arr = lams[z]
            rate_dep = mu * min(s, c)
            rate_sw = gam[z]
            rate_f = 0.0
            if self.fail is not None:
                rate_f = self.fail[0] if up else self.fail[1]
            tot = rate_arr + rate_dep + rate_sw + rate_f
            dt = rng.exponential(1.0 / tot)
            if t >= warm:
                area += s * dt; t_meas += dt
            elif t + dt > warm:
                area += s * (t + dt - warm); t_meas += t + dt - warm
            t += dt
            u = rng.random() * tot
            if u < rate_arr:
                tau = t - last_arr; last_arr = t
                tau_sum += tau - taus[ti]; taus[ti] = tau; ti = (ti + 1) % M
                lh = M / tau_sum
                j = 1 if rng.random() < q[1] else 0
                a = 0 if s >= K else int(policy(s, j, lh, z))
                if t >= warm:
                    narr += 1
                    if a:
                        tot_r += Rj[j]; nadm += 1
                    else:
                        tot_r -= Pj[j]; nblk += 1
                    if record:
                        recs.append((s, j, lh, z, a))
                if a:
                    s += 1
            elif u < rate_arr + rate_dep:
                s -= 1
            elif u < rate_arr + rate_dep + rate_sw:
                z = 1 - z
            else:
                up = not up
                c = c0 if up else self.fail[2]
        g = (tot_r - h * area) / t_meas
        out = dict(g=g, PB=nblk / max(narr, 1), L=area / t_meas)
        if record:
            out["recs"] = np.array(recs)
        return out


# ---------------- DQN environment (semi-Markov, arrival epochs) ----------------
class ArrivalEnv:
    """Step = one arrival decision. Returns obs, and after action: B_n (reward until next
    arrival incl. holding integral), tau_{n+1}, next obs. Optional regime-dependent fail."""

    def __init__(self, p, seed):
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.q, self.Rj, self.Pj = cls_par(p)
        self.reset()

    def reset(self, s0=None, z0=None):
        p = self.p
        self.s = int(self.rng.integers(0, p["K"] + 1)) if s0 is None else s0
        self.z = int(self.rng.integers(0, 2)) if z0 is None else z0
        self.taus = [1.0 / p["lams"][self.z]] * p["M"]
        self.ti = 0
        self.tau_sum = sum(self.taus)
        self.j = 1 if self.rng.random() < self.q[1] else 0
        return self.obs()

    def obs(self):
        return (self.s, self.j, self.p["M"] / self.tau_sum)

    def step(self, a):
        p, rng = self.p, self.rng
        c, K, mu, h = p["c"], p["K"], p["mu"], p["h"]
        if self.s >= K:
            a = 0
        B = self.Rj[self.j] if a else -self.Pj[self.j]
        if a:
            self.s += 1
        tau = 0.0
        hold = 0.0
        while True:
            ra = p["lams"][self.z]; rd = mu * min(self.s, c); rs = p["gam"][self.z]
            tot = ra + rd + rs
            dt = rng.exponential(1.0 / tot)
            hold += self.s * dt; tau += dt
            u = rng.random() * tot
            if u < ra:
                break
            elif u < ra + rd:
                self.s -= 1
            else:
                self.z = 1 - self.z
        B -= h * hold
        self.tau_sum += tau - self.taus[self.ti]; self.taus[self.ti] = tau
        self.ti = (self.ti + 1) % p["M"]
        self.j = 1 if rng.random() < self.q[1] else 0
        return B, tau, self.obs()
