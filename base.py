"""Base model (Sections 3.1-3.4, 5): exact evaluation, DP, structural checks."""
import numpy as np


def death(s, c, mu):
    return mu * min(s, c)


def eval_policy(pi, lam, mu, c, K, R, P, h):
    """Exact performance of a deterministic occupancy policy pi[s] in {0,1}, pi[K]=0.
    Birth rate lam*pi[s], death rate mu*min(s,c). Recurrent class {0..first reject state}."""
    pi = np.asarray(pi, dtype=int).copy()
    pi[K] = 0
    p = np.zeros(K + 1)
    p[0] = 1.0
    top = K
    for s in range(K):
        if pi[s] == 0:
            top = s
            break
        p[s + 1] = p[s] * lam / death(s + 1, c, mu)
    p[top + 1:] = 0.0
    p /= p.sum()
    s_arr = np.arange(K + 1)
    PB = float(p[pi == 0].sum())
    L = float((s_arr * p).sum())
    Theta = lam * (1 - PB)
    W = L / Theta if Theta > 0 else np.nan
    U = float((np.minimum(s_arr, c) * p).sum() / c)
    g = R * Theta - h * L - P * lam * PB
    return dict(PB=PB, L=L, Theta=Theta, W=W, U=U, g=g, T=top, dist=p)


def threshold_policy(T, K):
    return np.array([1 if s < T else 0 for s in range(K + 1)])


def best_threshold(lam, mu, c, K, R, P, h):
    best = None
    for T in range(0, K + 1):
        r = eval_policy(threshold_policy(T, K), lam, mu, c, K, R, P, h)
        if best is None or r["g"] > best[1]["g"] + 1e-12:
            best = (T, r)
    return best


def rvi_dp(lam, mu, c, K, R, P, h, tol=1e-12, maxit=200000):
    """Relative value iteration on the uniformized chain, reward (3). Returns
    rho*, h*, J*=-h*, greedy policy (ties -> reject), Q-advantage."""
    Lam = lam + c * mu
    s = np.arange(K + 1)
    d = np.array([death(x, c, mu) for x in s])
    hv = np.zeros(K + 1)
    for it in range(maxit):
        down = hv[np.maximum(s - 1, 0)]
        up = np.append(hv[1:], hv[K])  # unused at K
        q_rej = lam / Lam * (-P) - h * s / Lam + lam / Lam * hv + d / Lam * down + (1 - lam / Lam - d / Lam) * hv
        q_adm = lam / Lam * R - h * s / Lam + lam / Lam * up + d / Lam * down + (1 - lam / Lam - d / Lam) * hv
        q_adm[K] = -np.inf
        new = np.maximum(q_rej, q_adm)
        rho = new[0]
        new = new - rho
        if np.max(np.abs(new - hv)) < tol:
            hv = new
            break
        hv = new
    down = hv[np.maximum(s - 1, 0)]
    up = np.append(hv[1:], hv[K])
    q_rej = lam / Lam * (-P) - h * s / Lam + lam / Lam * hv + d / Lam * down + (1 - lam / Lam - d / Lam) * hv
    q_adm = lam / Lam * R - h * s / Lam + lam / Lam * up + d / Lam * down + (1 - lam / Lam - d / Lam) * hv
    q_adm[K] = -np.inf
    pol = (q_adm > q_rej + 1e-9).astype(int)
    pol[K] = 0
    return dict(rho=rho, g=rho * Lam, h=hv, J=-hv, policy=pol, q_adm=q_adm, q_rej=q_rej, iters=it)


def simulate_ctmc(pi, lam, mu, c, K, R, P, h, T_end, rng):
    """Discrete-event simulation of the controlled M/M/c/K; returns time-average metrics."""
    s = 0
    t = 0.0
    area = 0.0
    busy = 0.0
    adm = blk = 0
    narr = 0
    while True:
        rate = lam + death(s, c, mu)
        dt = rng.exponential(1 / rate)
        if t + dt > T_end:
            area += s * (T_end - t)
            busy += min(s, c) * (T_end - t)
            break
        area += s * dt
        busy += min(s, c) * dt
        t += dt
        if rng.random() < lam / rate:
            narr += 1
            if s < K and pi[s] == 1:
                s += 1
                adm += 1
            else:
                blk += 1
        else:
            s -= 1
    L = area / T_end
    Theta = adm / T_end
    g = (R * adm - P * blk) / T_end - h * L
    return dict(PB=blk / max(narr, 1), L=L, Theta=Theta, W=L / Theta, U=busy / (c * T_end), g=g)
