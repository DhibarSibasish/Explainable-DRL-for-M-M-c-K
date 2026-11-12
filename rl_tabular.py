"""Tabular learners on the uniformized base model.
rvi_q: Section 4.2.1 (RVI Q-learning, ref pair (0,1), alpha=(1+N)^-kappa, eps -> eps_min>0,
       uniformly drawn episode start state).
r_learning_legacy: the original submission's configuration (Schwartz R-learning, constant
       step sizes, gain updated on greedy steps only, episodes from the empty system,
       eps annealed 0.6 -> 0.02), kept as an ablation.
Greedy policies are evaluated EXACTLY with base.eval_policy."""
import numpy as np
from base import eval_policy


def _greedy(Q, K):
    pol = (Q[:, 1] > Q[:, 0]).astype(int)
    pol[K] = 0
    return pol


def rvi_q(lam, mu, c, K, R, P, h, seed, n_episodes=9000, ep_len=200, kappa=0.7,
          eps0=0.6, eps_min=0.05, eps_decay_frac=0.5, start="uniform", q0=0.0,
          eval_every=50, prm=None):
    rng = np.random.default_rng(seed)
    Lam = lam + c * mu
    pa = lam / Lam
    Q = np.full((K + 1, 2), q0, dtype=float)
    Q[K, 1] = -1e18  # infeasible, never used
    N = np.zeros((K + 1, 2), dtype=np.int64)
    total = n_episodes * ep_len
    U = rng.random(total)       # event
    E = rng.random(total)       # exploration
    A = rng.integers(0, 2, total)
    starts = rng.integers(0, K + 1, n_episodes)
    curve = []
    decay_steps = eps_decay_frac * total
    k = 0
    Ql = Q.tolist()
    Nl = N.tolist()
    for ep in range(n_episodes):
        s = int(starts[ep]) if start == "uniform" else 0
        for _ in range(ep_len):
            eps = max(eps_min, eps0 - (eps0 - eps_min) * k / decay_steps)
            if s == K:
                a = 0
            elif E[k] < eps:
                a = int(A[k])
            else:
                a = 1 if Ql[s][1] > Ql[s][0] else 0
            u = U[k]
            d = mu * (s if s < c else c) / Lam
            if u < pa:  # arrival
                if a == 1:
                    r = R - h * s / Lam
                    s2 = s + 1
                else:
                    r = -P - h * s / Lam
                    s2 = s
            elif u < pa + d:
                r = -h * s / Lam
                s2 = s - 1
            else:
                r = -h * s / Lam
                s2 = s
            row = Ql[s2]
            mx = row[0] if s2 == K else (row[0] if row[0] > row[1] else row[1])
            n = Nl[s][a]
            alpha = (1.0 + n) ** (-kappa)
            Nl[s][a] = n + 1
            Ql[s][a] += alpha * (r - Ql[0][1] + mx - Ql[s][a])
            s = s2
            k += 1
        if (ep + 1) % eval_every == 0:
            Qa = np.array(Ql)
            pol = _greedy(Qa, K)
            ev = eval_policy(pol, lam, mu, c, K, R, P, h)
            curve.append((k, ev["g"], Lam * Qa[0, 1], ev["T"]))
    Qa = np.array(Ql)
    pol = _greedy(Qa, K)
    return dict(Q=Qa, policy=pol, N=np.array(Nl), curve=np.array(curve), ghat=Lam * Qa[0, 1])


def r_learning_legacy(lam, mu, c, K, R, P, h, seed, n_episodes=9000, ep_len=200, alpha=0.1,
                      beta=0.01, eps0=0.6, eps_end=0.02, eval_every=50):
    rng = np.random.default_rng(seed)
    Lam = lam + c * mu
    pa = lam / Lam
    Ql = [[0.0, 0.0] for _ in range(K + 1)]
    Ql[K][1] = -1e18
    Nl = [[0, 0] for _ in range(K + 1)]
    rho = 0.0
    total = n_episodes * ep_len
    U = rng.random(total)
    E = rng.random(total)
    A = rng.integers(0, 2, total)
    curve = []
    k = 0
    for ep in range(n_episodes):
        s = 0
        for _ in range(ep_len):
            eps = eps0 + (eps_end - eps0) * k / total
            greedy_a = 0 if s == K else (1 if Ql[s][1] > Ql[s][0] else 0)
            if s == K:
                a = 0
            elif E[k] < eps:
                a = int(A[k])
            else:
                a = greedy_a
            u = U[k]
            d = mu * (s if s < c else c) / Lam
            if u < pa:
                if a == 1:
                    r = R - h * s / Lam; s2 = s + 1
                else:
                    r = -P - h * s / Lam; s2 = s
            elif u < pa + d:
                r = -h * s / Lam; s2 = s - 1
            else:
                r = -h * s / Lam; s2 = s
            row = Ql[s2]
            mx = row[0] if s2 == K else max(row)
            Nl[s][a] += 1
            Ql[s][a] += alpha * (r - rho + mx - Ql[s][a])
            if a == greedy_a:
                cur = Ql[s]
                cmx = cur[0] if s == K else max(cur)
                rho += beta * (r + mx - cmx - rho)
            s = s2
            k += 1
        if (ep + 1) % eval_every == 0:
            Qa = np.array(Ql)
            pol = _greedy(Qa, K)
            ev = eval_policy(pol, lam, mu, c, K, R, P, h)
            curve.append((k, ev["g"], Lam * rho, ev["T"]))
    Qa = np.array(Ql)
    pol = _greedy(Qa, K)
    return dict(Q=Qa, policy=pol, N=np.array(Nl), curve=np.array(curve), ghat=Lam * rho)
