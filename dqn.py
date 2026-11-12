"""NumPy DQN for the extended model (Section 4.2.2): semi-Markov average-reward target
y = B - g_hat * tau + max_a' Q_target(o', a'), Huber loss, replay buffer, target network,
Adam, feasibility masking at s = K, gain tracked as ratio of exponentially weighted sums (13)."""
import numpy as np
from extended import ArrivalEnv

HP = dict(hidden=64, lr=5e-4, batch=64, buffer=100_000, target_every=500, warmup=2_000,
          steps=300_000, eps0=1.0, eps_end=0.05, eps_frac=0.4, kappa_g=0.999, huber=1.0,
          reward_scale=10.0, lam_max=15.0, reset_every=2_000, double=True, lr_end_frac=1.0)


def feats(s, j, lh, K, lam_max):
    return np.array([s / K, float(j), min(lh, lam_max) / lam_max])


class MLP:
    def __init__(self, n_in, n_h, n_out, rng):
        self.W1 = rng.normal(0, np.sqrt(2 / n_in), (n_in, n_h)); self.b1 = np.zeros(n_h)
        self.W2 = rng.normal(0, np.sqrt(2 / n_h), (n_h, n_h)); self.b2 = np.zeros(n_h)
        self.W3 = rng.normal(0, np.sqrt(1 / n_h), (n_h, n_out)); self.b3 = np.zeros(n_out)
        self.params = ["W1", "b1", "W2", "b2", "W3", "b3"]
        self.m = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.v = {k: np.zeros_like(getattr(self, k)) for k in self.params}
        self.t = 0

    def forward(self, X):
        z1 = X @ self.W1 + self.b1; a1 = np.maximum(z1, 0)
        z2 = a1 @ self.W2 + self.b2; a2 = np.maximum(z2, 0)
        return a2 @ self.W3 + self.b3, (X, z1, a1, z2, a2)

    def copy_from(self, o):
        for k in self.params:
            setattr(self, k, getattr(o, k).copy())

    def step(self, cache, dout, lr):
        X, z1, a1, z2, a2 = cache
        g = {}
        g["W3"] = a2.T @ dout; g["b3"] = dout.sum(0)
        da2 = dout @ self.W3.T; dz2 = da2 * (z2 > 0)
        g["W2"] = a1.T @ dz2; g["b2"] = dz2.sum(0)
        da1 = dz2 @ self.W2.T; dz1 = da1 * (z1 > 0)
        g["W1"] = X.T @ dz1; g["b1"] = dz1.sum(0)
        self.t += 1
        b1, b2, e = 0.9, 0.999, 1e-8
        for k in self.params:
            self.m[k] = b1 * self.m[k] + (1 - b1) * g[k]
            self.v[k] = b2 * self.v[k] + (1 - b2) * g[k] ** 2
            mh = self.m[k] / (1 - b1 ** self.t); vh = self.v[k] / (1 - b2 ** self.t)
            setattr(self, k, getattr(self, k) - lr * mh / (np.sqrt(vh) + e))


class DQNAgent:
    def __init__(self, p, seed, hp=None):
        self.p = p
        self.hp = dict(HP, **(hp or {}))
        self.rng = np.random.default_rng(seed)
        self.net = MLP(3, self.hp["hidden"], 2, self.rng)
        self.tgt = MLP(3, self.hp["hidden"], 2, self.rng); self.tgt.copy_from(self.net)
        self.seed = seed

    def q(self, X):
        return self.net.forward(np.atleast_2d(X))[0]

    def act_greedy(self, s, j, lh, z=None):
        if s >= self.p["K"]:
            return 0
        qv = self.q(feats(s, j, lh, self.p["K"], self.hp["lam_max"]))[0]
        return int(qv[1] > qv[0])

    def greedy_batch(self, S, J, LH):
        K = self.p["K"]; lm = self.hp["lam_max"]
        X = np.column_stack([np.asarray(S) / K, np.asarray(J, float), np.minimum(LH, lm) / lm])
        qv = self.q(X)
        a = (qv[:, 1] > qv[:, 0]).astype(int)
        a[np.asarray(S) >= K] = 0
        return a

    def train(self, log_every=10_000, eval_fn=None, reset_sampler=None, g_init=None, env_seed=None):
        """reset_sampler(rng) -> (s0, z0) or None: optional restart distribution (used by the
        explanation-guided retraining experiment); g_init: optional initial gain (original units)."""
        hp, p, rng = self.hp, self.p, self.rng
        K = p["K"]; lm = hp["lam_max"]; sc = hp["reward_scale"]
        env = ArrivalEnv(p, seed=(self.seed + 7777) if env_seed is None else env_seed)
        n = hp["buffer"]
        bo = np.zeros((n, 3)); ba = np.zeros(n, int); bB = np.zeros(n); bt = np.zeros(n)
        bo2 = np.zeros((n, 3)); bfull2 = np.zeros(n, bool)
        ptr = size = 0
        G = Tt = 0.0; ghat = 0.0
        if g_init is not None:
            G, Tt, ghat = g_init / sc * 100.0, 100.0, g_init / sc
        o = env.reset()
        log = []
        upd = 0
        for k in range(hp["steps"]):
            if k % hp["reset_every"] == 0 and k > 0:
                if reset_sampler is not None and rng.random() < 0.5:
                    o = env.reset(*reset_sampler(rng))
                else:
                    o = env.reset()
            eps = max(hp["eps_end"], hp["eps0"] - (hp["eps0"] - hp["eps_end"]) * k / (hp["eps_frac"] * hp["steps"]))
            s, j, lh = o
            x = feats(s, j, lh, K, lm)
            if s >= K:
                a = 0
            elif rng.random() < eps:
                a = int(rng.integers(0, 2))
            else:
                qv = self.net.forward(x[None])[0][0]
                a = int(qv[1] > qv[0])
            B, tau, o2 = env.step(a)
            Bs = B / sc
            G = hp["kappa_g"] * G + Bs; Tt = hp["kappa_g"] * Tt + tau; ghat = G / Tt
            bo[ptr] = x; ba[ptr] = a; bB[ptr] = Bs; bt[ptr] = tau
            bo2[ptr] = feats(*o2, K, lm); bfull2[ptr] = o2[0] >= K
            ptr = (ptr + 1) % n; size = min(size + 1, n)
            o = o2
            if size >= hp["warmup"]:
                idx = rng.integers(0, size, hp["batch"])
                q2 = self.tgt.forward(bo2[idx])[0]
                q2[bfull2[idx], 1] = -np.inf
                if hp["double"]:
                    qo = self.net.forward(bo2[idx])[0]
                    qo[bfull2[idx], 1] = -np.inf
                    nxt = q2[np.arange(len(idx)), qo.argmax(1)]
                else:
                    nxt = q2.max(1)
                y = bB[idx] - ghat * bt[idx] + nxt
                qv, cache = self.net.forward(bo[idx])
                pred = qv[np.arange(len(idx)), ba[idx]]
                err = pred - y
                dl = np.clip(err, -hp["huber"], hp["huber"]) / len(idx)
                dout = np.zeros_like(qv); dout[np.arange(len(idx)), ba[idx]] = dl
                lr = hp["lr"] * (1 - (1 - hp["lr_end_frac"]) * k / hp["steps"])
                self.net.step(cache, dout, lr)
                upd += 1
                if upd % hp["target_every"] == 0:
                    self.tgt.copy_from(self.net)
            if (k + 1) % log_every == 0:
                entry = dict(step=k + 1, ghat=ghat * sc, eps=eps)
                if eval_fn is not None:
                    entry.update(eval_fn(self))
                log.append(entry)
        self.log = log
        self.ghat = ghat * sc
        return log
