"""Controllers for the extended model. All map (s, j, lam_hat, z) -> {0,1}; only the
regime-aware oracle uses the hidden z."""
import numpy as np
from extended import oracle_dp, eval_szj, ce_table
from dqn import DQNAgent

LAM_GRID = np.round(np.arange(0.5, 15.01, 0.1), 2)


def szj_from_thresholds(TL, TH, K):
    pol = np.zeros((K + 1, 2, 2), int)
    for s in range(K + 1):
        pol[s, :, 0] = s < TL
        pol[s, :, 1] = s < TH
    return pol


def best_static(p):
    K = p["K"]
    blind = max((eval_szj(szj_from_thresholds(T, T, K), p), T) for T in range(K + 1))
    cls = max((eval_szj(szj_from_thresholds(TL, TH, K), p), TL, TH) for TL in range(K + 1) for TH in range(K + 1))
    return blind, cls


def robust_static(p, models):
    """max over (T_L,T_H) of min over the model set (exact)."""
    K = p["K"]
    best = None
    for TL in range(K + 1):
        for TH in range(TL, K + 1):
            pol = szj_from_thresholds(TL, TH, K)
            worst = min(eval_szj(pol, m) for m in models)
            if best is None or worst > best[0]:
                best = (worst, TL, TH)
    return best


class Controllers:
    def __init__(self, p_train):
        self.p = p_train
        K = p_train["K"]
        self.K = K
        (gb, Tb), (gc, TL, TH) = best_static(p_train)
        self.static_blind = Tb
        self.static_cls = (TL, TH)
        self.ce = ce_table(p_train, LAM_GRID)  # [T_L, T_H] per lam
        self.orc = oracle_dp(p_train)
        self.zcut = 0.5 * (p_train["lams"][0] + p_train["lams"][1])
        self.robust = None

    def set_robust(self, models):
        self.robust = robust_static(self.p, models)

    def fcfs(self, s, j, lh, z):
        return 1

    def blind(self, s, j, lh, z):
        return int(s < self.static_blind)

    def cls(self, s, j, lh, z):
        return int(s < (self.static_cls[1] if j == 1 else self.static_cls[0]))

    def robust_pol(self, s, j, lh, z):
        _, TL, TH = self.robust
        return int(s < (TH if j == 1 else TL))

    def ce_thr(self, lh):
        i = int(np.clip(round((lh - 0.5) / 0.1), 0, len(LAM_GRID) - 1))
        return self.ce[i]

    def ce_dp(self, s, j, lh, z):
        TL, TH = self.ce_thr(lh)
        return int(s < (TH if j == 1 else TL))

    def plugin(self, s, j, lh, z):
        zh = 1 if lh > self.zcut else 0
        return int(self.orc["policy"][s, zh, j])

    def oracle(self, s, j, lh, z):
        return int(self.orc["policy"][s, z, j])


def load_agent(p, hp, weights, seed=0):
    ag = DQNAgent(p, seed=seed, hp=hp)
    for k, v in weights.items():
        setattr(ag.net, k, v.copy())
    return ag
