"""Exp 3a: train DQN (Section 4.2.2) on the extended model, 5 seeds; snapshot weights for learning curves."""
import pickle
import copy
from multiprocessing import Pool
import numpy as np
from extended import BASE
from dqn import DQNAgent

P_EXT = dict(BASE); P_EXT["lams"] = (4.0, 9.0); P_EXT["gam"] = (0.02, 0.02)
HP_MAIN = dict(double=True, reward_scale=50.0, lr=3e-4, batch=128, target_every=1000, lr_end_frac=0.1,
               steps=200_000)
SEEDS = [0, 1, 2, 3, 4]


def snap(agent):
    return {k: getattr(agent.net, k).copy() for k in agent.net.params}


def train(seed):
    ag = DQNAgent(P_EXT, seed=seed, hp=HP_MAIN)
    snaps = []

    def ev(a):
        snaps.append(snap(a))
        return {}
    ag.train(log_every=20_000, eval_fn=ev)
    return dict(seed=seed, log=ag.log, snaps=snaps, final=snap(ag), ghat=ag.ghat, hp=ag.hp)


if __name__ == "__main__":
    with Pool(2) as p:
        res = p.map(train, SEEDS)
    pickle.dump(dict(p=P_EXT, hp=HP_MAIN, runs=res), open("results/exp3_dqn.pkl", "wb"))
    for r in res:
        print(r["seed"], [round(e["ghat"], 1) for e in r["log"]])
