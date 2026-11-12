"""Render Section 6 (LaTeX) from results/numbers.json and the result files. Every number in
section6.tex is produced here; the template is section6_template.tex with <<key>> fields."""
import json
import re
import numpy as np

N = json.load(open("results/numbers.json"))
E1 = json.load(open("results/exp1.json"))
TB = json.load(open("results/table_base.json"))
V = {}


def f(x, d=2):
    return f"{x:.{d}f}"


def pct(x, d=1):
    return f"{100 * x:.{d}f}"


# ---------------- base exact table ----------------
rows = []
for t in TB:
    lam = t["lam"]; e = t["exact"]
    fc, op = e["FCFS"], e["DP"]
    imp = 100 * (op["g"] - fc["g"]) / fc["g"]
    loss = 100 * (op["g"] - fc["g"]) / op["g"]
    rows.append(f"{lam} & {lam/5:.1f} & {op['T']} & {f(fc['PB'],3)} & {f(fc['W'],3)} & {f(fc['U'],3)} & {f(fc['g'],2)} & "
                f"{f(op['PB'],3)} & {f(op['W'],3)} & {f(op['U'],3)} & {f(op['g'],2)} & {f(loss,1)} \\\\")
V["TAB_BASE_ROWS"] = "\n".join(rows)
g7 = {r["policy"]: r["g"] for r in E1["table"] if r["lam"] == 7}
V["G7_FCFS"] = f(g7["FCFS"]); V["G7_OPT"] = f(g7["DP"])
V["IMP7"] = f(100 * (g7["DP"] - g7["FCFS"]) / g7["FCFS"], 1)
V["LOSS7"] = f(100 * (g7["DP"] - g7["FCFS"]) / g7["DP"], 1)
g6 = {r["policy"]: r["g"] for r in E1["table"] if r["lam"] == 6}
g5 = {r["policy"]: r["g"] for r in E1["table"] if r["lam"] == 5}
V["LOSS6"] = f(100 * (g6["DP"] - g6["FCFS"]) / g6["DP"], 1); V["LOSS5"] = f(100 * (g5["DP"] - g5["FCFS"]) / g5["DP"], 1)
st = N["structure"]
V["MIN_D_RANGE"] = f"{min(s['min_d'] for s in st):.3f}"
V["MAX_RVI_EXACT_DIFF"] = f"{max(abs(s['g_rvi'] - s['g_exact_eval']) for s in st):.0e}".replace("e-0", "\\times 10^{-").replace("e-", "\\times 10^{-") + "}"
assert all(s["is_threshold"] and s["nondecreasing"] and s["convex"] and s["T_dp"] == s["s_star_eq23"] for s in st)
V["PROP3_COV"] = f"{N['prop3_covered']}"; V["PROP3_TOT"] = f"{N['prop3_total']}"

# ---------------- Prop 3 table ----------------
rows = []
for r in E1["prop3"]:
    if r["lam"] < 5:
        continue
    lab = "FCFS ($T=K$)" if r["policy"] == "FCFS" else "$T^*$"
    rows.append(f"{r['lam']} & {lab} & {f(r['PB_exact'],4)} & {f(r['PB_sim'],4)} $\\pm$ {f(r['PB_hw'],4)} & "
                f"{f(r['W_exact'],3)} & {f(r['W_sim'],3)} $\\pm$ {f(r['W_hw'],3)} & {f(r['g_exact'],2)} & {f(r['g_sim'],2)} $\\pm$ {f(r['g_hw'],2)} \\\\")
V["TAB_PROP3_ROWS"] = "\n".join(rows)

# ---------------- sensitivity table ----------------
S = E1["sens"]
def cell(vary, value, rho):
    for r in S:
        rv = tuple(r["value"]) if isinstance(r["value"], list) else r["value"]
        if r["vary"] == vary and r["rho"] == rho and rv == value:
            return f"{r['T_star']}; {r['g_opt']-r['g_fcfs']:.2f}"
    return "--"
rows = ["\\multicolumn{5}{l}{\\emph{Reference setting} ($c=5$, $K=20$, $R=10$, $P=5$, $h=1$)}\\\\",
        "Reference & " + " & ".join(cell("h", 1.0, rho) for rho in [0.8, 1.0, 1.2, 1.4]) + " \\\\",
        "\\multicolumn{5}{l}{\\emph{Holding cost} $h$}\\\\"]
for v in [0.25, 0.5, 2.0, 4.0]:
    rows.append(f"$h={v:g}$ & " + " & ".join(cell("h", v, rho) for rho in [0.8, 1.0, 1.2, 1.4]) + " \\\\")
rows.append("\\multicolumn{5}{l}{\\emph{Admission revenue} $R$}\\\\")
for v in [5.0, 20.0, 40.0]:
    rows.append(f"$R={v:g}$ & " + " & ".join(cell("R", v, rho) for rho in [0.8, 1.0, 1.2, 1.4]) + " \\\\")
rows.append("\\multicolumn{5}{l}{\\emph{Blocking penalty} $P$}\\\\")
for v in [0.0, 2.5, 10.0, 20.0]:
    rows.append(f"$P={v:g}$ & " + " & ".join(cell("P", v, rho) for rho in [0.8, 1.0, 1.2, 1.4]) + " \\\\")
rows.append("\\multicolumn{5}{l}{\\emph{System size} $(c,K)$}\\\\")
for v in [(2, 8), (5, 10), (5, 40), (10, 40), (20, 80)]:
    rows.append(f"$({v[0]},{v[1]})$ & " + " & ".join(cell("cK", v, rho) for rho in [0.8, 1.0, 1.2, 1.4]) + " \\\\")
V["TAB_SENS_ROWS"] = "\n".join(rows)

# ---------------- RVI-Q table ----------------
rows = []
for t in TB:
    a, l = t["rvi"], t["legacy"]
    rows.append(f"{t['lam']} & {t['exact']['DP']['T']} & {int(a['T_med'])} [{a['T_min']},{a['T_max']}] & {a['exact_hit']}/10 & "
                f"{f(a['gap_mean'],2)} & {f(a['gap_max'],2)} & {f(a['ghat'],2)} ({f(a['ghat_sd'],2)}) & "
                f"{int(l['T_med'])} [{l['T_min']},{l['T_max']}] & {f(l['gap_mean'],2)} & {f(l['gap_max'],2)} \\\\")
V["TAB_RL_ROWS"] = "\n".join(rows)
tb = {t["lam"]: t for t in TB}
V["RVI_MEAN_MAX"] = f(max(tb[l]["rvi"]["gap_mean"] for l in tb), 2)
V["RVI_WORST"] = f(max(tb[l]["rvi"]["gap_max"] for l in tb), 2)
V["LEG_MEAN6"] = f(tb[6]["legacy"]["gap_mean"], 1); V["LEG_MEAN7"] = f(tb[7]["legacy"]["gap_mean"], 1)
V["LEG_MAX7"] = f(tb[7]["legacy"]["gap_max"], 1)
V["FLAT5"] = "\\{" + ",".join(map(str, N["flat_1pct"]["5"])) + "\\}"
V["FLAT7"] = "\\{" + ",".join(map(str, N["flat_1pct"]["7"])) + "\\}"
V["VIS_LEG_MIN"] = f"{N['visits_legacy_9_19_min']:,}".replace(",", "{,}")
V["VIS_LEG_MEAN"] = f"{N['visits_legacy_9_19_mean']:,.0f}".replace(",", "{,}")
V["VIS_MAIN_MIN"] = f"{N['visits_main_9_19_min']:,}".replace(",", "{,}")
V["LEG_STEP_MIN7"] = f"{N['legacy_step_of_min_7']/1e3:.0f}"
se = N["sens"]
for lam in ["5", "7"]:
    for k in ["base", "k055", "k07", "k08", "k10", "e001", "e02", "empty", "q100", "x5"]:
        V[f"S{lam}_{k}_mean"] = f(se[lam][k]["mean"], 2); V[f"S{lam}_{k}_max"] = f(se[lam][k]["max"], 2)
V["MONO_MAIN7"] = str(N["monotone"]["main"]["7"]); V["MONO_MAIN6"] = str(N["monotone"]["main"]["6"])
V["MONO_MAIN5"] = str(N["monotone"]["main"]["5"])

# ---------------- extended: value-of-information table ----------------
rows = []
for s in N["voi_scan"]:
    star = "$^\\dagger$" if tuple(s["lams"]) == (4.0, 9.0) and s["gam"] == 0.02 else ""
    rows.append(f"$({s['lams'][0]:g},{s['lams'][1]:g})$ & {s['gam']:g}{star} & {f(s['oracle'])} & {f(s['cls'])} & "
                f"$({s['TL']},{s['TH']})$ & {f(s['gap_cls'],1)} & {f(s['gap_blind'],1)} & {f(s['gap_fcfs'],1)} \\\\")
V["TAB_VOI_ROWS"] = "\n".join(rows)
V["VOI_MAX"] = f(max(s["gap_cls"] for s in N["voi_scan"]), 1)
V["VOI_MIN"] = f(min(s["gap_cls"] for s in N["voi_scan"]), 1)
voi_main = [s for s in N["voi_scan"] if tuple(s["lams"]) == (4.0, 9.0) and s["gam"] == 0.02][0]
V["VOI_FCFS_MIN"] = f(min(s["gap_fcfs"] for s in N["voi_scan"]), 0); V["VOI_FCFS_MAX"] = f(max(s["gap_fcfs"] for s in N["voi_scan"]), 0)
V["VOI_MAIN"] = f(voi_main["gap_cls"], 1)
ot = N["oracle_T"]
V["OT_L_LOW"] = str(ot["z0_j0"]); V["OT_L_HIGH"] = str(ot["z1_j0"]); V["OT_H"] = str(ot["z0_j1"])
V["STATIC_BLIND"] = str(N["static_blind"]); V["STATIC_CLS"] = f"({N['static_cls'][0]},{N['static_cls'][1]})"
V["ROBUST"] = f"({N['robust'][1]},{N['robust'][2]})"

# ---------------- extended: scenario table ----------------
sc_keys = ["S0_train", "S1_fast_switch", "S2_demand_shock", "S3_service_degr", "S4_server_fail"]
names = [("fcfs", "FCFS"), ("blind", "Best static, class-blind ($T=" + V["STATIC_BLIND"] + "$)"),
         ("cls", "Best static, class-specific $(T_L,T_H)=" + V["STATIC_CLS"] + "$"),
         ("robust_pol", "Robust static $(T_L,T_H)=" + V["ROBUST"] + "$"),
         ("ce_dp", "Certainty-equivalent DP (re-solved at $\\hat\\lambda_n$)"),
         ("plugin", "Plug-in regime controller"), ("dqn", "DQN (5 seeds)"),
         ("oracle", "Regime-aware oracle, $S_0$ model (observes $Z_t$)")]
rows = []
for key, lab in names:
    cells = []
    for sc in sc_keys:
        r = N["scen"][sc][key]
        impl = ["cls", "robust_pol", "ce_dp", "plugin", "dqn", "blind", "fcfs"]
        bestv = max(N["scen"][sc][k]["m"] for k in impl)
        txt = f"{f(r['m'])} $\\pm$ {f(r['hw'])}"
        if key in impl and r["m"] >= bestv - 1e-9:
            txt = "\\textbf{" + f(r['m']) + "} $\\pm$ " + f(r['hw'])
        cells.append(txt)
    if key == "oracle":
        rows.append("\\midrule")
    rows.append(lab + " & " + " & ".join(cells) + " \\\\")
rows.append("Re-solved optimum for the shifted model (exact) & " + " & ".join(
    f(N["scen"][sc]["exact"]["oracle_resolved"]) if N["scen"][sc]["exact"] else "--" for sc in sc_keys) + " \\\\")
V["TAB_SCEN_ROWS"] = "\n".join(rows)
for i, sc in enumerate(sc_keys):
    for pol in ["cls", "ce_dp", "plugin"]:
        d = N["scen"][sc][f"dqn_minus_{pol}"]
        V[f"D{i}_{pol}"] = f"{d['m']:+.2f}"; V[f"D{i}_{pol}_hw"] = f(d["hw"])
    V[f"DQN{i}"] = f(N["scen"][sc]["dqn"]["m"]); V[f"DQN{i}_min"] = f(N["scen"][sc]["dqn"]["min"]); V[f"DQN{i}_max"] = f(N["scen"][sc]["dqn"]["max"])
    for pol in ["cls", "ce_dp", "plugin", "oracle", "fcfs", "robust_pol"]:
        V[f"G{i}_{pol}"] = f(N["scen"][sc][pol]["m"])
V["ORC0"] = f(N["scen"]["S0_train"]["exact"]["oracle_resolved"])
V["CLS0_EXACT"] = f(N["scen"]["S0_train"]["exact"]["cls"])
V["DQN_CURVE_FINAL"] = f(N["dqn_curve_final"]["mean"]); V["DQN_CURVE_100K"] = f(N["dqn_curve_100k"]["mean"])
V["DQN_CURVE_100K_MIN"] = f(N["dqn_curve_100k"]["min"])

# ---------------- explanation table ----------------
rows = []
lab = {"tree1": "Tree, depth 1", "tree2": "Tree, depth 2", "tree3": "Tree, depth 3", "tree4": "Tree, depth 4",
       "tree5": "Tree, depth 5", "logit": "Logistic regression"}
def a3(d):
    return f"{f(d['mean'],3)} [{f(d['min'],3)}, {f(d['max'],3)}]"
for k in ["tree1", "tree2", "tree3", "tree4", "tree5", "logit"]:
    F = N["fid"][k]
    rows.append(f"{lab[k]} & {f(F['train']['mean'],3)} & {a3(F['test'])} & {a3(F['near_boundary'])} & {f(F['z1_j0']['mean'],3)} & {a3(F['grid'])} \\\\")
V["TAB_FID_ROWS"] = "\n".join(rows)
V["FID4"] = f(N["fid"]["tree4"]["test"]["mean"], 3); V["FID4NB"] = f(N["fid"]["tree4"]["near_boundary"]["mean"], 3)
V["FRAC_NEAR"] = pct(N["frac_near"]["mean"], 0); V["ADMIT_RATE"] = pct(N["admit_rate"]["mean"], 0)
V["NTEST"] = f"{round(N['n_test']['mean'], -3):,.0f}".replace(",", "{,}"); V["NFIT"] = f"{round(N['n_fit']['mean'], -3):,.0f}".replace(",", "{,}")
pi = N["pi"]
V["PI_S"] = f(pi["occupancy s"]["mean"], 3); V["PI_S_R"] = f"{f(pi['occupancy s']['min'],3)}--{f(pi['occupancy s']['max'],3)}"
V["PI_J"] = f(pi["class j"]["mean"], 3); V["PI_J_R"] = f"{f(pi['class j']['min'],3)}--{f(pi['class j']['max'],3)}"
V["PI_L"] = f(pi["regime estimate lam_hat"]["mean"], 3); V["PI_L_R"] = f"{f(pi['regime estimate lam_hat']['min'],3)}--{f(pi['regime estimate lam_hat']['max'],3)}"
cf = N["cf"]
V["CF_S_EXISTS"] = pct(cf["agent_s_exists"]["mean"], 0); V["CF_S_MED"] = f(cf["agent_s_med"]["mean"], 1)
V["CF_S_LE1"] = pct(cf["agent_s_le1"]["mean"], 0)
V["CF_L_EXISTS"] = pct(cf["agent_l_exists"]["mean"], 0); V["CF_L_MED"] = f(cf["agent_l_med"]["mean"], 1)
V["CF_J"] = pct(cf["agent_j_flip"]["mean"], 0)
V["CF_VAL_S"] = pct(cf["sur_valid_s"]["mean"], 0); V["CF_VAL_S_R"] = f"{pct(cf['sur_valid_s']['min'],0)}--{pct(cf['sur_valid_s']['max'],0)}"
V["CF_VAL_L"] = pct(cf["sur_valid_l"]["mean"], 0); V["CF_VAL_L_R"] = f"{pct(cf['sur_valid_l']['min'],0)}--{pct(cf['sur_valid_l']['max'],0)}"
V["CF_VAL_J"] = pct(cf["sur_valid_j"]["mean"], 0); V["CF_VAL_J_R"] = f"{pct(cf['sur_valid_j']['min'],0)}--{pct(cf['sur_valid_j']['max'],0)}"
V["CF_EXACT_S"] = pct(cf["sur_exact_s"]["mean"], 0)
ex = N["cf_examples"][0]
V["EX_S"] = str(ex["s"]); V["EX_L"] = f(ex["lh"], 2); V["EX_DS"] = f"{int(ex['ds'])}" if ex["ds"] is not None else "--"
V["EX_DL"] = f(ex["lh"] + ex["dl"], 2) if ex["dl"] is not None else "--"
V["PROF_AGREE"] = pct(N["profile_agree"]["mean"], 0); V["PROF_AGREE_MIN"] = pct(N["profile_agree"]["min"], 0)
V["PROF_ABS"] = f(N["profile_absdiff"]["mean"], 2); V["PROF_ABS_MAX"] = f(N["profile_absdiff"]["max"], 2)
st = N["stability"]
V["STAB_S"] = pct(st["stab_surr"], 1); V["STAB_S_MIN"] = pct(st["stab_surr_min"], 1)
V["STAB_A"] = pct(st["stab_agent"], 1); V["STAB_A_MIN"] = pct(st["stab_agent_min"], 1)
V["THR_SD_A"] = f(st["thr_sd_agent"], 2); V["THR_SD_S"] = f(st["thr_sd_surr"], 2)
V["THR_SD_LOWBIN"] = f(st["thr_sd_agent_cells"][0][0], 1)
Ta = np.array(N["T_agent"])
V["TA_LOW"] = f(Ta[:, 0, 0].mean(), 1); V["TA_HIGH"] = f(Ta[:, 0, -1].mean(), 1)
V["TA_LOW_R"] = f"{Ta[:, 0, 0].min()}--{Ta[:, 0, 0].max()}"
V["TA_H_MIN"] = str(int(Ta[:, 1, :].min()))

# ---------------- patch / retraining table ----------------
rows = []
P = N["patch"]; R = N["retrain"]
for s in sorted(P, key=int):
    p = P[s]; r = R[s]
    rec = "--" if p["recovered"] is None else f(100 * p["recovered"], 0)
    rows.append(f"{s} & {f(p['gap'])} & {len(p['flagged'])} ({pct(p['flagged_w'],0)}\\%) & {p['patch_gain']:+.2f} $\\pm$ {f(p['patch_gain_hw'])} & "
                f"{rec} & {p['rand_gain']:+.2f} $\\pm$ {f(p['rand_gain_hw'])} & {r['guided_minus_control']:+.2f} $\\pm$ {f(r['gmc_hw'])} \\\\")
V["TAB_PATCH_ROWS"] = "\n".join(rows)
V["PATCH_POOL"] = pct(N["patch_pooled_recovered"], 0); V["RAND_POOL"] = pct(N["rand_pooled_recovered"], 0)
V["G_REF"] = f(P["0"]["g_ref"])
V["RT_GMC"] = f"{N['retrain_mean']['gmc']:+.2f}"
gm = [R[s]["guided_minus_control"] for s in R]
V["RT_WORST"] = f"{min(gm):+.2f}"
V["RT_POS"] = str(sum(1 for s in R if R[s]["guided_minus_control"] - R[s]["gmc_hw"] > 0))
V["RT_NEG"] = str(sum(1 for s in R if R[s]["guided_minus_control"] + R[s]["gmc_hw"] < 0))

tpl = open("section6_template.tex").read()
missing = sorted(set(re.findall(r"<<([A-Za-z0-9_]+)>>", tpl)) - set(V))
assert not missing, missing
out = re.sub(r"<<([A-Za-z0-9_]+)>>", lambda m: V[m.group(1)], tpl)
open("section6.tex", "w").write(out)
json.dump(V, open("results/rendered_values.json", "w"), indent=1)
print("rendered", len(V), "fields")
