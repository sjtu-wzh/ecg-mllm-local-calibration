"""Pre-registered analysis: is the local-calibration gain an in-domain vs out-of-domain effect?
Plan: outputs/clarity/preregistration_domain.json (saved before PTB-XL B0 was complete).
Implements P0-P4 (confirmatory) and S9/S10 (sensitivity). Other secondary analyses (S1-S8) are not implemented here.

Arms (same CPSC 9-label prompt and the same B1 Yes/No probe in both):
  CPSC : 2061 records, out-of-domain for PULSE.
  PTB  : 1357 PTB-XL fold-10 records with >= 1 mapped CPSC label, in-domain image source (patient-grouped).
Estimand G_a(n) = macroF1(S1@n) - macroF1(B0) over L8 = 9 labels minus STE, on the same test records.
"""
import ast, csv, json, os, re, runpy, sys
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LABELS = ["NORM", "AF", "I-AVB", "LBBB", "RBBB", "PAC", "PVC", "STD", "STE"]
L8 = [l for l in LABELS if l != "STE"]
SIZES = [25, 50, 100, 200, 300, 400, 600]
R, B = int(os.environ.get("DOMAIN_R", 20)), int(os.environ.get("DOMAIN_B", 200))  # env overrides only for dry runs
PTB_B0 = os.environ.get("DOMAIN_PTB_B0", "outputs/pulse7b/ptbxl-cpsc9/answers.jsonl")
GRID = np.round(np.arange(0.02, 0.97, 0.01), 2)
SESOI = 3.0
OOV = ["MI", "CD", "HYP", "STTC", "STC"]
J = lambda p: json.load(open(os.path.join(ROOT, p)))
compute_f1_auc = runpy.run_path(os.path.join(ROOT, "scripts/score_ecgbench.py"), run_name="lib")["compute_f1_auc"]


def jsonl(p):
    return [json.loads(l) for l in open(os.path.join(ROOT, p)) if l.strip()]


def parse(text):  # identical to clarity/learning_curve.py (substring match, no split on ';')
    return [L for L in LABELS if L in text]


# ------------------------------------------------------------------ data
def load_cpsc():
    data = J("data/ECGBench/cpsc-test.json")
    ids = [x["id"] for x in data]
    gold = {x["id"]: parse(x["conversations"][1]["value"]) for x in data}
    P = {r["id"]: r["probs"] for r in jsonl("outputs/clarity/b1_cpsc_dev.jsonl") + jsonl("outputs/clarity/b1_cpsc_test.jsonl")}
    b0 = {r["question_id"]: r["text"] for r in jsonl("outputs/pulse7b/cpsc-test/answers.jsonl")}
    return {"name": "CPSC", "ids": ids, "gold": gold, "P": P, "b0text": b0, "group": {i: i for i in ids}, "scp": None}


def load_ptb():
    data = J("data/derived/ptbxl_cpsc9.json")
    ids = [x["id"] for x in data]
    gold = {x["id"]: parse(x["conversations"][1]["value"]) for x in data}
    P = {r["id"]: r["probs"] for r in jsonl("outputs/clarity/b1_ptbxl_all.jsonl")}
    b0 = {r["question_id"]: r["text"] for r in jsonl(PTB_B0)}
    db = {int(r["ecg_id"]): r for r in csv.DictReader(open(os.path.join(ROOT, "data/ptbxl_meta/ptbxl_database.csv")))}
    eid = {i: int(re.match(r"ptb-xl-(\d+)", i).group(1)) for i in ids}
    group = {i: str(db[eid[i]]["patient_id"]) for i in ids}
    scp = {i: ast.literal_eval(db[eid[i]]["scp_codes"]) for i in ids}
    return {"name": "PTB", "ids": ids, "gold": gold, "P": P, "b0text": b0, "group": group, "scp": scp}


def arrays(arm, ids=None, gold=None):
    ids = ids or arm["ids"]; gold = gold or arm["gold"]
    lab_idx = {l: k for k, l in enumerate(LABELS)}
    G = np.zeros((len(ids), 9), bool); B0 = np.zeros((len(ids), 9), bool)
    for n, i in enumerate(ids):
        for l in gold[i]: G[n, lab_idx[l]] = True
        for l in parse(arm["b0text"][i]): B0[n, lab_idx[l]] = True
    p = np.clip(np.array([[arm["P"][i][l] for l in LABELS] for i in ids], float), 1e-4, 1 - 1e-4)
    return {"ids": list(ids), "G": G, "B0": B0, "p": p, "X": np.log(p / (1 - p)),
            "grp": np.array([arm["group"][i] for i in ids])}


# ------------------------------------------------------------------ metric
def macro_f1(G, Pr, w, cols):
    """Weighted macro-F1 over label columns `cols`: mean_L 2TP/(2TP+FP+FN)."""
    w = w[:, None]
    tp = (w * (G & Pr)).sum(0); fp = (w * (~G & Pr)).sum(0); fn = (w * (G & ~Pr)).sum(0)
    den = 2 * tp + fp + fn
    f = np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)
    return float(f[cols].mean()), f


def wf1_grid(y, s, w):
    """Weighted F1 for every threshold in GRID (vectorised)."""
    pr = s[None, :] >= GRID[:, None]
    tp = (pr & y[None, :]) @ w; fp = (pr & ~y[None, :]) @ w; fn = ((~pr) & y[None, :]) @ w
    den = 2 * tp + fp + fn
    return np.where(den > 0, 2 * tp / np.where(den > 0, den, 1), 0.0)


# ------------------------------------------------------------------ calibrator (S1)
def fit_s1(A, tr, te, w, seed):
    """Per label LR on 9 logits; OOF thresholds (StratifiedGroupKFold); fallback raw P>=0.5.
    Returns test scores (n_te x 9) and thresholds (9,)."""
    X, Gm, grp = A["X"], A["G"], A["grp"]
    S = np.zeros((len(te), 9)); T = np.zeros(9)
    for L in range(9):
        y = Gm[tr, L]; pos, neg = int(y.sum()), int((~y).sum())
        if pos < 2 or neg < 2:
            S[:, L] = A["p"][te, L]; T[L] = 0.5; continue
        k = min(5, pos, neg)
        oof = np.zeros(len(tr))
        cv = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
        for a, b in cv.split(X[tr], y, grp[tr]):
            if y[a].all() or (~y[a]).all():
                oof[b] = y[a].mean(); continue
            m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr][a], y[a], sample_weight=w[tr][a])
            oof[b] = m.predict_proba(X[tr][b])[:, 1]
        T[L] = float(GRID[int(np.argmax(wf1_grid(y, oof, w[tr])))])
        m = LogisticRegression(C=1.0, max_iter=2000).fit(X[tr], y, sample_weight=w[tr])
        S[:, L] = m.predict_proba(X[te])[:, 1]
    return S, T


def decide(S, T, at_least_one=True):
    Pr = S >= T[None, :]
    if at_least_one:
        empty = ~Pr.any(1)
        Pr[empty, np.argmax(S[empty] / T[None, :], 1)] = True
    return Pr


# ------------------------------------------------------------------ splits
def partitions(A, scored_cols):
    """20 outer partitions: pool ~600 / test rest, grouped; redraw until every scored label has >=10 positives in both."""
    out, grp = [], A["grp"]
    ug = np.unique(grp)
    for r in range(R):
        k = 0
        while True:
            rng = np.random.default_rng(r * 1000 + k)
            order = rng.permutation(ug)
            pool_g, cnt = set(), 0
            sizes = {g: int((grp == g).sum()) for g in order} if len(ug) < len(grp) else None
            for g in order:
                if cnt >= 600: break
                pool_g.add(g); cnt += sizes[g] if sizes else 1
            pool = np.where(np.isin(grp, list(pool_g)))[0]; test = np.where(~np.isin(grp, list(pool_g)))[0]
            ok = all(A["G"][pool, c].sum() >= 10 and A["G"][test, c].sum() >= 10 for c in scored_cols)
            if ok: break
            k += 1
        out.append((pool, test, k))
    return out


def cluster_index(grp_test):
    u, inv = np.unique(grp_test, return_inverse=True)
    members = [np.where(inv == c)[0] for c in range(len(u))]
    return members


def boot_diff(Gt, PrA, PrB, w, cols, members, rng):
    """Paired cluster bootstrap of macroF1(PrA)-macroF1(PrB). Returns B replicates (overall) and per-label (B x 9)."""
    reps, per = np.zeros(B), np.zeros((B, 9))
    nc = len(members)
    for b in range(B):
        idx = np.concatenate([members[c] for c in rng.integers(0, nc, nc)])
        fa, la = macro_f1(Gt[idx], PrA[idx], w[idx], cols)
        fb, lb = macro_f1(Gt[idx], PrB[idx], w[idx], cols)
        reps[b] = fa - fb; per[b] = la - lb
    return reps, per


# ------------------------------------------------------------------ one arm, full protocol
def run_arm(A, w, cols, parts, sizes=SIZES, tag="", decompose=False):
    res = {n: {"G": [], "reps": [], "per_label_point": [], "per_label_reps": []} for n in sizes}
    comp = {k: [] for k in ("a", "c", "b", "e", "G_L", "drank_S1", "drank_own", "r_L", "auc_own")} if decompose else None
    for r, (pool, test, _) in enumerate(parts):
        members = cluster_index(A["grp"][test])
        Gt, B0t, wt = A["G"][test], A["B0"][test], w[test]
        for n in sizes:
            rng = np.random.default_rng(100 * r + n)
            tr = pool if n >= len(pool) else np.sort(rng.choice(pool, size=n, replace=False))
            S, T = fit_s1(A, tr, test, w, seed=r)
            Pr = decide(S, T)
            f1s, lab_s = macro_f1(Gt, Pr, wt, cols); f1b, lab_b = macro_f1(Gt, B0t, wt, cols)
            reps, per = boot_diff(Gt, Pr, B0t, wt, cols, members, np.random.default_rng(10_000 * r + n))
            d = res[n]; d["G"].append(f1s - f1b); d["reps"].append(reps)
            d["per_label_point"].append(lab_s - lab_b); d["per_label_reps"].append(per)
            if decompose and n == sizes[-1]:
                own, thr_only = A["p"][test], decide(S, T, at_least_one=False)
                for key in comp: comp[key].append(np.full(9, np.nan))
                for L in cols:
                    kL = int(B0t[:, L].sum())
                    def topk(s):
                        o = np.zeros(len(s), bool)
                        if kL > 0: o[np.argsort(-s, kind="stable")[:kL]] = True
                        return o
                    f = lambda pr: macro_f1(Gt[:, [L]], pr[:, None], wt, [0])[0]
                    F_b0, F_own, F_s1k = f(B0t[:, L]), f(topk(own[:, L])), f(topk(S[:, L]))
                    F_thr, F_full = f(thr_only[:, L]), f(Pr[:, L])
                    comp["a"][-1][L] = F_own - F_b0; comp["c"][-1][L] = F_s1k - F_own
                    comp["b"][-1][L] = F_thr - F_s1k; comp["e"][-1][L] = F_full - F_thr
                    comp["G_L"][-1][L] = F_full - F_b0
                    y = Gt[:, L]
                    if y.any() and (~y).any():
                        neg_w, pos_w = wt[~y].sum(), wt[y].sum()
                        fpr_b0 = (wt * (B0t[:, L] & ~y)).sum() / neg_w; tpr_b0 = (wt * (B0t[:, L] & y)).sum() / pos_w
                        for key, s in (("drank_S1", S[:, L]), ("drank_own", own[:, L])):
                            fp_, tp_, _ = roc_curve(y, s, sample_weight=wt)
                            comp[key][-1][L] = float(np.interp(fpr_b0, fp_, tp_) - tpr_b0)
                        rate, prev = (wt * B0t[:, L]).sum() / wt.sum(), (wt * y).sum() / wt.sum()
                        comp["r_L"][-1][L] = float(np.log(max(rate, 1e-6) / prev))
                        comp["auc_own"][-1][L] = roc_auc_score(y, own[:, L], sample_weight=wt)
        print(f"  {A.get('name', '')}{tag} partition {r + 1}/{len(parts)}", flush=True)
    summ = {}
    for n in sizes:
        d = res[n]; pooled = np.concatenate(d["reps"])
        point = np.array(d["G"]) * 100
        summ[n] = {"G_mean": point.mean(), "G_sd_partitions": point.std(),
                   "G_p2.5_partitions": np.percentile(point, 2.5), "G_p97.5_partitions": np.percentile(point, 97.5),
                   "CI_low": np.percentile(pooled, 2.5) * 100, "CI_high": np.percentile(pooled, 97.5) * 100,
                   "pooled_reps": pooled * 100,
                   "per_label_mean": np.nanmean(np.array(d["per_label_point"]), 0) * 100,
                   "per_label_reps": np.concatenate(d["per_label_reps"]) * 100}
    out = {"summary": summ}
    if decompose:
        out["components"] = {k: np.nanmean(np.array(v), 0) * (1 if k in ("drank_S1", "drank_own", "r_L", "auc_own") else 100)
                             for k, v in comp.items()}
        out["components_partitions"] = {k: np.array(v) for k, v in comp.items()}
    return out


# ------------------------------------------------------------------ weights (P2)
def setkey(g):
    return tuple(sorted(g))


def casemix_weights(src_gold, src_ids, tgt_gold, tgt_ids):
    """w_i for records of `src` so that its label-set mix matches `tgt`; 0 for sets absent from tgt."""
    from collections import Counter
    cs = Counter(setkey(src_gold[i]) for i in src_ids); ct = Counter(setkey(tgt_gold[i]) for i in tgt_ids)
    return np.array([(ct[setkey(src_gold[i])] / len(tgt_ids)) / (cs[setkey(src_gold[i])] / len(src_ids))
                     if ct[setkey(src_gold[i])] else 0.0 for i in src_ids])


def ess(w):
    return float(w.sum() ** 2 / (w ** 2).sum()) if (w > 0).any() else 0.0


def ci(x):
    return [float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))]


def paired(a, b, f, seed):
    rng = np.random.default_rng(seed)
    return f(rng.permutation(a), rng.permutation(b))


# ================================================================== main
if __name__ == "__main__":
    report = {"plan": "outputs/clarity/preregistration_domain.json", "notes": []}
    cpsc, ptb = load_cpsc(), load_ptb()

    # ---------------- P0 prerequisites
    p0 = {}
    b0_ids = [r["question_id"] for r in jsonl(PTB_B0)]
    report["ptb_b0_file"] = PTB_B0; report["R"], report["B"] = R, B
    p0["ptb_b0_rows"], p0["ptb_b0_unique"] = len(b0_ids), len(set(b0_ids))
    missing = sorted(set(ptb["ids"]) - set(b0_ids))
    p0["ptb_b0_missing"] = missing[:20]
    if missing or len(set(b0_ids)) != 1357:
        print("P0 FAIL: PTB B0 incomplete", p0); json.dump(p0, open(os.path.join(ROOT, "outputs/clarity/domain_P0_fail.json"), "w")); sys.exit(1)
    Ac, Ap = arrays(cpsc), arrays(ptb)
    Ac["name"], Ap["name"] = "CPSC", "PTB"
    ones_c = np.ones(len(Ac["ids"]))
    mine9, _ = macro_f1(Ac["G"], Ac["B0"], ones_c, list(range(9)))
    ref9 = compute_f1_auc([parse(cpsc["b0text"][i]) for i in Ac["ids"]], [cpsc["gold"][i] for i in Ac["ids"]])[0]
    l8 = lambda L: [x for x in L if x != "STE"]
    mine8, _ = macro_f1(Ac["G"], Ac["B0"], ones_c, list(range(8)))
    ref8 = compute_f1_auc([l8(parse(cpsc["b0text"][i])) for i in Ac["ids"]], [l8(cpsc["gold"][i]) for i in Ac["ids"]])[0]
    p0["metric_check_9"] = {"mine": mine9, "official": ref9, "absdiff": abs(mine9 - ref9)}
    p0["metric_check_L8"] = {"mine": mine8, "official": ref8, "absdiff": abs(mine8 - ref8)}
    if abs(mine9 - ref9) > 1e-9 or abs(mine8 - ref8) > 1e-9:
        print("P0 FAIL: metric mismatch", p0); sys.exit(1)
    for A, arm in ((Ac, cpsc), (Ap, ptb)):
        txt = [arm["b0text"][i] for i in A["ids"]]
        p0[f"{A['name']}_b0_empty_parse"] = int((~A["B0"].any(1)).sum())
        p0[f"{A['name']}_b0_oov"] = {t: int(sum(bool(re.search(rf"\b{t}\b", x)) for x in txt)) for t in OOV}
    p0["fallback"] = "raw P(yes) >= 0.5 in both arms (ZS_THR not used)"
    report["P0"] = p0
    print("P0 passed:", json.dumps(p0)[:600], flush=True)

    COLS8 = list(range(8))
    parts_c, parts_p = partitions(Ac, COLS8), partitions(Ap, COLS8)
    report["redraws"] = {"CPSC": [k for *_, k in parts_c], "PTB": [k for *_, k in parts_p]}

    # ---------------- P1 raw arms (+ P3 decomposition at n=600)
    w_c, w_p = np.ones(len(Ac["ids"])), np.ones(len(Ap["ids"]))
    print("P1 CPSC ...", flush=True); RC = run_arm(Ac, w_c, COLS8, parts_c, decompose=True)
    print("P1 PTB raw ...", flush=True); RP = run_arm(Ap, w_p, COLS8, parts_p, decompose=True)

    # ---------------- P2 case-mix standardisation
    w_std = casemix_weights(ptb["gold"], Ap["ids"], cpsc["gold"], Ac["ids"])
    w_rev = casemix_weights(cpsc["gold"], Ac["ids"], ptb["gold"], Ap["ids"])
    print("P2 PTB standardised ...", flush=True); RS = run_arm(Ap, w_std, COLS8, parts_p, tag=" std", decompose=True)
    print("P2 CPSC -> PTB mix ...", flush=True); RV = run_arm(Ac, w_rev, COLS8, parts_c, sizes=[600], tag=" rev")

    def G(Rx, n=600): return Rx["summary"][n]
    did = paired(G(RC)["pooled_reps"], G(RP)["pooled_reps"], lambda a, b: a - b, 1)
    rstd = paired(G(RS)["pooled_reps"], G(RC)["pooled_reps"], lambda a, b: a / np.where(np.abs(b) > 1e-9, b, np.nan), 2)
    casemix_part = paired(G(RS)["pooled_reps"], G(RP)["pooled_reps"], lambda a, b: a - b, 3)
    nstar = {}
    for nm, Rx in (("CPSC", RC), ("PTB_raw", RP), ("PTB_std", RS)):
        nstar[nm] = next((n for n in SIZES if np.percentile(Rx["summary"][n]["pooled_reps"], 2.5) > 0), None)
    test_ess = [ess(w_std[test]) for _, test, _ in parts_p]

    P1 = {"G_CPSC_600": [G(RC)["G_mean"], G(RC)["CI_low"], G(RC)["CI_high"]],
          "G_PTBraw_600": [G(RP)["G_mean"], G(RP)["CI_low"], G(RP)["CI_high"]],
          "DiD_600": [float(np.nanmean(did)), *ci(did)], "n_star": nstar,
          "curves": {nm: {n: [Rx["summary"][n]["G_mean"], Rx["summary"][n]["CI_low"], Rx["summary"][n]["CI_high"],
                              Rx["summary"][n]["G_p2.5_partitions"], Rx["summary"][n]["G_p97.5_partitions"]] for n in SIZES}
                     for nm, Rx in (("CPSC", RC), ("PTB_raw", RP), ("PTB_std", RS))}}
    lo_raw, hi_raw = G(RP)["CI_low"], G(RP)["CI_high"]
    if lo_raw >= SESOI:
        P1["verdict"] = "G_PTB,raw lower CI >= 3: in-domain image with the same prompt still gains -> H_dom cannot be sole cause (supports H_task and/or H_label)"
    elif hi_raw < SESOI and ci(did)[0] > SESOI:
        P1["verdict"] = "G_PTB,raw upper CI < 3 and DiD lower CI > 3 -> go to P2 (H_dom vs H_prev)"
    else:
        P1["verdict"] = "neither P1 condition met; interpret via P2"
    report["P1"] = P1

    rs_mean, (rs_lo, rs_hi) = float(np.nanmean(rstd)), ci(rstd[~np.isnan(rstd)])
    rev_ratio = G(RV)["G_mean"] / G(RC)["G_mean"]
    P2 = {"ESS_overall": ess(w_std), "ESS_test_min": min(test_ess), "ESS_test_mean": float(np.mean(test_ess)),
          "kept_records": int((w_std > 0).sum()), "ESS_reverse": ess(w_rev),
          "G_PTBstd_600": [G(RS)["G_mean"], G(RS)["CI_low"], G(RS)["CI_high"]],
          "casemix_part_600": [float(np.mean(casemix_part)), *ci(casemix_part)],
          "residual_part_600": [G(RC)["G_mean"] - G(RS)["G_mean"]],
          "R_std": [rs_mean, rs_lo, rs_hi], "G_CPSC_to_PTBmix_600": [G(RV)["G_mean"], G(RV)["CI_low"], G(RV)["CI_high"]],
          "reverse_ratio": rev_ratio}
    if min(test_ess) < 100 or (rs_hi - rs_lo) > 0.8:
        P2["verdict"] = "INCONCLUSIVE (test-fold ESS < 100 or R_std CI width > 0.8)"
    elif rs_mean <= 0.3 and rs_hi < 0.5 and rev_ratio >= 0.6:
        P2["verdict"] = "H_dom supported as main cause"
    elif ci(casemix_part)[0] > 0 and np.mean(casemix_part) >= 3 and rs_mean >= 0.6 and rs_lo > 0.3 and rev_ratio <= 0.5:
        P2["verdict"] = "H_prev (case mix) supported as main cause"
    else:
        P2["verdict"] = "PARTIAL: both contribute / CIs straddle bounds (report shares, no single cause)"
    report["P2"] = P2

    # ---------------- P3 mechanism decomposition (n=600)
    P3 = {}
    for nm, Rx in (("CPSC", RC), ("PTB_raw", RP), ("PTB_std", RS)):
        c = Rx["components"]
        P3[nm] = {k: (float(np.nanmean(c[k][:8])) if k in ("a", "c", "b", "e", "G_L") else c[k][:8].round(3).tolist())
                  for k in c}
        P3[nm]["per_label_G_L"] = dict(zip(L8, c["G_L"][:8].round(1).tolist()))
    rC, rP = np.array(P3["CPSC"]["r_L"]), np.array(P3["PTB_raw"]["r_L"])
    sp_r = spearmanr(rC, rP).correlation
    signs = {L: (bool(np.sign(rC[L8.index(L)]) == np.sign(rP[L8.index(L)])), round(float(rC[L8.index(L)]), 2), round(float(rP[L8.index(L)]), 2))
             for L in ("AF", "STD", "I-AVB", "PVC")}
    v = []
    dr_ptb = np.array(P3["PTB_std"]["drank_S1"])
    if P3["CPSC"]["c"] - P3["PTB_std"]["c"] >= SESOI and (dr_ptb <= 0.02).sum() >= 5: v.append("(i) H_dom")
    if P3["PTB_std"]["c"] >= 0.6 * P3["CPSC"]["c"] and abs(P3["PTB_raw"]["c"]) < SESOI: v.append("(ii) H_prev")
    if P3["PTB_raw"]["b"] >= SESOI and sp_r >= 0.5 and all(s[0] for s in signs.values()): v.append("(iii) H_task (directional)")
    if abs(P3["PTB_raw"]["a"] - P3["CPSC"]["a"]) >= SESOI: v.append("(iv) H_head candidate (check it explains DiD)")
    P3["spearman_rL_CPSC_PTB"] = float(sp_r); P3["sign_check"] = signs; P3["rules_met"] = v or ["none"]
    report["P3"] = P3

    # ---------------- P4 label concordance + mapping robustness
    gC, gS = np.array(G(RC)["per_label_mean"][:8]), np.array(G(RS)["per_label_mean"][:8])
    sp1 = spearmanr(gC, gS).correlation
    sp2 = {nm: spearmanr(np.array(Rx["summary"][600]["per_label_mean"][:8]), np.array(P3[nm]["auc_own"])).correlation
           for nm, Rx in (("CPSC", RC), ("PTB_raw", RP), ("PTB_std", RS))}
    top3 = {nm: [L8[i] for i in np.argsort(-np.array(Rx["summary"][600]["per_label_mean"][:8]))[:3]]
            for nm, Rx in (("CPSC", RC), ("PTB_std", RS))}
    holm = {}
    for nm, Rx in (("CPSC", RC), ("PTB_raw", RP), ("PTB_std", RS)):
        reps = Rx["summary"][600]["per_label_reps"][:, :8]
        p = [2 * min((reps[:, k] <= 0).mean(), (reps[:, k] >= 0).mean()) for k in range(8)]
        order = np.argsort(p); adj = np.zeros(8); run = 0
        for rank, k in enumerate(order):
            run = max(run, min(1, (8 - rank) * p[k])); adj[k] = run
        holm[nm] = {L8[k]: [round(float(Rx["summary"][600]["per_label_mean"][k]), 1), *[round(x, 1) for x in ci(reps[:, k])], round(float(adj[k]), 4)] for k in range(8)}
    shared = set(top3["CPSC"]) & set(top3["PTB_std"]) & {"STD", "I-AVB", "PAC"}
    P4 = {"per_label_G_600 [mean, CI_lo, CI_hi, Holm_p]": holm, "spearman_G_CPSC_vs_PTBstd": float(sp1),
          "spearman_G_vs_probAUC": {k: float(x) for k, x in sp2.items()}, "top3": top3}
    P4["H_label_concept"] = bool(sp1 >= 0.6 and shared and all(sp2[k] <= -0.5 for k in ("CPSC", "PTB_std")))

    # mapping variants (PTB raw & std at n=600; CPSC recomputed only when the scored set changes)
    STD_BROAD = {"STD_", "NST_", "DIG", "INJAS", "INJAL", "INJIN", "INJLA", "INJIL"}

    def ptb_variant_gold(kind):
        g = {}
        for i in Ap["ids"]:
            codes, base = ptb["scp"][i], list(ptb["gold"][i])
            if kind == "rbbb_irbbb" and "IRBBB" in codes and "RBBB" not in base: base.append("RBBB")
            if kind == "std_broad" and "STD" not in base and any(c in STD_BROAD or c.startswith("ISC") for c in codes):
                base.append("STD")
            g[i] = [L for L in LABELS if L in base]
        return g

    variants = {}
    for kind, cols in (("L9", list(range(9))), ("L7_noSTE_noPAC", [0, 1, 2, 3, 4, 6, 7]),
                       ("norm_strict", COLS8), ("rbbb_irbbb", COLS8), ("std_broad", COLS8)):
        if kind == "norm_strict":
            keep = [i for i in Ap["ids"] if not ("NORM" in ptb["gold"][i] and (ptb["scp"][i].get("NORM", 0) < 100 or len(ptb["gold"][i]) > 1))]
            gold_v = ptb["gold"]
        else:
            keep = Ap["ids"]; gold_v = ptb_variant_gold(kind) if kind in ("rbbb_irbbb", "std_broad") else ptb["gold"]
        Av = arrays(ptb, ids=keep, gold=gold_v); Av["name"] = f"PTB[{kind}]"
        pv = partitions(Av, cols)
        rv = run_arm(Av, np.ones(len(keep)), cols, pv, sizes=[600], tag=" raw")
        wv = casemix_weights(gold_v, keep, cpsc["gold"], Ac["ids"])
        sv = run_arm(Av, wv, cols, pv, sizes=[600], tag=" std")
        if cols != COLS8:
            cv = run_arm(Ac, np.ones(len(Ac["ids"])), cols, partitions(Ac, cols), sizes=[600], tag=f" [{kind}]")
            gc = cv["summary"][600]["G_mean"]
        else:
            gc = G(RC)["G_mean"]
        rr = sv["summary"][600]["G_mean"] / gc
        cat = ("H_dom" if rr <= 0.3 else "H_prev" if rr >= 0.6 else "PARTIAL")
        variants[kind] = {"n_records": len(keep), "G_PTBraw": round(rv["summary"][600]["G_mean"], 2),
                          "G_PTBstd": round(sv["summary"][600]["G_mean"], 2), "G_CPSC": round(gc, 2),
                          "R_std_point": round(rr, 3), "category_point": cat, "ESS": round(ess(wv), 1)}
    base_cat = ("H_dom" if P2["R_std"][0] <= 0.3 else "H_prev" if P2["R_std"][0] >= 0.6 else "PARTIAL")
    P4["mapping_variants"] = variants; P4["primary_category_point"] = base_cat
    P4["mapping_sensitive"] = any(v["category_point"] != base_cat for k, v in variants.items() if k != "L9")
    report["P4"] = P4

    # ---------------- S9 / S10 sensitivity
    S9, S10 = {}, {}
    for nm, A, arm, parts in (("CPSC", Ac, cpsc, parts_c), ("PTB", Ap, ptb, parts_p)):
        nonempty = A["B0"].any(1)
        g_ne, g9 = [], []
        for r, (pool, test, _) in enumerate(parts):
            S, T = fit_s1(A, pool, test, np.ones(len(A["ids"])), seed=r); Pr = decide(S, T)
            m = nonempty[test]; wt = np.ones(m.sum())
            g_ne.append(macro_f1(A["G"][test][m], Pr[m], wt, COLS8)[0] - macro_f1(A["G"][test][m], A["B0"][test][m], wt, COLS8)[0])
            w1 = np.ones(len(test))
            g9.append(macro_f1(A["G"][test], Pr, w1, list(range(9)))[0] - macro_f1(A["G"][test], A["B0"][test], w1, list(range(9)))[0])
        S9[nm] = {"G_600_excluding_empty_B0": round(100 * float(np.mean(g_ne)), 2), "empty_B0_records": int((~nonempty).sum())}
        S10[nm] = {"G_600_official_9label": round(100 * float(np.mean(g9)), 2)}
    report["S9_B0_parse_sensitivity"] = S9; report["S10_continuity_9label"] = S10
    report["not_implemented"] = ["S1_crosslabel_AUC_lift", "S2_class_conditional_responses", "S3_threshold_vs_reranking_curve",
                                 "S4_structure_transfer", "S5_text_overlay_strata", "S6_grouped_vs_random_split",
                                 "S7_prior_only_correction", "S8_pooled_gap_regression", "CODE-15 descriptive P3 arm"]

    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items() if k not in ("pooled_reps", "per_label_reps")}
        if isinstance(o, (list, tuple)): return [clean(x) for x in o]
        if isinstance(o, np.ndarray): return clean(o.tolist())
        if isinstance(o, (np.floating, np.integer)): return o.item()
        return o
    out_name = "domain_results.json" if PTB_B0.startswith("outputs/pulse7b/") and (R, B) == (20, 200) else "domain_results_DRYRUN.json"
    json.dump(clean(report), open(os.path.join(ROOT, "outputs/clarity", out_name), "w"), indent=1)
    print(json.dumps(clean({k: report[k] for k in ("P1", "P2")}), indent=1)[:4000])
    print("P3 rules met:", report["P3"]["rules_met"], "| P4 H_label:", report["P4"]["H_label_concept"],
          "| mapping_sensitive:", report["P4"]["mapping_sensitive"])
