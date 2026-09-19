"""
Rail Corrugation — Step 9C: per-side reframing vs 3-class, same splits

Configs (all use identical outer file-level splits, stationary files gated to Normal):
  REF  step-8 config D on the OLD features (optional, --old-features)   reference
  F3   3-class XGB+ET blend + fixed mirror on the NEW file table        isolates features
  PS   per-side binary XGB+ET blend, t = 0.5                            isolates reframing
  PSN  PS with t chosen by nested inner CV (single parameter)
  PS-no_wl / PS-no_robust / PS-no_xc / PS-no_car / PS-no_other           ablations

Per-side decision rule:
    p_I, p_II = P(corrugated) for each side row of the file
    Normal if max(p_I, p_II) < t, else the side with the larger p.

Both side rows of a file always land in the same fold (splits are by file).

Metric: per repeat, pooled out-of-fold macro F1; mean ± std over repeats.
Differences below ~0.02 are inside seed noise (see step 8) — treat as ties.

Usage:
    python step9_evaluate.py --step9-dir outputs/step9 [--old-features outputs/step7/rail_features_enhanced.csv]
    python step9_evaluate.py --step9-dir outputs/step9 --only PS,PSN --repeats 3
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from step8_validation import build_mirror_plan, fit_predict as fit_predict_3class

RANDOM_STATE = 42
N_SPLITS = 5
INNER_SPLITS = 4
XGB_W = 0.65
T_GRID = np.linspace(0.10, 0.90, 33)
LABEL_TO_INT = {"Normal": 0, "Side I": 1, "Side II": 2}
META = {"filename", "side", "label", "file_label"}


# ============================================================
# PER-SIDE MODEL
# ============================================================

def fit_predict_side(X_tr, y_tr, X_va, seed):
    sw = compute_sample_weight("balanced", y_tr)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=3, learning_rate=0.035, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
        objective="binary:logistic", random_state=seed, n_jobs=-1, verbosity=0,
    ).fit(X_tr, y_tr, sample_weight=sw)
    et = ExtraTreesClassifier(
        n_estimators=300, max_depth=6, min_samples_split=3,
        class_weight="balanced", random_state=seed, n_jobs=-1,
    ).fit(X_tr, y_tr)
    return XGB_W * xgb.predict_proba(X_va)[:, 1] + (1 - XGB_W) * et.predict_proba(X_va)[:, 1]


def side_probs_to_class(p_I, p_II, t):
    pred = np.where(p_I >= p_II, 1, 2)
    pred[np.maximum(p_I, p_II) < t] = 0
    return pred


def tune_t(p_I, p_II, y):
    best, best_t = -1, 0.5
    for t in T_GRID:
        s = f1_score(y, side_probs_to_class(p_I, p_II, t), average="macro")
        if s > best + 1e-9 or (abs(s - best) <= 1e-9 and abs(t - 0.5) < abs(best_t - 0.5)):
            best, best_t = s, t
    return best_t


class SideData:
    """Index side rows by file position so file splits map to side rows."""

    def __init__(self, side_df, files):
        side_df = side_df.set_index(["filename", "side"])
        self.XI = side_df.xs("I", level="side").loc[files]
        self.XII = side_df.xs("II", level="side").loc[files]

    def rows(self, idx, cols):
        X = pd.concat([self.XI.iloc[idx][cols], self.XII.iloc[idx][cols]], ignore_index=True)
        y = np.concatenate([self.XI.iloc[idx]["label"].values, self.XII.iloc[idx]["label"].values])
        return X, y


def run_side_config(sd, cols, y_file, moving, splits, n_rep, nested):
    oof = np.zeros((n_rep, len(y_file)), int)
    oof_n = np.zeros((n_rep, len(y_file)), int)
    ts = []
    for k, (tr, va) in enumerate(splits):
        rep, seed = k // N_SPLITS, RANDOM_STATE + k
        tr_m, va_m = tr[moving[tr]], va[moving[va]]
        Xtr, ytr = sd.rows(tr_m, cols)
        Xva = pd.concat([sd.XI.iloc[va_m][cols], sd.XII.iloc[va_m][cols]], ignore_index=True)
        p = fit_predict_side(Xtr, ytr, Xva, seed)
        p_I, p_II = p[:len(va_m)], p[len(va_m):]

        pred = np.zeros(len(va), int)
        m = moving[va]
        pred[m] = side_probs_to_class(p_I, p_II, 0.5)
        oof[rep, va] = pred

        if nested:
            inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
            qI, qII = np.zeros(len(tr_m)), np.zeros(len(tr_m))
            for a, b in inner.split(tr_m, y_file[tr_m]):
                Xa, ya = sd.rows(tr_m[a], cols)
                Xb = pd.concat([sd.XI.iloc[tr_m[b]][cols], sd.XII.iloc[tr_m[b]][cols]], ignore_index=True)
                q = fit_predict_side(Xa, ya, Xb, seed)
                qI[b], qII[b] = q[:len(b)], q[len(b):]
            n_stat = int((~moving[tr]).sum())      # gated files count as correct Normals
            t = tune_t(np.r_[qI, np.zeros(n_stat)], np.r_[qII, np.zeros(n_stat)],
                       np.r_[y_file[tr_m], np.zeros(n_stat, int)])
            ts.append(t)
            pn = np.zeros(len(va), int)
            pn[m] = side_probs_to_class(p_I, p_II, t)
            oof_n[rep, va] = pn
    return oof, (oof_n if nested else None), ts


# ============================================================
# 3-CLASS ON A FILE TABLE
# ============================================================

def run_3class(X, y_file, moving, splits, n_rep):
    plan = build_mirror_plan(list(X.columns))
    oof = np.zeros((n_rep, len(y_file)), int)
    for k, (tr, va) in enumerate(splits):
        rep, seed = k // N_SPLITS, RANDOM_STATE + k
        tr_m, va_m = tr[moving[tr]], va[moving[va]]
        p = fit_predict_3class(X.iloc[tr_m], y_file[tr_m], X.iloc[va_m], plan, seed, XGB_W)
        pred = np.zeros(len(va), int)
        pred[moving[va]] = np.argmax(p, axis=1)
        oof[rep, va] = pred
    return oof


# ============================================================
# REPORTING
# ============================================================

def score(oof, y):
    per = np.array([f1_score(y, o, average=None, labels=[0, 1, 2]) for o in oof])
    mac = per.mean(1)
    return dict(macro=mac.mean(), macro_std=mac.std(),
                normal=per[:, 0].mean(), side_I=per[:, 1].mean(), side_II=per[:, 2].mean())


def show(name, s, extra=""):
    print(f"{name:<14} macro {s['macro']:.4f} ± {s['macro_std']:.4f} | "
          f"N {s['normal']:.3f} SI {s['side_I']:.3f} SII {s['side_II']:.3f} {extra}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step9-dir", default="outputs/step9")
    ap.add_argument("--old-features", default="")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--only", default="", help="comma list, e.g. PS,PSN,F3")
    args = ap.parse_args()

    d = Path(args.step9_dir)
    side_df = pd.read_csv(d / "side_table_train.csv")
    file_df = pd.read_csv(d / "file_table_train.csv")
    files = file_df["filename"].values
    y_file = file_df["label"].map(LABEL_TO_INT).values
    moving = file_df["wl_valid"].values > 0.5

    splits = list(RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=args.repeats,
                                          random_state=RANDOM_STATE).split(files, y_file))
    sd = SideData(side_df, files)

    side_cols = [c for c in side_df.columns if c not in META]
    is_wl = lambda c: "_wl_" in c or "tonality" in c or "peakwl" in c
    is_robust = lambda c: (c.endswith("_median") and not c.startswith("car_")) or c.endswith("_second")
    ablations = {
        "PS-no_wl": [c for c in side_cols if not is_wl(c)],
        "PS-no_robust": [c for c in side_cols if not is_robust(c)],
        "PS-no_xc": [c for c in side_cols if "_xc_" not in c],
        "PS-no_car": [c for c in side_cols if not c.startswith("car_")],
        "PS-no_other": [c for c in side_cols if not c.startswith("other_")],
    }

    hits = {}
    want = set(args.only.split(",")) if args.only else None
    run = lambda n: want is None or n in want
    rows = []

    if args.old_features and run("REF"):
        t0 = time.time()
        old = pd.read_csv(args.old_features).set_index("filename").loc[files].reset_index()
        Xo = old[[c for c in old.columns if c not in ("filename", "label") and "speed" not in c]]
        oof = run_3class(Xo, y_file, moving, splits, args.repeats)
        s = score(oof, y_file)
        show("REF(old,D)", s, f"({time.time()-t0:.0f}s)"); rows.append({"config": "REF", **s})
        hits["REF"] = (oof == y_file[None, :]).mean(0)

    if args.old_features and run("REF+WL"):
        # old step-7 features (speed-derived dropped) + wavelength features from the new
        # file table, mean/max aggregations only (median/second hurt in the ablation)
        t0 = time.time()
        old = pd.read_csv(args.old_features).set_index("filename").loc[files].reset_index()
        Xo = old[[c for c in old.columns if c not in ("filename", "label") and "speed" not in c]]
        wl_cols = [c for c in file_df.columns
                   if is_wl(c) and not c.endswith("_median") and not c.endswith("_second")
                   or (is_wl(c) and "_car_diff_median" in c)]
        Xc = pd.concat([Xo.reset_index(drop=True), file_df[wl_cols].reset_index(drop=True)], axis=1)
        oof = run_3class(Xc, y_file, moving, splits, args.repeats)
        s = score(oof, y_file)
        show("REF+WL", s, f"[+{len(wl_cols)} wl feats] ({time.time()-t0:.0f}s)")
        rows.append({"config": "REF+WL", **s, "n_feats": Xc.shape[1]})
        hits["REF+WL"] = (oof == y_file[None, :]).mean(0)

    if run("F3"):
        t0 = time.time()
        Xf = file_df.drop(columns=["filename", "label"])
        oof = run_3class(Xf, y_file, moving, splits, args.repeats)
        s = score(oof, y_file)
        show("F3", s, f"({time.time()-t0:.0f}s)"); rows.append({"config": "F3", **s})
        hits["F3"] = (oof == y_file[None, :]).mean(0)

    if run("PS") or run("PSN"):
        t0 = time.time()
        oof, oof_n, ts = run_side_config(sd, side_cols, y_file, moving, splits, args.repeats,
                                         nested=run("PSN"))
        s = score(oof, y_file)
        show("PS", s, f"({time.time()-t0:.0f}s)"); rows.append({"config": "PS", **s})
        hits["PS"] = (oof == y_file[None, :]).mean(0)
        if oof_n is not None:
            sn = score(oof_n, y_file)
            show("PSN", sn, f"t = {np.mean(ts):.2f} ± {np.std(ts):.2f}")
            rows.append({"config": "PSN", **sn, "t_mean": np.mean(ts), "t_std": np.std(ts)})

    for name, cols in ablations.items():
        if not run(name):
            continue
        t0 = time.time()
        oof, _, _ = run_side_config(sd, cols, y_file, moving, splits, args.repeats, nested=False)
        s = score(oof, y_file)
        show(name, s, f"[{len(cols)} feats] ({time.time()-t0:.0f}s)")
        rows.append({"config": name, **s, "n_feats": len(cols)})

    pd.DataFrame(rows).to_csv(d / "step9_results.csv", index=False)
    if hits:
        h = pd.DataFrame({"filename": files, "label": file_df["label"].values,
                          "speed_kmh": file_df["speed_kmh"].values, **hits})
        h = h[h["label"] != "Normal"].sort_values("speed_kmh")
        h.to_csv(d / "step9_fault_hits.csv", index=False)
        print("\nFault-file hit rates (look at Train121 / 180 / 185):")
        print(h.round(2).to_string(index=False))
    print(f"\nSaved {d / 'step9_results.csv'}")


if __name__ == "__main__":
    main()
