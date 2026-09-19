"""
Rail Corrugation — Step 8: Honest validation + targeted fixes

What this does (all configs share IDENTICAL outer CV splits):

  1. Fixes the bilateral mirror augmentation.
     The Step 7 version silently skipped:
       - *_cars_sideI_dominant_count  (11 cols) -> should become 8 - x
       - vib_power_per_speed_sq_diff  (1 col)   -> should be negated
     Cause: the "sideI" branch matches `cars_sideI_dominant_count` first,
     finds no `sideII` twin, and the elif for dominant_count never runs.
     Every mirrored Side I -> Side II sample therefore carried a
     Side-I-dominant vote count while labelled Side II.

  2. Speed gate. 44/234 Normal files are stationary (speed ~0, vibration at
     the noise floor, RMS ~0.07 vs ~0.3 moving). Every fault file is at
     >= 35 km/h. Corrugation cannot excite vibration without rolling, so:
       speed < GATE_MPS  -> predict Normal, never shown to the model
       speed >= GATE_MPS -> model
     This also removes the clip-at-1.0 artefact in the per-speed features.

  3. Speed-feature ablation (drop all speed-derived columns).

  4. Nested threshold tuning: class scale factors are chosen on an inner CV
     of the OUTER TRAINING fold only, then applied to the outer validation
     fold. This is the number you can actually quote.

  5. Metrics: per repeat, OOF predictions are pooled across the 5 folds and
     macro F1 is computed once (~14 Side I per pool instead of ~3 per fold).
     Reported as mean ± std over repeats. Fold-mean F1 is also kept so it is
     comparable with Step 7.

Usage:
    python step8_validation.py --features outputs/step7/rail_features_enhanced.csv
    python step8_validation.py --features ... --repeats 5 --quick   (skip nested)
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


RANDOM_STATE = 42
N_SPLITS = 5
INNER_SPLITS = 4
GATE_MPS = 1.0          # stationary threshold (all stationary files are ~0)
EPS = 1e-6
LABEL_TO_INT = {"Normal": 0, "Side I": 1, "Side II": 2}
INT_TO_LABEL = {v: k for k, v in LABEL_TO_INT.items()}

SPEED_COL = "estimated_speed_mps"


# ============================================================
# FIXED MIRROR AUGMENTATION
# ============================================================

def build_mirror_plan(columns):
    """
    Decide once, per column, how it transforms under Side I <-> Side II.
    Returns (swaps, negate, invert, eight_minus, swap_neg, swap_inv).
    Order of checks matters: most specific patterns first.
    """
    cols = set(columns)
    swaps, swap_neg, swap_inv = [], [], []
    negate, invert, eight_minus = [], [], []
    done = set()

    for c in columns:
        if c in done:
            continue

        if "sideI_dominant_count" in c:
            eight_minus.append(c); done.add(c); continue

        if "sideI_decisive_count" in c:
            o = c.replace("sideI_decisive_count", "sideII_decisive_count")
            if o in cols:
                swaps.append((c, o)); done |= {c, o}
            continue

        if "car_diff_max" in c:
            o = c.replace("car_diff_max", "car_diff_min")
            if o in cols:
                swap_neg.append((c, o)); done |= {c, o}
            continue

        if "car_asym_max" in c:
            o = c.replace("car_asym_max", "car_asym_min")
            if o in cols:
                swap_neg.append((c, o)); done |= {c, o}
            continue

        if "top2_cars_ratio_mean" in c:
            o = c.replace("top2_cars_ratio_mean", "bottom2_cars_ratio_mean")
            if o in cols:
                swap_inv.append((c, o)); done |= {c, o}
            continue

        if "sideI_" in c and "sideII_" not in c:
            o = c.replace("sideI_", "sideII_")
            if o in cols:
                swaps.append((c, o)); done |= {c, o}
            continue

        if ("I_minus_II" in c or "asym" in c or "car_diff_median" in c
                or c == "vib_power_per_speed_sq_diff"):
            negate.append(c); done.add(c); continue

        if "I_div_II" in c:
            invert.append(c); done.add(c); continue

    return dict(swaps=swaps, swap_neg=swap_neg, swap_inv=swap_inv,
                negate=negate, invert=invert, eight_minus=eight_minus)


def mirror(X, y, plan):
    Xm = X.copy()
    for a, b in plan["swaps"]:
        Xm[a], Xm[b] = X[b].values, X[a].values
    for mx, mn in plan["swap_neg"]:          # new max = -old min, new min = -old max
        Xm[mx], Xm[mn] = -X[mn].values, -X[mx].values
    for top, bot in plan["swap_inv"]:        # new top2 = 1/old bottom2, etc.
        Xm[top], Xm[bot] = 1.0 / (X[bot].values + EPS), 1.0 / (X[top].values + EPS)
    for c in plan["negate"]:
        Xm[c] = -X[c].values
    for c in plan["invert"]:
        Xm[c] = 1.0 / (X[c].values + EPS)
    for c in plan["eight_minus"]:
        Xm[c] = 8.0 - X[c].values
    ym = y.copy()
    ym[y == 1], ym[y == 2] = 2, 1
    return Xm, ym


def legacy_mirror_plan(plan):
    """Reproduce the Step 7 bug: dominant_count and speed diff untouched."""
    p = {k: list(v) for k, v in plan.items()}
    p["eight_minus"] = []
    p["negate"] = [c for c in p["negate"] if c != "vib_power_per_speed_sq_diff"]
    return p


# ============================================================
# MODEL
# ============================================================

def fit_predict(X_tr, y_tr, X_va, plan, seed, xgb_w=0.65, side_i_boost=1.2):
    fault = (y_tr == 1) | (y_tr == 2)
    Xm, ym = mirror(X_tr[fault], y_tr[fault], plan)
    X_fit = pd.concat([X_tr, Xm], ignore_index=True)
    y_fit = np.concatenate([y_tr, ym])

    sw = compute_sample_weight("balanced", y_fit)
    sw[y_fit == 1] *= side_i_boost

    xgb = XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=0.035,
        min_child_weight=3, subsample=0.8, colsample_bytree=0.6,
        reg_alpha=0.1, reg_lambda=1.0, objective="multi:softprob",
        num_class=3, random_state=seed, n_jobs=-1, verbosity=0,
    )
    xgb.fit(X_fit, y_fit, sample_weight=sw)
    p = xgb.predict_proba(X_va)

    if xgb_w < 1.0:
        et = ExtraTreesClassifier(
            n_estimators=300, max_depth=6, min_samples_split=3,
            class_weight="balanced", random_state=seed, n_jobs=-1,
        )
        et.fit(X_fit, y_fit)
        p = xgb_w * p + (1 - xgb_w) * et.predict_proba(X_va)
    return p


# ============================================================
# THRESHOLDS
# ============================================================

T1_GRID = np.linspace(0.15, 0.60, 19)
T2_GRID = np.linspace(0.15, 0.60, 19)


def apply_scales(p, t):
    return np.argmax(np.column_stack([p[:, 0], p[:, 1] / t[0], p[:, 2] / t[1]]), axis=1)


def tune_scales(p, y):
    """Grid search over Side I / Side II scale factors. Ties -> closest to 1/3
    (i.e. the least aggressive shift), which reduces variance on tiny data."""
    best, best_t = -1.0, (1 / 3, 1 / 3)
    for t1 in T1_GRID:
        for t2 in T2_GRID:
            s = f1_score(y, apply_scales(p, (t1, t2)), average="macro")
            dist = abs(t1 - 1 / 3) + abs(t2 - 1 / 3)
            if s > best + 1e-9 or (abs(s - best) <= 1e-9 and
                                   dist < abs(best_t[0] - 1 / 3) + abs(best_t[1] - 1 / 3)):
                best, best_t = s, (t1, t2)
    return best_t


# ============================================================
# ONE CONFIG
# ============================================================

def run_config(cfg, X, y, speed, splits, n_repeats, plan_fixed, plan_legacy):
    cols = cfg["columns"]
    Xc = X[cols]
    plan = plan_legacy if cfg["legacy_mirror"] else plan_fixed
    plan = {k: [e for e in v if (all(x in cols for x in e) if isinstance(e, tuple) else e in cols)]
            for k, v in plan.items()}
    moving = speed >= GATE_MPS if cfg["gate"] else np.ones(len(y), bool)

    oof_default = np.zeros((n_repeats, len(y)), int)
    oof_nested = np.zeros((n_repeats, len(y)), int)
    fold_scores = []
    chosen_t = []

    for k, (tr, va) in enumerate(splits):
        rep = k // N_SPLITS
        seed = RANDOM_STATE + k
        tr_m = tr[moving[tr]]
        va_m = va[moving[va]]

        pred_def = np.zeros(len(va), int)       # stationary -> Normal (0)
        pred_nst = np.zeros(len(va), int)

        if len(va_m):
            p = fit_predict(Xc.iloc[tr_m], y[tr_m], Xc.iloc[va_m], plan, seed, cfg["xgb_w"])
            mask = moving[va]
            pred_def[mask] = np.argmax(p, axis=1)

            if cfg["nested"]:
                inner = StratifiedKFold(INNER_SPLITS, shuffle=True, random_state=seed)
                p_in = np.zeros((len(tr_m), 3))
                for itr, iva in inner.split(tr_m, y[tr_m]):
                    a, b = tr_m[itr], tr_m[iva]
                    p_in[iva] = fit_predict(Xc.iloc[a], y[a], Xc.iloc[b], plan, seed, cfg["xgb_w"])
                # Tune on inner OOF; stationary train files are counted as correct Normals
                # when gating, so include them to keep the F1 on the same footing.
                y_in = y[tr_m]
                if cfg["gate"]:
                    n_stat = int((~moving[tr]).sum())
                    y_in = np.concatenate([y_in, np.zeros(n_stat, int)])
                    p_in = np.vstack([p_in, np.tile([1.0, 0.0, 0.0], (n_stat, 1))])
                t = tune_scales(p_in, y_in)
                chosen_t.append(t)
                pred_nst[mask] = apply_scales(p, t)

        oof_default[rep, va] = pred_def
        oof_nested[rep, va] = pred_nst
        fold_scores.append(f1_score(y[va], pred_def, average="macro"))

    def pooled(oof):
        per = np.array([f1_score(y, oof[r], average=None, labels=[0, 1, 2]) for r in range(n_repeats)])
        mac = per.mean(axis=1)
        return mac, per

    out = {"config": cfg["name"]}
    mac, per = pooled(oof_default)
    out.update(macro_pooled=mac.mean(), macro_pooled_std=mac.std(),
               normal=per[:, 0].mean(), side_I=per[:, 1].mean(), side_II=per[:, 2].mean(),
               macro_fold_mean=np.mean(fold_scores))
    if cfg["nested"]:
        mac_n, per_n = pooled(oof_nested)
        ts = np.array(chosen_t)
        out.update(nested_macro=mac_n.mean(), nested_macro_std=mac_n.std(),
                   nested_side_I=per_n[:, 1].mean(), nested_side_II=per_n[:, 2].mean(),
                   nested_normal=per_n[:, 0].mean(),
                   t_sideI_mean=ts[:, 0].mean(), t_sideI_std=ts[:, 0].std(),
                   t_sideII_mean=ts[:, 1].mean(), t_sideII_std=ts[:, 1].std())
    return out, oof_default, oof_nested


# ============================================================
# MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="outputs/step7/rail_features_enhanced.csv")
    ap.add_argument("--out", default="outputs/step8")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="skip nested threshold runs")
    ap.add_argument("--extra-features", default="",
                    help="CSV with filename + extra columns (e.g. wavelength features) -> adds config F")
    ap.add_argument("--only", default="", help="comma-separated config letters, e.g. B,E")
    args = ap.parse_args()

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.features)
    extra_cols = []
    if args.extra_features:
        ex = pd.read_csv(args.extra_features)
        extra_cols = [c for c in ex.columns if c != "filename"]
        df = df.merge(ex, on="filename", how="left", validate="one_to_one")
        assert df[extra_cols].notna().all().all(), "extra features missing for some files"
    y = df["label"].map(LABEL_TO_INT).values
    X = df.drop(columns=["filename", "label"]).copy()
    speed = X[SPEED_COL].values

    all_cols = list(X.columns)
    speed_cols = [c for c in all_cols if "speed" in c]
    no_speed = [c for c in all_cols if c not in speed_cols and c not in extra_cols]
    no_extra = [c for c in all_cols if c not in extra_cols]

    plan_fixed = build_mirror_plan(all_cols)
    plan_legacy = legacy_mirror_plan(plan_fixed)

    splits = list(RepeatedStratifiedKFold(
        n_splits=N_SPLITS, n_repeats=args.repeats, random_state=RANDOM_STATE
    ).split(X, y))

    base = dict(columns=no_extra, legacy_mirror=False, gate=False, xgb_w=0.65, nested=False)
    configs = [
        {**base, "name": "A_step7_reproduction (buggy mirror)", "legacy_mirror": True},
        {**base, "name": "B_fixed_mirror"},
        {**base, "name": "C_fixed_mirror + speed_gate", "gate": True},
        {**base, "name": "D_C + no_speed_features", "gate": True, "columns": no_speed},
    ]
    if extra_cols:
        configs += [
            {**base, "name": "F_C + wavelength_features", "gate": True, "columns": all_cols},
        ]
    if not args.quick:
        configs += [
            {**base, "name": "E_C + nested_thresholds", "gate": True, "nested": True},
        ]

    if args.only:
        keep = set(args.only.split(","))
        configs = [c for c in configs if c["name"][0] in keep]

    rows = []
    for cfg in configs:
        t0 = time.time()
        res, oof_d, _ = run_config(cfg, X, y, speed, splits, args.repeats, plan_fixed, plan_legacy)
        rows.append(res)
        msg = (f"{cfg['name']:<40} pooled macro {res['macro_pooled']:.4f} ± {res['macro_pooled_std']:.4f} | "
               f"SI {res['side_I']:.3f} SII {res['side_II']:.3f} N {res['normal']:.3f} | "
               f"fold-mean {res['macro_fold_mean']:.4f}")
        if cfg["nested"]:
            msg += (f"\n{'':<40} NESTED macro {res['nested_macro']:.4f} ± {res['nested_macro_std']:.4f} | "
                    f"SI {res['nested_side_I']:.3f} SII {res['nested_side_II']:.3f} | "
                    f"t_SI {res['t_sideI_mean']:.2f}±{res['t_sideI_std']:.2f} "
                    f"t_SII {res['t_sideII_mean']:.2f}±{res['t_sideII_std']:.2f}")
        print(msg + f"  ({time.time() - t0:.0f}s)", flush=True)

        # per-file hit rate for the fault files, for hard-case analysis
        hits = (oof_d == y[None, :]).mean(axis=0)
        pd.DataFrame({"filename": df["filename"], "label": df["label"],
                      "speed_kmh": speed * 3.6, "hit_rate": hits}) \
            .query("label != 'Normal' or hit_rate < 1") \
            .sort_values("hit_rate") \
            .to_csv(out_dir / f"hits_{cfg['name'].split()[0]}.csv", index=False)

    pd.DataFrame(rows).to_csv(out_dir / "step8_summary.csv", index=False)
    print(f"\nSaved {out_dir / 'step8_summary.csv'}")


if __name__ == "__main__":
    main()
