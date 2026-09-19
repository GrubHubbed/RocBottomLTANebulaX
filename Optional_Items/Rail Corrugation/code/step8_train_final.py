"""
Rail Corrugation — Step 8C: Train and package the final model

Fixes a deployment mismatch in step7_model_refinement.py:
    The champion was selected and threshold-tuned as the XGB + ExtraTrees
    BLEND, but the saved joblib contains only the XGBoost model. Any predict.py
    loading it would run a different model from the one you validated, with
    thresholds tuned for the blend.

This script saves everything predict.py needs, and exposes one function:
    RailModel.predict(feature_df) -> list of "Normal" / "Side I" / "Side II"

Thresholds:
    Pass --t-side-i / --t-side-ii from step8_validation's NESTED run
    (use the reported means). Default 1/3 each == plain argmax, i.e. no tuning.
    Only use tuned values if the nested run beat the untuned run.

Usage:
    python step8_train_final.py --features outputs/step7/rail_features_enhanced.csv \
        [--extra-features outputs/step8/wavelength_features.csv] \
        [--drop-speed-features] [--t-side-i 0.30 --t-side-ii 0.33]
"""

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from step8_validation import build_mirror_plan, mirror, GATE_MPS, SPEED_COL, LABEL_TO_INT, INT_TO_LABEL

RANDOM_STATE = 42


class RailModel:
    def __init__(self, xgb, et, xgb_w, columns, thresholds, gate):
        self.xgb, self.et, self.xgb_w = xgb, et, xgb_w
        self.columns, self.thresholds, self.gate = columns, thresholds, gate

    def predict_proba(self, feats):
        X = feats[self.columns]
        p = self.xgb.predict_proba(X)
        if self.et is not None:
            p = self.xgb_w * p + (1 - self.xgb_w) * self.et.predict_proba(X)
        if self.gate:
            stationary = feats[SPEED_COL].values < GATE_MPS
            p[stationary] = [1.0, 0.0, 0.0]
        return p

    def predict(self, feats):
        p = self.predict_proba(feats)
        t1, t2 = self.thresholds
        idx = np.argmax(np.column_stack([p[:, 0], p[:, 1] / t1, p[:, 2] / t2]), axis=1)
        return [INT_TO_LABEL[i] for i in idx]


def train(X, y, speed, xgb_w=0.65, gate=True, thresholds=(1 / 3, 1 / 3), side_i_boost=1.2):
    keep = speed >= GATE_MPS if gate else np.ones(len(y), bool)
    Xk, yk = X[keep], y[keep]

    plan = build_mirror_plan(list(X.columns))
    fault = (yk == 1) | (yk == 2)
    Xm, ym = mirror(Xk[fault], yk[fault], plan)
    X_fit = pd.concat([Xk, Xm], ignore_index=True)
    y_fit = np.concatenate([yk, ym])
    sw = compute_sample_weight("balanced", y_fit)
    sw[y_fit == 1] *= side_i_boost

    xgb = XGBClassifier(
        n_estimators=160, max_depth=3, learning_rate=0.035, min_child_weight=3,
        subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
        objective="multi:softprob", num_class=3, random_state=RANDOM_STATE, n_jobs=-1,
    ).fit(X_fit, y_fit, sample_weight=sw)

    et = None
    if xgb_w < 1.0:
        et = ExtraTreesClassifier(
            n_estimators=500, max_depth=6, min_samples_split=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        ).fit(X_fit, y_fit)

    return RailModel(xgb, et, xgb_w, list(X.columns), thresholds, gate)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--extra-features", default="")
    ap.add_argument("--drop-speed-features", action="store_true")
    ap.add_argument("--no-gate", action="store_true")
    ap.add_argument("--xgb-weight", type=float, default=0.65)
    ap.add_argument("--t-side-i", type=float, default=1 / 3)
    ap.add_argument("--t-side-ii", type=float, default=1 / 3)
    ap.add_argument("--out", default="outputs/step8/rail_model.joblib")
    args = ap.parse_args()

    df = pd.read_csv(args.features)
    if args.extra_features:
        df = df.merge(pd.read_csv(args.extra_features), on="filename", validate="one_to_one")
    y = df["label"].map(LABEL_TO_INT).values
    speed = df[SPEED_COL].values
    X = df.drop(columns=["filename", "label"])
    if args.drop_speed_features:
        X = X[[c for c in X.columns if "speed" not in c]]

    model = train(X, y, speed, args.xgb_weight, not args.no_gate, (args.t_side_i, args.t_side_ii))

    # sanity: training-set predictions should be near-perfect; if not, something is broken
    feats_for_pred = X.copy()
    feats_for_pred[SPEED_COL] = speed
    train_acc = np.mean(np.array(model.predict(feats_for_pred)) == df["label"].values)
    print(f"Training-set accuracy (sanity only, not a score): {train_acc:.3f}")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.out)
    print(f"Saved {args.out}  | columns={len(model.columns)} gate={model.gate} "
          f"xgb_w={model.xgb_w} thresholds={model.thresholds}")


if __name__ == "__main__":
    main()
