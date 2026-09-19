"""
rail_model.py — shared feature pipeline + model wrapper.

Used by BOTH train_final.py and predict.py, so training and test features are
computed by exactly the same code (no train/test feature drift).

Feature configs:
    REF     step-3 + step-7 features, speed-derived columns removed
    REF+WL  REF + wavelength features from step 9 (mean/max/per-car aggregations)

Model: 3-class XGBoost + ExtraTrees blend (0.65 / 0.35), trained with the fixed
bilateral mirror augmentation, averaged over several seeds.
Stationary files (speed < 1 m/s) are always predicted Normal.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from step1_data_cleaning import load_rail_file
from step3_feature_engineering import extract_file_features
from step7_enhanced_features import (
    compute_car_invariant_features,
    compute_normalized_asymmetry_features,
    compute_speed_normalized_features,
)
from step8_validation import build_mirror_plan, mirror
from step9_side_features import file_side_rows, to_file_table

GATE_MPS = 1.0
SPEED_COL = "estimated_speed_mps"
CLASSES = ["Normal", "Side I", "Side II"]
LABEL_TO_INT = {c: i for i, c in enumerate(CLASSES)}
CONFIGS = ("REF", "REF+WL")


# ============================================================
# FEATURES
# ============================================================

def is_wl_feature(c):
    """Wavelength features from the step-9 file table, minus the median /
    second-largest aggregations (those hurt in the step-9 ablation)."""
    wl = "_wl_" in c or "tonality" in c or "peakwl" in c
    if not wl:
        return False
    if "_car_diff_median" in c:
        return True
    return not (c.endswith("_median") or c.endswith("_second"))


def file_sort_key(p):
    digits = "".join(ch for ch in Path(p).stem if ch.isdigit())
    return int(digits) if digits else 0


def build_features(csv_paths, with_wl=True, verbose=True):
    """Raw CSV paths -> one feature row per file (all columns any config needs).
    Returns DataFrame with 'filename' plus features."""
    base_rows, side_rows = [], []
    for i, path in enumerate(csv_paths, 1):
        path = Path(path)
        base = extract_file_features(path)
        base["filename"] = path.name
        base_rows.append(base)
        if with_wl:
            raw = load_rail_file(path).to_numpy(dtype=np.float32)
            for r in file_side_rows(raw):
                r["filename"] = path.name
                side_rows.append(r)
        if verbose and i % 20 == 0:
            print(f"  features: {i}/{len(csv_paths)}", flush=True)

    base_df = pd.DataFrame(base_rows)
    step7 = pd.concat([
        base_df,
        compute_car_invariant_features(base_df, channel="vib"),
        compute_normalized_asymmetry_features(base_df),
        compute_speed_normalized_features(base_df),
    ], axis=1)

    if with_wl:
        ft = to_file_table(pd.DataFrame(side_rows))
        wl_cols = [c for c in ft.columns if is_wl_feature(c)]
        step7 = step7.merge(ft[["filename"] + wl_cols], on="filename", how="left",
                            validate="one_to_one")
    return step7


def select_columns(feature_df, config):
    """Model input columns for a config. Speed-derived columns are excluded
    (they added nothing in step 8 and are a shortcut risk)."""
    if config not in CONFIGS:
        raise ValueError(f"config must be one of {CONFIGS}")
    cols = [c for c in feature_df.columns
            if c not in ("filename", "label") and "speed" not in c]
    wl = [c for c in cols if is_wl_feature(c)]
    base = [c for c in cols if c not in wl and c != "wl_valid"]
    return base + wl if config == "REF+WL" else base


# ============================================================
# MODEL
# ============================================================

class RailModel:
    """Seed-averaged XGB + ET blend. predict(feature_df) -> list of class names."""

    def __init__(self, config, columns, members, xgb_w):
        self.config, self.columns, self.members, self.xgb_w = config, columns, members, xgb_w

    def predict_proba(self, feature_df):
        X = feature_df[self.columns]
        p = np.zeros((len(X), 3))
        for xgb, et in self.members:
            p += self.xgb_w * xgb.predict_proba(X) + (1 - self.xgb_w) * et.predict_proba(X)
        p /= len(self.members)
        stationary = feature_df[SPEED_COL].values < GATE_MPS
        p[stationary] = [1.0, 0.0, 0.0]
        return p

    def predict(self, feature_df):
        return [CLASSES[i] for i in np.argmax(self.predict_proba(feature_df), axis=1)]


def train_model(feature_df, labels, config, n_seeds=5, xgb_w=0.65, side_i_boost=1.2, seed=42):
    """feature_df: output of build_features; labels: array of class names."""
    cols = select_columns(feature_df, config)
    y = pd.Series(labels).map(LABEL_TO_INT).values
    moving = feature_df[SPEED_COL].values >= GATE_MPS
    X, y = feature_df.loc[moving, cols].reset_index(drop=True), y[moving]

    plan = build_mirror_plan(cols)
    fault = (y == 1) | (y == 2)
    Xm, ym = mirror(X[fault], y[fault], plan)
    X_fit = pd.concat([X, Xm], ignore_index=True)
    y_fit = np.concatenate([y, ym])
    sw = compute_sample_weight("balanced", y_fit)
    sw[y_fit == 1] *= side_i_boost

    members = []
    for k in range(n_seeds):
        xgb = XGBClassifier(
            n_estimators=160, max_depth=3, learning_rate=0.035, min_child_weight=3,
            subsample=0.8, colsample_bytree=0.6, reg_alpha=0.1, reg_lambda=1.0,
            objective="multi:softprob", num_class=3, random_state=seed + k,
            n_jobs=-1, verbosity=0,
        ).fit(X_fit, y_fit, sample_weight=sw)
        et = ExtraTreesClassifier(
            n_estimators=300, max_depth=6, min_samples_split=3,
            class_weight="balanced", random_state=seed + k, n_jobs=-1,
        ).fit(X_fit, y_fit)
        members.append((xgb, et))
    return RailModel(config, cols, members, xgb_w)
