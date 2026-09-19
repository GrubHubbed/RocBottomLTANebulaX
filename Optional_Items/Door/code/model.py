"""Model definitions for the Door subsystem.

Two complementary detectors, matching the system architecture:

* **Fault classifier** (supervised) - Normal vs Abnormal resistance, trained on the 110
  labelled cycles. Several candidates are benchmarked and the best is selected by
  cross-validation rather than by assertion.
* **Anomaly detector** (unsupervised, "fast monitoring mode") - an Isolation Forest fitted
  on *normal* cycles only, so it flags unusual cycles without needing fault labels. It
  also covers fault modes the labelled set has never seen.

``OperationNormalizer`` implements the operating-state normalisation from the
noise-suppression design: a cycle is judged against the baseline of its *own* operating
state (Open vs Close), which is what stops an ordinary heavy close from looking like a
fault. It is a fitted transformer, so inside a Pipeline the baselines are learned from the
training fold only - no leakage into validation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import (
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    IsolationForest,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

POSITIVE_LABEL = "Abnormal resistance"
NEGATIVE_LABEL = "Normal"
RANDOM_STATE = 42

# Features that are meaningful to normalise against the same-operation baseline.
_RATIO_SAFE_PREFIXES = ("cur_", "emf_", "volt_", "speed_", "work_", "power_", "travel")


class OperationNormalizer(BaseEstimator, TransformerMixin):
    """Append ``feature / median(feature | same operation)`` columns.

    Expects a DataFrame carrying an ``is_open`` column. Baseline medians are learned in
    ``fit`` (training fold only) and reused unchanged in ``transform``.
    """

    def __init__(self, enabled: bool = True):
        self.enabled = enabled

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.ratio_cols_ = [
            c
            for c in X.columns
            if c != "is_open" and c.startswith(_RATIO_SAFE_PREFIXES)
        ]
        self.baselines_ = {}
        for op in (0.0, 1.0):
            mask = X["is_open"] > 0.5 if op == 1.0 else X["is_open"] <= 0.5
            sub = X.loc[mask, self.ratio_cols_]
            med = sub.median() if len(sub) else X[self.ratio_cols_].median()
            self.baselines_[op] = med.replace(0, np.nan)
        self.global_baseline_ = X[self.ratio_cols_].median().replace(0, np.nan)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X)[self.columns_].copy()
        if not self.enabled:
            return X.fillna(0.0)
        is_open = X["is_open"] > 0.5
        ratios = pd.DataFrame(index=X.index, columns=self.ratio_cols_, dtype=float)
        for op, mask in ((1.0, is_open), (0.0, ~is_open)):
            if not mask.any():
                continue
            base = self.baselines_[op].fillna(self.global_baseline_)
            ratios.loc[mask, :] = X.loc[mask, self.ratio_cols_].div(base, axis=1).to_numpy()
        ratios = ratios.add_prefix("rel_")
        out = pd.concat([X, ratios], axis=1)
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def get_feature_names_out(self, input_features=None):
        return np.array(list(self.columns_) + [f"rel_{c}" for c in self.ratio_cols_])


def _tree(clf):
    return Pipeline([("norm", OperationNormalizer()), ("clf", clf)])


def _scaled(clf):
    return Pipeline(
        [("norm", OperationNormalizer()), ("scale", StandardScaler()), ("clf", clf)]
    )


def candidate_models() -> dict[str, Pipeline]:
    """The benchmark set. All are class-weight aware - the data is 80/30 imbalanced."""
    return {
        "logistic_l2": _scaled(
            LogisticRegression(
                max_iter=5000, class_weight="balanced", C=1.0, random_state=RANDOM_STATE
            )
        ),
        "random_forest": _tree(
            RandomForestClassifier(
                n_estimators=600,
                min_samples_leaf=2,
                class_weight="balanced_subsample",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "extra_trees": _tree(
            ExtraTreesClassifier(
                n_estimators=600,
                min_samples_leaf=2,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        ),
        "gradient_boosting": _tree(
            GradientBoostingClassifier(
                n_estimators=300, learning_rate=0.05, max_depth=2, random_state=RANDOM_STATE
            )
        ),
        "hist_gradient_boosting": _tree(
            HistGradientBoostingClassifier(
                max_iter=300,
                learning_rate=0.05,
                max_depth=3,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
            )
        ),
    }


def build_anomaly_detector(contamination: float = 0.05) -> Pipeline:
    """Isolation Forest for fast-monitoring mode (fitted on normal cycles only)."""
    return Pipeline(
        [
            ("norm", OperationNormalizer()),
            ("scale", StandardScaler()),
            (
                "iso",
                IsolationForest(
                    n_estimators=400,
                    contamination=contamination,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
