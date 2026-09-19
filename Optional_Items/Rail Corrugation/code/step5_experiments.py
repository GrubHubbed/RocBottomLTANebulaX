"""
Rail Corrugation — Step 5: Model Experiments

Purpose:
    Systematically test different modelling choices.

We compare:
    - Extra Trees
    - vibration-only Extra Trees
    - feature selection
    - different class weights
    - XGBoost

Evaluation:
    Repeated Stratified 5-Fold Cross-Validation.

IMPORTANT:
    Feature selection happens INSIDE each training fold to avoid
    data leakage.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.metrics import f1_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline

from xgboost import XGBClassifier


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

# Start with 5 repeats.
#
# That means:
#
# 5 folds × 5 repetitions = 25 model evaluations
#
# Later you can increase this to 10.
N_REPEATS = 5

CLASS_ORDER = [
    "Normal",
    "Side I",
    "Side II",
]


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(features):

    feature_columns = [
        c for c in features.columns
        if c not in {
            "filename",
            "label"
        }
    ]

    X = features[
        feature_columns
    ].copy()

    y = features[
        "label"
    ].copy()

    return X, y


# ============================================================
# MANUAL CROSS-VALIDATION
# ============================================================

def evaluate_experiment(
    name,
    model,
    X,
    y,
    cv,
):
    """
    Run one experiment over all cross-validation folds.

    We calculate F1 separately for:

        Normal
        Side I
        Side II

    plus Macro F1.

    We save EVERY fold's score so we can measure how stable
    the model is.
    """

    fold_results = []

    print()
    print("=" * 60)
    print(name)
    print("=" * 60)

    for fold_number, (
    train_index,
    val_index
    ) in enumerate(
        cv,
        start=1
    ):

        X_train = X.iloc[
            train_index
        ]

        X_val = X.iloc[
            val_index
        ]

        y_train = y.iloc[
            train_index
        ]

        y_val = y.iloc[
            val_index
        ]

        model.fit(
            X_train,
            y_train
        )

        predictions = model.predict(
            X_val
        )

        # ----------------------------------------------
        # Macro F1
        # ----------------------------------------------

        macro_f1 = f1_score(
            y_val,
            predictions,
            average="macro",
            zero_division=0,
        )

        # ----------------------------------------------
        # Individual classes
        # ----------------------------------------------

        class_scores = f1_score(
            y_val,
            predictions,
            labels=CLASS_ORDER,
            average=None,
            zero_division=0,
        )

        result = {

            "experiment": name,

            "fold": fold_number,

            "macro_f1": macro_f1,

            "normal_f1":
                class_scores[0],

            "side_I_f1":
                class_scores[1],

            "side_II_f1":
                class_scores[2],
        }

        fold_results.append(
            result
        )

        print(
            f"Fold {fold_number:2d}: "
            f"Macro={macro_f1:.3f} | "
            f"I={class_scores[1]:.3f} | "
            f"II={class_scores[2]:.3f}"
        )

    return pd.DataFrame(
        fold_results
    )


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

def create_experiments(
    X
):
    """
    Define all experiments in one place.

    Each experiment contains:

        model
        feature columns to use
    """

    all_columns = list(
        X.columns
    )

    # ----------------------------------------------
    # VIBRATION ONLY
    # ----------------------------------------------

    vibration_columns = [
        c
        for c in X.columns
        if c.startswith("vib_")
    ]

    experiments = []


    # ========================================================
    # EXPERIMENT 1
    # Baseline Extra Trees
    # ========================================================

    experiments.append({

        "name":
            "ExtraTrees_all_features",

        "columns":
            all_columns,

        "model":
            ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
    })


    # ========================================================
    # EXPERIMENT 2
    # Vibration only
    # ========================================================

    experiments.append({

        "name":
            "ExtraTrees_vibration_only",

        "columns":
            vibration_columns,

        "model":
            ExtraTreesClassifier(
                n_estimators=500,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
    })


    # ========================================================
    # EXPERIMENT 3
    # Stronger weighting for Side I
    # ========================================================

    experiments.append({

        "name":
            "ExtraTrees_sideI_weighted",

        "columns":
            all_columns,

        "model":
            ExtraTreesClassifier(

                n_estimators=500,

                class_weight={
                    "Normal": 1,
                    "Side I": 8,
                    "Side II": 4,
                },

                random_state=RANDOM_STATE,

                n_jobs=-1,
            )
    })


    # ========================================================
    # EXPERIMENT 4
    # Top 100 features
    # ========================================================

    experiments.append({

        "name":
            "ExtraTrees_top100",

        "columns":
            all_columns,

        "model":
            Pipeline([

                (
                    "feature_selection",

                    SelectKBest(
                        mutual_info_classif,
                        k=100
                    )
                ),

                (
                    "classifier",

                    ExtraTreesClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                )
            ])
    })


    # ========================================================
    # EXPERIMENT 5
    # Top 50 features
    # ========================================================

    experiments.append({

        "name":
            "ExtraTrees_top50",

        "columns":
            all_columns,

        "model":
            Pipeline([

                (
                    "feature_selection",

                    SelectKBest(
                        mutual_info_classif,
                        k=50
                    )
                ),

                (
                    "classifier",

                    ExtraTreesClassifier(
                        n_estimators=500,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )
                )
            ])
    })


    # ========================================================
    # EXPERIMENT 6
    # XGBoost
    # ========================================================

    # XGBoost expects numeric labels instead of:
    #
    # Normal
    # Side I
    # Side II
    #
    # We handle that separately later.

    return experiments


# ============================================================
# XGBOOST EVALUATION
# ============================================================

def evaluate_xgboost(
    X,
    y,
    cv
):
    """
    Evaluate XGBoost.

    XGBoost uses integer class labels:

        Normal  -> 0
        Side I  -> 1
        Side II -> 2
    """

    label_to_number = {
        "Normal": 0,
        "Side I": 1,
        "Side II": 2,
    }

    number_to_label = {
        0: "Normal",
        1: "Side I",
        2: "Side II",
    }

    y_numeric = y.map(
        label_to_number
    )

    results = []

    print()
    print("=" * 60)
    print("XGBoost_all_features")
    print("=" * 60)

    for fold_number, (
    train_index,
    val_index
    ) in enumerate(
        cv,
        start=1
    ):

        X_train = X.iloc[
            train_index
        ]

        X_val = X.iloc[
            val_index
        ]

        y_train = y_numeric.iloc[
            train_index
        ]

        y_val = y_numeric.iloc[
            val_index
        ]

        # ----------------------------------------------
        # Give more importance to rare classes
        # ----------------------------------------------

        weights = y_train.map({

            0: 1.0,   # Normal

            1: 8.0,   # Side I

            2: 4.0,   # Side II
        })

        model = XGBClassifier(

            n_estimators=400,

            max_depth=4,

            learning_rate=0.03,

            subsample=0.8,

            colsample_bytree=0.8,

            objective="multi:softprob",

            num_class=3,

            eval_metric="mlogloss",

            random_state=RANDOM_STATE,

            n_jobs=-1,
        )

        model.fit(
            X_train,
            y_train,
            sample_weight=weights,
        )

        predictions_numeric = (
            model.predict(
                X_val
            )
        )

        # Convert back into names.

        predictions = pd.Series(
            predictions_numeric
        ).map(
            number_to_label
        )

        y_true = pd.Series(
            y_val.values
        ).map(
            number_to_label
        )

        macro_f1 = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        class_scores = f1_score(
            y_true,
            predictions,
            labels=CLASS_ORDER,
            average=None,
            zero_division=0,
        )

        results.append({

            "experiment":
                "XGBoost_all_features",

            "fold":
                fold_number,

            "macro_f1":
                macro_f1,

            "normal_f1":
                class_scores[0],

            "side_I_f1":
                class_scores[1],

            "side_II_f1":
                class_scores[2],
        })

        print(
            f"Fold {fold_number:2d}: "
            f"Macro={macro_f1:.3f} | "
            f"I={class_scores[1]:.3f} | "
            f"II={class_scores[2]:.3f}"
        )

    return pd.DataFrame(
        results
    )


# ============================================================
# SUMMARIZE RESULTS
# ============================================================

def summarize_results(
    all_results
):
    """
    Turn all individual fold results into one easy-to-read
    comparison table.
    """

    summary = (

        all_results

        .groupby(
            "experiment"
        )

        .agg(

            macro_f1_mean=(
                "macro_f1",
                "mean"
            ),

            macro_f1_std=(
                "macro_f1",
                "std"
            ),

            side_I_f1_mean=(
                "side_I_f1",
                "mean"
            ),

            side_II_f1_mean=(
                "side_II_f1",
                "mean"
            ),

            normal_f1_mean=(
                "normal_f1",
                "mean"
            ),
        )

        .reset_index()
    )

    summary = summary.sort_values(
        "macro_f1_mean",
        ascending=False
    )

    return summary


# ============================================================
# STEP 5
# ============================================================

def run_step5(
    features,
    output_dir
):

    print()
    print("=" * 60)
    print("STEP 5 — MODEL EXPERIMENTS")
    print("=" * 60)

    output_dir = Path(
        output_dir
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    X, y = prepare_data(
        features
    )

    print(
        f"\nRecordings: {len(X)}"
    )

    print(
        f"Features: {X.shape[1]}"
    )

    # --------------------------------------------------------
    # SAME CV SPLITS FOR EVERY EXPERIMENT
    # --------------------------------------------------------

    cv = RepeatedStratifiedKFold(

        n_splits=N_SPLITS,

        n_repeats=N_REPEATS,

        random_state=RANDOM_STATE,
    )

    # Important:
    #
    # Convert the generator into an actual list.
    #
    # This guarantees EVERY experiment sees exactly the
    # same train/validation splits.

    splits = list(
        cv.split(X, y)
    )

    experiments = (
        create_experiments(X)
    )

    all_results = []

    # --------------------------------------------------------
    # EXTRA TREES EXPERIMENTS
    # --------------------------------------------------------

    for experiment in experiments:

        experiment_X = X[
            experiment["columns"]
        ]

        result = evaluate_experiment(

            name=experiment["name"],

            model=experiment["model"],

            X=experiment_X,

            y=y,

            cv=splits,
        )

        all_results.append(
            result
        )

    # --------------------------------------------------------
    # XGBOOST
    # --------------------------------------------------------

    xgb_results = (
        evaluate_xgboost(
            X,
            y,
            splits
        )
    )

    all_results.append(
        xgb_results
    )

    # --------------------------------------------------------
    # COMBINE RESULTS
    # --------------------------------------------------------

    all_results = pd.concat(
        all_results,
        ignore_index=True
    )

    detailed_path = (
        output_dir
        / "experiment_fold_results.csv"
    )

    all_results.to_csv(
        detailed_path,
        index=False
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = summarize_results(
        all_results
    )

    summary_path = (
        output_dir
        / "experiment_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    print()
    print("=" * 80)
    print("EXPERIMENT SUMMARY")
    print("=" * 80)

    print(
        summary.to_string(
            index=False
        )
    )

    print()
    print(
        f"Saved:\n{summary_path}"
    )

    print(
        f"\nDetailed fold results:\n"
        f"{detailed_path}"
    )

    print()
    print("=" * 60)
    print("STEP 5 COMPLETE")
    print("=" * 60)

    return summary