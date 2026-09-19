"""
Rail Corrugation — Step 6: XGBoost Refinement

Purpose:
    Refine the best-performing model from Step 5.

We investigate:

    1. All features vs vibration-only features
    2. Feature selection
    3. Small XGBoost hyperparameter variations
    4. Repeated cross-validation
    5. Per-file error analysis

The goal is NOT to blindly search hundreds of combinations.

Instead, we make a small number of controlled experiments and
measure which changes actually improve Macro F1.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
)

from sklearn.metrics import (
    f1_score,
)

from sklearn.model_selection import (
    RepeatedStratifiedKFold,
)

from xgboost import (
    XGBClassifier,
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

N_SPLITS = 5

N_REPEATS = 5

CLASS_ORDER = [
    "Normal",
    "Side I",
    "Side II",
]

LABEL_TO_NUMBER = {
    "Normal": 0,
    "Side I": 1,
    "Side II": 2,
}

NUMBER_TO_LABEL = {
    0: "Normal",
    1: "Side I",
    2: "Side II",
}


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(
    features
):

    feature_columns = [
        c
        for c in features.columns
        if c not in {
            "filename",
            "label",
        }
    ]

    X = features[
        feature_columns
    ].copy()

    y = features[
        "label"
    ].copy()

    filenames = features[
        "filename"
    ].copy()

    return (
        X,
        y,
        filenames,
    )


# ============================================================
# XGBOOST MODEL
# ============================================================

def create_xgboost_model(
    max_depth=4,
    learning_rate=0.03,
    n_estimators=400,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
):
    """
    Create one XGBoost classifier.

    max_depth:
        Maximum depth of each decision tree.

        Larger = more complex model.

    learning_rate:
        How aggressively each new tree corrects previous
        mistakes.

        Smaller values learn more gradually.

    n_estimators:
        Number of trees.

    min_child_weight:
        Controls how easily XGBoost creates highly specific
        branches.

        Larger values make the model more conservative.

    subsample:
        Fraction of training recordings used by each tree.

    colsample_bytree:
        Fraction of features available to each tree.
    """

    return XGBClassifier(

        n_estimators=
            n_estimators,

        max_depth=
            max_depth,

        learning_rate=
            learning_rate,

        min_child_weight=
            min_child_weight,

        subsample=
            subsample,

        colsample_bytree=
            colsample_bytree,

        objective=
            "multi:softprob",

        num_class=3,

        eval_metric=
            "mlogloss",

        random_state=
            RANDOM_STATE,

        n_jobs=-1,
    )


# ============================================================
# FEATURE SUBSETS
# ============================================================

def get_feature_subset(
    X,
    subset
):
    """
    Select a broad feature family.

    Available:

        all
        vibration
        vibration_and_speed
    """

    if subset == "all":

        return list(
            X.columns
        )

    if subset == "vibration":

        return [
            c
            for c in X.columns
            if c.startswith(
                "vib_"
            )
        ]

    if subset == "vibration_and_speed":

        return [
            c
            for c in X.columns
            if (
                c.startswith(
                    "vib_"
                )
                or
                c.startswith(
                    "estimated_speed"
                )
                or
                c == "speed_transition_count"
            )
        ]

    raise ValueError(
        f"Unknown feature subset: "
        f"{subset}"
    )


# ============================================================
# EXPERIMENT DEFINITIONS
# ============================================================

def create_experiments():
    """
    Small, controlled experiment set.

    We intentionally avoid a massive hyperparameter search.
    """

    return [

        # ----------------------------------------------------
        # Baseline from Step 5
        # ----------------------------------------------------

        {
            "name":
                "XGB_all_baseline",

            "feature_subset":
                "all",

            "top_k":
                None,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        # ----------------------------------------------------
        # Vibration only
        # ----------------------------------------------------

        {
            "name":
                "XGB_vibration_only",

            "feature_subset":
                "vibration",

            "top_k":
                None,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        # ----------------------------------------------------
        # Vibration + speed
        # ----------------------------------------------------

        {
            "name":
                "XGB_vibration_speed",

            "feature_subset":
                "vibration_and_speed",

            "top_k":
                None,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        # ----------------------------------------------------
        # Feature selection
        # ----------------------------------------------------

        {
            "name":
                "XGB_top75",

            "feature_subset":
                "all",

            "top_k":
                75,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        {
            "name":
                "XGB_top100",

            "feature_subset":
                "all",

            "top_k":
                100,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        {
            "name":
                "XGB_top125",

            "feature_subset":
                "all",

            "top_k":
                125,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        {
            "name":
                "XGB_top150",

            "feature_subset":
                "all",

            "top_k":
                150,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        # ----------------------------------------------------
        # Shallower trees
        # ----------------------------------------------------

        {
            "name":
                "XGB_depth3",

            "feature_subset":
                "all",

            "top_k":
                None,

            "params": {
                "max_depth": 3,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        # ----------------------------------------------------
        # Faster learning
        # ----------------------------------------------------

        {
            "name":
                "XGB_lr005",

            "feature_subset":
                "all",

            "top_k":
                None,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.05,
                "n_estimators": 300,
                "min_child_weight": 1,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },

        # ----------------------------------------------------
        # More conservative tree splitting
        # ----------------------------------------------------

        {
            "name":
                "XGB_child3",

            "feature_subset":
                "all",

            "top_k":
                None,

            "params": {
                "max_depth": 4,
                "learning_rate": 0.03,
                "n_estimators": 400,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
        },
    ]


# ============================================================
# FEATURE SELECTION INSIDE TRAINING FOLD
# ============================================================

def select_features_for_fold(
    X_train,
    X_val,
    y_train,
    top_k,
):
    """
    Feature selection MUST happen using training data only.

    Otherwise information from the validation set could leak
    into model development.
    """

    if top_k is None:

        return (
            X_train,
            X_val,
            list(X_train.columns),
        )

    actual_k = min(
        top_k,
        X_train.shape[1]
    )

    selector = SelectKBest(
        score_func=
            mutual_info_classif,
        k=actual_k,
    )

    X_train_selected = (
        selector.fit_transform(
            X_train,
            y_train
        )
    )

    X_val_selected = (
        selector.transform(
            X_val
        )
    )

    selected_columns = (
        X_train.columns[
            selector.get_support()
        ].tolist()
    )

    return (
        X_train_selected,
        X_val_selected,
        selected_columns,
    )


# ============================================================
# EVALUATE ONE EXPERIMENT
# ============================================================

def evaluate_experiment(
    experiment,
    X,
    y,
    filenames,
    splits,
):
    """
    Evaluate one configuration over exactly the same repeated
    CV splits as every other experiment.

    Also record predictions for EACH FILE so we can identify
    repeatedly difficult recordings.
    """

    name = experiment[
        "name"
    ]

    print()
    print("=" * 70)
    print(name)
    print("=" * 70)

    columns = get_feature_subset(
        X,
        experiment[
            "feature_subset"
        ]
    )

    X_subset = X[
        columns
    ]

    fold_results = []
    prediction_results = []

    for fold_number, (
        train_index,
        val_index
    ) in enumerate(
        splits,
        start=1
    ):

        X_train = (
            X_subset.iloc[
                train_index
            ]
        )

        X_val = (
            X_subset.iloc[
                val_index
            ]
        )

        y_train_text = (
            y.iloc[
                train_index
            ]
        )

        y_val_text = (
            y.iloc[
                val_index
            ]
        )

        filenames_val = (
            filenames.iloc[
                val_index
            ]
        )

        y_train = (
            y_train_text.map(
                LABEL_TO_NUMBER
            )
        )

        # ----------------------------------------------------
        # Feature selection
        # ----------------------------------------------------

        (
            X_train_model,
            X_val_model,
            selected_columns,
        ) = select_features_for_fold(
            X_train,
            X_val,
            y_train,
            experiment[
                "top_k"
            ],
        )

        # ----------------------------------------------------
        # Model
        # ----------------------------------------------------

        model = (
            create_xgboost_model(
                **experiment[
                    "params"
                ]
            )
        )

        # ----------------------------------------------------
        # Sample weights
        # ----------------------------------------------------
        #
        # Same weighting strategy as our successful Step 5
        # XGBoost experiment.

        sample_weights = (
            y_train.map({
                0: 1.0,
                1: 8.0,
                2: 4.0,
            })
        )

        model.fit(
            X_train_model,
            y_train,
            sample_weight=
                sample_weights,
        )

        prediction_numbers = (
            model.predict(
                X_val_model
            )
        )

        predictions = pd.Series(
            prediction_numbers
        ).map(
            NUMBER_TO_LABEL
        ).to_numpy()

        y_true = (
            y_val_text
            .to_numpy()
        )

        # ----------------------------------------------------
        # Scores
        # ----------------------------------------------------

        macro_f1 = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0,
        )

        class_f1 = f1_score(
            y_true,
            predictions,
            labels=CLASS_ORDER,
            average=None,
            zero_division=0,
        )

        fold_results.append({

            "experiment":
                name,

            "fold":
                fold_number,

            "macro_f1":
                macro_f1,

            "normal_f1":
                class_f1[0],

            "side_I_f1":
                class_f1[1],

            "side_II_f1":
                class_f1[2],

            "n_features":
                len(
                    selected_columns
                ),
        })

        # ----------------------------------------------------
        # Record every validation prediction
        # ----------------------------------------------------

        for (
            filename,
            true_label,
            predicted_label
        ) in zip(
            filenames_val,
            y_true,
            predictions,
        ):

            prediction_results.append({

                "experiment":
                    name,

                "fold":
                    fold_number,

                "filename":
                    filename,

                "true_label":
                    true_label,

                "prediction":
                    predicted_label,

                "correct":
                    (
                        true_label
                        == predicted_label
                    ),
            })

        print(
            f"Fold {fold_number:2d}: "
            f"Macro={macro_f1:.3f} | "
            f"I={class_f1[1]:.3f} | "
            f"II={class_f1[2]:.3f}"
        )

    return (
        pd.DataFrame(
            fold_results
        ),
        pd.DataFrame(
            prediction_results
        ),
    )


# ============================================================
# SUMMARISE EXPERIMENTS
# ============================================================

def summarize_experiments(
    fold_results
):

    summary = (

        fold_results

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

            normal_f1_mean=(
                "normal_f1",
                "mean"
            ),

            side_I_f1_mean=(
                "side_I_f1",
                "mean"
            ),

            side_II_f1_mean=(
                "side_II_f1",
                "mean"
            ),

            n_features=(
                "n_features",
                "mean"
            ),
        )

        .reset_index()

        .sort_values(
            "macro_f1_mean",
            ascending=False
        )
    )

    return summary


# ============================================================
# PER-FILE ERROR ANALYSIS
# ============================================================

def create_file_error_analysis(
    predictions
):
    """
    Determine how often each recording was predicted correctly.

    With 5 repeats, each file appears in validation exactly
    5 times per experiment.

    Example:

        Train62.csv
        Side I

        correct 1 / 5

    tells us that this is a consistently difficult recording.
    """

    analysis = (

        predictions

        .groupby([
            "experiment",
            "filename",
            "true_label",
        ])

        .agg(

            times_tested=(
                "correct",
                "count"
            ),

            times_correct=(
                "correct",
                "sum"
            ),
        )

        .reset_index()
    )

    analysis[
        "accuracy"
    ] = (
        analysis[
            "times_correct"
        ]
        /
        analysis[
            "times_tested"
        ]
    )

    return analysis


# ============================================================
# STEP 6
# ============================================================

def run_step6(
    features,
    output_dir
):

    print()
    print("=" * 70)
    print(
        "STEP 6 — XGBOOST REFINEMENT"
    )
    print("=" * 70)

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

    (
        X,
        y,
        filenames,
    ) = prepare_data(
        features
    )

    print()
    print(
        f"Recordings: {len(X)}"
    )

    print(
        f"Available features: "
        f"{X.shape[1]}"
    )

    # --------------------------------------------------------
    # REPEATED CV
    # --------------------------------------------------------

    cv = RepeatedStratifiedKFold(

        n_splits=N_SPLITS,

        n_repeats=N_REPEATS,

        random_state=
            RANDOM_STATE,
    )

    # SAME splits for every experiment.

    splits = list(
        cv.split(
            X,
            y
        )
    )

    experiments = (
        create_experiments()
    )

    all_fold_results = []
    all_predictions = []

    # --------------------------------------------------------
    # RUN EXPERIMENTS
    # --------------------------------------------------------

    for experiment in experiments:

        (
            fold_results,
            predictions,
        ) = evaluate_experiment(

            experiment,
            X,
            y,
            filenames,
            splits,
        )

        all_fold_results.append(
            fold_results
        )

        all_predictions.append(
            predictions
        )

    # --------------------------------------------------------
    # COMBINE
    # --------------------------------------------------------

    fold_results = pd.concat(
        all_fold_results,
        ignore_index=True
    )

    predictions = pd.concat(
        all_predictions,
        ignore_index=True
    )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    summary = (
        summarize_experiments(
            fold_results
        )
    )

    # --------------------------------------------------------
    # FILE ERROR ANALYSIS
    # --------------------------------------------------------

    file_errors = (
        create_file_error_analysis(
            predictions
        )
    )

    # --------------------------------------------------------
    # SAVE EVERYTHING
    # --------------------------------------------------------

    fold_results.to_csv(

        output_dir
        / "xgb_fold_results.csv",

        index=False,
    )

    summary.to_csv(

        output_dir
        / "xgb_experiment_summary.csv",

        index=False,
    )

    predictions.to_csv(

        output_dir
        / "xgb_predictions_by_file.csv",

        index=False,
    )

    file_errors.to_csv(

        output_dir
        / "xgb_file_error_analysis.csv",

        index=False,
    )

    # --------------------------------------------------------
    # DISPLAY SUMMARY
    # --------------------------------------------------------

    print()
    print("=" * 100)
    print(
        "XGBOOST EXPERIMENT SUMMARY"
    )
    print("=" * 100)

    print(
        summary.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # DIFFICULT SIDE I FILES
    # --------------------------------------------------------

    best_experiment = (
        summary.iloc[0][
            "experiment"
        ]
    )

    difficult_side_I = (

        file_errors[
            (
                file_errors[
                    "experiment"
                ]
                == best_experiment
            )
            &
            (
                file_errors[
                    "true_label"
                ]
                == "Side I"
            )
        ]

        .sort_values(
            "accuracy"
        )
    )

    print()
    print("=" * 100)
    print(
        f"SIDE I ERROR ANALYSIS — "
        f"{best_experiment}"
    )
    print("=" * 100)

    print(
        difficult_side_I.to_string(
            index=False
        )
    )

    print()
    print(
        "Saved Step 6 outputs to:"
    )

    print(
        output_dir
    )

    print()
    print("=" * 70)
    print(
        "STEP 6 COMPLETE"
    )
    print("=" * 70)

    return (
        summary,
        file_errors,
    )